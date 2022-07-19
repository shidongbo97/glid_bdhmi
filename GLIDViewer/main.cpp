//#define EIGEN_USE_BLAS
//#define EIGEN_USE_MKL
//#define EIGEN_MKL_NO_DIRECT_CALL
//#define EIGEN_USE_MKL_ALL 
// for 64bit version
// link to mkl_intel_lp64.lib;mkl_intel_thread.lib;mkl_core.lib;libiomp5mt.lib
// not mkl_intel_ilp64.lib

#define FREEGLUT_STATIC
#include "gl_core_3_3.h"
#include <GL/glut.h>
#include <GL/freeglut_ext.h>

#define TW_STATIC
#include <AntTweakBar.h>


#include <ctime>
#include <memory>
#include <vector>
#include <string>
#include <cstdlib>

#include "glprogram.h"
#include "MyImage.h"
#include "VAOImage.h"
#include "VAOMesh.h"

#include "matlabDeformer.h"
#include "BDHMInterp.h"


using namespace std;

string textureFile;

std::shared_ptr<Deformer> deformer;
BDHMInterp* bdhmInterp;

GLProgram MyMesh::prog, MyMesh::pickProg, MyMesh::pointSetProg;
GLTexture MyMesh::colormapTex;

MyMesh M;
int viewport[4] = { 0, 0, 1280, 960 };
int actPrimType = MyMesh::PE_VERTEX;

int saveImageResolution = 3072;

bool showATB = true;
int deformerType = 0;
bool continuousDeform = true;
int numIterationPerDeform = 1;

void loadP2PConstraints();

std::vector<std::string> dataset_names;
int idataset = 0;
std::string datadir;
std::string meshName = "mesh.obj";
std::string p2pFileName = "p2p";

void saveMesh(const std::string &filename, bool srcMesh=false)
{
    using MapMatX2f = Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, 2,Eigen:: RowMajor> >;
    using MapMatX3i = Eigen::Map<const Eigen::Matrix<int, Eigen::Dynamic, 3, Eigen::RowMajor> >;

    if (srcMesh){
        matlabEval("x=X;");
    }
    else{
        eigen2matlab("x", MapMatX2f(M.mesh.X.data(), M.nVertex, 2).cast<double>());
    }
    eigen2matlab("t", MapMatX3i(M.mesh.T.data(), M.nFace,3).cast<double>());

    if (!M.mesh.UV.empty())
        eigen2matlab("uv", MapMatX2f(M.mesh.UV.data(), M.nVertex, 2).cast<double>());

    matlabEval( "writeOBJ('" +filename + "', struct('V', x, 'F', t+1, 'UV', uv));" );
    //matlabEval( "writeOBJ('" +filename + "', struct('V', [x uv(:,2)], 'F', t+1, 'UV', uv));" );
}

void resetDeform()
{
    Eigen::Matrix<float, Eigen::Dynamic, 2, Eigen::RowMajor> x;
    matlab2eigen("single(fC2R(X))", x, true);
    if (x.rows() == 0) return;
    M.upload(x.data(), x.rows(), nullptr, M.nFace, nullptr);

    if (deformer) deformer->resetDeform();
}

void createDeformer()
{
    deformer.reset();
    switch (deformerType)
    {
    case 0:
        deformer.reset(new MatlabDeformer(M));
        break;
   case 2:
        deformer.reset(new LIMDeformer(M));
        break;
   case 1:
        deformer.reset(new AKVFDeformer(M));
        break;
    default:
        break;
    }
}

void createBDHMInterp()
{
	bdhmInterp = new BDHMInterp(M);
}

void updatePosConstraints(int newConstraint)
{
    if (deformer)   deformer->updateP2PConstraints(newConstraint);
}

void deformMesh()
{
    if (deformer)   deformer->deform();
}

void loadP2PConstraints()
{
    if (!getMatEngine().hasVar("P2PVtxIds"))
        return;
    
    Eigen::MatrixXi p2pidx;
    matlab2eigen("int32(P2PVtxIds)-1", p2pidx, true);

    Eigen::Matrix<float, Eigen::Dynamic, 2, Eigen::RowMajor> p2pdst;
    matlab2eigen("single(fC2R(P2PCurrentPositions))", p2pdst, true);

    M.setConstraintVertices(p2pidx.data(), p2pdst.data(), p2pidx.size());


    // load anchors from matlab if it's there
    if (getMatEngine().hasVar("interpAnchID")) 
        M.auxVtxIdxs=matlab2vector<int>("int32(interpAnchID)-1", true);

    updatePosConstraints(1);
}

void loadData(std::string dataset)
{
    double numMeshVertex = matlab2scalar("numMeshVertex", 1e4);

    matlabEval("clear;");
    scalar2matlab("numMeshVertex", numMeshVertex);
    string2matlab("working_dataset", dataset);
    matlabEval("deformer_main;");   ///////

    dataset = matlab2string("working_dataset");
    for (int i = 0; i < dataset_names.size(); i++) if (dataset_names[i] == dataset) { idataset = i; break; }

    string2matlab("working_dataset", dataset);

    datadir = matlab2string("datadir");


    using namespace Eigen;
    Matrix<float, Dynamic, 2, RowMajor> x, uv;
    Matrix<int, Dynamic, 3, RowMajor> t;

    matlabEval("assert(~isreal(X) && ~isreal(uv));");
    matlab2eigen("single(fC2R(X))", x, true);
    matlab2eigen("single(fC2R(uv))", uv, true);
    matlab2eigen("int32(T)-1", t, true);

    if (x.rows() == 0 || t.rows() == 0) return;

    textureFile = matlab2string("imgfilepath");
    M.tex.setImage(textureFile);
    M.tex.setClamping(GL_CLAMP_TO_EDGE);

    M.mTextureScale = 1.f;
    if (getMatEngine().hasVar("textureScale"))
        M.mTextureScale = float(matlab2scalar("textureScale"));

    if (getMatEngine().hasVar("textureClampingFlag")) {
        int i = matlab2scalar("textureClampingFlag");
        if(i==1)  M.tex.setClamping(GL_REPEAT);
        else if(i==2)  M.tex.setClamping(GL_MIRRORED_REPEAT);
    }

    M.showTexture = true;
    M.edgeWidth = 0;

    M.upload(x, t, uv.data());

    M.updateBBox();

    createDeformer();

    loadP2PConstraints();

	createBDHMInterp();
}

/////////////////////////////////////////////////////////////////////////
void dumpGLInfo(bool dumpExtensions=false) 
{
	GLint major, minor;
    glGetIntegerv(GL_MAJOR_VERSION, &major);
    glGetIntegerv(GL_MINOR_VERSION, &minor);

    printf("GL Vendor    : %s\n", glGetString( GL_VENDOR ));
    printf("GL Renderer  : %s\n", glGetString( GL_RENDERER ));
    printf("GL Version   : %s\n", glGetString( GL_VERSION ));
    printf("GL Version   : %d.%d\n", major, minor);
    printf("GLSL Version : %s\n", glGetString( GL_SHADING_LANGUAGE_VERSION ));

    if( dumpExtensions ) {
        GLint nExtensions;
        glGetIntegerv(GL_NUM_EXTENSIONS, &nExtensions);
        for( int i = 0; i < nExtensions; i++ ) {
            printf("%s\n", glGetStringi(GL_EXTENSIONS, i));
        }
    }
}

int mousePressButton;
int mouseButtonDown;
int mousePos[2];

float fps = 0;
bool msaa = true;

void display()
{
	static clock_t lastTime = 0;
	static int numRenderedFrames = 0;
	const int numFramesPerReport = 100;
	if( (numRenderedFrames+1) % numFramesPerReport  == 0 ) {
		clock_t now = clock();
		float dual = (now - lastTime) / float(CLOCKS_PER_SEC);
        fps = numFramesPerReport / dual;
		//printf("%d frames in %f sec (%f fps)\n", numFramesPerReport, dual, fps);
		lastTime = now;
	}

    glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);

    if (msaa) glEnable(GL_MULTISAMPLE);
    else glDisable(GL_MULTISAMPLE);

    glViewport(0, 0, viewport[2], viewport[3]);
	M.draw(viewport);

    if(showATB) TwDraw();
	glutSwapBuffers();

	//glFinish();
	numRenderedFrames++;
}

void onKeyboard(unsigned char code, int x, int y)
{
    if (!TwEventKeyboardGLUT(code, x, y)){
        //if (code == 'q' && (GLUT_ACTIVE_CTRL & glutGetModifiers())){
        switch (code){
        case 17:
            exit(0);
        case 'f':
            glutFullScreenToggle();
            break;

        case 'd':
            for (int i = 0; i < numIterationPerDeform; i++){
                deformMesh();
                display();
            }
            break;
        case ' ':
            showATB = !showATB;
            //if (showATB)   TwDefine(" glvu visible=true ");
            break;
        }
    }

    glutPostRedisplay();
}

void onMouseButton(int button, int updown, int x, int y)
{ 
    if (!showATB || !TwEventMouseButtonGLUT(button, updown, x, y)){
        mousePressButton = button;
        mouseButtonDown = updown;

        if (updown == GLUT_DOWN){
            if (button == GLUT_LEFT_BUTTON){
                if (glutGetModifiers()&GLUT_ACTIVE_CTRL){
                    M.pick(x, y, viewport, actPrimType, M.PO_NONE);

                    if (M.actVertex >= 0){
                        if (M.auxVtxIdxs.size() > 1) M.auxVtxIdxs.erase(M.auxVtxIdxs.begin()+1, M.auxVtxIdxs.end());
                        M.auxVtxIdxs.insert(M.auxVtxIdxs.begin(), M.actVertex);
                    }
                }
                else{
                    int r = M.pick(x, y, viewport, M.PE_VERTEX, M.PO_ADD);
                    updatePosConstraints(r);
                    //if (continuousDeform) deformMesh(); // may waste computation
                }
            }
            else if(button == GLUT_RIGHT_BUTTON) {
                M.pick(x, y, viewport, M.PE_VERTEX, M.PO_REMOVE);
                updatePosConstraints(-1);  // one vertex removed

                for (int i = 0; i < numIterationPerDeform; i++)
                    deformMesh();
            }
        }
        else{ // updown == GLUT_UP
            if (!continuousDeform && button == GLUT_LEFT_BUTTON)
                updatePosConstraints(0);
        }

        mousePos[0] = x;
        mousePos[1] = y;
    }

    glutPostRedisplay();
}


void onMouseMove(int x, int y)
{
    if (!showATB || !TwEventMouseMotionGLUT(x, y)){
        if (mouseButtonDown == GLUT_DOWN){
            if (mousePressButton == GLUT_MIDDLE_BUTTON){
                float ss = M.drawscale();
                M.mTranslate[0] += (x - mousePos[0]) * 2 / ss / viewport[2];
                M.mTranslate[1] -= (y - mousePos[1]) * 2 / ss / viewport[3];
            }
            else if(mousePressButton == GLUT_LEFT_BUTTON){
                M.moveCurrentVertex(x, y, viewport);
                if (continuousDeform){
                    updatePosConstraints(0);
                    deformMesh();
                    display();
                }
            }
        }
    }

    mousePos[0] = x; mousePos[1] = y;

    glutPostRedisplay();
}


void onMouseWheel(int wheel_number, int direction, int x, int y)
{
    if (glutGetModifiers() & GLUT_ACTIVE_CTRL){
    }
    else
        M.mMeshScale *= direction > 0 ? 1.1f : 0.9f;

    glutPostRedisplay();
}

int initGL(int argc, char **argv) 
{
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_MULTISAMPLE);
    glutInitWindowSize(960, 960);
    glutInitWindowPosition( 200, 50 );
    glutCreateWindow(argv[0]);

    // !Load the OpenGL functions. after the opengl context has been created
    if( ogl_LoadFunctions() == ogl_LOAD_FAILED ) 
        return -1;

    glClearColor(1.f, 1.f, 1.f, 0.f);

	glutReshapeFunc([](int w, int h){ viewport[2] = w; viewport[3] = h; TwWindowSize(w, h);});
    glutDisplayFunc(display);
    glutKeyboardFunc(onKeyboard);
    glutMouseFunc( onMouseButton );
    glutMotionFunc( onMouseMove );
    glutMouseWheelFunc( onMouseWheel );
    glutCloseFunc([]() {exit(0); });
    return 0;
}


void createTweakbar()
{
    TwBar *bar = TwGetBarByName("glvu");
    if (bar)    TwDeleteBar(bar);

     //Create a tweak bar
    bar = TwNewBar("glvu");
    TwDefine(" GLOBAL help='This is mesh viewer based on OpenGL(GLSL)' "); // Message added to the help bar.
    TwDefine(" glvu size='220 320' color='0 128 255' text=dark alpha=128 position='5 5'"); // change default tweak bar size and color

    TwAddVarRO(bar, "#Vertex", TW_TYPE_INT32, &M.nVertex, " group=Mesh ");
    TwAddVarRO(bar, "#Face", TW_TYPE_INT32, &M.nFace, " group=Mesh ");
    TwAddVarRW(bar, "Texture Scale", TW_TYPE_FLOAT, &M.mTextureScale, " group=Mesh min=0 step=0.02");

    TwEnumVal textureClampings[] = { { GL_CLAMP_TO_EDGE, "To Edge" }, { GL_CLAMP_TO_BORDER, "To Border" }, { GL_REPEAT, "Repeat" }, { GL_MIRRORED_REPEAT, "Mirrored Repeat" } };
    TwType texClamp = TwDefineEnum("Texture Clamping", textureClampings, sizeof(textureClampings) / sizeof(textureClampings[0]));
    TwAddVarCB(bar, "Texture Clamping", texClamp,
        [](const void *v, void *) { M.tex.setClamping(*(unsigned int*)(v)); },
        [](void *v, void *)       { *(unsigned int*)(v) = M.tex.clamping(); },
        nullptr, " group=Mesh help='Set texture clamping' ");


    TwAddVarRW(bar, "Point Size", TW_TYPE_FLOAT, &M.pointSize, " group='Mesh View' ");
    TwAddVarRW(bar, "Edge Width", TW_TYPE_FLOAT, &M.edgeWidth, " group='Mesh View' ");
    TwAddVarRW(bar, "Depth Mode", TW_TYPE_INT32, &M.depthMode, " group='Mesh View' min=0 max=3 keyincr=v keydecr=V");

    TwAddVarRW(bar, "Vertex Color", TW_TYPE_COLOR4F, M.vertexColor.data(), " group='Mesh View' help='mesh vertex color' ");
    TwAddVarRW(bar, "Edge Color", TW_TYPE_COLOR4F, M.edgeColor.data(), " group='Mesh View' help='mesh edge color' ");
    TwAddVarRW(bar, "Face Color", TW_TYPE_COLOR4F, M.faceColor.data(), " group='Mesh View' help='mesh face color' ");

    TwAddVarRW(bar, "aux Point Size", TW_TYPE_FLOAT, &M.auxPointSize, " group='Mesh View' ");
    TwAddVarRW(bar, "TargetP2P", TW_TYPE_BOOLCPP, &M.drawTargetP2P, " group='Mesh View' ");

    TwAddVarCB(bar, "Use Texture", TW_TYPE_BOOLCPP, 
        [](const void *v, void *) { M.showTexture = *(bool*)(v); if (M.showTexture){ M.tex.setImage( MyImage(textureFile) ); } },
        [](void *v, void *)       { *(bool*)(v) = M.showTexture; },
        nullptr, " group='Mesh View' help='show texture on mesh' ");


    //////////////////////////////////////////////////////////////////////////
    TwAddSeparator(bar, " Deformer ", " ");
    TwType DeformerType = TwDefineEnumFromString("Deformer", "P2P harmonic, AKVF, LIM");

    TwAddVarCB(bar, "Deformer", DeformerType,
        [](const void *v, void *) {  deformerType = *(unsigned int*)v;  createDeformer(); },
        [](void *v, void *)       { *(unsigned int*)(v) = deformerType; },
        nullptr, " ");

    TwAddVarRW(bar, "Continuous deform", TW_TYPE_BOOLCPP, &continuousDeform, " key=c ");

    TwAddVarCB(bar, "Iteration", TW_TYPE_BOOLCPP, 
        [](const void *v, void *) {  if (deformer) deformer->needIteration = *(bool*)(v); },
        [](void *v, void *)       { *(bool*)(v) = deformer?deformer->needIteration:false; },
        nullptr, " key=i ");

    //TwAddButton(bar, "Load P2P", [](void *d){ loadP2PConstraints(); }, nullptr, " ");

    TwAddVarCB(bar, "iP2P", TW_TYPE_INT32,
        [](const void *v, void *d) {
        if (!matEngineConnected()) return;
        scalar2matlab("iP2P", *(const int*)(v)+1);
        matlabEval("deformer_updateP2P");
        loadP2PConstraints();
    },
        [](void *v, void *) { if (matEngineConnected()) *(int*)(v) = matlab2scalar("iP2P", 1)-1; },
        nullptr, " min=0 max=1000 step=1 keyincr=. keydecr=, ");

    TwAddButton(bar, "set P2P", [](void *d) { matlabEval("deformer_setP2P;"); }, nullptr, " key=s ");
    TwAddButton(bar, "add P2P", [](void *d) { matlabEval("deformer_addP2P;"); }, nullptr, " key=a ");
    TwAddButton(bar, "del P2P", [](void *d) { matlabEval("deformer_delP2P;"); }, nullptr, " key=D ");

    TwAddButton(bar, "Clear P2P", [](void *d){ M.constrainVertices.clear(); M.actConstrainVertex = -1; updatePosConstraints(-1); }, nullptr, " ");

    TwAddSeparator(bar, "LoadSave", " ");


    TwType datasetTWType = TwDefineEnumFromString("Datasets", catStr(dataset_names).c_str());
    TwAddVarCB(bar, "Shape", datasetTWType,
        [](const void *v, void *d) {
        idataset = *(int*)v;
        if (idataset < dataset_names.size()) loadData(dataset_names[idataset]);
    },
        [](void *v, void *) { *(int*)v = idataset; },
        nullptr, " ");


    TwAddVarCB(bar, "Mesh Resolution", TW_TYPE_INT32,
        [](const void *v, void *d) { scalar2matlab("numMeshVertex", *(const int*)(v)); },
        [](void *v, void *) { *(int*)(v) = matlab2scalar("numMeshVertex"); },
        nullptr, " min=3 ");

    TwAddButton(bar, "Reset", [](void *d){ resetDeform(); }, nullptr, " key=R ");

    TwAddVarRW(bar, "Image Save Resolution", TW_TYPE_INT32, &saveImageResolution, " min=1024 max=1000000 ");
    TwAddButton(bar, "Save Mesh/Image", [](void *d) {
        std::string meshfilebasename = meshName;

        if (meshName.size() > 4 && !strcmp(meshName.data() + meshName.size() - 4, ".obj"))
            meshfilebasename = meshName.substr(0, meshName.size() - 4);

        std::string imgfile = datadir + (deformer ? deformer->name() : meshfilebasename) + "_p2p"+std::to_string( int(matlab2scalar("iP2P", 1) - 1) ) + ".png";

        //M.updateBBox(); // TODO: do not update bbox so that the scale is the same for different algs
        M.saveResultImage(imgfile.c_str(), saveImageResolution); },
        nullptr, " key=p ");

    TwAddButton(bar, "Save data", [](void *d){
        if (deformer){
            deformer->saveData();
            //std::string objfile = datadir + deformer->name() + "_src.obj";
            ////fprintf(stdout, "saving source mesh to %s\n", objfile.c_str());
            //saveMesh(objfile, true);
            //objfile = datadir + deformer->name() + ".obj";
            ////fprintf(stdout, "saving result mesh to %s\n", objfile.c_str());
            //saveMesh(objfile, false);
        }

        //const std::string filename = getFileName(textureFile);
        //const size_t szFileName = filename.length();
        //const char *extname = (szFileName > 4 && filename[szFileName - 4] == '.') ? filename.c_str() + szFileName - 4 : ".png";
        //std::string imgfile(filename.data(), filename.data() + szFileName - 4);
        //imgfile += std::string("_result") + extname;
        },
        nullptr, " key=S ");

    TwDefine(" glvu/Mesh opened=false ");
    TwDefine(" glvu/'Mesh View' opened=false ");
}

int main(int argc, char *argv[])
{
    SetConsoleScreenBufferSize(GetStdHandle(STD_OUTPUT_HANDLE), { 100, 5000 });

    if (initGL(argc, argv)) {
        fprintf(stderr, "!Failed to initialize OpenGL!Exit...");
        exit(-1);
    }
    dumpGLInfo();


    MyMesh::buildShaders();

    float x[] = { -1, -1, 1, -1, -1, 1, 1, 1 };
    float uv[] = { 0, 0, 1, 0, 0, 1, 1, 1 };
    int t[] = { 0, 1, 2, 2, 1, 3 };
    M.upload(x, 4, t, 2, uv);

    //////////////////////////////////////////////////////////////////////////
    TwInit(TW_OPENGL_CORE, NULL);
    //Send 'glutGetModifers' function pointer to AntTweakBar;
    //required because the GLUT key event functions do not report key modifiers states.
    TwGLUTModifiersFunc(glutGetModifiers);
    glutSpecialFunc([](int key, int x, int y) { TwEventSpecialGLUT(key, x, y); glutPostRedisplay(); }); // important for special keys like UP/DOWN/LEFT/RIGHT ...
    TwCopyStdStringToClientFunc([](std::string& dst, const std::string& src) {dst = src; });


    //////////////////////////////////////////////////////////////////////////
    atexit([] { deformer.reset(); TwDeleteAllBars();  TwTerminate(); glutExit(); });  // Called after glutMainLoop ends

    //glutIdleFunc([]() { if (matEngineConnected()) meshDeformInMatlab(M);  glutPostRedisplay(); });
    glutIdleFunc([]() { if (deformer && deformer->needIteration && !deformer->converged()) deformer->deform();  glutPostRedisplay(); });


    glutTimerFunc(1000, [](int) {
        getMatEngine().connect("");
        matlabEval("list_datasets;");
        dataset_names = matlab2strings("datasets");
        createTweakbar();
        loadData("");
    },
        0);


    //////////////////////////////////////////////////////////////////////////
    glutMainLoop();

    return 0;
}


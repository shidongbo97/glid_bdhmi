#pragma once

#include "deformer.h"
#include "matlab_utils.h"
#include <thread>
#include <chrono>


int numInterpFrames = 200;

extern bool showATB;
extern int viewport[4];
void display();
void loadP2PConstraints();

inline std::string catStr(const std::vector<std::string> &names)
{
    std::string str;
    for (int i = 0; i < names.size(); i++) {
        str += names[i];
        if (i < names.size() - 1) str += ", ";
    }
    return str;
}

struct MatlabDeformer : public Deformer
{
    std::vector<std::string> solver_names;    //const char *[] = { "AQP", "cuAQP single", "cuAQP double", "CVX", "Direct Mosek" };
    std::vector<std::string> energy_names;    //const char *[] = { 'ARAP', 'BARAP', 'ISO', 'EISO', 'AMIPS', 'BETA'};
    float cage_offset;
    int nVirtualVertex;
    int nFixedSample;
    int nDenseEvaluationSample;
    int nActSetPoolSample;
    float p2p_weight;
    float sigma1_upper_bound;
    float sigma2_lower_bound;
    float k_upper_bound;
    bool solver_output;

    bool binarySearchValidMap;
    float timet;

	unsigned int interpAlgorithm;

    float AQP_kappa;
    bool softP2P;
    int AQP_IterPerUIUpdate;

    bool initiated;

    Eigen::MatrixXcd C; // Cauchy Coordiantes for vertices

    int solver = 0;
    int energy_type = 0;

    MyMesh &M;

    MatlabDeformer(MatlabDeformer&) = delete;

    MatlabDeformer(MyMesh &m) :M(m), initiated(false),
        solver(0), cage_offset(2e-2f), 
        nVirtualVertex(1), nFixedSample(1), nActSetPoolSample(2000), 
        p2p_weight(100000.f), sigma1_upper_bound(7.f), sigma2_lower_bound(0.35f), k_upper_bound(0.99f),
        AQP_kappa(1), softP2P(true), AQP_IterPerUIUpdate(10),
		solver_output(false), binarySearchValidMap(true), timet(0), 
        interpAlgorithm(0) {

        using deformerptr = MatlabDeformer*;

        TwBar *bar = TwNewBar("MatlabDeformer");

        TwDefine(" MatlabDeformer size='220 380' color='255 0 255' text=dark alpha=128 position='5 380' label='BDH Deformer'"); // change default tweak bar size and color
        //TwAddVarRW(bar, "k", TW_TYPE_FLOAT, &k4bdh, " min=0 max=1 step=0.1 ");

        //////////////////////////////////////////////////////////////////////////

        TwAddVarCB(bar, "cage offset", TW_TYPE_FLOAT,
            [](const void *v, void *d) {  deformerptr(d)->cage_offset = *(const float*)(v); deformerptr(d)->preprocess(); deformerptr(d)->updateP2PConstraints(0);},
            [](void *v, void *d)       { *(float*)(v) = deformerptr(d)->cage_offset; },
            this, " ");

        TwAddVarCB(bar, "numVirtualVertex", TW_TYPE_INT32,
            [](const void *v, void *d) { deformerptr(d)->nVirtualVertex = *(const int*)(v); deformerptr(d)->preprocess(); deformerptr(d)->updateP2PConstraints(0);},
            [](void *v, void *d)       { *(int*)(v) = deformerptr(d)->nVirtualVertex; },
            this, " ");

        k_upper_bound = matlab2scalar("k_upper_bound");

        solver_names = matlab2strings("harmonic_map_solvers");
        std::string defaultsolver = matlab2string("default_harmonic_map_solver");
        for (int i = 0; i < solver_names.size(); i++) if (defaultsolver == solver_names[i]) solver = i;

        energy_names = matlab2strings("harmonic_map_energies");
        std::string defaultenergy = matlab2string("default_harmonic_map_energy");
        for (int i = 0; i < energy_names.size(); i++) if (defaultenergy == energy_names[i]) energy_type = i;

        TwType energyType = TwDefineEnumFromString("Energy", catStr(energy_names).c_str());
        TwAddVarRW(bar, "Energy", energyType, &energy_type, " ");
 
        TwType solverType = TwDefineEnumFromString("Solver", catStr(solver_names).c_str());
        TwAddVarRW(bar, "Solver", solverType, &solver, " ");


        TwAddButton(bar, "Reset View", [](void *d) {
            deformerptr(d)->M.updateBBox();
            deformerptr(d)->M.mMeshScale = 1.f;
            deformerptr(d)->M.mTranslate.assign(0.f);
        }, this, " ");


        TwAddButton(bar, "Reset Shape", [](void *d) {
            deformerptr(d)->resetDeform(); 
            matlabEval("NLO_preprocessed = false; P2P_Deformation_Converged = 0;"); }, this, " key=r ");

        TwAddVarCB(bar, "vis distortion", TW_TYPE_BOOLCPP, 
            [](const void *v, void *d) { scalar2matlab("update_distortions_plots", *(const bool*)(v)); },
            [](void *v, void *)       { *(bool*)(v) = matlab2scalar("update_distortions_plots"); },
            nullptr, " ");

        //////////////////////////////////////////////////////////////////////////

        //TwAddVarRW(bar, "Solver Output", TW_TYPE_BOOLCPP, &solver_output, " group=Parameters ");
        //////////////////////////////////////////////////////////////////////////
        TwAddVarCB(bar, "P2P weight", TW_TYPE_FLOAT, 
            [](const void *v, void *) { scalar2matlab("p2p_weight", *(const float*)(v)); },
            [](void *v, void *) { *(float*)(v) = matlab2scalar("p2p_weight"); },
            nullptr, " min=0 ");

        TwAddVarCB(bar, "#Samples", TW_TYPE_INT32, 
            [](const void *v, void *d) { scalar2matlab("numEnergySamples", *(const int*)(v)); 
                matlabEval("p2p_harmonic_prep;");
                },
            [](void *v, void *) { *(int*)(v) = matlab2scalar("numEnergySamples"); },
            nullptr, " group=Parameters min=100 ");

        TwAddVarCB(bar, "hessianSample%", TW_TYPE_FLOAT, 
            [](const void *v, void *d) { scalar2matlab("hessianSampleRate", *(const float*)(v)/100); },
            [](void *v, void *)       { *(float*)(v) = matlab2scalar("hessianSampleRate")*100; },
            nullptr, " group=Parameters min=0.0001 max=100 help='sub sample for faster hessian assembly' ");

        TwAddVarCB(bar, "kappa", TW_TYPE_FLOAT, 
            [](const void *v, void *d) { scalar2matlab("AQP_kappa", *(const float*)(v)); },
            [](void *v, void *)       { *(float*)(v) = matlab2scalar("AQP_kappa"); },
            nullptr, " group=Parameters min=1 help='kappa=1 means no acceleration' ");

        TwAddVarCB(bar, "softP2P", TW_TYPE_BOOLCPP, 
            [](const void *v, void *d) { scalar2matlab("softP2P", *(const bool*)(v)); },
            [](void *v, void *)       { *(bool*)(v) = matlab2scalar("softP2P"); },
            nullptr, " group=Parameters ");

        TwAddVarCB(bar, "numIter/Update", TW_TYPE_INT32, 
            [](const void *v, void *d) { scalar2matlab("numIterations", *(const int*)(v)); },
            [](void *v, void *)       { *(int*)(v) = matlab2scalar("numIterations"); },
            nullptr, " group=Parameters min=1 help='#iterations before updating p2p deformer result' ");

        TwAddVarCB(bar, "energy param", TW_TYPE_FLOAT, 
            [](const void *v, void *d) { scalar2matlab("energy_parameter", *(const float*)(v)); },
            [](void *v, void *)       { *(float*)(v) = matlab2scalar("energy_parameter"); },
            nullptr, " group=Parameters min=0 help='change the parameter for some energies, e.g. AMIPS, BARAP, power iso' ");




        //////////////////////////////////////////////////////////////////////////
        TwAddVarCB(bar, "numFixedSample", TW_TYPE_INT32,
            [](const void *v, void *d) { deformerptr(d)->nFixedSample = *(const int*)(v); deformerptr(d)->preprocess(); },
            [](void *v, void *d)       { *(int*)(v) = deformerptr(d)->nFixedSample; },
            this, " group=BDHM ");

        TwAddVarCB(bar, "numDenseEvaluationSample", TW_TYPE_INT32,
            [](const void *v, void *d) { deformerptr(d)->nDenseEvaluationSample = *(const int*)(v); deformerptr(d)->preprocess(); },
            [](void *v, void *d)       { *(int*)(v) = deformerptr(d)->nDenseEvaluationSample; },
            this, " group=BDHM ");

        TwAddVarCB(bar, "numActiveSetPoolSample", TW_TYPE_INT32,
            [](const void *v, void *d) { deformerptr(d)->nActSetPoolSample = *(const int*)(v);  deformerptr(d)->preprocess(); },
            [](void *v, void *d)       { *(int*)(v) = deformerptr(d)->nActSetPoolSample; },
            this, " group=BDHM ");

        TwAddVarRW(bar, "Validate Map", TW_TYPE_BOOLCPP, &binarySearchValidMap, " group=BDHM ");
        TwAddVarRW(bar, "Sigma1", TW_TYPE_FLOAT, &sigma1_upper_bound, " group=BDHM ");
        TwAddVarRW(bar, "Sigma2", TW_TYPE_FLOAT, &sigma2_lower_bound, " group=BDHM ");
        TwAddVarRW(bar, "k", TW_TYPE_FLOAT, &k_upper_bound, " group=BDHM ");

        //////////////////////////////////////////////////////////////////////////
        //TwAddSeparator(bar, " Interpolation ", " ");

        TwAddButton(bar, "Add Keyframe", [](void *){
			matlabEval("PhiPsyKF(:,end+(1:2)) = [Phi Psy];"); }, nullptr, " group=Interpolator ");

        TwAddButton(bar, "Set as Keyframe", [](void *){
			matlabEval("PhiPsyKF(:,ikeyframe*2+(-1:0)) = [Phi Psy];"); }, nullptr, " group=Interpolator ");

        TwAddButton(bar, "Save all Keyframes", [](void *){
			matlabEval("save([datadir 'PhiPsyKF'], 'PhiPsyKF');"); }, nullptr, " group=Interpolator ");

        TwAddVarCB(bar, "View Keyframe", TW_TYPE_INT32,
			[](const void *v, void *d) {
	        scalar2matlab("ikeyframe", *(const int*)(v)); 
			matlabEval("Phi = PhiPsyKF(:, ikeyframe*2-1);  Psy = PhiPsyKF(:, ikeyframe*2); rot_trans=[1; 0];");
			deformerptr(d)->deformResultFromMaltab("PhiPsy");

			matlabEval("P2PCurrentPositions = C(P2PVtxIds, :)*Phi + conj(C(P2PVtxIds, :)*Psy);");
			//loadP2PConstraints();
		},
            [](void *v, void *)       { *(int*)(v) = matlab2scalar("ikeyframe"); },
            this, " min=1 max=1000 step=1 group=Interpolator ");


        TwAddVarCB(bar, "t", TW_TYPE_FLOAT,
            [](const void *v, void *d) {
            deformerptr(d)->timet = *(const float*)(v);
			deformerptr(d)->interpAtTime();
        },
            [](void *v, void *d)       { *(float*)(v) = deformerptr(d)->timet; },
            this, " min=-0.1 max=1.1 step=0.002 keyincr=RIGHT keydecr=LEFT group=Interpolator ");


		TwType InterpAlg = TwDefineEnumFromString("InterpAlgorithm", "BDH/metric, BDH/eta, BDH/nu, SIG13, BDH_GM, ARAP, FFMP");
		TwAddVarCB(bar, "InterpAlg", InterpAlg, 
            [](const void *v, void *d) {
			deformerptr(d)->interpAlgorithm = *(unsigned int*)(v);
			deformerptr(d)->interpAtTime();
        },
            [](void *v, void *d)       { *(unsigned int*)(v) = deformerptr(d)->interpAlgorithm; },
			
			this, " group=Interpolator ");


        TwAddVarRW(bar, "numFrame", TW_TYPE_INT32, &numInterpFrames, " group=Interpolator ");


        TwAddButton(bar, "Generate sequence", [](void *d){
            auto * pthis = deformerptr(d);

            vector2matlab("interpAnchID", pthis->M.auxVtxIdxs);
            matlabEval("interpAnchID= interpAnchID+1;"); 
            matlabEval("bdhInterpPrep");

            auto &M = pthis->M;
            auto constrainVertices0 = M.constrainVertices;
            auto constrainVerticesRef0 = M.constrainVerticesRef;
            auto auxVtxIdxs0 = M.auxVtxIdxs;
            M.auxVtxIdxs.clear();
            M.constrainVertices.clear();
            M.constrainVerticesRef.clear();


            bool showATB0 = showATB;
            float edgeWidth0 = M.edgeWidth;
            //float meshScale0 = M.mMeshScale;
            //auto translate = M.mTranslate;

           
            {

                int vw = viewport[2]/8*8, vh = viewport[3];
                std::vector<unsigned char> pixels(vw*vh * 3);


                for (int i = 0; i < numInterpFrames; i++) {
                    pthis->timet = i / (numInterpFrames - 1.f);
                    pthis->interpAtTime();
                    //std::string imgfile = datadir + "morph_" + std::to_string(i) + ".jpg";

                    display();

                    glReadPixels(viewport[0], viewport[1], vw, vh, GL_BGR, GL_UNSIGNED_BYTE, &pixels[0]);
                }
            }


            M.constrainVertices = constrainVertices0;
            M.constrainVerticesRef = constrainVerticesRef0;
            M.auxVtxIdxs = auxVtxIdxs0;
        }, this, " key=m group=Interpolator ");

        TwDefine(" MatlabDeformer/Interpolator opened=false ");
        TwDefine(" MatlabDeformer/BDHM opened=false ");

        preprocess();
        initiated = true;
    }

    ~MatlabDeformer(){
        TwBar *bar = TwGetBarByName("MatlabDeformer");
        if (bar)    TwDeleteBar(bar); 
    }

    virtual std::string name(){ return "P2PHarmonic"; }

    const char* currentInterpAlgorithm() const
    {
        const char* algNames[] = { "BDHI/metric", "BDHI/eta",  "BDHI/nu",   "Chen13", "ARAP", "ARAP_LG", "FFMP", "BDH_GM", "Unknown" }; // sync with TwType InterpAlg
        return algNames[std::min<int>(std::size(algNames) - 1, interpAlgorithm)];
    }

    virtual void preprocess() 
    {
        if (initiated){
            scalar2matlab("numVirtualVertices", nVirtualVertex);
            scalar2matlab("numFixedSamples", nFixedSample);

            scalar2matlab("cage_offset", cage_offset);
            scalar2matlab("numDenseEvaluationSamples", nDenseEvaluationSample);
            scalar2matlab("numActiveSetPoolSamples", nActSetPoolSample);
        }

        matlabEval("p2p_harmonic_prep;");

        C = matlab2eigenComplex("C");

        nVirtualVertex = (int)matlab2scalar("numVirtualVertices");
        nFixedSample = (int)matlab2scalar("numFixedSamples");

        if (!initiated){
            cage_offset = (float)matlab2scalar("cage_offset");
            nDenseEvaluationSample = (int)matlab2scalar("numDenseEvaluationSamples");
            nActSetPoolSample = (int)matlab2scalar("numActiveSetPoolSamples");

            sigma1_upper_bound = (float)matlab2scalar("sigma1_upper_bound");
            sigma2_lower_bound = (float)matlab2scalar("sigma2_lower_bound");
            k_upper_bound = (float)matlab2scalar("k_upper_bound");
        }

        //deformResultFromMaltab("PhiPsy");
        deformResultFromMaltab("XP2PDeform"); // even here it is not working, something is wrong with eigen/mkl
    }


	void interpAtTime() {
        float t = timet;

		const char *bdhimethods[] = { "metric", "eta", "nu" };
		switch (interpAlgorithm) {
		case 0:			// BDH / metric
		case 1:			// BDH / eta
		case 2:			// BDH / nu

			matlabEval(std::string("bdhiMethod='")+bdhimethods[interpAlgorithm]+"';");

			// TODO: clear rot_trans mess, can be merged into Phi Psy
			//matlabEval("[Phi, Psy, rot_trans] = fBdhInterp(" + std::to_string(t) + ");");
			matlabEval("XBDHI = fBdhInterpX(" + std::to_string(t) + ");");
			deformResultFromMaltab("XBDHI");
            if (M.vizVtxData) {
                matlabEval("k = fBdhInterpkX(" + std::to_string(t) + ");");
                auto k = matlab2vector<float>("single(k)", true);
                M.setVertexDataViz(k.data());
            }
			break;
		case 3:			// SIG13
			matlabEval("XSIG13 = fSIG13Interp(" + std::to_string(t) + ");");
			deformResultFromMaltab("XSIG13");
			break;
        case 4:         // BDH_GM, BDHI for general map other than harmonic map
			matlabEval("XBDHGM = fGBDHInterp(" + std::to_string(t) + ");");
			deformResultFromMaltab("XBDHGM");
			break;
		case 5:			// ARAP
			matlabEval("XARAP = fARAPInterp(" + std::to_string(t) + ");");
			deformResultFromMaltab("XARAP");
			break;
		case 6:			// FFMP
			matlabEval("XFFMP = fFFMPInterp(" + std::to_string(t) + ");");
			deformResultFromMaltab("XFFMP");
			break;

		}
	}



    virtual void updateP2PConstraints(int) 
    {
        using namespace Eigen;
        const size_t nConstrain = M.constrainVertices.size();
        eigen2matlab("P2PVtxIds", (Map<VectorXi>(M.getConstrainVertexIds().data(), nConstrain) + VectorXi::Ones(nConstrain)).cast<double>());
        matlabEval("CauchyCoordinatesAtP2Phandles = C(P2PVtxIds,:);");

        MatrixX2d p2pdst = Map<Matrix<float, Dynamic, 2, RowMajor> >(M.getConstrainVertexCoords().data(), nConstrain, 2).cast<double>();
        eigen2matlabComplex("P2PCurrentPositions", p2pdst.col(0), p2pdst.col(1));
#if 0
        eigen2matlab(matName("handleIds"), (Map<VectorXi>(M.getConstrainVertexIds().data(), nConstrain) + VectorXi::Ones(nConstrain)).cast<double>());
        eigen2matlab(matName("handlePos"), Map<MatrixXf>(M.getConstrainVertexCoords().data(), nConstrain, 2).cast<double>());

        bool updatePlotInMatlab = true;
        if (updatePlotInMatlab){
            matlabEval("fUpdatePlotVertices(uihandle.hconstrains," + matName("handlePos") + ");\n"
                + "fUpdatePlotVertices(uihandle.hconstraint,uihandle.hmt.Vertices(" + matName("handleIds") + ",:));\n"
                + "fUpdatePlotVertices(hcs,fC2R(z(" + matName("handleIds") + ")));");
        }
#endif
    }


    void deformResultFromMaltab(std::string resVarName)
    {
        using namespace Eigen;
		if ( !resVarName.compare("PhiPsy") ) { // do all interpolation computation in matlab, for better performance with # virtual vertex > 1
			MatrixXcd Phi = matlab2eigenComplex("Phi");
			MatrixXcd Psy = matlab2eigenComplex("Psy");

			if (Phi.rows() == 0 || Psy.rows() == 0 || C.rows() == 0) return;

			//using Vec = Eigen::Map < Eigen::VectorXcd > ;
			//Eigen::VectorXcd x = C*Vec(Phi.data(), Phi.rows()) + (C*Vec(Psy.data(), Psy.rows())).conjugate();
			Eigen::VectorXcd x = C*Phi + (C*Psy).conjugate();

			if (getMatEngine().hasVar("rot_trans")) {
				// for interpolation
				Vector2cd rot_trans = matlab2eigenComplex("rot_trans");
				x = x.array()*rot_trans(0) + rot_trans(1);
			}

			if (x.rows() == 0) return;

			Matrix<float, Dynamic, 2, RowMajor> xr(x.rows(), 2);
			xr.col(0) = x.real().cast<float>();
			xr.col(1) = x.imag().cast<float>();
			M.upload(xr, Eigen::MatrixXi(), nullptr);
		}
		else {
			MatrixXcd x = matlab2eigenComplex(resVarName);
			if (x.rows() == 0) return;

			Matrix<float, Dynamic, 2, RowMajor> xr(x.rows(), 2);
			xr.col(0) = x.real().cast<float>();
			xr.col(1) = x.imag().cast<float>();
			M.upload(xr, Eigen::MatrixXi(), nullptr);
		}
    }

    virtual void deform()
    {
        string2matlab("solver_type", solver_names[solver]);
        string2matlab("energy_type", energy_names[energy_type]);
        scalar2matlab("no_output", !solver_output);
        scalar2matlab("p2p_weight", p2p_weight);
        scalar2matlab("sigma1_upper_bound", sigma1_upper_bound);
        scalar2matlab("sigma2_lower_bound", sigma2_lower_bound);
        scalar2matlab("k_upper_bound", k_upper_bound);
        scalar2matlab("binarySearchValidMap", binarySearchValidMap);

        matlabEval("p2p_harmonic;");
        matlabEval("clear rot_trans;");

        //deformResultFromMaltab("PhiPsy");  // todo: still buggy, matrix C may change sizes 
        deformResultFromMaltab("XP2PDeform");
    }

    virtual bool converged() {
        return !getMatEngine().hasVar("P2P_Deformation_Converged")  ||  matlab2scalar("P2P_Deformation_Converged") > 0;  // treat case where preprocessing fails as "converged"
    }

    virtual void resetDeform() {
		matlabEval("Phi = vv; Psy = Phi * 0; XP2PDeform = X; phipsyIters = []; clear rot_trans;");
        deformResultFromMaltab("XP2PDeform"); 
    }
    virtual void getResult() {}
    virtual void saveData()   { matlabEval("p2p_harmonic_savedata;"); }
};



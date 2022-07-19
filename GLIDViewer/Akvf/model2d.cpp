#define NOMINMAX

#include "model2d.h"
#include "logspiral.h"
#include <cmath>
#include <fstream>
#include <string>
#include <algorithm>
#include <complex>
#include <limits>

//#define EIGEN_CHOLMOD_SUPPORT
#define EIGEN_USE_MKL_ALL
#include <Eigen/SparseCore>
#include <Eigen/CholmodSupport>
#include <Eigen/PardisoSupport>

template<class T>
std::vector < Eigen::Triplet<T> > getIJVTriplet(int rows, int cols, int nnz, const int *rowStart, const int *colIdxs, const T* values)
{
    std::vector<Eigen::Triplet<T>> ijv;
    ijv.reserve(nnz);

	for ( int i = 0 ; i < rows ; i++ ) {
        for (int j = rowStart[i]; j < rowStart[i + 1]; ++j) {
            ijv.push_back( Eigen::Triplet<T>(i, colIdxs[j], values[j]) );
		}
	}

    return ijv;
}


using namespace std;

template <class T>
void addConstraint(Eigen::SparseMatrix<T, Eigen::RowMajor>& m, const vector<int> &rows, T alpha) {
    int nr = m.rows();
    m.conservativeResize(nr + rows.size(), m.cols());

    for (int i = 0; i < rows.size(); i++) {
        //m.startVec(nr + i);
        //m.insertBack(nr + i, rows[i]) = alpha;
        m.insert(nr + i, rows[i]) = alpha;
    }
}


template <class T>
double infinityNorm(const Eigen::SparseMatrix<T, Eigen::RowMajor>& m)
{
    double nmax = 0;
    for (int k = 0; k < m.outerSize(); k++) {
        double rowsum = 0;
        for (Eigen::SparseMatrix<T, Eigen::RowMajor>::InnerIterator  it(m, k); it; ++it) 
            rowsum += abs(it.value());

        nmax = std::max(nmax, rowsum);
    }

    return nmax;
}

template <class T>
void zeroOutColumns(Eigen::SparseMatrix<T, Eigen::RowMajor>& m, 
                    const std::vector<bool> &colflags) 
{
    for (int k = 0; k < m.outerSize(); k++) {
        for (Eigen::SparseMatrix<T, Eigen::RowMajor>::InnerIterator it(m, k); it; ++it)
            if (colflags[it.col()]) it.valueRef() = 0;
    }
}


template<class T>
void Model2D<T>::displaceMesh(const vector<int> &indices, const vector< vec2<T> > &displacements, T alpha) {
    if (indices.size() == 1) { // when only one vertex is constrained, move parallel
        for (auto &v:vertices) v += displacements[0];
        return;
    }

    // solver have problem with all 0 rhs
    bool nomove = true;
    for (auto i: displacements)
        if ( i.norm() > 1e-8f) 
            nomove = false;

    if (nomove) return;

    SparseMat P = getP();
    SparseMat Pcopy = P;

    alpha = alpha / (2*indices.size()) * infinityNorm(P);

    Eigen::VectorXd rhs = Eigen::VectorXd::Zero(2*numVertices);

    for (unsigned int i = 0; i < indices.size(); i++) {
        rhs[ indices[i] ] = displacements[i][0]*alpha*alpha;
        rhs[ indices[i]+numVertices ] = displacements[i][1]*alpha*alpha;
    }

    vector<int> indices2;
    for (auto i:indices) {
        indices2.push_back(i);
        indices2.push_back(i + numVertices);
    }

    addConstraint(P, indices2, double(alpha));

    Eigen::SparseMatrix<double> matP = P.transpose()*P;

    Eigen::CholmodSimplicialLLT<Eigen::SparseMatrix<double>, Eigen::Upper> lltSolver;
    //Eigen::PardisoLLT<Eigen::SparseMatrix<double>, Eigen::Upper> lltSolver;
    lltSolver.analyzePattern(matP);
    lltSolver.factorize(matP);

    if (lltSolver.info() != Eigen::Success) return;
    Eigen::VectorXd Xx = lltSolver.solve(rhs);



    // NOW, DIRICHLET SOLVE ////////////////////////////////////////////////////

    Eigen::VectorXd  boundaryX = Eigen::VectorXd::Zero(numVertices*2);

    for (auto i: boundaryVertices) {
        boundaryX[i] = Xx[i];
        boundaryX[i + numVertices] = Xx[i + numVertices];
    }

    Eigen::VectorXd rhsMove = Pcopy*boundaryX;

    rhsMove *= -1;

    auto vtxFlags = boundaryVtxFlag;
    vtxFlags.insert(vtxFlags.end(), boundaryVtxFlag.cbegin(), boundaryVtxFlag.cend());

    zeroOutColumns(Pcopy, vtxFlags);

    Eigen::VectorXd Bx = Pcopy.transpose()*rhsMove;

    vector<int> constrained;
    for (auto i: boundaryVertices) {
        int bv = i;
        constrained.push_back(i);
        constrained.push_back(i+numVertices);
        Bx[bv] += Xx[bv];
        Bx[bv+numVertices] += Xx[bv+numVertices];
    }

    addConstraint(Pcopy, constrained, 1.);

    matP = Pcopy.transpose()*Pcopy;

    lltSolver.analyzePattern(matP);
    lltSolver.factorize(matP);
    if (lltSolver.info() != Eigen::Success) return;

    Xx = lltSolver.solve(Bx);


    ////////////////////////////////////////////////////////////////////////////

    // Try complex number trick /////

    // This log spiral implementation is slow and double-counts edges -- eventually we can make it
    // faster and parallelize

    newPoints.resize(numVertices);
    counts.resize(numVertices);

    for (int i = 0; i < numVertices; i++) {
        counts[i] = 0;
        newPoints[i] = vec2<T>(0,0);
    }

    for (int i = 0; i < numFaces; i++)
        for (int j = 0; j < 3; j++) {
            int e1 = faces[i][j];
            int e2 = faces[i][(j+1)%3];
            int vtx = faces[i][(j+2)%3];

            complex<double> v1(Xx[e1], Xx[e1+numVertices]);
            complex<double> v2(Xx[e2], Xx[e2+numVertices]);
            complex<double> p1(vertices[e1][0], vertices[e1][1]);
            complex<double> p2(vertices[e2][0], vertices[e2][1]);

            complex<double> z = (v1-v2)/(p1-p2);
            complex<double> p0 = (p2*v1-p1*v2)/(v1-v2);

            double c = z.real();
            double alpha = z.imag();
            vec2<T> p(p0.real(),p0.imag());
            vec2<T> l1(vertices[e1][0], vertices[e1][1]);
            vec2<T> l2(vertices[e2][0], vertices[e2][1]);

            LogSpiral<T> spiral;
            spiral.p0 = p;
            spiral.c = c;
            spiral.alpha = alpha;

            vec2<T> result1 = spiral.evaluate(l1,1);//logSpiral(p,c,alpha,l1,1);
            vec2<T> result2 = spiral.evaluate(l2,1);//logSpiral(p,c,alpha,l2,1);

            // compute cotangent weights
            vec2<T> d1 = vertices[e1] - vertices[vtx];
            vec2<T> d2 = vertices[e2] - vertices[vtx];
            double angle = fabs(rotationAngle(d1,d2));
            double cotangent = 1;// / tan(angle);

            counts[e1] += cotangent;
            counts[e2] += cotangent;

            newPoints[e1] += result1*cotangent;
            newPoints[e2] += result2*cotangent;
        }

    /////////////////////////////////

    vf.resize(numVertices);

    for (int i = 0; i < numVertices; i++) {
        vertices[i] = newPoints[i] / counts[i];
    }
}

template<class T>
typename Model2D<T>::SparseMat Model2D<T>::getP() 
{
    const double SQRT_2 = 1.41421356;
    using triplet = Eigen::Triplet < T > ;

    std::vector<triplet> P_ijv, dx_ijv, dy_ijv;
    P_ijv.reserve(numFaces * 4);
    dx_ijv.reserve(3 * numFaces);
    dy_ijv.reserve(3 * numFaces);

    for (int f = 0; f < numFaces; f++) {
        int i = faces[f][0], j = faces[f][1], k = faces[f][2];

        if (i > j) std::swap(i, j);
        if (i > k) std::swap(i, k);
        if (j > k) std::swap(j, k);

        vec2<T> d1 = vertices[i] - vertices[k],
                d2 = vertices[j] - vertices[i];

        T area = fabs(d1[1] * d2[0] - d1[0] * d2[1]);
        vec2<T>  c1(-d1[1]/area,d1[0]/area),
                 c2(-d2[1]/area,d2[0]/area);

        P_ijv.insert(P_ijv.end(), { triplet(3 * f, f, 2),
            triplet(3 * f + 1, f + numFaces, SQRT_2),
            triplet(3 * f + 1, f + 2 * numFaces, SQRT_2),
            triplet(3 * f + 2, f + 3 * numFaces, 2) });

        dx_ijv.insert(dx_ijv.end(), { triplet(f, i, -c1[0] - c2[0]),
            triplet(f, j, c1[0]),
            triplet(f, k, c2[0]) });

        dy_ijv.insert(dy_ijv.end(), { triplet(f, i, -c1[1] - c2[1]),
            triplet(f, j, c1[1]),
            triplet(f, k, c2[1]) });
    }


    SparseMat P2a(numFaces * 3, numFaces * 4);
    P2a.setFromTriplets(P_ijv.cbegin(), P_ijv.cend());

    std::vector<triplet> dxy2 = dx_ijv;
    dxy2.reserve(4 * numFaces);

    for (auto i : dy_ijv) dxy2.push_back(triplet(i.row() + numFaces, i.col(), i.value()));
    for (auto i : dx_ijv) dxy2.push_back(triplet(i.row() + numFaces * 2, i.col() + numVertices, i.value()));
    for (auto i : dy_ijv) dxy2.push_back(triplet(i.row() + numFaces * 3, i.col() + numVertices, i.value()));

    SparseMat stack2(4 * numFaces, 2 * numVertices);
    stack2.setFromTriplets(dxy2.cbegin(), dxy2.cend());
    //SparseMat prod2 = P2a*stack2;

    //prod = SimpleSparseMatrix<T>(prod2.rows(), prod2.cols(), prod2.nonZeros(), prod2.outerIndexPtr(), prod2.innerIndexPtr(), prod2.valuePtr());

    return P2a*stack2;
}

template<class T>
void Model2D<T>::initialize() {
    minX = numeric_limits<T>::max();
    maxX = numeric_limits<T>::min();
    minY = numeric_limits<T>::max();
    maxY = numeric_limits<T>::min();

    for (int i = 0; i < numVertices; i++) {
        minX = min(minX, vertices[i][0]);
        minY = min(minY, vertices[i][1]);
        maxX = max(maxX, vertices[i][0]);
        maxY = max(maxY, vertices[i][1]);
    }

    neighbors.resize(numVertices);
    map< int, map<int, int> > edgeCount;
    for (int i = 0; i < numFaces; i++) {
        int a = faces[i][0], b = faces[i][1], c = faces[i][2];
        neighbors[a].insert(b);
        neighbors[a].insert(c);
        neighbors[b].insert(a);
        neighbors[b].insert(c);
        neighbors[c].insert(a);
        neighbors[c].insert(b);
        edgeCount[a][b]++;
        edgeCount[b][a]++;
        edgeCount[a][c]++;
        edgeCount[c][a]++;
        edgeCount[c][b]++;
        edgeCount[b][c]++;
    }

    boundaryVtxFlag.assign(numVertices, false);
    for (int i = 0; i < numFaces; i++) {
        int a = faces[i][0], b = faces[i][1], c = faces[i][2];
        if (edgeCount[a][b] == 1) {
            boundaryVtxFlag[a] = true;
            boundaryVtxFlag[b] = true;

            boundaryVertices.insert(a);
            boundaryVertices.insert(b);
        }
        if (edgeCount[b][c] == 1) {
            boundaryVtxFlag[b] = true;
            boundaryVtxFlag[c] = true;

            boundaryVertices.insert(b);
            boundaryVertices.insert(c);
        }
        if (edgeCount[a][c] == 1) {
            boundaryVtxFlag[a] = true;
            boundaryVtxFlag[c] = true;

            boundaryVertices.insert(a);
            boundaryVertices.insert(c);
        }
    }

}

template<class T>
Model2D<T>::Model2D(Model2D<T> &m) {
    numVertices = m.numVertices;
    numFaces = m.numFaces;
    vertices = m.vertices;
    faces = m.faces;
    initialize();
}

template<class T>
void Model2D<T>::save(const std::string &filename) {
	ofstream outfile(filename);
    outfile << "OFF\n";
    outfile << numVertices << ' ' << numFaces << " 0\n"; // don't bother counting edges

    for (int i = 0; i < numVertices; i++)
        outfile << vertices[i][0] << ' ' << vertices[i][1] << " 0\n";

    for (int i = 0; i < numFaces; i++)
        outfile << "3 " << faces[i][0] << ' ' << faces[i][1] << ' ' << faces[i][2] << endl;
}

template<class T>
Model2D<T>::Model2D(const T* x, int nv, const int *t, int nf) : numVertices(nv), numFaces(nf)
{
    vertices.resize(numVertices);

    for (int i = 0; i < numVertices; i++)
        vertices[i] = vec2<T>(x[i*2], x[i*2+1]);

    faces.resize(numFaces);
    for (int i = 0; i < numFaces; i++)
        faces[i] = { t[i * 3], t[i * 3 + 1], t[i * 3 + 2] };

    if (nv > 0)
        initialize();
}

template class Model2D<float>;
template class Model2D<double>;

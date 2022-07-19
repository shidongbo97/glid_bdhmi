#ifndef MODEL2D_H
#define MODEL2D_H


#include "logspiral.h"
#include <set>
#include <vector>
#include <array>
#include <limits>
#include <Eigen/SparseCore>


template<class T>
class Model2D {
public:
    using Face = Eigen::Vector3i;
    using SparseMat = Eigen::SparseMatrix < double, Eigen::RowMajor > ;

    Model2D(const T* x=nullptr, int nv=0, const int *t=nullptr, int nf=0);
    void save(const std::string &filename);
    void replacePoints(const T* x);
    Model2D(Model2D<T> &m);
    ~Model2D() {}

    T getMinX() { return minX; }
    T getMaxX() { return maxX; }
    T getMinY() { return minY; }
    T getMaxY() { return maxY; }
    T getWidth() { return maxX - minX; }
    T getHeight() { return maxY - minY; }
    vec2<T> &getVertex(int i) { return vertices[i]; }
    int getNumVertices() { return numVertices; }
    int getNumFaces() { return numFaces; }
    Face &getFace(int i) { return faces[i]; }

    int getClosestVertex(vec2<T> point, T dist);

    void displaceMesh(const vector<int> &indices, const vector<vec2<T>> &displacements, T alpha);

    SparseMat getP();
    void updateCovariance();

    void initialize();
    void copyPositions(Model2D<T> &m) {
        for (int i = 0; i < numVertices; i++)
            vertices[i] = m.vertices[i];
    }

    void reuseVF() {
        for (int i = 0; i < numVertices; i++)
            for (int j = 0; j < 2; j++)
                vertices[i][j] += vf[i][j] * .5;
    }

private:
    std::vector<vec2<T>> vertices;
    std::vector<Face> faces;
    int numVertices, numFaces;

    T minX, maxX, minY, maxY;

    std::vector<vec2<T>> newPoints;
    std::vector<double> counts;
    std::vector<vec2<T>> vf;

    std::vector< set<int> > neighbors;
    std::set<int> boundaryVertices;
    std::vector<bool> boundaryVtxFlag;
};

#endif // MODEL2D_H

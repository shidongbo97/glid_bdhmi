#pragma once

#include "vaomesh.h"
#include "Akvf/model2d.h"
#include "LIM/LIMdeform.h"


struct Deformer
{
    bool needIteration = true;
    virtual std::string name(){ return "UNKNOWN"; }
    virtual void preprocess() {}
    virtual void updateP2PConstraints(int){}
    virtual void deform() = 0;
    virtual bool converged() { return false; }
    virtual void resetDeform(){}
    virtual void getResult(){}
    virtual void saveData(){}
};

struct AKVFDeformer : public Deformer
{
    MyMesh &M;
    std::vector<float> XDeformed;

    using vec2f = vec2 < float >;
    Model2D < float > deformer;
    float alpha;

    std::vector<int> p2pIdxs;
    std::vector<float> p2pCoords;

    AKVFDeformer(MyMesh &m) :M(m), alpha(1.f),
        deformer(m.mesh.X.data(), m.nVertex, m.mesh.T.data(), m.nFace)
    {
        preprocess();

        TwBar *bar = TwNewBar("AKVFDeformation");
        TwDefine(" AKVFDeformation size='220 100' color='76 76 127' position='5 600' label='AKVF Deformation' ");

        TwAddVarRW(bar, "Alpha", TW_TYPE_FLOAT, &alpha, " min=0 step=0.1 ");
    }

    ~AKVFDeformer() {
        TwBar *bar = TwGetBarByName("AKVFDeformation");
        if (bar) TwDeleteBar(bar);
    }

    virtual std::string name() { return "AKVF"; }
    virtual void preprocess() { XDeformed = M.mesh.X; }

    virtual void updateP2PConstraints(int) {
        p2pIdxs = M.getConstrainVertexIds();
        p2pCoords = M.getConstrainVertexCoords();
    }


    virtual void deform() {
        std::vector<vec2f> displacements(p2pIdxs.size());
        for (int i = 0; i < p2pIdxs.size(); i++)
            displacements[i] = vec2f(p2pCoords[i * 2] - XDeformed[p2pIdxs[i] * 2], p2pCoords[i * 2 + 1] - XDeformed[p2pIdxs[i] * 2 + 1]);


        deformer.displaceMesh(p2pIdxs, displacements, alpha);

        for (int i = 0; i < M.nVertex; i++) {
            auto xi = deformer.getVertex(i);
            XDeformed[i * 2] = xi[0];
            XDeformed[i * 2 + 1] = xi[1];
        }

        updateP2PConstraints(0);

        M.upload(XDeformed.data(), M.nVertex, nullptr, M.nFace, nullptr);
    }

    virtual void resetDeform() {
        XDeformed = M.mesh.X;
        deformer = Model2D<float>(M.mesh.X.data(), M.nVertex, M.mesh.T.data(), M.nFace);
    }
    virtual void getResult() {}
    virtual void saveData() {}
};


struct LIMDeformer : public Deformer
{
    MyMesh &M;
    std::vector<float> XDeformed;

    std::shared_ptr<LIMDeformerWrap> deformer;

    std::vector<int> p2pIdxs;
    std::vector<float> p2pCoords;

    LIMDeformer(MyMesh &m) :M(m)
    {
        preprocess();
        deformer.reset(new LIMDeformerWrap);
        deformer->setMesh(m.mesh.X.data(), m.nVertex, m.mesh.T.data(), m.nFace);
    }

    virtual std::string name() { return deformer->energyName(); }

    virtual void preprocess() { XDeformed = M.mesh.X; }

    virtual void updateP2PConstraints(int) {
        bool newConstrain = (p2pIdxs.size() != M.constrainVertices.size());

        for (int i = 0; (!newConstrain) && i < p2pIdxs.size(); i++)
            if (M.constrainVertices.count(p2pIdxs[i]) == 0) {
                newConstrain = true;
                break;
            }

        p2pIdxs = M.getConstrainVertexIds();
        p2pCoords = M.getConstrainVertexCoords();

        if (newConstrain)
            deformer->UpdatePositionalConstraints(p2pIdxs);
        else {
            using Mat = Eigen::Map < Eigen::Matrix<float, Eigen::Dynamic, 2, Eigen::RowMajor> >;
            deformer->UpdateConstraintVertexPositions(p2pIdxs, Mat(p2pCoords.data(), p2pIdxs.size(), 2).cast<double>());
        }
    }


    virtual void deform() {
        deformer->solve(XDeformed.data(), M.nVertex);
        M.upload(XDeformed.data(), M.nVertex, nullptr, M.nFace, nullptr);
    }

    virtual void resetDeform() {
        XDeformed = M.mesh.X;
        deformer.reset(); // problem with anttweakbar, to be fixed, make sure d'ctr is called before c'ctr
        deformer.reset(new LIMDeformerWrap);
        deformer->setMesh(M.mesh.X.data(), M.nVertex, M.mesh.T.data(), M.nFace);
    }
    virtual void getResult() {}
    virtual void saveData() {}
};

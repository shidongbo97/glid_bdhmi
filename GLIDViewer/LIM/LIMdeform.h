// Copyright 2013 - Christian Schüller 2013, schuellc@inf.ethz.ch
// Interactive Geometry Lab - ETH Zurich

#pragma once

#include <AntTweakBar.h>
#include <igl/boundary_faces.h>
#include "LIM/TriangleMesh.h"
#include "LIM/LGARAP_LIMSolver2D.h"
#include "LIM/GreenStrain_LIMSolver2D.h"
#include "LIM/LSConformal_LIMSolver2D.h"


struct LIMDeformerWrap
{
    //enum class LIM_ENERGY_TYPE : unsigned { ARAP = 0, LSCM = 1, GREEN = 2 };
    enum LIM_ENERGY_TYPE { ARAP = 0, LSCM = 1, GREEN = 2 };

    LIMSolver* solver;
    TriangleMesh* mesh;
    double barrierWeight;
    bool enableBarriers;
    double alpha;
    int energyType;
    const int dim = 2;
    bool runSolver;
    bool enableAlphaUpdate;
    bool enableSubStepping;
    bool enableOutput;
    double error;

    LIMDeformerWrap() :solver(nullptr), mesh(nullptr), energyType(ARAP), runSolver(true), enableBarriers(true),
        enableSubStepping(true), enableAlphaUpdate(true), error(0.), enableOutput(true)
    {
        TwBar *bar = TwNewBar("LIMDeformation");
        TwDefine(" LIMDeformation size='220 300' color='76 76 127' position='5 600' label='LIM Deformation' "); // change default tweak bar size and color

        const int EnergyCount = 3;
        TwEnumVal energyEV[EnergyCount] = { { GREEN, "Green Strain" }, { ARAP, "ARAP" }, { LSCM, "LSC" } };
        TwType energy = TwDefineEnum("Energy", energyEV, EnergyCount);
        TwAddVarCB(bar, "Energy", energy, 
            [](const void *v, void *deformer) { LIMDeformerWrap* d = (LIMDeformerWrap*)deformer; d->energyType = *(unsigned int*)(v); d->initSolver();  },
            [](void *v, void *deformer)       { *(unsigned int*)(v) = ((LIMDeformerWrap*)deformer)->energyType; }, 
            this,  "");
        TwAddVarRW(bar, "RunSolver", TW_TYPE_BOOLCPP, &runSolver, "label='Run Solver'");

        TwAddVarRW(bar, "Barriers", TW_TYPE_BOOLCPP, &enableBarriers, "group='Solver Options'");
        TwAddVarRW(bar, "BarrierWeights", TW_TYPE_DOUBLE, &barrierWeight, "group='Solver Options'");
        TwAddVarRW(bar, "SubStepping", TW_TYPE_BOOLCPP, &enableSubStepping, "group='Solver Options'");
        TwAddVarRW(bar, "AlphaUpdate", TW_TYPE_BOOLCPP, &enableAlphaUpdate, "group='Solver Options'");
        TwAddVarRW(bar, "Alpha/Ratio", TW_TYPE_DOUBLE, &alpha, "group='Solver Options'");
        TwAddVarRW(bar, "Output", TW_TYPE_BOOLCPP, &enableOutput, "group='Solver Options'");

        TwAddVarRO(bar, "PCError", TW_TYPE_DOUBLE, &error, "group='Solver Options'");
    }

    ~LIMDeformerWrap() {
        if (mesh) delete(mesh);
        if (solver) delete(solver);
        TwBar *bar = TwGetBarByName("LIMDeformation");
        if (bar) TwDeleteBar(bar);
    }


    std::string energyName(){
        const char* energyNames[] = { "ARAP", "LSCM", "GREEN" };
        return std::string(enableBarriers ? "LIM" : "") + energyNames[energyType];
    }



    void setMesh(const float *x, int nv, const int *t, int nf)
    {
        using namespace Eigen;
        using namespace std;
        // init triangle mesh
        if (mesh) delete(mesh);
        mesh = new TriangleMesh();

        MatrixX3d X(nv, 3);
        X.leftCols(2) = Map<const Matrix<float, Dynamic, 2, RowMajor>>(x, nv, 2).cast<double>();
        X.rightCols(1).setZero();

        mesh->InitalVertices = new MatrixX3d(X);
        mesh->DeformedVertices = new MatrixX3d(X);
        mesh->PredictedVertices = new MatrixX3d(X);

        mesh->Triangles = new MatrixX3i(nf, 3);
        *(mesh->Triangles) = Map < const Matrix<int, Dynamic, 3, RowMajor> >(t, nf, 3);

        vector<vector<int> > verticesVec(nf);
        for (int r = 0; r < nf; r++) {
            vector<int> v;
            for (int c = 0; c < 3; c++)
                v.push_back(mesh->Triangles->coeff(r, c));

            verticesVec[r] = v;
        }

        vector<vector<int> > borderEdges;
        igl::boundary_faces(verticesVec, borderEdges);

        map<int, int> edges;
        for (int i = 0; i < borderEdges.size(); i++)
            edges[borderEdges[i][0]] = borderEdges[i][1];

        mesh->BorderVertices = new Matrix<int, Dynamic, 1>(borderEdges.size());

        map<int, int>::iterator iter = edges.begin();
        int count = 0;
        do{
            mesh->BorderVertices->coeffRef(count++) = iter->first;
            iter = edges.find(iter->second);
        } while (iter != edges.begin());

        mesh->ConstraintMatrix = new SparseMatrix<double>();
        mesh->ConstraintTargets = new Matrix<double, Dynamic, 1>();
        mesh->InitMesh();

        //////////////////////////////////////////////////////////////////////////
        mesh->ConstraintMatrix->resize(mesh->InitalVertices->rows()*dim, mesh->InitalVertices->rows()*dim);
        mesh->ConstraintTargets->resize(mesh->InitalVertices->rows()*dim);
        mesh->ConstraintTargets->setZero();

        // init with identity matrix in order to reserve single vertex constraints
        vector<Eigen::Triplet<double> > triplets;
        for (int i = 0; i < mesh->InitalVertices->rows()*dim; i++)
            triplets.push_back(Triplet<double>(i, i, 1));
        mesh->ConstraintMatrix->setFromTriplets(triplets.begin(), triplets.end());


        initSolver();
    }

    void initSolver()
    {
        switch (energyType){
        case LIM_ENERGY_TYPE::GREEN:
            solver = new GreenStrain_LIMSolver2D();
            break;
        case LIM_ENERGY_TYPE::ARAP:
            solver = new LGARAP_LIMSolver2D();
            break;
        case LIM_ENERGY_TYPE::LSCM:
            solver = new LSConformal_LIMSolver2D();
            break;
        }

        if (solver->EnableAlpaUpdate)
            alpha = solver->AlphaRatio;
        else
            alpha = solver->Alpha;

        barrierWeight = solver->Beta;

        if (mesh)
            solver->Init(mesh);

        // free all constraint vertices as now hessian structure is already reserved
        for (int i = 0; i < mesh->InitalVertices->rows()*dim; i++)
            mesh->ConstraintMatrix->coeffRef(i, i) = 0;

        solver->UpdatePositionalConstraintMatrix();
    }


    void UpdateConstraintVertexPositions(int constraintVtx, const float *positions)
    {
        int idx = constraintVtx;
        for (int d = 0; d < dim; d++)
            mesh->ConstraintTargets->coeffRef(idx*dim + d) = positions[d];

        solver->Restart();
    }


    void UpdateConstraintVertexPositions(const std::vector<int>& constraintVertices, const Eigen::MatrixX2d& positions)
    {
        for (int i = 0; i < constraintVertices.size(); i++){
            int idx = constraintVertices[i];
            for (int d = 0; d < dim; d++)
                mesh->ConstraintTargets->coeffRef(idx*dim + d) = positions.coeff(i, d);
        }

        solver->Restart();
    }


    void UpdatePositionalConstraints(const std::vector<int>& constraintVertices)
    {
        // free all constraint vertices
        for (int i = 0; i < mesh->InitalVertices->rows()*dim; i++)
            mesh->ConstraintMatrix->coeffRef(i, i) = 0;

        mesh->ConstraintTargets->setZero();

        // set new constraint vertices
        for (auto idx : constraintVertices) {
            for (int c = 0; c < dim; c++) {
                mesh->ConstraintMatrix->coeffRef(idx*dim + c, idx*dim + c) = 1;
                mesh->ConstraintTargets->coeffRef(idx*dim + c) = mesh->DeformedVertices->coeff(idx, c);
            }
        }

        solver->UpdatePositionalConstraintMatrix();

        solver->Restart();
    }



    void solve(float *x, int nv)
    {
        if (!mesh || !runSolver) return;

        if (solver->EnableBarriers != enableBarriers
            || solver->EnableSubstepping != enableSubStepping
            || solver->EnableAlpaUpdate != enableAlphaUpdate
            || (solver->EnableAlpaUpdate && solver->AlphaRatio != alpha)
            || (!solver->EnableAlpaUpdate && solver->Alpha != alpha)
            || solver->Beta != barrierWeight)
        {
            if (solver->EnableAlpaUpdate != enableAlphaUpdate) {
                if (enableAlphaUpdate)
                    alpha = solver->AlphaRatio;
                else
                    alpha = 1e8;
            }

            solver->EnableBarriers = enableBarriers;
            solver->EnableSubstepping = enableSubStepping;
            solver->EnableAlpaUpdate = enableAlphaUpdate;

            if (solver->EnableAlpaUpdate)
                solver->AlphaRatio = alpha;
            else
                solver->Alpha = alpha;

            solver->Beta = barrierWeight;

            solver->Restart();
        }
        solver->EnableOutput = enableOutput;

        solver->Solve();

        error = solver->CurrentPositionalEnergy;

        // switch vertex buffers
        std::swap(mesh->DeformedVertices, mesh->PredictedVertices);

        using namespace Eigen;
        Map<Matrix<float, Dynamic, 2, RowMajor>>(x, nv, 2) = mesh->DeformedVertices->leftCols(2).cast<float>();
    }
};

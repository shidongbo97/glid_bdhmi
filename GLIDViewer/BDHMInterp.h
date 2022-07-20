#pragma once

#include "vaomesh.h"
#include "matlab_utils.h"
#include <thread>
#include <chrono>

int numBDHInterpFrames = 12;

extern bool showATB;
extern int viewport[4];
void display();

struct BDHMInterp
{
	float bdhmiHessianSample = 20.0f;
	int videoFPS = 30;
	float timet;

	unsigned int interpAlgorithm;

	Eigen::MatrixXcd C; // Cauchy Coordiantes for vertices

	MyMesh& M;

	BDHMInterp(MyMesh& m):M(m),timet(0), interpAlgorithm(0)
	{
		using bdhmiptr = BDHMInterp*;

		TwBar* bar = TwNewBar("BDHMI");

		scalar2matlab("bdhmiHessianSampleRate", bdhmiHessianSample / 100);
		scalar2matlab("bdhmiUseGPU", true);
		scalar2matlab("video_fps", videoFPS);

		TwDefine(" BDHMI size='220 220' color='0 0 255' text=dark alpha=128 position='5 770' label='BDHMI'");
		
		///////////////////////////////////////////////////////////////////
		TwAddVarCB(bar, "bdhmiHessianSample%", TW_TYPE_FLOAT,
			[](const void* v, void* d) { scalar2matlab("bdhmiHessianSampleRate", *(const float*)(v) / 100); },
			[](void* v, void*) { *(float*)(v) = matlab2scalar("bdhmiHessianSampleRate") * 100; },
			nullptr, " group=Parameters min=0.0001 max=100 help='sub sample for faster hessian assembly' ");

		TwAddVarCB(bar, "bdhmUseGPUCompute", TW_TYPE_BOOLCPP,
			[](const void* v, void* d) { scalar2matlab("bdhmiUseGPU", *(const bool*)(v)); },
			[](void* v, void*) { *(bool*)(v) = matlab2scalar("bdhmiUseGPU"); },
			nullptr, " group=Parameters help='is using gpu code for bdhmInterporation' ");

		TwAddButton(bar, "Add Keyframe", [](void*) {
			matlabEval("PhiPsyKF(:,end+(1:2)) = [Phi Psy];"); }, nullptr, " group=Interpolator ");

		TwAddButton(bar, "Set as Keyframe", [](void*) {
			matlabEval("PhiPsyKF(:,ikeyframe*2+(-1:0)) = [Phi Psy];"); }, nullptr, " group=Interpolator ");

		TwAddButton(bar, "Save all Keyframes", [](void*) {
			matlabEval("save([datadir 'PhiPsyKF'], 'PhiPsyKF');"); }, nullptr, " group=Interpolator ");

		TwAddVarCB(bar, "View Keyframe", TW_TYPE_INT32,
			[](const void* v, void* d) {
				scalar2matlab("ikeyframe", *(const int*)(v));
				matlabEval("Phi = PhiPsyKF(:, ikeyframe*2-1);  Psy = PhiPsyKF(:, ikeyframe*2); rot_trans=[1; 0];");
				bdhmiptr(d)->deformResultFromMaltab("PhiPsy");

				matlabEval("P2PCurrentPositions = C(P2PVtxIds, :)*Phi + conj(C(P2PVtxIds, :)*Psy);");
				//loadP2PConstraints();
			},
			[](void* v, void*) { *(int*)(v) = matlab2scalar("ikeyframe"); },
				this, " min=1 max=1000 step=1 group=Interpolator ");

		TwAddVarCB(bar, "t", TW_TYPE_FLOAT,
			[](const void* v, void* d) {
				bdhmiptr(d)->timet = *(const float*)(v);
				bdhmiptr(d)->interpAtTime();
			},
			[](void* v, void* d) { *(float*)(v) = bdhmiptr(d)->timet; },
				this, " min=-0.1 max=1.1 step=0.002 keyincr=RIGHT keydecr=LEFT group=Interpolator ");


		TwType InterpAlg = TwDefineEnumFromString("InterpAlgorithm", "BDH/metric, GE , ARAP, FFMP");
		TwAddVarCB(bar, "InterpAlg", InterpAlg,
			[](const void* v, void* d) {
				bdhmiptr(d)->interpAlgorithm = *(unsigned int*)(v);
				bdhmiptr(d)->interpAtTime();
			},
			[](void* v, void* d) { *(unsigned int*)(v) = bdhmiptr(d)->interpAlgorithm; },

				this, " group=Interpolator ");


		TwAddVarRW(bar, "numFrame", TW_TYPE_INT32, &numBDHInterpFrames, " group=Interpolator ");

		TwAddVarCB(bar, "video fps%", TW_TYPE_INT32,
			[](const void* v, void* d) { scalar2matlab("video_fps", *(const int*)(v)); },
			[](void* v, void*) { *(int*)(v) = matlab2scalar("video_fps"); },
			nullptr, " group=Interpolator help='#frames per second for video' ");

		TwAddButton(bar, "Generate sequence", [](void* d) {
			auto* pthis = bdhmiptr(d);

			vector2matlab("interpAnchID", pthis->M.auxVtxIdxs);
			matlabEval("interpAnchID= interpAnchID+1;");
			matlabEval("bdhInterpPrepMulti");

			auto& M = pthis->M;
			auto constrainVertices0 = M.constrainVertices;
			auto constrainVerticesRef0 = M.constrainVerticesRef;
			auto auxVtxIdxs0 = M.auxVtxIdxs;
			M.auxVtxIdxs.clear();
			M.constrainVertices.clear();
			M.constrainVerticesRef.clear();


			bool showATB0 = showATB;
			float edgeWidth0 = M.edgeWidth;

			{

				int vw = viewport[2] / 8 * 8, vh = viewport[3];
				std::vector<unsigned char> pixels(vw * vh * 3);

				std::string datadir = matlab2string("datadir");
				for (int i = 0; i < numBDHInterpFrames; i++) {
					pthis->timet = i / (numBDHInterpFrames - 1.f);
					pthis->interpAtTime();

					std::string imgfile = datadir + "_" + std::to_string(pthis->interpAlgorithm) + "_morph_" + std::to_string(i) + ".png";// +".jpg";

					display();

					glReadPixels(viewport[0], viewport[1], vw, vh, GL_BGR, GL_UNSIGNED_BYTE, &pixels[0]);
					
					M.saveResultImage(imgfile.c_str(), 3072);
				}
			}
			scalar2matlab("interpAlgorithm", pthis->interpAlgorithm);
			scalar2matlab("numframes", numBDHInterpFrames);
			matlabEval("test_image2video;");

			M.constrainVertices = constrainVertices0;
			M.constrainVerticesRef = constrainVerticesRef0;
			M.auxVtxIdxs = auxVtxIdxs0;
			}, this, " key=m group=Interpolator ");

		TwDefine(" BDHMI/Interpolator opened=false ");
		//TwDefine(" BDHMI/BDHM opened=false ");

		preprocess();

	}

	~BDHMInterp() {
		TwBar* bar = TwGetBarByName("BDHMI");
		if (bar)    TwDeleteBar(bar);
	}

	void preprocess()
	{
		matlabEval("p2p_harmonic_prep;");

		C = matlab2eigenComplex("C");

		deformResultFromMaltab("XP2PDeform");
	}

	void interpAtTime() {
		float t = timet;

		const char* bdhimethods[] = { "metric" };
		switch (interpAlgorithm) {
		case 0:			// BDH / metric
			matlabEval(std::string("bdhiMethod='") + bdhimethods[interpAlgorithm] + "';");

			matlabEval("XBDHI = fBdhInterpX(" + std::to_string(t) + ");");
			deformResultFromMaltab("XBDHI");
			if (M.vizVtxData) {
				matlabEval("k = fBdhInterpkX(" + std::to_string(t) + ");");
				auto k = matlab2vector<float>("single(k)", true);
				M.setVertexDataViz(k.data());
			}
			break;
		case 1:			// SIG13
			matlabEval("XGE19 = fANALYTICInterp(" + std::to_string(t) + ");");
			deformResultFromMaltab("XGE19");
			break;
		case 2:			// ARAP
			matlabEval("XARAP = fARAPInterp(" + std::to_string(t) + ");");
			deformResultFromMaltab("XARAP");
			break;
		case 3:			// FFMP
			matlabEval("XFFMP = fFFMPInterp(" + std::to_string(t) + ");");
			deformResultFromMaltab("XFFMP");
			break;

		}
	}

	void deformResultFromMaltab(std::string resVarName)
	{
		using namespace Eigen;
		if (!resVarName.compare("PhiPsy")) { // do all interpolation computation in matlab, for better performance with # virtual vertex > 1
			MatrixXcd Phi = matlab2eigenComplex("Phi");
			MatrixXcd Psy = matlab2eigenComplex("Psy");
			C = matlab2eigenComplex("C");

			if (Phi.rows() == 0 || Psy.rows() == 0 || C.rows() == 0) return;

			//using Vec = Eigen::Map < Eigen::VectorXcd > ;
			//Eigen::VectorXcd x = C*Vec(Phi.data(), Phi.rows()) + (C*Vec(Psy.data(), Psy.rows())).conjugate();
			Eigen::VectorXcd x = C * Phi + (C * Psy).conjugate();

			if (getMatEngine().hasVar("rot_trans")) {
				// for interpolation
				Vector2cd rot_trans = matlab2eigenComplex("rot_trans");
				x = x.array() * rot_trans(0) + rot_trans(1);
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
};
#pragma once

#include "cuda_runtime.h"
#include "mesh/MeshTriGPU.h"
#include "mesh/DMOMeshTri.h"
#include "mesh/DMOMeshTet.h"
#include "mesh/DMOMeshQuad.h"
#include "ConfigUsing.h"
#include "DMOConfig.h"
#include "DMOQuality.h"
#include "Timer.h"


using namespace DMO;

namespace DMO {

	//void discreteMeshOptimization(DMOMeshTri& dmo_mesh, QualityCriterium qualityCriterium = MEAN_RATIO, const float gridScale = 0.5f, int n_iter = 100);

	//void displayQualityGPU(DMOMeshTri& mesh, int n_cols);

	
	class DMOBaseClass {
	public:
		//DMOBaseClass() = delete;
		DMOBaseClass(QualityCriterium qualityCriterium_ = QualityCriterium::MEAN_RATIO, const float gridScale_ = 0.5f, int n_iter_ = 100)
		: qualityCriterium(qualityCriterium_)
		, gridScale(gridScale_)
		, n_iter(n_iter_)
		{

		}
		
		//void discreteMeshOptimization();
		virtual void displayQualityGPU(int n_cols = 10) = 0;
		virtual bool isDone() = 0;
		virtual void doIteration() = 0;
		virtual void getEstimateLocalSurfacePoints(int vid, int nu, int nv, void* outSurfacePoints) = 0;
		virtual void getLocalSurfacePoints(int vid, int nu, int nv, void* outSurfacePoints, int featureSid) = 0;
		virtual void getElementQualities(void* outFloatBuffer) = 0;
		float getQuality() const { return lastMinQuality; }
		virtual void getQualityHistogram(std::vector<int>& vec, int n_cols) = 0;

		void optimize() {
			cout << "Running DMO, Iterations=" << n_iter << endl;
			Timer t;
			cout << "0," << lastMinQuality << endl;
			while (curr_it < n_iter) { // incremented in called func
				doIteration();
			}
			cout << "DMO Time: " << t.timeInSeconds() << "s" << endl;
		}

	protected:
		QualityCriterium qualityCriterium;
		const float gridScale;
		int n_iter;
		int curr_it = 0;
		float lastMinQuality = 0.f;

		void printFormattedQuality(host_vector<int> q_vecHost, host_vector<float> q_minHost) {
			int n_cols = (int)q_vecHost.size();
			int maxnum = *std::max_element(q_vecHost.begin(), q_vecHost.end());
			int maxlen = std::max((int)log10(maxnum) + 1, 5);
			cout << "interval ";
			for (int i = 0; i < n_cols; ++i) {
				float intervalstart = (float)i / (float)q_vecHost.size();
				std::string sinterval = std::to_string(intervalstart).substr(0, 4);
				cout << sinterval;
				for (int j = 0; j < maxlen - (int)sinterval.size(); ++j) {
					cout << " |";
				}
			}
			cout << endl;
			cout << "amount   ";
			for (size_t i = 0; i < q_vecHost.size(); ++i) {
				int num = q_vecHost[i];
				std::string snum = std::to_string(num);
				cout << snum;
				for (int j = 0; j < maxlen - (int)snum.size(); ++j) {
					cout << " ";
				}
				cout << "|";
			}
			cout << endl;
			cout << "q_min = " << q_minHost[0] << endl;
		}
	};

}



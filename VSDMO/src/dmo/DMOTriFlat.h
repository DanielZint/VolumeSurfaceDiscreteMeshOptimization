#pragma once

#include "cuda_runtime.h"
#include "mesh/MeshTriGPU.h"
#include "ConfigUsing.h"
#include "DMOConfig.h"
#include "DMOQuality.h"
#include "DMOBase.h"
#include "SurfaceConfig.h"


//using namespace SurfLS;
//using namespace Surf1D;

namespace DMO {

	//void discreteMeshOptimization(DMOMeshTri& dmo_mesh, QualityCriterium qualityCriterium = MEAN_RATIO, const float gridScale = 0.5f, int n_iter = 100);

	//void displayQualityGPU(DMOMeshTri& mesh, int n_cols);

	
	class DMOTriFlatClass : public DMOBaseClass {
	public:
		DMOTriFlatClass() = delete;
		DMOTriFlatClass(DMOMeshTri& dmo_mesh_, QualityCriterium qualityCriterium_ = QualityCriterium::MIN_ANGLE, const float gridScale_ = 0.5f, int n_iter_ = 30);

		bool isDone() { return curr_it >= n_iter; }
		virtual void doIteration();

		void getEstimateLocalSurfacePoints(int vid, int nu, int nv, void* outSurfacePoints) override;
		void getLocalSurfacePoints(int vid, int nu, int nv, void* outSurfacePoints, int featureSid) override;

		void displayQualityGPU(int n_cols = 10);
		float findMinimumQuality() const;
		void updateCurrentQualities();
		void getElementQualities(void* outFloatBuffer);
		void getQualityHistogram(std::vector<int>& vec, int n_cols) override;

		//float getQuality() const override { return lastMinQuality; }

	protected:
		void init();

		DMOMeshTri& dmo_mesh;
		//QualityCriterium qualityCriterium;
		//const float gridScale;
		//int n_iter;
		//int curr_it = 0;

		//float lastMinQuality;

		float affineFactor = 1.f / (float)(DMO_NQ - 1);

		device_vector<Vec3f> vertexPointsInit; // copy of initial vertex positions
		device_vector<localSurface1d> localSurfacesInit1d; // initial local surfaces1d
		device_vector<int> nearestNeighbors;
		device_vector<bool> optimizeFeatureVertex;


		device_vector<float> currentQualities;

	};

}



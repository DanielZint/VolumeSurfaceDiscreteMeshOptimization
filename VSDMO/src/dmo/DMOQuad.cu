#include "DMOQuad.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <thrust/count.h>
#include <thrust/sequence.h>
#include "CudaUtil.h"
#include "CudaAtomic.h"
#include "DMOCommon.h"
#include "SurfaceConfig.h"

#include "io/FileWriter.h"
#include "Serializer.h"

#include "DMOQuadImplCub.h"


#ifdef USECUB
using namespace DMOImplCub;
#else
using namespace DMOImplAtomic;
#endif

namespace DMO {
	

	// counts the number of elements with quality between 0 - 0.1 and so on.
	__global__ static void k_qualityHistogram(MeshQuadDevice* mesh, ArrayView<int> q_vec, ArrayView<float> q_min, int n_cols, QualityCriterium q_crit) {
		for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < mesh->nQuads; idx += blockDim.x * gridDim.x) {
			const Quad& quad = mesh->quads[idx];
			const Vec3f points[4] = { mesh->vertexPoints[quad.v0], mesh->vertexPoints[quad.v1], mesh->vertexPoints[quad.v2], mesh->vertexPoints[quad.v3] };
			float q = qualityQuad(points, q_crit); // TODO qualityQuad
			myAtomicMin(&q_min[0], q);

			q = fminf(0.9999f, q);
			size_t index = size_t(q * n_cols);
			atomicAdd(&q_vec[index], 1);
		}
	}

	// finds the minimum element quality
	__global__ static void k_findMinimumQuality(MeshQuadDevice* mesh, ArrayView<float> q_min, QualityCriterium q_crit) {
		for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < mesh->nQuads; idx += blockDim.x * gridDim.x) {
			const Quad& quad = mesh->quads[idx];
			const Vec3f points[4] = { mesh->vertexPoints[quad.v0], mesh->vertexPoints[quad.v1], mesh->vertexPoints[quad.v2], mesh->vertexPoints[quad.v3] };
			float q = qualityQuad(points, q_crit);
			myAtomicMin(&q_min[0], q);
		}
	}

	// checks for non decreasing quality of every element
	__global__ static void k_updateCurrentQualities(MeshQuadDevice* mesh, ArrayView<float> currentQualities, float lastMinQuality, ArrayView<bool> failure, QualityCriterium q_crit) {
		for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < mesh->nQuads; idx += blockDim.x * gridDim.x) {
			const Quad& quad = mesh->quads[idx];
			const Vec3f points[4] = { mesh->vertexPoints[quad.v0], mesh->vertexPoints[quad.v1], mesh->vertexPoints[quad.v2], mesh->vertexPoints[quad.v3] };
			float q = qualityQuad(points, q_crit);
			if (q < lastMinQuality) {
				failure[0] = true;
				printf("minimum quality decreased! quad %i vertices %i %i %i q %f\n", idx, quad.v0, quad.v1, quad.v2, q);
				assert(0);
			}
			currentQualities[idx] = q;
		}
	}

	__global__ static void k_getElementQualities(MeshQuadDevice* mesh, float* outBuffer, QualityCriterium q_crit) {
		for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < mesh->nQuads; idx += blockDim.x * gridDim.x) {
			const Quad& quad = mesh->quads[idx];
			const Vec3f points[4] = { mesh->vertexPoints[quad.v0], mesh->vertexPoints[quad.v1], mesh->vertexPoints[quad.v2], mesh->vertexPoints[quad.v3] };
			outBuffer[idx] = qualityQuad(points, q_crit);
		}
	}


	
	// ######################################################################## //
	// ### DMOTriClass ######################################################## //
	// ######################################################################## //

	DMOQuadClass::DMOQuadClass(DMOMeshQuad& dmo_mesh_, QualityCriterium qualityCriterium_, const float gridScale_, int n_iter_)
		: DMOBaseClass(qualityCriterium_, gridScale_, n_iter_)
		, dmo_mesh(dmo_mesh_)
		//, qualityCriterium(qualityCriterium_)
		//, gridScale(gridScale_)
		//, n_iter(n_iter_)
	{
		init();
	}

	void DMOQuadClass::init() {
		vertexPointsInit = device_vector<Vec3f>(dmo_mesh.vertexPoints, dmo_mesh.vertexPoints + dmo_mesh.nVerticesSurf); // copy of initial vertex positions

		localSurfacesInit1d = device_vector<localSurface1d>(dmo_mesh.nVerticesSurf); // initial local surfaces1d
		computeLocalSurfaces1d(dmo_mesh, localSurfacesInit1d);

		localSurfacesInit = device_vector<localSurface>(dmo_mesh.nVerticesSurf); // initial local surfaces
		computeLocalSurfaces(dmo_mesh, localSurfacesInit);

		// new
		if (dmo_mesh.nVerticesFeature > 0)
			initLocalSurfacesFeature(dmo_mesh, surfacesRowPtr, localSurfacesFeatureInit, table);

		//writeSurfaces(localSurfacesInit, "res/surfaces/surfaces2d2.binary");

		nearestNeighbors = device_vector<int>(dmo_mesh.nVerticesSurf);
		thrust::sequence(nearestNeighbors.begin(), nearestNeighbors.end(), 0);

		optimizeFeatureVertex = device_vector<bool>(dmo_mesh.nVerticesSurf, true);
		calcOptFeatureVec(dmo_mesh, optimizeFeatureVertex);

		lastMinQuality = findMinimumQuality();

		cout << "Total points " << dmo_mesh.nVerticesSurf << endl;
		cout << "Free Surf points " << dmo_mesh.nVerticesSurf - dmo_mesh.nVerticesFeature << endl;
		cout << "Feature Surf points " << dmo_mesh.nVerticesFeature << endl;

		cout << "Colors Free " << dmo_mesh.nColorsFree << endl;
		cout << "Colors Feature " << dmo_mesh.nColorsFeature << endl;

		currentQualities = device_vector<float>(dmo_mesh.nQuads, -FLT_MAX);

		cout << "0," << lastMinQuality << endl;
	}


	void DMOQuadClass::doIteration() {
		int dynMemSize2D = (2 * dmo_mesh.maxNumHalfedges + 1) * 3 * sizeof(float) + dmo_mesh.maxNumHalfedges * sizeof(int);
		int dynMemSize1D = (2 * dmo_mesh.maxNumHalfedges + 1) * 3 * sizeof(float) + dmo_mesh.maxNumHalfedges * sizeof(int);

		for (int cid = 0; cid < dmo_mesh.nColorsFree; ++cid) {
			//cout << "color offsets " << (dmo_mesh.colorOffsetsFree[cid + 1] - dmo_mesh.colorOffsetsFree[cid]) << endl;
			//cout << dynMemSize2D << endl;
			k_optimizeHierarchical2D<DMO_NQ* DMO_NQ/2, USE_SURF_OF_NN> << <dmo_mesh.colorOffsetsFree[cid + 1] - dmo_mesh.colorOffsetsFree[cid], DMO_NQ* DMO_NQ/2, dynMemSize2D >> >
				(dmo_mesh.colorOffsetsFree[cid], dmo_mesh.colorOffsetsFree[cid + 1], dmo_mesh.d_mesh, affineFactor, qualityCriterium, gridScale,
					raw_pointer_cast(vertexPointsInit.data()), raw_pointer_cast(localSurfacesInit.data()), raw_pointer_cast(nearestNeighbors.data()),
					raw(table), raw(surfacesRowPtr), raw(localSurfacesFeatureInit), 2 * dmo_mesh.maxNumHalfedges + 1);
			gpuErrchk(cudaPeekAtLastError());
			gpuErrchk(cudaDeviceSynchronize());
		}

		for (int cid = 0; cid < dmo_mesh.nColorsFeature; ++cid) {
			k_optimizeHierarchical1D<DMO_NQ> << <dmo_mesh.colorOffsetsFeature[cid + 1] - dmo_mesh.colorOffsetsFeature[cid], DMO_NQ, dynMemSize1D >> >
				(dmo_mesh.colorOffsetsFeature[cid], dmo_mesh.colorOffsetsFeature[cid + 1], dmo_mesh.d_mesh, affineFactor, qualityCriterium, gridScale,
					raw_pointer_cast(vertexPointsInit.data()), raw_pointer_cast(localSurfacesInit1d.data()), raw_pointer_cast(nearestNeighbors.data()),
					raw_pointer_cast(optimizeFeatureVertex.data()), 2 * dmo_mesh.maxNumHalfedges + 1, dmo_mesh.maxNumHalfedges);
			gpuErrchk(cudaPeekAtLastError());
			gpuErrchk(cudaDeviceSynchronize());
		}


		updateNearestNeighbor(dmo_mesh, vertexPointsInit, nearestNeighbors);
		dmo_mesh.updateNormals();
		++curr_it;

		updateCurrentQualities();
		float newQuality = findMinimumQuality();
		//cout << "new quality: " << newQuality << endl;
		cout << curr_it << "," << newQuality << endl;
		assert(newQuality >= lastMinQuality);
		lastMinQuality = newQuality;
	}





	void DMOQuadClass::getEstimateLocalSurfacePoints(int vid, int nu, int nv, void* outSurfacePoints) {
		int dynMemSize = dmo_mesh.maxNumHalfedges * sizeof(int);
		k_fillEstimateLocalSurfacePoints<USE_SURF_OF_NN> << <1, nu* nv, dynMemSize>> > (vid, nu, nv, dmo_mesh.d_mesh, affineFactor, gridScale,
			raw_pointer_cast(vertexPointsInit.data()), raw_pointer_cast(localSurfacesInit.data()), raw_pointer_cast(nearestNeighbors.data()), (Vec3f*)outSurfacePoints,
			raw(table), raw(surfacesRowPtr), raw(localSurfacesFeatureInit));
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
	}

	void DMOQuadClass::getLocalSurfacePoints(int vid, int nu, int nv, void* outSurfacePoints, int featureSid) {
		k_fillLocalSurfacePoints << <1, nu* nv >> > (vid, featureSid, nu, nv, dmo_mesh.d_mesh, affineFactor, gridScale,
			raw(localSurfacesInit), raw(nearestNeighbors), (Vec3f*)outSurfacePoints, raw(localSurfacesFeatureInit), raw(surfacesRowPtr));
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
	}

	void DMOQuadClass::displayQualityGPU(int n_cols) {
		device_vector<int> q_vec(n_cols, 0);
		device_vector<float> q_min(1, FLT_MAX);
		const int BLOCK_SIZE = 128;
		k_qualityHistogram << <getBlockCount(dmo_mesh.nQuads, BLOCK_SIZE), BLOCK_SIZE >> > (dmo_mesh.d_mesh, q_vec, q_min, n_cols, qualityCriterium);
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());

		host_vector<int> q_vecHost(q_vec);
		host_vector<float> q_minHost(q_min);

		printFormattedQuality(q_vecHost, q_minHost);
	}

	void DMOQuadClass::getQualityHistogram(std::vector<int>& vec, int n_cols) {
		device_vector<int> q_vec(n_cols, 0);
		device_vector<float> q_min(1, FLT_MAX);
		const int BLOCK_SIZE = 128;
		k_qualityHistogram << <getBlockCount(dmo_mesh.nQuads, BLOCK_SIZE), BLOCK_SIZE >> > (dmo_mesh.d_mesh, q_vec, q_min, n_cols, qualityCriterium);
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());

		host_vector<int> q_vecHost(q_vec);
		thrust::copy(q_vec.begin(), q_vec.end(), vec.begin());
	}

	float DMOQuadClass::findMinimumQuality() const {
		device_vector<float> q_min(1, FLT_MAX);
		const int BLOCK_SIZE = 128;
		k_findMinimumQuality << <getBlockCount(dmo_mesh.nQuads, BLOCK_SIZE), BLOCK_SIZE >> > (dmo_mesh.d_mesh, q_min, qualityCriterium);
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
		return q_min[0];
	}

	void DMOQuadClass::updateCurrentQualities() {
		const int BLOCK_SIZE = 128;
		device_vector<bool> failure(1, false);
		k_updateCurrentQualities << <getBlockCount(dmo_mesh.nQuads, BLOCK_SIZE), BLOCK_SIZE >> > (dmo_mesh.d_mesh, currentQualities, lastMinQuality, failure, qualityCriterium);
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
		if (failure[0]) {
			writeOFF("res/fail_debug", dmo_mesh);
			throw 1;
		}
	}

	void DMOQuadClass::getElementQualities(void* outFloatBuffer) {
		const int BLOCK_SIZE = 128;
		k_getElementQualities << <getBlockCount(dmo_mesh.nQuads, BLOCK_SIZE), BLOCK_SIZE >> > (dmo_mesh.d_mesh, (float*)outFloatBuffer, qualityCriterium);
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
	}

}



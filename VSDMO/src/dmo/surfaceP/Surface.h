#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cublas_v2.h>


#include "mesh/MeshTriGPU.h"
#include "mesh/DMOMeshTri.h"
#include "mesh/DMOMeshTet.h"
#include "ConfigUsing.h"
#include "CudaUtil.h"
#include "dmo/SurfaceCommon.h"


namespace SurfLS {

	struct localSurface {
		// Origin of the local coordinate system
		Vec3f p0;
		// Axes of the local coordinate system
		Vec3f ei[3];
		// Coefficients for the polynomial function
		float a, b, c;
		// compute z-coordinate
		//		z = a * x * x + b * x * y + c * y * y
		// transforming back to global coordinates:
		//		glob = p + u * x + v * y + w * z
		int numPoints;

		// Compute v(u) = a*u*u
		__host__ __device__ float polynomialFunction(const float u, const float v) const {
			return a * u * u + b * u * v + c * v * v;
		}

		// Transform from local to world coordinates.
		__host__ __device__ Vec3f localToWorldCoords(const float u, const float v, const float w) const {
			return p0 + u * ei[0] + v * ei[1] + w * ei[2];
		}

		__host__ __device__ Vec3f localToWorldCoords(const float u, const float v) const {
			float w = polynomialFunction(u, v);
			return localToWorldCoords(u, v, w);
		}

		__host__ __device__ Vec3f worldToLocalCoords(const Vec3f& worldCoords) const {
			Vec3f p1 = worldCoords - p0;
			float u = p1.dot(ei[0]);
			float v = p1.dot(ei[1]);
			float w = p1.dot(ei[2]);

			return { u,v,w };
		}

		__host__ __device__ void setZero() {
			a = 0.f;
			b = 0.f;
			c = 0.f;
		}
	};





	__device__ inline void computeLocalCoordinates(MeshBaseDevice* mesh, const int vid, const int neigh, Vec3f normal, Vec3f* xi1, Vec3f* xi2, Vec3f* xi3) {
		// construct a random vector
		Vec3f randVec = mesh->vertexPoints[neigh] - mesh->vertexPoints[vid];

		if (normal.cross(randVec).norm() < 10.f * EPSILON) {
			printf("computeLocalCoordinates: normal vector needs to be recalculated\n");
			//assert(0);
			randVec = { normal[0], normal[2], -normal[1] };
		}

		// compute tanget vectors
		Vec3f t1 = normal.cross(randVec);
		Vec3f t2 = normal.cross(t1);

		// normalize
		normal.normalize();
		t1.normalize();
		t2.normalize();

		if (std::abs(t1.cross(t2).dot(normal) - 1) > 10.f * EPSILON) {
			printf("computeLocalCoordinates: computation of local coordinates failed for vertex %i\n", vid);
			assert(0);
		}

		*xi1 = t1;
		*xi2 = t2;
		*xi3 = normal;
	}


	__device__ inline const localSurface& getNearestNeighborSurf(localSurface* localSurfacesInit, int* nearestNeighbors, const int vid) {
		// surface of nearest neighbor of vh
		int nn = nearestNeighbors[vid];
		return localSurfacesInit[nn];
	}






	void inline checkcuBLASError(cublasStatus_t status, const char* msg) {
		if (status != CUBLAS_STATUS_SUCCESS)
		{
			printf("%s", msg);
			exit(EXIT_FAILURE);
		}
	}

	__global__ void k_computeLocalSurfaces1(MeshBaseDevice* mesh, localSurface* localSurfaces, float* outMatrixData, float* outRhsData, float* outMatrixPtrs[], float* outRhsPtrs[]);

	__global__ void k_computeLocalSurfaces2(MeshBaseDevice* mesh, localSurface* localSurfaces, float* abcVec);



	template<class DMOMesh>
	__host__ void computeLocalSurfaces(DMOMesh& mesh, device_vector<localSurface>& localSurfaces) {
		const int BLOCK_SIZE = 128;
		int nVerticesSurf = mesh.nVerticesSurf;

		float* ATAData;
		float* ATbData;
		int* pivotArray;
		int* infoArray;
		float** outMatrixPtrs;
		float** outRhsPtrs;

		gpuErrchk(cudaMalloc(&ATAData, nVerticesSurf * 9 * sizeof(float)));
		gpuErrchk(cudaMalloc(&ATbData, nVerticesSurf * 3 * sizeof(float)));
		gpuErrchk(cudaMalloc(&pivotArray, nVerticesSurf * 3 * sizeof(int)));
		gpuErrchk(cudaMalloc(&infoArray, nVerticesSurf * sizeof(int)));
		gpuErrchk(cudaMalloc(&outMatrixPtrs, nVerticesSurf * sizeof(float*)));
		gpuErrchk(cudaMalloc(&outRhsPtrs, nVerticesSurf * sizeof(float*)));

		k_computeLocalSurfaces1 << <getBlockCount(mesh.nVerticesSurf, BLOCK_SIZE), BLOCK_SIZE >> > (mesh.d_mesh, raw_pointer_cast(localSurfaces.data()), ATAData, ATbData, outMatrixPtrs, outRhsPtrs);
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());

		cublasHandle_t handle;
		checkcuBLASError(cublasCreate(&handle), "cublasCreate() failed!\n");

		int lda = 3;
		checkcuBLASError(cublasSgetrfBatched(handle, 3, outMatrixPtrs, lda, pivotArray, infoArray, nVerticesSurf), "cublasSgetrfBatched() failed!");

		int info;
		checkcuBLASError(cublasSgetrsBatched(handle, CUBLAS_OP_N, 3, 1, outMatrixPtrs, lda, pivotArray, outRhsPtrs, lda, &info, nVerticesSurf), "cublasSgetrsBatched() failed!");
		if (info) {
			printf("cublasSgetrsBatched info %i", info);
			exit(EXIT_FAILURE);
		}
		//ATbData now holds the solution (x) data of ATA * x = ATb

		k_computeLocalSurfaces2 << <getBlockCount(mesh.nVerticesSurf, BLOCK_SIZE), BLOCK_SIZE >> > (mesh.d_mesh, raw_pointer_cast(localSurfaces.data()), ATbData);
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());

		gpuErrchk(cudaFree(ATAData));
		gpuErrchk(cudaFree(ATbData));
		gpuErrchk(cudaFree(pivotArray));
		gpuErrchk(cudaFree(infoArray));
		gpuErrchk(cudaFree(outMatrixPtrs));
		gpuErrchk(cudaFree(outRhsPtrs));
		cublasDestroy(handle);
	}



}

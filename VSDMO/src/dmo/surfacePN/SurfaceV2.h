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


namespace SurfV2 {

	struct localSurface {
		// Origin of the local coordinate system
		Vec3f p0;
		// Axes of the local coordinate system
		Vec3f ei[3];
		// Coefficients for the polynomial function
		float b0, b1, b2, c0, c1, c2, c3;
		// compute z-coordinate
		//		z = a * x * x + b * x * y + c * y * y
		// transforming back to global coordinates:
		//		glob = p + u * x + v * y + w * z
		int numPoints;

		// Compute v(u) = a*u*u
		__host__ __device__ float polynomialFunction(const float u, const float v) const {
			return b0 * u * u + b1 * u * v + b2 * v * v + c0 * u * u * u + c1 * u * u * v + c2 * u * v * v + c3 * v * v * v;
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
			b0 = 0.f;
			b1 = 0.f;
			b2 = 0.f;
			c0 = 0.f;
			c1 = 0.f;
			c2 = 0.f;
			c3 = 0.f;
		}
	};





	__device__ inline void computeLocalCoordinates(MeshBaseDevice* mesh, const int vid, const int neigh, Vec3f normal, Vec3f* xi1, Vec3f* xi2, Vec3f* xi3) {
		//Vec3f normal = mesh->vertexNormals[vid];

		// construct a random vector
		//const Halfedge& halfedge = *(mesh->he_ccw_order_begin(vid));
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
		//const int BLOCK_SIZE = 128;
		//int nVerticesSurf = mesh.nVerticesSurf;
		//
		//float* ATAData;
		//float* ATbData;
		//int* infoArray;
		//float** outMatrixPtrs;
		//float** outRhsPtrs;
		//
		//gpuErrchk(cudaMalloc(&ATAData, nVerticesSurf * 49 * sizeof(float)));
		//gpuErrchk(cudaMalloc(&ATbData, nVerticesSurf * 7 * sizeof(float)));
		//gpuErrchk(cudaMalloc(&infoArray, nVerticesSurf * sizeof(int)));
		//gpuErrchk(cudaMalloc(&outMatrixPtrs, nVerticesSurf * sizeof(float*)));
		//gpuErrchk(cudaMalloc(&outRhsPtrs, nVerticesSurf * sizeof(float*)));
		//
		//k_computeLocalSurfaces1 << <getBlockCount(mesh.nVerticesSurf, BLOCK_SIZE), BLOCK_SIZE >> > (mesh.d_mesh, raw_pointer_cast(localSurfaces.data()), ATAData, ATbData, outMatrixPtrs, outRhsPtrs);
		//gpuErrchk(cudaPeekAtLastError());
		//gpuErrchk(cudaDeviceSynchronize());
		//
		//cublasHandle_t handle;
		//checkcuBLASError(cublasCreate(&handle), "cublasCreate() failed!\n");
		//
		//int lda = 7;
		//int info;
		//checkcuBLASError(cublasSgelsBatched(handle, CUBLAS_OP_N, 7, 7, 1, outMatrixPtrs, lda, outRhsPtrs, lda, &info, infoArray, nVerticesSurf), "cublasSgelsBatched() failed!");
		//if (info) {
		//	printf("cublasSgelsBatched info %i", info);
		//	exit(EXIT_FAILURE);
		//}
		////ATbData now holds the least squares solution (x) data of ATA * x = ATb
		//
		//k_computeLocalSurfaces2 << <getBlockCount(mesh.nVerticesSurf, BLOCK_SIZE), BLOCK_SIZE >> > (mesh.d_mesh, raw_pointer_cast(localSurfaces.data()), ATbData);
		//gpuErrchk(cudaPeekAtLastError());
		//gpuErrchk(cudaDeviceSynchronize());
		//
		//gpuErrchk(cudaFree(ATAData));
		//gpuErrchk(cudaFree(ATbData));
		//gpuErrchk(cudaFree(infoArray));
		//gpuErrchk(cudaFree(outMatrixPtrs));
		//gpuErrchk(cudaFree(outRhsPtrs));
		//cublasDestroy(handle);



		const int BLOCK_SIZE = 128;
		int nVerticesSurf = mesh.nVerticesSurf;

		float* ATAData;
		float* ATbData;
		int* pivotArray;
		int* infoArray;
		float** outMatrixPtrs;
		float** outRhsPtrs;

		gpuErrchk(cudaMalloc(&ATAData, nVerticesSurf * 49 * sizeof(float)));
		gpuErrchk(cudaMalloc(&ATbData, nVerticesSurf * 7 * sizeof(float)));
		gpuErrchk(cudaMalloc(&pivotArray, nVerticesSurf * 7 * sizeof(int)));
		gpuErrchk(cudaMalloc(&infoArray, nVerticesSurf * sizeof(int)));
		gpuErrchk(cudaMalloc(&outMatrixPtrs, nVerticesSurf * sizeof(float*)));
		gpuErrchk(cudaMalloc(&outRhsPtrs, nVerticesSurf * sizeof(float*)));

		k_computeLocalSurfaces1 << <getBlockCount(mesh.nVerticesSurf, BLOCK_SIZE), BLOCK_SIZE >> > (mesh.d_mesh, raw_pointer_cast(localSurfaces.data()), ATAData, ATbData, outMatrixPtrs, outRhsPtrs);
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());

		cublasHandle_t handle;
		checkcuBLASError(cublasCreate(&handle), "cublasCreate() failed!\n");

		int lda = 7;
		checkcuBLASError(cublasSgetrfBatched(handle, 7, outMatrixPtrs, lda, pivotArray, infoArray, nVerticesSurf), "cublasSgetrfBatched() failed!");

		int info;
		checkcuBLASError(cublasSgetrsBatched(handle, CUBLAS_OP_N, 7, 1, outMatrixPtrs, lda, pivotArray, outRhsPtrs, lda, &info, nVerticesSurf), "cublasSgetrsBatched() failed!");
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



	__device__ inline void addPoint(const int vid, MeshBaseDevice* mesh, int dst, const Vec3f& p0, const Vec3f& n0, Vec3f* ei, float* outMatrixData, float* outRhsData) {
		Vec3f p = mesh->vertexPoints[dst] - p0;
		const Vec3f& n = mesh->vertexNormals[dst];

		const float u = p.dot(ei[0]);
		const float v = p.dot(ei[1]);
		const float w = p.dot(ei[2]);
		const float n1 = n.dot(ei[0]);
		const float n2 = n.dot(ei[1]);
		const float n3 = n.dot(ei[2]);

		float a[7] = { u * u, u * v, v * v, u * u * u, u * u * v, u * v * v, v * v * v };
		float b[7] = { n3 * 2.f * u, n3 * v, 0.f, n3 * 3.f * u * u, n3 * 2.f * u * v, n3 * v * v, 0.f };
		float c[7] = { 0.f, n3 * u, n3 * 2.f * v, 0.f, n3 * u * u, n3 * 2.f * u * v, n3 * 3.f * v * v };

		for (int row = 0; row < 7; ++row) {
			for (int col = 0; col < 7; ++col) {
				outMatrixData[vid * 49 + (col * 7 + row)] += a[row] * a[col] + b[row] * b[col] + c[row] * c[col];
			}
		}
		for (int col = 0; col < 7; ++col) {
			outRhsData[vid * 7 + col] += w * a[col] - n1 * b[col] - n2 * c[col];
		}
	}



}

#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cublas_v2.h>


#include "mesh/MeshTriGPU.h"
#include "ConfigUsing.h"
#include "CudaUtil.h"
#include "SurfaceV2.h"


namespace SurfV2 {



	__global__ void k_computeLocalSurfacesFeature1(MeshBaseDevice* mesh, int nSurfaces, localSurface* localSurfaces, float* outMatrixData, float* outRhsData, float* outMatrixPtrs[], float* outRhsPtrs[], int* neighbors, int* numNeighbors);

	__global__ void k_computeLocalSurfacesFeature2(MeshBaseDevice* mesh, int nSurfaces, localSurface* localSurfaces, float* abcVec);

	template<class DMOMesh>
	__host__ void computeLocalSurfacesFeature(DMOMesh& mesh, device_vector<localSurface>& localSurfaces, device_vector<int>& neighbors, device_vector<int>& numNeighbors) {
		//return;//TODO
		const int BLOCK_SIZE = 128;
		int nSurfaces = (int)localSurfaces.size();

		float* ATAData;
		float* ATbData;
		int* pivotArray;
		int* infoArray;
		float** outMatrixPtrs;
		float** outRhsPtrs;

		gpuErrchk(cudaMalloc(&ATAData, nSurfaces * 49 * sizeof(float)));
		gpuErrchk(cudaMalloc(&ATbData, nSurfaces * 7 * sizeof(float)));
		gpuErrchk(cudaMalloc(&pivotArray, nSurfaces * 7 * sizeof(int)));
		gpuErrchk(cudaMalloc(&infoArray, nSurfaces * sizeof(int)));
		gpuErrchk(cudaMalloc(&outMatrixPtrs, nSurfaces * sizeof(float*)));
		gpuErrchk(cudaMalloc(&outRhsPtrs, nSurfaces * sizeof(float*)));

		k_computeLocalSurfacesFeature1 << <getBlockCount(nSurfaces, BLOCK_SIZE), BLOCK_SIZE >> > (mesh.d_mesh, nSurfaces, raw(localSurfaces), ATAData, ATbData, outMatrixPtrs, outRhsPtrs, raw(neighbors), raw(numNeighbors));
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());

		cublasHandle_t handle;
		checkcuBLASError(cublasCreate(&handle), "cublasCreate() failed!\n");

		int lda = 7;
		checkcuBLASError(cublasSgetrfBatched(handle, 7, outMatrixPtrs, lda, pivotArray, infoArray, nSurfaces), "cublasSgetrfBatched() failed!");

		int info;
		checkcuBLASError(cublasSgetrsBatched(handle, CUBLAS_OP_N, 7, 1, outMatrixPtrs, lda, pivotArray, outRhsPtrs, lda, &info, nSurfaces), "cublasSgetrsBatched() failed!");
		if (info) {
			printf("cublasSgetrsBatched info %i", info);
			exit(EXIT_FAILURE);
		}
		//ATbData now holds the solution (x) data of ATA * x = ATb

		k_computeLocalSurfacesFeature2 << <getBlockCount(nSurfaces, BLOCK_SIZE), BLOCK_SIZE >> > (mesh.d_mesh, nSurfaces, raw(localSurfaces), ATbData);
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

	__global__ void k_countLocalSurfacesFeature(MeshBaseDevice* mesh, int* surfaceCounts, int n, int start);


	__global__ void k_setNeighborsFeature(MeshBaseDevice* mesh, const int n, const int start, const int* rowPtr, int* neighbors, int* numNeighbors, int* table);


	template<class DMOMesh>
	inline void initLocalSurfacesFeature(DMOMesh& dmo_mesh, device_vector<int>& rowPtr, device_vector<localSurface>& surfaces, device_vector<int>& table) {
		const int BLOCK_SIZE = 128;
		int nVerticesFeature = dmo_mesh.nVerticesFeature;
		int nVerticesSurfFree = dmo_mesh.nVerticesSurf - dmo_mesh.nVerticesFeature;
		int startFeatureVertices = nVerticesSurfFree;
		device_vector<int> counts(nVerticesFeature);
		// count number of surfaces per feature node
		k_countLocalSurfacesFeature << <getBlockCount(nVerticesFeature, BLOCK_SIZE), BLOCK_SIZE >> > (dmo_mesh.d_mesh, raw(counts), nVerticesFeature, startFeatureVertices);
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
		// compress to column pointer format
		rowPtr = device_vector<int>(nVerticesFeature + 1);
		rowPtr[0] = 0;
		thrust::inclusive_scan(counts.begin(), counts.end(), rowPtr.begin() + 1);
		int numSurfaces = rowPtr[nVerticesFeature];
		//int numSurfacesTotal = numSurfaces + nVerticesSurfFree;
		device_vector<int> neighbors(numSurfaces * MAX_NEIGHBORS);
		device_vector<int> numNeighbors(numSurfaces);
		table = device_vector<int>(nVerticesFeature * TABLE_SIZE, -1);
		// calc neighbors, numNeighbors, table
		k_setNeighborsFeature << <getBlockCount(nVerticesFeature, BLOCK_SIZE), BLOCK_SIZE >> >
			(dmo_mesh.d_mesh, nVerticesFeature, startFeatureVertices, raw(rowPtr), raw(neighbors), raw(numNeighbors), raw(table));
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
		// compute surfaces from neighbors
		surfaces = device_vector<localSurface>(numSurfaces);
		computeLocalSurfacesFeature(dmo_mesh, surfaces, neighbors, numNeighbors);

	}








}

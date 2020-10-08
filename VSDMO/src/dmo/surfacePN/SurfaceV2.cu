#include "SurfaceV2.h"
#include "CudaUtil.h"
#include <cuda_runtime.h>
//#include <helper_cuda.h>
#include "device_launch_parameters.h"

namespace SurfV2 {


	__device__ void computeLocalSurface1(const int vid, MeshBaseDevice* mesh, localSurface* localSurfaces, float* outMatrixData, float* outRhsData) {
		const Vec3f& p0 = mesh->vertexPoints[vid];
		const Vec3f& n0 = mesh->vertexNormals[vid];
		Vec3f ei[3];
		int neighId = mesh->he_ccw_order_begin(vid)->targetVertex;
		computeLocalCoordinates(mesh, vid, neighId, mesh->vertexNormals[vid], &ei[0], &ei[1], &ei[2]);

		for (int row = 0; row < 7; ++row) {
			for (int col = 0; col < 7; ++col) {
				outMatrixData[vid * 49 + (col * 7 + row)] = 0.f;
			}
			outRhsData[vid * 7 + row] = 0.f;
		}
	
		int sz = 1 + (size_t)mesh->halfedges.rowPtr_[vid + 1] - (size_t)mesh->halfedges.rowPtr_[vid];
		int nrEqs = 3 * sz;
		int unknowns = 7;
	
		addPoint(vid, mesh, vid, p0, n0, ei, outMatrixData, outRhsData);
		for (auto it = mesh->he_ccw_order_begin(vid); it != mesh->he_ccw_order_end(vid); ++it) {
			int dst = it->targetVertex;
			addPoint(vid, mesh, dst, p0, n0, ei, outMatrixData, outRhsData);
		}


		localSurface& para = localSurfaces[vid];
		para.p0 = p0;
		para.ei[0] = ei[0];
		para.ei[1] = ei[1];
		para.ei[2] = ei[2];
		para.numPoints = sz;
	}

	__device__ void computeLocalSurface2(const int vid, localSurface* localSurfaces, float* abcVec) {
		localSurface& para = localSurfaces[vid];
		para.b0 = abcVec[vid * 7 + 0];
		para.b1 = abcVec[vid * 7 + 1];
		para.b2 = abcVec[vid * 7 + 2];
		para.c0 = abcVec[vid * 7 + 3];
		para.c1 = abcVec[vid * 7 + 4];
		para.c2 = abcVec[vid * 7 + 5];
		para.c3 = abcVec[vid * 7 + 6];

		//printf("%f %f %f %f %f %f %f\n", para.b0, para.b1, para.b2, para.c0, para.c1, para.c2, para.c3);
		// this is also a hack.
		// sometimes ATA * x = ATb is underdetermined and we get NANs

		if (para.numPoints <= 4) {
			para.b0 = 0;
			para.b1 = 0;
			para.b2 = 0;
			para.c0 = 0;
			para.c1 = 0;
			para.c2 = 0;
			para.c3 = 0;
		}

		if (para.b0 != para.b0) {
			para.b0 = 0;
			printf("b0 nan, numPoints: %i\n", para.numPoints);
		}
		if (para.b1 != para.b1) {
			para.b1 = 0;
			printf("b1 nan, numPoints: %i\n", para.numPoints);
		}
		if (para.b2 != para.b2) {
			para.b2 = 0;
			printf("b2 nan, numPoints: %i\n", para.numPoints);
		}
		if (para.c0 != para.c0) {
			para.c0 = 0;
			printf("c0 nan, numPoints: %i\n", para.numPoints);
		}
		if (para.c1 != para.c1) {
			para.c1 = 0;
			printf("c1 nan, numPoints: %i\n", para.numPoints);
		}
		if (para.c2 != para.c2) {
			para.c2 = 0;
			printf("c2 nan, numPoints: %i\n", para.numPoints);
		}
		if (para.c3 != para.c3) {
			para.c3 = 0;
			printf("c3 nan, numPoints: %i\n", para.numPoints);
		}
	}

	__global__ void k_computeLocalSurfaces1(MeshBaseDevice* mesh, localSurface* localSurfaces, float* outMatrixData, float* outRhsData, float* outMatrixPtrs[], float* outRhsPtrs[])
	{
		for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < mesh->nVerticesSurf; idx += blockDim.x * gridDim.x) {
			computeLocalSurface1(idx, mesh, localSurfaces, outMatrixData, outRhsData);
			outMatrixPtrs[idx] = &outMatrixData[idx * 49];
			outRhsPtrs[idx] = &outRhsData[idx * 7];
		}
	}

	__global__ void k_computeLocalSurfaces2(MeshBaseDevice* mesh, localSurface* localSurfaces, float* abcVec)
	{
		for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < mesh->nVerticesSurf; idx += blockDim.x * gridDim.x) {
			computeLocalSurface2(idx, localSurfaces, abcVec);
		}
	}




}

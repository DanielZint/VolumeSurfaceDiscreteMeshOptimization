#include "Surface.h"
#include "CudaUtil.h"
#include <cuda_runtime.h>
//#include <helper_cuda.h>
#include "device_launch_parameters.h"

namespace SurfLS {



	__device__ void computeLocalSurface1(const int vid, MeshBaseDevice* mesh, localSurface* localSurfaces, float* outMatrixData, float* outRhsData) {
		Vec3f p_center = mesh->vertexPoints[vid];

		// neighbors (also contains vh itself)
		Vec3f p_neighbors[MAX_NEIGHBORS];
		int pos = 0;
		p_neighbors[pos++] = mesh->vertexPoints[vid];
		// collect one-ring
		for (auto it = mesh->he_ccw_order_begin(vid); it != mesh->he_ccw_order_end(vid); ++it) {
			int dst = it->targetVertex;
			p_neighbors[pos++] = mesh->vertexPoints[dst];
		}

		if (pos == 5) { // 4 neighbors
			Vec3f p1 = p_neighbors[2];
			Vec3f p2 = p_neighbors[3];
			p_neighbors[pos++] = 0.5f * (p1 + p2); // Vec3f(0.5 * (p1[0] + p2[0]), 0.5 * (p1[1] + p2[1]), 0.5 * (p1[2] + p2[2]));
		}
		if (pos >= MAX_NEIGHBORS) {
			printf("vid %i has too many neighs: %i\n", vid, pos);
			assert(0);
		}

		Vec3f ei[3];

		int neighId = mesh->he_ccw_order_begin(vid)->targetVertex;
		computeLocalCoordinates(mesh, vid, neighId, mesh->vertexNormals[vid], &ei[0], &ei[1], &ei[2]);

		//convert points to frenet basis & fitting function f(x,y) = ax + bxy +cy -> (x xy y)(a, b, c)T = z =^= Ax=b -> ATA x = ATb
		// column major storage
		outMatrixData[vid * 9 + 0] = 0.f;
		outMatrixData[vid * 9 + 1] = 0.f;
		outMatrixData[vid * 9 + 2] = 0.f;
		outMatrixData[vid * 9 + 4] = 0.f;
		outMatrixData[vid * 9 + 5] = 0.f;
		outMatrixData[vid * 9 + 8] = 0.f;
		outRhsData[vid * 3 + 0] = 0.f;
		outRhsData[vid * 3 + 1] = 0.f;
		outRhsData[vid * 3 + 2] = 0.f;

		for (auto i = 0; i < pos; ++i) {
			Vec3f p = p_neighbors[i] - p_center;
			const float u = ei[0].dot(p);
			const float v = ei[1].dot(p);
			const float w = ei[2].dot(p);
			outRhsData[vid * 3 + 0] += u * u * w;
			outRhsData[vid * 3 + 1] += u * v * w;
			outRhsData[vid * 3 + 2] += v * v * w;
			outMatrixData[vid * 9 + 0] += u * u * u * u;
			outMatrixData[vid * 9 + 1] += u * u * u * v;
			outMatrixData[vid * 9 + 4] += u * u * v * v;
			outMatrixData[vid * 9 + 2] += u * u * v * v;
			outMatrixData[vid * 9 + 5] += u * v * v * v;
			outMatrixData[vid * 9 + 8] += v * v * v * v;
		}

		outMatrixData[vid * 9 + 3] = outMatrixData[vid * 9 + 1];
		outMatrixData[vid * 9 + 6] = outMatrixData[vid * 9 + 2];
		outMatrixData[vid * 9 + 7] = outMatrixData[vid * 9 + 5];

		localSurface& para = localSurfaces[vid];
		para.p0 = p_center;
		para.ei[0] = ei[0];
		para.ei[1] = ei[1];
		para.ei[2] = ei[2];
		para.numPoints = pos;
	}

	__device__ void computeLocalSurface2(const int vid, localSurface* localSurfaces, float* abcVec) {
		localSurface& para = localSurfaces[vid];
		para.a = abcVec[vid * 3 + 0];
		para.b = abcVec[vid * 3 + 1];
		para.c = abcVec[vid * 3 + 2];
		// this is also a hack.
		// sometimes ATA * x = ATb is underdetermined and we get NANs

		if (para.numPoints <= 4) {
			para.a = 0;
			para.b = 0;
			para.c = 0;
		}

		if (para.a != para.a) {
			para.a = 0;
			printf("a nan, numPoints: %i\n", para.numPoints);
		}
		if (para.b != para.b) {
			para.b = 0;
			printf("b nan, numPoints: %i\n", para.numPoints);
		}
		if (para.c != para.c) {
			para.c = 0;
			printf("c nan, numPoints: %i\n", para.numPoints);
		}
	}

	__global__ void k_computeLocalSurfaces1(MeshBaseDevice* mesh, localSurface* localSurfaces, float* outMatrixData, float* outRhsData, float* outMatrixPtrs[], float* outRhsPtrs[])
	{
		for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < mesh->nVerticesSurf; idx += blockDim.x * gridDim.x) {
			computeLocalSurface1(idx, mesh, localSurfaces, outMatrixData, outRhsData);
			outMatrixPtrs[idx] = &outMatrixData[idx * 9];
			outRhsPtrs[idx] = &outRhsData[idx * 3];
		}
	}

	__global__ void k_computeLocalSurfaces2(MeshBaseDevice* mesh, localSurface* localSurfaces, float* abcVec)
	{
		for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < mesh->nVerticesSurf; idx += blockDim.x * gridDim.x) {
			computeLocalSurface2(idx, localSurfaces, abcVec);
		}
	}



}

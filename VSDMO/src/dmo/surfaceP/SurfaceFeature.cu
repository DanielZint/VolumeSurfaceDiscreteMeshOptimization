#include "SurfaceFeature.h"
#include "CudaUtil.h"
#include <cuda_runtime.h>
//#include <helper_cuda.h>
#include "device_launch_parameters.h"

namespace SurfLS {




	__device__ void computeLocalSurfaceFeature1(const int surfIdx, MeshBaseDevice* mesh, localSurface* localSurfaces, float* outMatrixData, float* outRhsData, int* neighbors, int numNeighbors) {
		const int vid = neighbors[0];
		if (numNeighbors > MAX_NEIGHBORS) {
			printf("too many neighbors, vid %i idx %i\n", vid, surfIdx);
			assert(0);
		}
		Vec3f p_center = mesh->vertexPoints[vid];

		// neighbors (also contains vh itself)
		Vec3f p_neighbors[MAX_NEIGHBORS];

		for (int i = 0; i < numNeighbors; ++i) {
			p_neighbors[i] = mesh->vertexPoints[neighbors[i]];
		}

		if (numNeighbors == 5) { // 4 neighbors
			Vec3f p1 = p_neighbors[2];
			Vec3f p2 = p_neighbors[3];
			p_neighbors[numNeighbors] = 0.5f * (p1 + p2); // Vec3f(0.5 * (p1[0] + p2[0]), 0.5 * (p1[1] + p2[1]), 0.5 * (p1[2] + p2[2]));
			++numNeighbors;
		}

		// we need normal of this subsurface described by the open fan from neighbors[1] to neighbors[numNeighbors-1] around neighbors[0]
		Vec3f normal;
		for (auto heit = mesh->he_ccw_order_begin(vid); heit != mesh->he_ccw_order_end(vid); ++heit) {
			if (heit->targetVertex == neighbors[1]) {
				normal = mesh->faceNormals[heit->incidentFace];
			}
		}

		//Vec3f normal = mesh->faceNormals[heit->incidentFace];
		Vec3f ei[3];

		computeLocalCoordinates(mesh, vid, neighbors[1], normal, &ei[0], &ei[1], &ei[2]);

		//convert points to frenet basis & fitting function f(x,y) = ax + bxy +cy -> (x xy y)(a, b, c)T = z =^= Ax=b -> ATA x = ATb
		// column major storage
		outMatrixData[surfIdx * 9 + 0] = 0.f;
		outMatrixData[surfIdx * 9 + 1] = 0.f;
		outMatrixData[surfIdx * 9 + 2] = 0.f;
		outMatrixData[surfIdx * 9 + 4] = 0.f;
		outMatrixData[surfIdx * 9 + 5] = 0.f;
		outMatrixData[surfIdx * 9 + 8] = 0.f;
		outRhsData[surfIdx * 3 + 0] = 0.f;
		outRhsData[surfIdx * 3 + 1] = 0.f;
		outRhsData[surfIdx * 3 + 2] = 0.f;


		for (auto i = 0; i < numNeighbors; ++i) {
			Vec3f p = p_neighbors[i] - p_center;
			const float u = ei[0].dot(p);
			const float v = ei[1].dot(p);
			const float w = ei[2].dot(p);
			outRhsData[surfIdx * 3 + 0] += u * u * w;
			outRhsData[surfIdx * 3 + 1] += u * v * w;
			outRhsData[surfIdx * 3 + 2] += v * v * w;
			outMatrixData[surfIdx * 9 + 0] += u * u * u * u;
			outMatrixData[surfIdx * 9 + 1] += u * u * u * v;
			outMatrixData[surfIdx * 9 + 4] += u * u * v * v;
			outMatrixData[surfIdx * 9 + 2] += u * u * v * v;
			outMatrixData[surfIdx * 9 + 5] += u * v * v * v;
			outMatrixData[surfIdx * 9 + 8] += v * v * v * v;
		}

		outMatrixData[surfIdx * 9 + 3] = outMatrixData[surfIdx * 9 + 1];
		outMatrixData[surfIdx * 9 + 6] = outMatrixData[surfIdx * 9 + 2];
		outMatrixData[surfIdx * 9 + 7] = outMatrixData[surfIdx * 9 + 5];

		localSurface& para = localSurfaces[surfIdx];
		para.p0 = p_center;
		para.ei[0] = ei[0];
		para.ei[1] = ei[1];
		para.ei[2] = ei[2];
		para.numPoints = numNeighbors;
	}

	__global__ void k_computeLocalSurfacesFeature1(MeshBaseDevice* mesh, int nSurfaces, localSurface* localSurfaces, float* outMatrixData, float* outRhsData, float* outMatrixPtrs[], float* outRhsPtrs[], int* neighbors, int* numNeighbors)
	{
		for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < nSurfaces; idx += blockDim.x * gridDim.x) {
			computeLocalSurfaceFeature1(idx, mesh, localSurfaces, outMatrixData, outRhsData, &neighbors[MAX_NEIGHBORS * idx], numNeighbors[idx]);
			outMatrixPtrs[idx] = &outMatrixData[idx * 9];
			outRhsPtrs[idx] = &outRhsData[idx * 3];
		}
	}

	__device__ void computeLocalSurfaceFeature2(const int vid, localSurface* localSurfaces, float* abcVec) {
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
			printf("feat a nan, numPoints: %i\n", para.numPoints);
		}
		if (para.b != para.b) {
			para.b = 0;
			printf("feat b nan, numPoints: %i\n", para.numPoints);
		}
		if (para.c != para.c) {
			para.c = 0;
			printf("feat c nan, numPoints: %i\n", para.numPoints);
		}

	}

	__global__ void k_computeLocalSurfacesFeature2(MeshBaseDevice* mesh, int nSurfaces, localSurface* localSurfaces, float* abcVec)
	{
		for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < nSurfaces; idx += blockDim.x * gridDim.x) {
			computeLocalSurfaceFeature2(idx, localSurfaces, abcVec);
		}
	}


	__global__ void k_countLocalSurfacesFeature(MeshBaseDevice* mesh, int* surfaceCounts, int n, int start)
	{
		for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < n; idx += blockDim.x * gridDim.x) {
			int vid = start + idx;
			int sid = 0;

			//int nHalfedges = mesh->halfedges.rowPtr_[vid + 1] - mesh->halfedges.rowPtr_[vid];
			auto begin = mesh->he_ccw_order_begin(vid);
			++begin;
			for (auto it = begin; it != mesh->he_ccw_order_end(vid); ++it) {
				const Halfedge& halfedge = *it;
				const int vdst = halfedge.targetVertex;
				if (mesh->isFeature(vdst)) {
					++sid;
				}
			}

			// if vid is not boundary1d (closed fan) add last vertex
			if (!mesh->vertexIsBoundary1D[vid]) {
				++sid;
			}
			// number of feature neighbors. +1 if boundary vertex.
			surfaceCounts[idx] = sid;
		}
	}

	__global__ void k_setNeighborsFeature(MeshBaseDevice* mesh, const int n, const int start, const int* rowPtr, int* neighbors, int* numNeighbors, int* table)
	{
		for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < n; idx += blockDim.x * gridDim.x) {
			int vid = start + idx;
			int sid = 0;
			int nneigh = 0; // num of neighs of current surface
			bool lastWasFeature = true;

			auto begin = mesh->he_ccw_order_begin(vid);
			auto lastHE = begin;
			++begin;
			int vindex = 1;
			table[TABLE_SIZE * idx + 0] = sid; // TEST
			for (auto it = begin; it != mesh->he_ccw_order_end(vid); ++it) {
				const Halfedge& halfedge = *it;
				const int vdst = halfedge.targetVertex;
				if (lastWasFeature) {
					nneigh = 0;
					assert(rowPtr[idx] + sid < rowPtr[idx + 1]);
					neighbors[MAX_NEIGHBORS * (rowPtr[idx] + sid) + nneigh] = vid;
					++nneigh;
					neighbors[MAX_NEIGHBORS * (rowPtr[idx] + sid) + nneigh] = lastHE->targetVertex;
					++nneigh;
				}
				if (mesh->isFeature(vdst)) {
					assert(rowPtr[idx] + sid < rowPtr[idx + 1]);
					table[TABLE_SIZE * idx + vindex] = sid; // TEST
					neighbors[MAX_NEIGHBORS * (rowPtr[idx] + sid) + nneigh] = vdst;
					++nneigh;
					numNeighbors[rowPtr[idx] + sid] = nneigh;
					++sid;
					nneigh = 0;
					lastWasFeature = true;
				}
				else {
					assert(vindex < MAX_NEIGHBORS);
					assert(rowPtr[idx] + sid < rowPtr[idx + 1]);
					table[TABLE_SIZE * idx + vindex] = sid;
					neighbors[MAX_NEIGHBORS * (rowPtr[idx] + sid) + nneigh] = vdst;
					++nneigh;
					lastWasFeature = false;
				}
				++lastHE;
				++vindex;
			}
			if (!mesh->vertexIsBoundary1D[vid]) {
				if (lastWasFeature) {
					nneigh = 0;
					assert(rowPtr[idx] + sid < rowPtr[idx + 1]);
					neighbors[MAX_NEIGHBORS * (rowPtr[idx] + sid) + nneigh] = vid;
					++nneigh;
					neighbors[MAX_NEIGHBORS * (rowPtr[idx] + sid) + nneigh] = lastHE->targetVertex;
					++nneigh;
				}
				assert(rowPtr[idx] + sid < rowPtr[idx + 1]);
				neighbors[MAX_NEIGHBORS * (rowPtr[idx] + sid) + nneigh] = mesh->he_ccw_order_begin(vid)->targetVertex;
				++nneigh;
				numNeighbors[rowPtr[idx] + sid] = nneigh;
			}

		}
	}

}

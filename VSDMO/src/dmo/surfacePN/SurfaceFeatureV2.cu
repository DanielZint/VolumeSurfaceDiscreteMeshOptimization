#include "SurfaceFeatureV2.h"
#include "CudaUtil.h"
#include <cuda_runtime.h>
//#include <helper_cuda.h>
#include "device_launch_parameters.h"

namespace SurfV2 {




	__device__ void computeLocalSurfaceFeature1(const int surfIdx, MeshBaseDevice* mesh, localSurface* localSurfaces, float* outMatrixData, float* outRhsData, int* neighbors, int numNeighbors) {
		const int vid = neighbors[0];
		if (numNeighbors > MAX_NEIGHBORS) {
			printf("too many neighbors, vid %i idx %i\n", vid, surfIdx);
			assert(0);
		}

		// we need normal of this subsurface described by the open fan from neighbors[1] to neighbors[numNeighbors-1] around neighbors[0]
		Vec3f n0;
		for (auto heit = mesh->he_ccw_order_begin(vid); heit != mesh->he_ccw_order_end(vid); ++heit) {
			if (heit->targetVertex == neighbors[1]) {
				n0 = mesh->faceNormals[heit->incidentFace];
			}
		}

		const Vec3f& p0 = mesh->vertexPoints[vid];
		//const Vec3f& n0 = mesh->vertexNormals[vid];
		Vec3f ei[3];
		computeLocalCoordinates(mesh, vid, neighbors[1], n0, &ei[0], &ei[1], &ei[2]);

		for (int row = 0; row < 7; ++row) {
			for (int col = 0; col < 7; ++col) {
				outMatrixData[surfIdx * 49 + (col * 7 + row)] = 0.f;
			}
			outRhsData[surfIdx * 7 + row] = 0.f;
		}

		int sz = numNeighbors; // 1 + (size_t)mesh->halfedges.rowPtr_[vid + 1] - (size_t)mesh->halfedges.rowPtr_[vid];
		int nrEqs = 3 * sz;
		int unknowns = 7;

		for (int i = 0; i < numNeighbors; ++i) {
			addPoint(surfIdx, mesh, neighbors[i], mesh->vertexPoints[vid], n0, ei, outMatrixData, outRhsData);
		}

		localSurface& para = localSurfaces[surfIdx];
		para.p0 = p0;
		para.ei[0] = ei[0];
		para.ei[1] = ei[1];
		para.ei[2] = ei[2];
		para.numPoints = sz;
	}

	__global__ void k_computeLocalSurfacesFeature1(MeshBaseDevice* mesh, int nSurfaces, localSurface* localSurfaces, float* outMatrixData, float* outRhsData, float* outMatrixPtrs[], float* outRhsPtrs[], int* neighbors, int* numNeighbors)
	{
		for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < nSurfaces; idx += blockDim.x * gridDim.x) {
			computeLocalSurfaceFeature1(idx, mesh, localSurfaces, outMatrixData, outRhsData, &neighbors[MAX_NEIGHBORS * idx], numNeighbors[idx]);
			outMatrixPtrs[idx] = &outMatrixData[idx * 49];
			outRhsPtrs[idx] = &outRhsData[idx * 7];
		}
	}

	__device__ void computeLocalSurfaceFeature2(const int vid, localSurface* localSurfaces, float* abcVec) {
		localSurface& para = localSurfaces[vid];
		para.b0 = abcVec[vid * 7 + 0];
		para.b1 = abcVec[vid * 7 + 1];
		para.b2 = abcVec[vid * 7 + 2];
		para.c0 = abcVec[vid * 7 + 3];
		para.c1 = abcVec[vid * 7 + 4];
		para.c2 = abcVec[vid * 7 + 5];
		para.c3 = abcVec[vid * 7 + 6];

		para.b0 = 0;
		para.b1 = 0;
		para.b2 = 0;
		para.c0 = 0;
		para.c1 = 0;
		para.c2 = 0;
		para.c3 = 0;
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
			printf("feat b0 nan, numPoints: %i\n", para.numPoints);
		}
		if (para.b1 != para.b1) {
			para.b1 = 0;
			printf("feat b1 nan, numPoints: %i\n", para.numPoints);
		}
		if (para.b2 != para.b2) {
			para.b2 = 0;
			printf("feat b2 nan, numPoints: %i\n", para.numPoints);
		}
		if (para.c0 != para.c0) {
			para.c0 = 0;
			printf("feat c0 nan, numPoints: %i\n", para.numPoints);
		}
		if (para.c1 != para.c1) {
			para.c1 = 0;
			printf("feat c1 nan, numPoints: %i\n", para.numPoints);
		}
		if (para.c2 != para.c2) {
			para.c2 = 0;
			printf("feat c2 nan, numPoints: %i\n", para.numPoints);
		}
		if (para.c3 != para.c3) {
			para.c3 = 0;
			printf("feat c3 nan, numPoints: %i\n", para.numPoints);
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

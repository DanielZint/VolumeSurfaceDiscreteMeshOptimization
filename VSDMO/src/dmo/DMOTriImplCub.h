#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <thrust/count.h>
#include <thrust/sequence.h>
#include "CudaUtil.h"
#include "CudaAtomic.h"
#include "DMOCommon.h"
#include "SurfaceConfig.h"
#include "SurfaceEstimation.h"

#include "io/FileWriter.h"
#include "Serializer.h"

#include <cub/cub.cuh>

//using namespace SurfLS;
//using namespace Surf1D;



namespace DMOImplCub {

	template<int BLOCK_SIZE, bool SurfOfNN>
	__global__ void k_optimizeHierarchical2D(int cStart, int cEnd, MeshTriDevice* mesh, float affineFactor, QualityCriterium q_crit, float grid_scale,
		Vec3f* vertexPointsInit, localSurface* localSurfacesInit, int* nearestNeighbors, int* table, int* surfacesRowPtr, localSurface* localSurfacesFeatureInit, int maxOneRingSize)
	{
		if (blockIdx.x >= cEnd - cStart) {
			return;
		}

		const int vid = cStart + blockIdx.x;
		const int nnvid = nearestNeighbors[vid];

		float qMax = -FLT_MAX;

		__shared__ Vec3f s_currPos;
		__shared__ float s_maxDist;
		//__shared__ float s_minDist;

		__shared__ localSurface s_localSurf;
		__shared__ Vec3f s_currLocalPos;

		__shared__ cub::KeyValuePair<int, float> s_argMaxVal;

		__shared__ int s_nOneRing;
		extern __shared__ Vec3f s_oneRing[];

		__shared__ int s_nSurfVh;
		int* s_surfVh = (int*)&s_oneRing[maxOneRingSize];

		// collect surface
		int surfCenterVid = vid;
		if constexpr (SurfOfNN) {
			surfCenterVid = nnvid;
		}
		const int heStartSurf = mesh->halfedges.rowPtr_[surfCenterVid];
		const int nHalfedgesSurf = mesh->halfedges.rowPtr_[surfCenterVid + 1] - heStartSurf;

		for (int i = threadIdx.x; i < nHalfedgesSurf; i += blockDim.x) {
			const Halfedge& halfedge = mesh->halfedges.values_[heStartSurf + i];
			const int vdst = halfedge.targetVertex;
			s_surfVh[i] = vdst;
		}
		if (threadIdx.x == 0) {
			s_nSurfVh = nHalfedgesSurf;
		}


		if (threadIdx.x == 0) {
			//s_minDist = FLT_MAX;
			s_maxDist = -FLT_MAX;
			s_currPos = mesh->vertexPoints[vid];
		}

		sync<BLOCK_SIZE>();

		// min/max search + loading oneRing
		const int heStartRing = mesh->halfedges.rowPtr_[vid];
		const int nHalfedgesRing = mesh->halfedges.rowPtr_[vid + 1] - heStartRing;

		float tlocalMin = FLT_MAX;
		float tlocalMax = -FLT_MAX;
		for (int i = threadIdx.x; i < nHalfedgesRing; i += blockDim.x) {
			const Halfedge& halfedge = mesh->halfedges.values_[heStartRing + i];
			const int vdst = halfedge.targetVertex;
			Vec3f oneRingPos = mesh->vertexPoints[vdst];
			s_oneRing[i] = oneRingPos;

			float d = (oneRingPos - s_currPos).norm();
			tlocalMin = fminf(tlocalMin, d);
			tlocalMax = fmaxf(tlocalMax, d);
		}

		//reduce
		typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
		__shared__ typename BlockReduce::TempStorage temp_storage;
		float tblockMin = BlockReduce(temp_storage).Reduce(tlocalMin, cub::Min());
		sync<BLOCK_SIZE>();
		float tblockMax = BlockReduce(temp_storage).Reduce(tlocalMax, cub::Max());
		sync<BLOCK_SIZE>();

		// close fan
		if (threadIdx.x == 0) {
			//s_minDist = tblockMin;
			s_maxDist = tblockMax;
			s_oneRing[nHalfedgesRing] = s_oneRing[0];
			s_nOneRing = nHalfedgesRing + 1;

			s_localSurf = localSurfacesInit[nearestNeighbors[vid]];
			Vec3f currLocalPosVec(s_currPos - s_localSurf.p0);
			s_currLocalPos = { s_localSurf.ei[0].dot(currLocalPosVec), s_localSurf.ei[1].dot(currLocalPosVec), s_localSurf.ei[2].dot(currLocalPosVec) };
		}


		sync<BLOCK_SIZE>();

		// start depth iteration
		float depth_scale = grid_scale;

		const int i1 = threadIdx.x / DMO_NQ;
		const int j1 = threadIdx.x % DMO_NQ;
		const int i2 = i1 + 4;
		const int j2 = j1;

		typedef cub::BlockReduce<cub::KeyValuePair<int, float>, BLOCK_SIZE> BlockReduce2;
		__shared__ typename BlockReduce2::TempStorage temp_storage2;

		for (int depth = 0; depth < DMO_DEPTH; ++depth) {
			float uMax, uMin, vMax, vMin;
			uMax = +depth_scale * s_maxDist;
			uMin = -depth_scale * s_maxDist;
			vMax = +depth_scale * s_maxDist;
			vMin = -depth_scale * s_maxDist;

			float u1 = s_currLocalPos[0] + affineFactor * (i1 * uMin + (DMO_NQ - 1 - i1) * uMax);
			float v1 = s_currLocalPos[1] + affineFactor * (j1 * vMin + (DMO_NQ - 1 - j1) * vMax);
			float u2 = s_currLocalPos[0] + affineFactor * (i2 * uMin + (DMO_NQ - 1 - i2) * uMax);
			float v2 = s_currLocalPos[1] + affineFactor * (j2 * vMin + (DMO_NQ - 1 - j2) * vMax);


			Vec3f pWorld1;
			Vec3f pWorld2;
			if constexpr (SurfOfNN) {
				pWorld1 = localSurfaceEstimationWithFeatureNew(vertexPointsInit, localSurfacesInit, nearestNeighbors, mesh, vid, s_surfVh, s_nSurfVh, u1, v1, table, surfacesRowPtr, localSurfacesFeatureInit);
				pWorld2 = localSurfaceEstimationWithFeatureNew(vertexPointsInit, localSurfacesInit, nearestNeighbors, mesh, vid, s_surfVh, s_nSurfVh, u2, v2, table, surfacesRowPtr, localSurfacesFeatureInit);
			}
			else {
				pWorld1 = localSurfaceEstimationWithFeature(vertexPointsInit, localSurfacesInit, nearestNeighbors, mesh, vid, s_surfVh, s_nSurfVh, u1, v1, table, surfacesRowPtr, localSurfacesFeatureInit);
				pWorld2 = localSurfaceEstimationWithFeature(vertexPointsInit, localSurfacesInit, nearestNeighbors, mesh, vid, s_surfVh, s_nSurfVh, u2, v2, table, surfacesRowPtr, localSurfacesFeatureInit);
			}

			float qLast = qualityTriRing(s_nOneRing, s_oneRing, s_currPos, mesh->vertexNormals[nnvid], q_crit);
			float q1 = qualityTriRing(s_nOneRing, s_oneRing, pWorld1, mesh->vertexNormals[nnvid], q_crit);
			float q2 = qualityTriRing(s_nOneRing, s_oneRing, pWorld2, mesh->vertexNormals[nnvid], q_crit);

			Vec2f uvMax(u1, v1);
			qMax = q1;
			if (q2 > q1) {
				qMax = q2;
				uvMax = Vec2f(u2, v2);
			}

			auto tblockMaxQ = BlockReduce2(temp_storage2).Reduce(cub::KeyValuePair<int, float>(threadIdx.x, qMax), cub::ArgMax());
			if (threadIdx.x == 0) {
				s_argMaxVal = tblockMaxQ;
			}

			sync<BLOCK_SIZE>();

			if (threadIdx.x == s_argMaxVal.key && qLast < qMax) {
				float u = uvMax.x;
				float v = uvMax.y;
				float w = s_localSurf.polynomialFunction(u, v);

				Vec3f pWorld;
				if constexpr (SurfOfNN) {
					pWorld = localSurfaceEstimationWithFeatureNew(vertexPointsInit, localSurfacesInit, nearestNeighbors, mesh, vid, s_surfVh, s_nSurfVh, u, v, table, surfacesRowPtr, localSurfacesFeatureInit);
				}
				else {
					pWorld = localSurfaceEstimationWithFeature(vertexPointsInit, localSurfacesInit, nearestNeighbors, mesh, vid, s_surfVh, s_nSurfVh, u, v, table, surfacesRowPtr, localSurfacesFeatureInit);
				}

				s_currPos = pWorld;
				s_currLocalPos = { u,v,w };
			}

			sync<BLOCK_SIZE>();

			//depth dependent scaling factor
			depth_scale = depth_scale * (2.f / (DMO_NQ - 1));
		}

		// set new position if it is better than the old one
		const Vec3f oldPos = mesh->vertexPoints[vid];
		float qOld = qualityTriRing(s_nOneRing, s_oneRing, oldPos, q_crit);
		if (threadIdx.x == s_argMaxVal.key && qOld < qMax) {
			mesh->vertexPoints[vid] = s_currPos;
		}
	}





	template<int BLOCK_SIZE>
	__global__ void k_optimizeHierarchical1D(int cStart, int cEnd, MeshTriDevice* mesh, float affineFactor, QualityCriterium q_crit, float grid_scale,
		Vec3f* vertexPointsInit, localSurface1d* localSurfacesInit1d, int* nearestNeighbors, bool* optimizeFeatureVertex, int maxOneRingSize, int maxSurfRingSize)
	{
		if (blockIdx.x >= cEnd - cStart) {
			return;
		}
		const int vid = cStart + blockIdx.x;
		const int nnvid = nearestNeighbors[vid];

		if (!optimizeFeatureVertex[vid]) {
			return;
		}
		const int i1 = threadIdx.x % DMO_NQ;

		float qMax = -FLT_MAX;


		__shared__ Vec3f s_currPos;
		__shared__ float s_maxDist;

		__shared__ localSurface1d s_localSurf;

		__shared__ Vec2f s_currLocalPos;

		__shared__ cub::KeyValuePair<int, float> s_argMaxVal;

		__shared__ int s_nOneRing;
		extern __shared__ Vec3f s_oneRing[];

		__shared__ int s_nSurfVh;
		int* s_surfVh = (int*)&s_oneRing[maxOneRingSize];

		////
		//for (int i = threadIdx.x; i < maxSurfRingSize; i += blockDim.x) {
		//	s_surfVh[i] = INT_MAX; // init for sort later.
		//}

		if (threadIdx.x == 0) {
			s_currPos = mesh->vertexPoints[vid];
			s_nSurfVh = 0;
		}
		sync<BLOCK_SIZE>();

		const int heStartRing = mesh->halfedges.rowPtr_[vid];
		const int nHalfedgesRing = mesh->halfedges.rowPtr_[vid + 1] - heStartRing;

		float tlocalMin = FLT_MAX;
		float tlocalMax = -FLT_MAX;
		for (int i = threadIdx.x; i < nHalfedgesRing; i += blockDim.x) {
			const Halfedge& halfedge = mesh->halfedges.values_[heStartRing + i];
			const int vdst = halfedge.targetVertex;
			Vec3f oneRingPos = mesh->vertexPoints[vdst];
			s_oneRing[i] = oneRingPos;

			float xDist = abs(s_currPos.x - oneRingPos.x);
			float yDist = abs(s_currPos.y - oneRingPos.y);
			float zDist = abs(s_currPos.z - oneRingPos.z);
			float d = std::sqrtf(xDist * xDist + yDist * yDist + zDist * zDist);
			tlocalMin = fminf(tlocalMin, d);
			tlocalMax = fmaxf(tlocalMax, d);

			//if (mesh->vertexIsFeature[vdst]) {
			//	int nNeigh = 0;
			//	for (auto it2 = mesh->he_ccw_order_begin(vdst); it2 != mesh->he_ccw_order_end(vdst); ++it2) {
			//		int vdst2 = it2->targetVertex;
			//		if (mesh->vertexIsFeature[vdst2]) {
			//			++nNeigh;
			//		}
			//	}
			//	if (nNeigh == 2) {
			//		int currPos = atomicAdd(&s_nSurfVh, 1); // currPos not deterministic?
			//		s_surfVh[currPos] = vdst;
			//	}
			//}
		}

		//constexpr int MAX_SURF_RING = 32;
		//constexpr int SortItemsPerThread = MAX_SURF_RING / BLOCK_SIZE;
		//typedef cub::BlockRadixSort<int, BLOCK_SIZE, SortItemsPerThread> BlockRadixSort;
		//__shared__ typename BlockRadixSort::TempStorage temp_storageS;
		//BlockRadixSort(temp_storageS).Sort(&s_surfVh[threadIdx.x * SortItemsPerThread]);
		//sync<BLOCK_SIZE>();

		//reduce
		typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
		__shared__ typename BlockReduce::TempStorage temp_storage;
		float tblockMin = BlockReduce(temp_storage).Reduce(tlocalMin, cub::Min());
		sync<BLOCK_SIZE>();
		float tblockMax = BlockReduce(temp_storage).Reduce(tlocalMax, cub::Max());
		sync<BLOCK_SIZE>();

		if (threadIdx.x == 0) {
			for (int i = 0; i < nHalfedgesRing; ++i) {
				const Halfedge& halfedge = mesh->halfedges.values_[heStartRing + i];
				const int vdst = halfedge.targetVertex;

				if (mesh->vertexIsFeature[vdst]) {
					int nNeigh = 0;
					for (auto it2 = mesh->he_ccw_order_begin(vdst); it2 != mesh->he_ccw_order_end(vdst); ++it2) {
						int vdst2 = it2->targetVertex;
						if (mesh->vertexIsFeature[vdst2]) {
							++nNeigh;
						}
					}
					if (nNeigh == 2) {
						s_surfVh[s_nSurfVh++] = vdst;
					}
				}
			}
			//s_minDist = tblockMin;
			s_maxDist = tblockMax;
			s_nOneRing = nHalfedgesRing;
			if (!mesh->vertexIsBoundary1D[vid]) {
				// close fan
				s_oneRing[nHalfedgesRing] = s_oneRing[0];
				++s_nOneRing;
			}
			s_localSurf = localSurfacesInit1d[nearestNeighbors[vid]];
			Vec3f currLocalPosVec = s_currPos - s_localSurf.p0;
			s_currLocalPos = { s_localSurf.ei[0].dot(currLocalPosVec), s_localSurf.ei[1].dot(currLocalPosVec) };
		}


		sync<BLOCK_SIZE>();


		// start depth iteration
		float depth_scale = grid_scale;

		typedef cub::BlockReduce<cub::KeyValuePair<int, float>, BLOCK_SIZE> BlockReduce2;
		__shared__ typename BlockReduce2::TempStorage temp_storage2;

		for (int depth = 0; depth < DMO_DEPTH; ++depth) {
			float uMax, uMin;
			uMax = +depth_scale * s_maxDist;
			uMin = -depth_scale * s_maxDist;

			float u1 = s_currLocalPos[0] + affineFactor * (i1 * uMin + (DMO_NQ - 1 - i1) * uMax);
			Vec3f pWorld = localSurfaceEstimation1d(vertexPointsInit, localSurfacesInit1d, nearestNeighbors, vid, s_surfVh, s_nSurfVh, u1);
			qMax = qualityTriRing(s_nOneRing, s_oneRing, pWorld, mesh->vertexNormals[nnvid], q_crit);

			float qLast = qualityTriRing(s_nOneRing, s_oneRing, s_currPos, mesh->vertexNormals[nnvid], q_crit);

			auto tblockMaxQ = BlockReduce2(temp_storage2).Reduce(cub::KeyValuePair<int, float>(threadIdx.x, qMax), cub::ArgMax());
			if (threadIdx.x == 0) {
				s_argMaxVal = tblockMaxQ;
			}

			sync<BLOCK_SIZE>();

			if (threadIdx.x == s_argMaxVal.key && qLast < qMax) {
				float u = s_currLocalPos[0] + affineFactor * (i1 * uMin + (DMO_NQ - 1 - i1) * uMax);
				float v = s_localSurf.polynomialFunction1d(u);
				Vec3f pVecWorld = localSurfaceEstimation1d(vertexPointsInit, localSurfacesInit1d, nearestNeighbors, vid, s_surfVh, s_nSurfVh, u);
				s_currPos = pVecWorld;
				s_currLocalPos = { u,v };
			}

			sync<BLOCK_SIZE>();

			//depth dependent scaling factor
			depth_scale = depth_scale * (2.f / (DMO_NQ - 1));
		}


		// set new position if it is better than the old one
		const Vec3f& oldPos = mesh->vertexPoints[vid];
		float qOld = qualityTriRing(s_nOneRing, s_oneRing, oldPos, q_crit);
		if (threadIdx.x == s_argMaxVal.key && qOld < qMax) {
			mesh->vertexPoints[vid] = s_currPos;
		}
	} // k_optimizeHierarchical1D



}
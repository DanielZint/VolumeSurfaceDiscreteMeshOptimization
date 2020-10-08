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

	template<int BLOCK_SIZE>
	__global__ void k_optimizeHierarchical2DFlat(int cStart, int cEnd, MeshTriDevice* mesh, float affineFactor, QualityCriterium q_crit, float grid_scale)
	{
		if (blockIdx.x >= cEnd - cStart) {
			return;
		}

		const int vid = cStart + blockIdx.x;

		float qMax = -FLT_MAX;


		__shared__ Vec3f s_currPos;
		__shared__ Vec3f s_maxDist;
		__shared__ Vec3f s_minDist;

		__shared__ cub::KeyValuePair<int, float> s_argMaxVal;

		__shared__ int s_nOneRing;
		extern __shared__ Vec3f s_oneRing[];


		if (threadIdx.x == 0) {
			s_currPos = mesh->vertexPoints[vid];
		}

		sync<BLOCK_SIZE>();

		// min/max search + loading oneRing
		const int heStartRing = mesh->halfedges.rowPtr_[vid];
		const int nHalfedgesRing = mesh->halfedges.rowPtr_[vid + 1] - heStartRing;

		Vec3f tlocalMin = Vec3f(FLT_MAX, FLT_MAX, FLT_MAX);
		Vec3f tlocalMax = Vec3f(-FLT_MAX, -FLT_MAX, -FLT_MAX);
		for (int i = threadIdx.x; i < nHalfedgesRing; i += blockDim.x) {
			const Halfedge& halfedge = mesh->halfedges.values_[heStartRing + i];
			const int vdst = halfedge.targetVertex;
			Vec3f oneRingPos = mesh->vertexPoints[vdst];
			s_oneRing[i] = oneRingPos;
			for (int k = 0; k < 3; ++k) { // for x,y,z
				tlocalMin[k] = fminf(tlocalMin[k], fabsf(oneRingPos[k] - s_currPos[k]));
				tlocalMax[k] = fmaxf(tlocalMax[k], fabsf(oneRingPos[k] - s_currPos[k]));
			}

			//float d = (oneRingPos - s_currPos).norm();
			//tlocalMin = fminf(tlocalMin, d);
			//tlocalMax = fmaxf(tlocalMax, d);
		}

		//reduce
		typedef cub::BlockReduce<Vec3f, BLOCK_SIZE> BlockReduce;
		__shared__ typename BlockReduce::TempStorage temp_storage;
		Vec3f tblockMin = BlockReduce(temp_storage).Reduce(tlocalMin, Vec3fMin());
		sync<BLOCK_SIZE>();
		Vec3f tblockMax = BlockReduce(temp_storage).Reduce(tlocalMax, Vec3fMax());
		sync<BLOCK_SIZE>();

		// close fan
		if (threadIdx.x == 0) {
			s_minDist = tblockMin;
			s_maxDist = tblockMax;
			s_oneRing[nHalfedgesRing] = s_oneRing[0];
			s_nOneRing = nHalfedgesRing + 1;
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
			Vec3f gridMax = s_currPos + depth_scale * s_maxDist;
			Vec3f gridMin = s_currPos - depth_scale * s_maxDist;

			Vec3f pWorld1;
			Vec3f pWorld2;
			pWorld1.x = affineFactor * (i1 * gridMin.x + (DMO_NQ - 1 - i1) * gridMax.x);
			pWorld1.y = affineFactor * (j1 * gridMin.y + (DMO_NQ - 1 - j1) * gridMax.y);
			pWorld2.x = affineFactor * (i2 * gridMin.x + (DMO_NQ - 1 - i2) * gridMax.x);
			pWorld2.y = affineFactor * (j2 * gridMin.y + (DMO_NQ - 1 - j2) * gridMax.y);

			float qLast = qualityTri2DRing(s_nOneRing, s_oneRing, s_currPos, q_crit);
			float q1 = qualityTri2DRing(s_nOneRing, s_oneRing, pWorld1, q_crit);
			float q2 = qualityTri2DRing(s_nOneRing, s_oneRing, pWorld2, q_crit);

			qMax = q1;
			if (q2 > q1) {
				qMax = q2;
			}

			auto tblockMaxQ = BlockReduce2(temp_storage2).Reduce(cub::KeyValuePair<int, float>(threadIdx.x, qMax), cub::ArgMax());
			if (threadIdx.x == 0) {
				s_argMaxVal = tblockMaxQ;
			}


			sync<BLOCK_SIZE>(); // probably not necessary, since BlockReduce syncs

			if (threadIdx.x == s_argMaxVal.key && qLast < qMax) {
				if (q1 > q2) {
					s_currPos = pWorld1;
				}
				else {
					s_currPos = pWorld2;
				}
			}

			sync<BLOCK_SIZE>();

			//depth dependent scaling factor
			depth_scale = depth_scale * (2.f / (DMO_NQ - 1));
		}

		// set new position if it is better than the old one
		const Vec3f oldPos = mesh->vertexPoints[vid];
		float qOld = qualityTri2DRing(s_nOneRing, s_oneRing, oldPos, q_crit);
		
		if (threadIdx.x == s_argMaxVal.key && qOld < qMax) {
			mesh->vertexPoints[vid] = s_currPos;
		}
	} // k_optimizeHierarchical2D





	template<int BLOCK_SIZE>
	__global__ void k_optimizeHierarchical1DFlat(int cStart, int cEnd, MeshTriDevice* mesh, float affineFactor, QualityCriterium q_crit, float grid_scale,
		Vec3f* vertexPointsInit, localSurface1d* localSurfacesInit1d, int* nearestNeighbors, bool* optimizeFeatureVertex, int maxOneRingSize, int maxSurfRingSize)
	{
		if (blockIdx.x >= cEnd - cStart) {
			return;
		}
		const int vid = cStart + blockIdx.x;
		//const int nnvid = nearestNeighbors[vid];

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
			qMax = qualityTri2DRing(s_nOneRing, s_oneRing, pWorld, q_crit);

			float qLast = qualityTri2DRing(s_nOneRing, s_oneRing, s_currPos, q_crit);

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
		float qOld = qualityTri2DRing(s_nOneRing, s_oneRing, oldPos, q_crit);
		if (threadIdx.x == s_argMaxVal.key && qOld < qMax) {
			mesh->vertexPoints[vid] = s_currPos;
		}
	} // k_optimizeHierarchical1D



}
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

#include <cub/cub.cuh>

//using namespace SurfLS;
//using namespace Surf1D;

namespace DMOImplCub {

	template<int BLOCK_SIZE>
	__global__ void k_optimizeHierarchicalInnerTet(int cStart, int cEnd, MeshTetDevice* mesh, float affineFactor, QualityCriterium q_crit, const float grid_scale)
	{
		if (blockIdx.x >= cEnd - cStart) {
			return;
		}

		const int vid = cStart + blockIdx.x;
		const int pointsPerThread = (DMO_NQ * DMO_NQ * DMO_NQ) / blockDim.x;


		float qMax = -FLT_MAX;

		__shared__ Vec3f s_currPos;
		__shared__ Vec3f s_maxDist;
		__shared__ Vec3f s_minDist;

		__shared__ cub::KeyValuePair<int, float> s_argMaxVal;

		__shared__ int s_nOneRing;
		extern __shared__ Vec3f s_oneRing[];

		const int hfStart = mesh->halffaces.rowPtr_[vid];
		const int nHalffaces = mesh->halffaces.rowPtr_[vid + 1] - hfStart;

		if (threadIdx.x == 0) {
			s_currPos = mesh->vertexPoints[vid];
			s_nOneRing = nHalffaces;
		}

		sync<BLOCK_SIZE>();

		Vec3f tlocalMin = Vec3f(FLT_MAX, FLT_MAX, FLT_MAX);
		Vec3f tlocalMax = Vec3f(-FLT_MAX, -FLT_MAX, -FLT_MAX);
		for (int i = threadIdx.x; i < nHalffaces; i += blockDim.x) {
			const Halfface& halfface = mesh->halffaces.values_[hfStart + i];
			Vec3f posRing[3] = { mesh->vertexPoints[halfface.v0], mesh->vertexPoints[halfface.v1], mesh->vertexPoints[halfface.v2] };
			for (int j = 0; j < 3; ++j) { // for each point in halfface
				s_oneRing[3 * i + j] = posRing[j];
				for (int k = 0; k < 3; ++k) { // for x,y,z
					tlocalMin[k] = fminf(tlocalMin[k], fabsf(posRing[j][k] - s_currPos[k]));
					tlocalMax[k] = fmaxf(tlocalMax[k], fabsf(posRing[j][k] - s_currPos[k]));
				}
			}
		}

		//reduce
		typedef cub::BlockReduce<Vec3f, BLOCK_SIZE> BlockReduce;
		__shared__ typename BlockReduce::TempStorage temp_storage;
		Vec3f tblockMin = BlockReduce(temp_storage).Reduce(tlocalMin, Vec3fMin());
		sync<BLOCK_SIZE>();
		Vec3f tblockMax = BlockReduce(temp_storage).Reduce(tlocalMax, Vec3fMax());


		if (threadIdx.x == 0) {
			s_minDist = tblockMin;
			s_maxDist = tblockMax;
		}
		sync<BLOCK_SIZE>();

		// start depth iteration
		float depth_scale = grid_scale;

		typedef cub::BlockReduce<cub::KeyValuePair<int, float>, BLOCK_SIZE> BlockReduce2;
		__shared__ typename BlockReduce2::TempStorage temp_storage2;

		for (int depth = 0; depth < DMO_DEPTH; ++depth) {
			Vec3f gridMax = s_currPos + depth_scale * s_maxDist;
			Vec3f gridMin = s_currPos - depth_scale * s_maxDist;

			float qLast = qualityTetRing(s_nOneRing, s_oneRing, s_currPos, q_crit);

			Vec3f pMax;
			for (int idx = threadIdx.x * pointsPerThread; idx < (threadIdx.x + 1) * pointsPerThread; ++idx) {
				int idx_copy = idx;
				int i = idx_copy % DMO_NQ;
				idx_copy /= DMO_NQ;
				int j = idx_copy % DMO_NQ;
				idx_copy /= DMO_NQ;
				int k = idx_copy;
				Vec3f p;
				p.x = affineFactor * (i * gridMin.x + (DMO_NQ - 1 - i) * gridMax.x);
				p.y = affineFactor * (j * gridMin.y + (DMO_NQ - 1 - j) * gridMax.y);
				p.z = affineFactor * (k * gridMin.z + (DMO_NQ - 1 - k) * gridMax.z);
				float q = qualityTetRing(s_nOneRing, s_oneRing, p, q_crit);
				if (q > qMax) {
					qMax = q;
					pMax = p;
				}
			}



			auto tblockMaxQ = BlockReduce2(temp_storage2).Reduce(cub::KeyValuePair<int, float>(threadIdx.x, qMax), cub::ArgMax());
			if (threadIdx.x == 0) {
				s_argMaxVal = tblockMaxQ;
			}


			sync<BLOCK_SIZE>();

			if (threadIdx.x == s_argMaxVal.key && qLast < qMax) {
				s_currPos = pMax;
			}

			sync<BLOCK_SIZE>();

			//depth dependent scaling factor
			depth_scale = depth_scale * (2.f / (DMO_NQ - 1));
		}

		// set new position if it is better than the old one
		const Vec3f& oldPos = mesh->vertexPoints[vid];
		float qOld = qualityTetRing(s_nOneRing, s_oneRing, oldPos, q_crit);
		if (threadIdx.x == s_argMaxVal.key && qOld < qMax) {
			mesh->vertexPoints[vid] = s_currPos;
		}
	} // optimizeHierarchicalInner



	template<int BLOCK_SIZE, bool SurfOfNN>
	__global__ void k_optimizeHierarchical2DTet(int cStart, int cEnd, MeshTetDevice* mesh, float affineFactor, QualityCriterium q_crit, float grid_scale,
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
		const int hfStart = mesh->halffaces.rowPtr_[vid];
		const int nHalffaces = mesh->halffaces.rowPtr_[vid + 1] - mesh->halffaces.rowPtr_[vid];

		float tlocalMin = FLT_MAX;
		float tlocalMax = -FLT_MAX;
		for (int i = threadIdx.x; i < nHalffaces; i += blockDim.x) {
			const Halfface& halfface = mesh->halffaces.values_[hfStart + i];

			Vec3f posRing[3] = { mesh->vertexPoints[halfface.v0], mesh->vertexPoints[halfface.v1], mesh->vertexPoints[halfface.v2] };
			for (int j = 0; j < 3; ++j) { // for each point in halfface
				s_oneRing[3 * i + j] = posRing[j];
				float d = (posRing[j] - s_currPos).norm();
				tlocalMin = fminf(tlocalMin, d);
				tlocalMax = fmaxf(tlocalMax, d);
			}
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
			s_nOneRing = nHalffaces;

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

			float qLast = qualityTetRing(s_nOneRing, s_oneRing, s_currPos, q_crit);

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

			float q1 = qualityTetRing(s_nOneRing, s_oneRing, pWorld1, q_crit);
			float q2 = qualityTetRing(s_nOneRing, s_oneRing, pWorld2, q_crit);

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

				//if (vid == 1503)
				//	printf("%.10g %.10g %.10g\n", u,v,w);

				Vec3f pWorld;
				if constexpr (SurfOfNN) {
					pWorld = localSurfaceEstimationWithFeatureNew(vertexPointsInit, localSurfacesInit, nearestNeighbors, mesh, vid, s_surfVh, s_nSurfVh, u, v, table, surfacesRowPtr, localSurfacesFeatureInit);
				}
				else {
					pWorld = localSurfaceEstimationWithFeature(vertexPointsInit, localSurfacesInit, nearestNeighbors, mesh, vid, s_surfVh, s_nSurfVh, u, v, table, surfacesRowPtr, localSurfacesFeatureInit);
				}

				//if (vid == 1503)
				//	printf("pWorld %.10g %.10g %.10g\n", XYZ(pWorld));

				s_currPos = pWorld;
				s_currLocalPos = { u,v,w };
			}

			sync<BLOCK_SIZE>();

			//depth dependent scaling factor
			depth_scale = depth_scale * (2.f / (DMO_NQ - 1));
		}

		// set new position if it is better than the old one
		const Vec3f oldPos = mesh->vertexPoints[vid];
		float qOld = qualityTetRing(s_nOneRing, s_oneRing, oldPos, q_crit);
		if (threadIdx.x == s_argMaxVal.key && qOld < qMax) {
			//if (qOld < 0.f) printf("vid %i qold %.10g qmax %.10g\n", vid, qOld, qMax);
			mesh->vertexPoints[vid] = s_currPos;
		}
	} // k_optimizeHierarchical2D



	template<int BLOCK_SIZE>
	__global__ void k_optimizeHierarchical1DTet(int cStart, int cEnd, MeshTetDevice* mesh, float affineFactor, QualityCriterium q_crit, float grid_scale,
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
		//__shared__ float s_minDist;

		__shared__ localSurface1d s_localSurf;

		__shared__ Vec2f s_currLocalPos;

		__shared__ cub::KeyValuePair<int, float> s_argMaxVal;

		__shared__ int s_nOneRing;
		extern __shared__ Vec3f s_oneRing[];

		__shared__ int s_nSurfVh;
		int* s_surfVh = (int*)&s_oneRing[maxOneRingSize];


		if (threadIdx.x == 0) {
			s_currPos = mesh->vertexPoints[vid];
			s_nSurfVh = 0;
		}
		sync<BLOCK_SIZE>();

		const int heStartRing = mesh->halfedges.rowPtr_[vid];
		const int nHalfedgesRing = mesh->halfedges.rowPtr_[vid + 1] - heStartRing;

		//for (int i = threadIdx.x; i < nHalfedgesRing; i += blockDim.x) {
		//	const Halfedge& halfedge = mesh->halfedges.values_[heStartRing + i];
		//	const int vdst = halfedge.targetVertex;
		//	if (mesh->vertexIsFeature[vdst]) {
		//		int nNeigh = 0;
		//		for (auto it2 = mesh->he_ccw_order_begin(vdst); it2 != mesh->he_ccw_order_end(vdst); ++it2) {
		//			int vdst2 = it2->targetVertex;
		//			if (mesh->vertexIsFeature[vdst2]) {
		//				++nNeigh;
		//			}
		//		}
		//		if (nNeigh == 2) {
		//			int currPos = atomicAdd(&s_nSurfVh, 1);
		//			s_surfVh[currPos] = vdst;
		//		}
		//	}
		//}


		// min/max search + loading oneRing
		const int hfStart = mesh->halffaces.rowPtr_[vid];
		const int nHalffaces = mesh->halffaces.rowPtr_[vid + 1] - hfStart;

		float tlocalMin = FLT_MAX;
		float tlocalMax = -FLT_MAX;
		for (int i = threadIdx.x; i < nHalffaces; i += blockDim.x) {
			const Halfface& halfface = mesh->halffaces.values_[hfStart + i];

			Vec3f posRing[3] = { mesh->vertexPoints[halfface.v0], mesh->vertexPoints[halfface.v1], mesh->vertexPoints[halfface.v2] };
			for (int j = 0; j < 3; ++j) { // for each point in halfface
				s_oneRing[3 * i + j] = posRing[j];
				float d = (posRing[j] - s_currPos).norm();
				tlocalMin = fminf(tlocalMin, d);
				tlocalMax = fmaxf(tlocalMax, d);
			}
		}

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
			s_nOneRing = nHalffaces;

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
			qMax = qualityTetRing(s_nOneRing, s_oneRing, pWorld, q_crit);


			float qLast = qualityTetRing(s_nOneRing, s_oneRing, s_currPos, q_crit);

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
		float qOld = qualityTetRing(s_nOneRing, s_oneRing, oldPos, q_crit);
		if (threadIdx.x == s_argMaxVal.key && qOld < qMax) {
			mesh->vertexPoints[vid] = s_currPos;
		}
	} // k_optimizeHierarchical1D


}
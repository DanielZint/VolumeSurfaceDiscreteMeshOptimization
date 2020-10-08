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



//using namespace SurfLS;
//using namespace Surf1D;

namespace DMOImplAtomic {

	template<int BLOCK_SIZE, bool SurfOfNN>
	__global__ void k_optimizeHierarchical2D(int cStart, int cEnd, MeshTriDevice* mesh, float affineFactor, QualityCriterium q_crit, float grid_scale,
		Vec3f* vertexPointsInit, localSurface* localSurfacesInit, int* nearestNeighbors, int* table, int* surfacesRowPtr, localSurface* localSurfacesFeatureInit, int maxOneRingSize)
	{
		if (blockIdx.x >= cEnd - cStart) return;
		const int myidx = threadIdx.x;

		const int vid = cStart + blockIdx.x;
		const int nnvid = nearestNeighbors[vid];

		float qMax = -FLT_MAX;


		__shared__ Vec3f s_currPos;
		__shared__ float s_maxDist;
		__shared__ float s_minDist;

		__shared__ localSurface s_localSurf;
		__shared__ Vec3f s_currLocalPos;

		__shared__ my_atomics s_argMaxVal;

		__shared__ int s_nOneRing;
		extern __shared__ Vec3f s_oneRing[];

		__shared__ int s_nSurfVh;
		int* s_surfVh = (int*)&s_oneRing[maxOneRingSize];


		// min/max search + loading oneRing
		if (threadIdx.x == 0) {
			s_minDist = FLT_MAX;
			s_maxDist = -FLT_MAX;
			s_argMaxVal.value = -FLT_MAX;
			s_argMaxVal.index = DMO_NQ * DMO_NQ;

			Vec3f myPos = mesh->vertexPoints[vid];
			s_currPos = myPos;
			int k = 0;

			int surfCenterVid = vid;
			if constexpr (SurfOfNN) {
				surfCenterVid = nnvid;
			}

			//int nHalfedges = mesh->halfedges.rowPtr_[vid + 1] - mesh->halfedges.rowPtr_[vid];
			for (auto it = mesh->he_ccw_order_begin(surfCenterVid); it != mesh->he_ccw_order_end(surfCenterVid); ++it) {
				const Halfedge& halfedge = *it;
				const int vdst = halfedge.targetVertex;
				s_surfVh[k] = vdst;
				++k;
			}
			s_nSurfVh = k;

			k = 0;
			for (auto it = mesh->he_ccw_order_begin(vid); it != mesh->he_ccw_order_end(vid); ++it) {
				const Halfedge& halfedge = *it;
				const int vdst = halfedge.targetVertex;
				Vec3f oneRingPos = mesh->vertexPoints[vdst];
				s_oneRing[k] = oneRingPos;

				float d = (oneRingPos - myPos).norm();
				s_minDist = fminf(s_minDist, d);
				s_maxDist = fmaxf(s_maxDist, d);

				++k;
			}

			// close fan
			s_oneRing[k] = s_oneRing[0];
			s_nOneRing = k + 1;

			s_localSurf = localSurfacesInit[nearestNeighbors[vid]];
			Vec3f currLocalPosVec(s_currPos - s_localSurf.p0);
			s_currLocalPos = { s_localSurf.ei[0].dot(currLocalPosVec), s_localSurf.ei[1].dot(currLocalPosVec), s_localSurf.ei[2].dot(currLocalPosVec) };
		}

		__syncwarp();

		// start depth iteration
		float depth_scale = grid_scale;

		const int i1 = threadIdx.x / DMO_NQ;
		const int j1 = threadIdx.x % DMO_NQ;
		const int i2 = i1 + 4;
		const int j2 = j1;

		for (int depth = 0; depth < DMO_DEPTH; ++depth) {
			float uMax, uMin, vMax, vMin;
			uMax = +depth_scale * s_minDist;
			uMin = -depth_scale * s_minDist;
			vMax = +depth_scale * s_minDist;
			vMin = -depth_scale * s_minDist;

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

			float q1 = qualityTriRing(s_nOneRing, s_oneRing, pWorld1, mesh->vertexNormals[nnvid], q_crit);
			float q2 = qualityTriRing(s_nOneRing, s_oneRing, pWorld2, mesh->vertexNormals[nnvid], q_crit);

			Vec2f uvMax(u1, v1);
			qMax = q1;
			if (q2 > q1) {
				qMax = q2;
				uvMax = Vec2f(u2, v2);
			}

			my_atomicArgMax(&(s_argMaxVal.ulong), qMax, myidx);

			float qLast = qualityTriRing(s_nOneRing, s_oneRing, s_currPos, mesh->vertexNormals[nnvid], q_crit);

			__syncwarp();

			if (myidx == s_argMaxVal.index && qLast < qMax) {
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

			__syncwarp();

			//depth dependent scaling factor
			depth_scale = depth_scale * (2.f / (DMO_NQ - 1));
		}

		// set new position if it is better than the old one
		const Vec3f oldPos = mesh->vertexPoints[vid];
		float qOld = qualityTriRing(s_nOneRing, s_oneRing, oldPos, q_crit); // normals change between iterations, so dont use normal in this quality calculation
		if (myidx == s_argMaxVal.index && qOld < qMax) {
			mesh->vertexPoints[vid] = s_currPos;
		}
	} // k_optimizeHierarchical2D



	template<int BLOCK_SIZE>
	__global__ void k_optimizeHierarchical1D(int cStart, int cEnd, MeshTriDevice* mesh, float affineFactor, QualityCriterium q_crit, float grid_scale,
		Vec3f* vertexPointsInit, localSurface1d* localSurfacesInit1d, int* nearestNeighbors, bool* optimizeFeatureVertex, int maxOneRingSize, int maxSurfRingSize)
	{
		if (blockIdx.x >= cEnd - cStart) return;
		const int vid = cStart + blockIdx.x;
		const int nnvid = nearestNeighbors[vid];

		if (!optimizeFeatureVertex[vid]) {
			return;
		}
		const int myidx = threadIdx.x;
		const int i1 = threadIdx.x % DMO_NQ;


		float qMax = -FLT_MAX;


		__shared__ Vec3f s_currPos;
		__shared__ float s_maxDist;

		__shared__ localSurface1d s_localSurf;

		__shared__ Vec2f s_currLocalPos; //u,v

		__shared__ my_atomics s_argMaxVal;

		__shared__ int s_nOneRing;
		extern __shared__ Vec3f s_oneRing[];

		__shared__ int s_nSurfVh;
		int* s_surfVh = (int*)&s_oneRing[maxOneRingSize];


		// min/max search + loading oneRing
		if (threadIdx.x == 0) {
			s_argMaxVal.value = -FLT_MAX;
			s_argMaxVal.index = DMO_NQ * DMO_NQ;
			s_maxDist = -FLT_MAX;

			Vec3f myPos = mesh->vertexPoints[vid];
			s_currPos = myPos;
			int k = 0;
			int surfVhPos = 0;
			//int nHalfedges = mesh->halfedges.rowPtr_[vid + 1] - mesh->halfedges.rowPtr_[vid];
			for (auto it = mesh->he_ccw_order_begin(vid); it != mesh->he_ccw_order_end(vid); ++it) {
				const Halfedge& halfedge = *it;
				const int vdst = halfedge.targetVertex;
				Vec3f oneRingPos = mesh->vertexPoints[vdst];
				s_oneRing[k] = oneRingPos;

				float xDist = abs(myPos.x - oneRingPos.x);
				float yDist = abs(myPos.y - oneRingPos.y);
				float zDist = abs(myPos.z - oneRingPos.z);
				s_maxDist = fmaxf(s_maxDist, std::sqrtf(xDist * xDist + yDist * yDist + zDist * zDist));
				++k;

				if (mesh->vertexIsFeature[vdst]) {
					int nNeigh = 0;
					for (auto it2 = mesh->he_ccw_order_begin(vdst); it2 != mesh->he_ccw_order_end(vdst); ++it2) {
						int vdst2 = it2->targetVertex;
						if (mesh->vertexIsFeature[vdst2]) {
							++nNeigh;
						}
					}
					if (nNeigh == 2) {
						s_surfVh[surfVhPos++] = vdst;
					}
				}
			}

			if (!mesh->vertexIsBoundary1D[vid]) {
				// close fan
				s_oneRing[k] = s_oneRing[0];
				++k;
			}

			s_nOneRing = k;
			s_nSurfVh = surfVhPos;

			s_localSurf = localSurfacesInit1d[nearestNeighbors[vid]];
			Vec3f currLocalPosVec = s_currPos - s_localSurf.p0;
			s_currLocalPos = { s_localSurf.ei[0].dot(currLocalPosVec), s_localSurf.ei[1].dot(currLocalPosVec) };

		}

		__syncthreads();


		// start depth iteration
		float depth_scale = grid_scale;

		for (int depth = 0; depth < DMO_DEPTH; ++depth) {
			float uMax, uMin;
			uMax = +depth_scale * s_maxDist;
			uMin = -depth_scale * s_maxDist;

			float u1 = s_currLocalPos[0] + affineFactor * (i1 * uMin + (DMO_NQ - 1 - i1) * uMax);

			Vec3f pWorld = localSurfaceEstimation1d(vertexPointsInit, localSurfacesInit1d, nearestNeighbors, vid, s_surfVh, s_nSurfVh, u1);

			qMax = qualityTriRing(s_nOneRing, s_oneRing, pWorld, mesh->vertexNormals[nnvid], q_crit);

			my_atomicArgMax(&(s_argMaxVal.ulong), qMax, myidx);

			float qLast = qualityTriRing(s_nOneRing, s_oneRing, s_currPos, mesh->vertexNormals[nnvid], q_crit);

			__syncthreads();

			if (myidx == s_argMaxVal.index && qLast < qMax) {
				float u = s_currLocalPos[0] + affineFactor * (i1 * uMin + (DMO_NQ - 1 - i1) * uMax);
				float v = s_localSurf.polynomialFunction1d(u);
				Vec3f pVecWorld = localSurfaceEstimation1d(vertexPointsInit, localSurfacesInit1d, nearestNeighbors, vid, s_surfVh, s_nSurfVh, u);
				s_currPos = pVecWorld;
				s_currLocalPos = { u,v };
			}

			__syncthreads();

			//depth dependent scaling factor
			depth_scale = depth_scale * (2.f / (DMO_NQ - 1));
		}


		// set new position if it is better than the old one
		const Vec3f& oldPos = mesh->vertexPoints[vid];
		float qOld = qualityTriRing(s_nOneRing, s_oneRing, oldPos, q_crit);
		if (myidx == s_argMaxVal.index && qOld < qMax) {
			mesh->vertexPoints[vid] = s_currPos;
		}
	} // k_optimizeHierarchical1D



}
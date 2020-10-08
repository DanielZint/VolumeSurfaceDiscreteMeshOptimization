#pragma once

#include "cuda_runtime.h"
#include "ConfigUsing.h"
#include "DMOConfig.h"
#include "SurfaceConfig.h"
#include "SurfaceEstimation.h"


//using namespace SurfLS;
//using namespace Surf1D;

namespace DMO {

	template<int BLOCK_SIZE>
	__device__ __forceinline__ void sync() {
		if constexpr (BLOCK_SIZE <= 32) {
			__syncwarp();
		}
		else {
			__syncthreads();
		}
	}

	struct Vec3fMin
	{
		__host__ __device__ __forceinline__ Vec3f operator()(const Vec3f& a, const Vec3f& b) const
		{
			Vec3f ret;
			ret[0] = fminf(a[0], b[0]);
			ret[1] = fminf(a[1], b[1]);
			ret[2] = fminf(a[2], b[2]);
			return ret;
		}
	};

	struct Vec3fMax
	{
		__host__ __device__ __forceinline__ Vec3f operator()(const Vec3f& a, const Vec3f& b) const
		{
			Vec3f ret;
			ret[0] = fmaxf(a[0], b[0]);
			ret[1] = fmaxf(a[1], b[1]);
			ret[2] = fmaxf(a[2], b[2]);
			return ret;
		}
	};


	typedef union {
		struct {
			float value;
			int index;
		};
		unsigned long long int ulong;    // for atomic update
	} my_atomics;

	__device__ inline int getThirdVertex(Triangle tri, int src, int currDst) {
		if (tri.v0 != src && tri.v0 != currDst) {
			return tri.v0;
		}
		else if (tri.v1 != src && tri.v1 != currDst) {
			return tri.v1;
		}
		else {
			return tri.v2;
		}
	}

	__device__ inline unsigned long long int my_atomicArgMax(unsigned long long int* address, float val, int arg)
	{
		my_atomics loc, loctest;
		loc.value = val;
		loc.index = arg;
		loctest.ulong = *address;
		while (loctest.value < val)
			loctest.ulong = atomicCAS(address, loctest.ulong, loc.ulong);
		return loctest.ulong;
	}


	template<class MeshDevice>
	__global__ static void k_calcOptFeatureVecOld(bool* optimizeFeatureVertex, MeshDevice* mesh) {
		for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < mesh->nVerticesSurf; idx += blockDim.x * gridDim.x) {
			if (!mesh->vertexIsFeature[idx]) {
				optimizeFeatureVertex[idx] = false;
				continue;
			}

			int nNeigh = 0;
			bool lastVertexWasFeature = false;
			bool isLoop = false;
			
			for (auto it = mesh->he_ccw_order_begin(idx); it != mesh->he_ccw_order_end(idx); ++it) {
				int dst = it->targetVertex;
				if (mesh->vertexIsFeature[dst]) {
					if (lastVertexWasFeature) {
						isLoop = true;
						break;
					}
					++nNeigh;
					lastVertexWasFeature = true;
				}
				else {
					lastVertexWasFeature = false;
				}
			}

			auto last = mesh->he_ccw_order_rbegin(idx);
			if (last->incidentFace != -1) { // closed fan, check for loop with first neighbor
				auto first = mesh->he_ccw_order_begin(idx);
				int firstdst = first->targetVertex;
				if (mesh->vertexIsFeature[firstdst]) {
					if (lastVertexWasFeature) {
						isLoop = true;
					}
				}
			}

			if (nNeigh != 2) {
				optimizeFeatureVertex[idx] = false;
			}
			if (isLoop) {
				optimizeFeatureVertex[idx] = false;
			}
		}
	}

	template<class MeshDevice>
	__global__ static void k_calcOptFeatureVec(bool* optimizeFeatureVertex, MeshDevice* mesh) {
		for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < mesh->nVerticesSurf; idx += blockDim.x * gridDim.x) {
			if (!mesh->vertexIsFeature[idx]) {
				optimizeFeatureVertex[idx] = false;
				continue;
			}

			int nNeigh = 0;
			Vec3f p0 = mesh->vertexPoints[idx];
			Vec3f p[2];

			for (auto it = mesh->he_ccw_order_begin(idx); it != mesh->he_ccw_order_end(idx); ++it) {
				int dst = it->targetVertex;
				if (mesh->vertexIsFeature[dst]) {
					if (nNeigh < 2) {
						p[nNeigh] = mesh->vertexPoints[dst];
					}
					++nNeigh;
				}
			}

			optimizeFeatureVertex[idx] = false;
			if (nNeigh == 2) {
				// calc angle
				Vec3f a = p[0] - p0;
				Vec3f b = p[1] - p0;
				float dot = a.dot(b);
				float len = sqrtf(a.squaredNorm() * b.squaredNorm());
				float angle = 180.f / M_PI * acosf(dot / len);
				if (angle >= 160.f) {
					optimizeFeatureVertex[idx] = true;
				}
			}
			
		}
	}

	// for each vertex decide whether it's position can be changed by Opt1D (includes feature verts, always false in that case)
	template<class DMOMesh>
	inline void calcOptFeatureVec(DMOMesh& dmo_mesh, device_vector<bool>& optimizeFeatureVertex) {
		int nVerticesSurf = dmo_mesh.nVerticesSurf;
		const int BLOCK_SIZE = 128;
		k_calcOptFeatureVec << <getBlockCount(nVerticesSurf, BLOCK_SIZE), BLOCK_SIZE >> > (raw_pointer_cast(optimizeFeatureVertex.data()), dmo_mesh.d_mesh);
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
	}

	__device__ inline float squaredDist(const Vec3f& p1, const Vec3f& p2) {
		const Vec3f p(p1 - p2);
		return p.dot(p);
	}

	__global__ void k_updateNearestNeighbor(MeshBaseDevice* mesh, Vec3f* vertexPointsInit, int* nearestNeighbors);
	__global__ void k_updateNearestNeighborAll(MeshBaseDevice* mesh, Vec3f* vertexPointsInit, int* nearestNeighbors);

	template<class DMOMesh>
	inline void updateNearestNeighbor(DMOMesh& dmo_mesh, device_vector<Vec3f>& vertexPointsInit, device_vector<int>& nearestNeighbors) {
		const int BLOCK_SIZE = 128;
		k_updateNearestNeighbor << <getBlockCount(dmo_mesh.nVerticesSurf, BLOCK_SIZE), BLOCK_SIZE >> > (dmo_mesh.d_mesh, raw_pointer_cast(vertexPointsInit.data()), raw_pointer_cast(nearestNeighbors.data()));
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
	}

	


	template<bool SurfOfNN = true>
	__global__ void k_fillEstimateLocalSurfacePoints(int vid, int nu, int nv, MeshBaseDevice* mesh, float affineFactor, float grid_scale,
		Vec3f* vertexPointsInit, localSurface* localSurfacesInit, int* nearestNeighbors, Vec3f* outSurfacePoints, int* table, int* surfacesRowPtr, localSurface* localSurfacesFeatureInit)
	{
		const int nPointsPerThread = (nu * nv) / blockDim.x;
		const float affineFactorU = 1.f / float(nu - 1);
		const float affineFactorV = 1.f / float(nv - 1);

		const int nnvid = nearestNeighbors[vid];

		__shared__ float s_maxDist;
		__shared__ float s_minDist;

		__shared__ localSurface s_localSurf;
		__shared__ Vec3f s_currLocalPos;

		__shared__ int s_nSurfVh;
		extern __shared__ int s_surfVh[];

		int surfCenterVid = vid;
		if constexpr (SurfOfNN) {
			surfCenterVid = nnvid;
		}

		// min/max search + loading oneRing
		if (threadIdx.x == 0) {
			s_minDist = FLT_MAX;
			s_maxDist = -FLT_MAX;

			Vec3f myPos = mesh->vertexPoints[vid];
			int k = 0;
			//int nHalfedges = mesh->halfedges.rowPtr_[vid + 1] - mesh->halfedges.rowPtr_[vid];
			for (auto it = mesh->he_ccw_order_begin(surfCenterVid); it != mesh->he_ccw_order_end(surfCenterVid); ++it) {
				const Halfedge& halfedge = *it;
				const int vdst = halfedge.targetVertex;
				s_surfVh[k] = vdst;
				Vec3f oneRingPos = mesh->vertexPoints[vdst];
				float d = (oneRingPos - myPos).norm();
				s_minDist = fminf(s_minDist, d);
				s_maxDist = fmaxf(s_maxDist, d);
				++k;
			}
			s_nSurfVh = k;
			s_localSurf = localSurfacesInit[nearestNeighbors[vid]];

			Vec3f currLocalPosVec(myPos - s_localSurf.p0);
			s_currLocalPos = { s_localSurf.ei[0].dot(currLocalPosVec), s_localSurf.ei[1].dot(currLocalPosVec), s_localSurf.ei[2].dot(currLocalPosVec) };
		}

		__syncthreads();


		float depth_scale = grid_scale;

		for (int idxppt = 0; idxppt < nPointsPerThread; ++idxppt) {
			int idx = threadIdx.x * nPointsPerThread + idxppt;
			int j = idx % nu;
			int i = idx / nu; //swap these if wrong

			float uMax, uMin, vMax, vMin;
			uMax = +depth_scale * s_maxDist;
			uMin = -depth_scale * s_maxDist;
			vMax = +depth_scale * s_maxDist;
			vMin = -depth_scale * s_maxDist;

			float u = s_currLocalPos[0] + affineFactorU * (i * uMin + (nu - 1 - i) * uMax);
			float v = s_currLocalPos[1] + affineFactorV * (j * vMin + (nv - 1 - j) * vMax);

			//Vec3f pWorld = localSurfaceEstimationWithFeature(vertexPointsInit, localSurfacesInit, nearestNeighbors, mesh, vid, s_surfVh, s_nSurfVh, u, v, table, surfacesRowPtr, localSurfacesFeatureInit);
			
			Vec3f pWorld;
			if constexpr (SurfOfNN) {
				pWorld = localSurfaceEstimationWithFeatureNew(vertexPointsInit, localSurfacesInit, nearestNeighbors, mesh, vid, s_surfVh, s_nSurfVh, u, v, table, surfacesRowPtr, localSurfacesFeatureInit);
			}
			else {
				pWorld = localSurfaceEstimationWithFeature(vertexPointsInit, localSurfacesInit, nearestNeighbors, mesh, vid, s_surfVh, s_nSurfVh, u, v, table, surfacesRowPtr, localSurfacesFeatureInit);
			}
			
			outSurfacePoints[idx] = pWorld;
		}
	} // k_fillEstimateLocalSurfacePoints


	__global__ void k_fillLocalSurfacePoints(int vid, int featureSid, int nu, int nv, MeshBaseDevice* mesh, float affineFactor, float grid_scale,
		localSurface* localSurfacesInit, int* nearestNeighbors, Vec3f* outSurfacePoints, localSurface* localSurfacesFeatureInit, int* rowPtr);
	
}



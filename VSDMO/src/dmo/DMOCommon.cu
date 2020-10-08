#include "DMOCommon.h"
#include "SurfaceConfig.h"

//using namespace SurfLS;
//using namespace Surf1D;

namespace DMO {

	
	// only considers topological neighbors
	__global__ void k_updateNearestNeighbor(MeshBaseDevice* mesh, Vec3f* vertexPointsInit, int* nearestNeighbors)
	{
		for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < mesh->nVerticesSurf; idx += blockDim.x * gridDim.x) {
			Vec3f p = mesh->vertexPoints[idx];
			int vh_init = nearestNeighbors[idx];
			if (vh_init < 0 || vh_init >= mesh->nVerticesSurf) {
				printf("vh_init out of bounds\n");
			}
			Vec3f p_init = vertexPointsInit[vh_init];

			float squared_dist = squaredDist(p, p_init);
			int vh_new;
			do {
				vh_new = vh_init;
				auto begin = mesh->he_ccw_order_begin(vh_init);
				auto end = mesh->he_ccw_order_end(vh_init);
				for (auto it = begin; it != end; ++it) {
					int dst = it->targetVertex;
					if (mesh->isFeature(idx) != mesh->isFeature(dst)) {
						continue;
					}
					Vec3f p_dst = vertexPointsInit[dst];
					float sd_vv = squaredDist(p, p_dst);
					if (sd_vv < squared_dist) {
						squared_dist = sd_vv;
						vh_init = dst;
					}
				}
			} while (vh_new != vh_init);
			nearestNeighbors[idx] = vh_new;
		}
	}


	__global__ void k_updateNearestNeighborAll(MeshBaseDevice* mesh, Vec3f* vertexPointsInit, int* nearestNeighbors)
	{
		for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < mesh->nVerticesSurf; idx += blockDim.x * gridDim.x) {
			Vec3f p = mesh->vertexPoints[idx];
			int vh_init = nearestNeighbors[idx];
			if (vh_init < 0 || vh_init >= mesh->nVerticesSurf) {
				printf("vh_init out of bounds\n");
			}
			Vec3f p_init = vertexPointsInit[vh_init];

			float squared_dist = squaredDist(p, p_init);
			int curr_nn = vh_init;
			
			for (int i = 0; i < mesh->nVerticesSurf; ++i) {
				if (mesh->isFeature(idx) == mesh->isFeature(i)) {
					Vec3f p_curr = vertexPointsInit[vh_init];
					float dist = squaredDist(p, p_curr);
					if (dist < squared_dist) {
						squared_dist = dist;
						curr_nn = i;
					}
				}
			}
			nearestNeighbors[idx] = curr_nn;
		}
	}


	__global__ void k_fillLocalSurfacePoints(int vid, int featureSid, int nu, int nv, MeshBaseDevice* mesh, float affineFactor, float grid_scale,
		localSurface* localSurfacesInit, int* nearestNeighbors, Vec3f* outSurfacePoints, localSurface* localSurfacesFeatureInit, int* rowPtr)
	{
		const int nnvid = nearestNeighbors[vid];
		const int nPointsPerThread = (nu * nv) / blockDim.x;
		const float affineFactorU = 1.f / float(nu - 1);
		const float affineFactorV = 1.f / float(nv - 1);

		__shared__ Vec3f s_currPos;
		__shared__ float s_maxDist;
		__shared__ float s_minDist;

		__shared__ localSurface s_localSurf;
		__shared__ Vec3f s_currLocalPos;


		// min/max search + loading oneRing
		if (threadIdx.x == 0) {
			s_minDist = FLT_MAX;
			s_maxDist = -FLT_MAX;

			s_currPos = mesh->vertexPoints[vid];

			// if we are a feature node, lookup localsurface offset in rowPtr
			if (nnvid < mesh->nVerticesSurfFree) {
				s_localSurf = localSurfacesInit[nnvid];
			}
			else {
				int vidOffset = nnvid - mesh->nVerticesSurfFree;
				int surfOffset = rowPtr[vidOffset];
				int nSurfaces = rowPtr[vidOffset + 1] - rowPtr[vidOffset];
				if (featureSid >= nSurfaces) {
					featureSid = nSurfaces - 1;
				}
				s_localSurf = localSurfacesFeatureInit[surfOffset + featureSid];
			}

			Vec3f currLocalPosVec(s_currPos - s_localSurf.p0);
			s_currLocalPos = { s_localSurf.ei[0].dot(currLocalPosVec), s_localSurf.ei[1].dot(currLocalPosVec), s_localSurf.ei[2].dot(currLocalPosVec) };

			for (auto it = mesh->he_ccw_order_begin(vid); it != mesh->he_ccw_order_end(vid); ++it) {
				const Halfedge& halfedge = *it;
				const int vdst = halfedge.targetVertex;
				Vec3f oneRingPos = mesh->vertexPoints[vdst];
				float d = (oneRingPos - s_currPos).norm();
				s_minDist = fminf(s_minDist, d);
				s_maxDist = fmaxf(s_maxDist, d);
			}
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
			uMin *= 2.f;
			uMax *= 2.f;
			vMin *= 2.f;
			vMax *= 2.f;

			float u = s_currLocalPos[0] + affineFactorU * (i * uMin + (nu - 1 - i) * uMax);
			float v = s_currLocalPos[1] + affineFactorV * (j * vMin + (nv - 1 - j) * vMax);

			Vec3f pWorld = s_localSurf.localToWorldCoords(u, v);

			outSurfacePoints[idx] = pWorld;
		}
	} // k_fillLocalSurfacePoints
	
}



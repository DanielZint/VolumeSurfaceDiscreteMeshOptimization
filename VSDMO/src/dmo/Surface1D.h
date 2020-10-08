#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"


#include "mesh/MeshTriGPU.h"
#include "mesh/DMOMeshTri.h"
#include "mesh/DMOMeshTet.h"
#include "ConfigUsing.h"
#include "CudaUtil.h"
#include "SurfaceCommon.h"

namespace Surf1D {
	struct localSurface1d {
		// Origin of the local coordinate system
		Vec3f p0;
		// Axes of the local coordinate system
		Vec3f ei[2];
		// Coefficient for the polynomial function
		float a;
		// compute z-coordinate
		//		z = a * x * x
		// transforming back to global coordinates:
		//		glob = p + u * x + v * y

			// Compute v(u) = a*u*u
		__host__ __device__ float polynomialFunction1d(const float u) const {
			return a * u * u;
		}

		// Transform from local to world coordinates.
		__host__ __device__ Vec3f localToWorldCoords1d(const float u, const float v) const {
			return p0 + u * ei[0] + v * ei[1];
		}

		__host__ __device__ Vec2f worldToLocalCoords1d(const Vec3f& worldCoords) const {
			Vec3f p1 = worldCoords - p0;
			float u = p1.dot(ei[0]);
			float v = p1.dot(ei[1]);
			return { u,v };
		}
	};



	__device__ inline const localSurface1d& getNearestNeighborSurf1d(localSurface1d* localSurfacesInit, const int* nearestNeighbors, const int vid) {
		// surface of nearest neighbor of vh
		int nn = nearestNeighbors[vid];
		return localSurfacesInit[nn];
	}

	
	__device__ inline Vec3f localSurfaceEstimation1d(Vec3f* vertexPointsInit, localSurface1d* localSurfacesInit, const int* nearestNeighbors, const int vid,
		const int* neighbors, const int nneigh, const float u)
	{
		int nn = nearestNeighbors[vid];
		const localSurface1d& localSurf = getNearestNeighborSurf1d(localSurfacesInit, nearestNeighbors, vid);
		const float v = localSurf.polynomialFunction1d(u);

		Vec3f p_world = localSurf.localToWorldCoords1d(u, v);		// position of (u,v) in world coordinates

		Vec3f p_localOfNeighbors[MAX_NEIGHBORS + 1];
		float invDist[MAX_NEIGHBORS + 1];
		float invDistSum = 0.f;

		// add us to estimation
		p_localOfNeighbors[0] = p_world;
		invDist[0] = (1.f / ((vertexPointsInit[nn] - p_world).squaredNorm() + EPSILON));
		invDistSum += invDist[0];

		for (int i = 0; i < nneigh; ++i) {
			int vneigh = neighbors[i];
			nn = nearestNeighbors[vneigh];
			Vec3f pNeighbor_world = vertexPointsInit[nn];

			invDist[i + 1] = (1.f / ((pNeighbor_world - p_world).squaredNorm() + EPSILON));
			invDistSum += invDist[i + 1];

			const localSurface1d& localSurfNeighbor = getNearestNeighborSurf1d(localSurfacesInit, nearestNeighbors, vneigh);
			Vec2f p_localOfNeighbor = localSurfNeighbor.worldToLocalCoords1d(p_world);
			p_localOfNeighbor[1] = localSurfNeighbor.polynomialFunction1d(p_localOfNeighbor[0]);

			Vec3f p_worldNeigh = localSurfNeighbor.localToWorldCoords1d(p_localOfNeighbor[0], p_localOfNeighbor[1]);

			p_localOfNeighbors[i + 1] = p_worldNeigh;
		}

		Vec3f projectedNeighborsSum(0,0,0);
		float invInvDistSum = 1.f / invDistSum;
		for (int i = 0; i < nneigh + 1; ++i) {
			projectedNeighborsSum += p_localOfNeighbors[i] * invDist[i] * invInvDistSum;
		}
		return projectedNeighborsSum;
	}



	__global__ void k_computeLocalSurfaces1d(MeshBaseDevice* mesh, ArrayView<localSurface1d> localSurfaces);

	__global__ void k_updateNearestNeighbor1d(MeshBaseDevice* mesh, Vec3f* vertexPointsInit, int* nearestNeighbors);

	template<class DMOMesh>
	inline void computeLocalSurfaces1d(DMOMesh& mesh, device_vector<localSurface1d>& localSurfaces) {
		const int BLOCK_SIZE = 128;
		k_computeLocalSurfaces1d << <getBlockCount(mesh.nVerticesSurf, BLOCK_SIZE), BLOCK_SIZE >> >
			(mesh.d_mesh, localSurfaces);
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
	}



}

#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "ConfigUsing.h"
#include "CudaUtil.h"
#include "SurfaceConfig.h"


__device__ inline localSurface surfaceHack(MeshBaseDevice* mesh, const int vid, const int vNeigh) {
	Halfedge halfedge;
	for (auto it = mesh->he_ccw_order_begin(vid); it != mesh->he_ccw_order_end(vid); ++it) {
		halfedge = *it;
		if (halfedge.targetVertex == vNeigh) {
			break;
		}
	}

	int face1 = halfedge.incidentFace;
	int face2 = mesh->halfedges.values_[halfedge.oppositeHE].incidentFace;
	Vec3f fn1 = mesh->faceNormals[face1];
	Vec3f fn2 = mesh->faceNormals[face2];
	Vec3f normal = 0.5f * (fn1 + fn2);

	auto it = mesh->he_ccw_order_begin(vid);
	Halfedge otherHE = *it;
	if (otherHE.targetVertex == vNeigh) {
		// dont want same neighbor point
		++it;
		otherHE = *it;
	}
	int vOther = otherHE.targetVertex;
	// construct a random vector
	Vec3f randVec = mesh->vertexPoints[vOther] - mesh->vertexPoints[vid];

	normal.normalize();
	//randVec.normalize(); // TODO ?
	if (normal.cross(randVec).norm() < 10.f * EPSILON) {
		printf("normal vector needs to be recalculated\n");
		assert(0);
		randVec = { normal[0], normal[2], -normal[1] };
	}

	// compute tanget vectors
	Vec3f t1 = normal.cross(randVec);
	Vec3f t2 = normal.cross(t1);

	// normalize
	normal.normalize();
	t1.normalize();
	t2.normalize();

	Vec3f pNeigh = mesh->vertexPoints[vNeigh];

	localSurface ls;
	ls.p0 = pNeigh;
	ls.setZero();
	ls.ei[0] = normal;
	ls.ei[1] = t1;
	ls.ei[2] = t2;
	return ls;
}



// Old
__device__ inline Vec3f localSurfaceEstimation(Vec3f* vertexPointsInit, localSurface* localSurfacesInit, int* nearestNeighbors, MeshBaseDevice* mesh, const int vid,
	int* neighbors, const int nneigh, const float u, const float v)
{
	int nn = nearestNeighbors[vid];
	const localSurface& localSurf = getNearestNeighborSurf(localSurfacesInit, nearestNeighbors, vid);
	const float w = localSurf.polynomialFunction(u, v);

	Vec3f p_world = localSurf.localToWorldCoords(u, v, w);		// position of (u,v) in world coordinates

	Vec3f p_localOfNeighbors[MAX_NEIGHBORS + 1];
	float invDist[MAX_NEIGHBORS + 1];
	float invDistSum = 0;

	// add vh to estimation
	p_localOfNeighbors[0] = p_world;
	invDist[0] = (1.f / ((vertexPointsInit[nn] - p_world).squaredNorm() + EPSILON));
	invDistSum += invDist[0];

	for (int i = 0; i < nneigh; ++i) {
		int vNeigh = neighbors[i];
		nn = nearestNeighbors[vNeigh];
		Vec3f pNeighbor_world = vertexPointsInit[nn];

		invDist[i + 1] = (1.f / ((pNeighbor_world - p_world).squaredNorm() + EPSILON));
		invDistSum += invDist[i + 1];

		localSurface localSurfNeighbor;

		if (mesh->isFeature(vNeigh)) {
			localSurfNeighbor = surfaceHack(mesh, vid, vNeigh);
		}
		else {
			localSurfNeighbor = getNearestNeighborSurf(localSurfacesInit, nearestNeighbors, neighbors[i]);
		}

		Vec3f p_localOfNeighbor = localSurfNeighbor.worldToLocalCoords(p_world);
		p_localOfNeighbor[2] = localSurfNeighbor.polynomialFunction(p_localOfNeighbor[0], p_localOfNeighbor[1]);

		Vec3f p_world2 = localSurfNeighbor.localToWorldCoords(p_localOfNeighbor[0], p_localOfNeighbor[1]);

		p_localOfNeighbors[i + 1] = p_world2;
	}

	Vec3f projectedNeighborsSum = { 0,0,0 };
	float invInvDistSum = 1.f / invDistSum;
	for (int i = 0; i < nneigh + 1; ++i) {
		projectedNeighborsSum += p_localOfNeighbors[i] * invDist[i] * invInvDistSum;
	}
	return projectedNeighborsSum;
}





__device__ inline Vec3f localSurfaceEstimationWithFeature(Vec3f* vertexPointsInit, localSurface* localSurfacesInit, int* nearestNeighbors, MeshBaseDevice* mesh, const int vid,
	int* neighbors, const int nneigh, const float u, const float v, const int* table, const int* rowPtr, const localSurface* localSurfacesFeatureInit)
{
	const int featureStartOffset = mesh->nVerticesSurfFree;
	const int nnvid = nearestNeighbors[vid];
	const localSurface& localSurf = getNearestNeighborSurf(localSurfacesInit, nearestNeighbors, vid);
	const float w = localSurf.polynomialFunction(u, v);

	const Vec3f p_world = localSurf.localToWorldCoords(u, v, w);		// position of (u,v) in world coordinates

	Vec3f p_localOfNeighbors[MAX_NEIGHBORS + 1];
	float invDist[MAX_NEIGHBORS + 1];
	float invDistSum = 0;

	// add vh to estimation
	p_localOfNeighbors[0] = p_world;
	invDist[0] = (1.f / ((vertexPointsInit[nnvid] - p_world).squaredNorm() + EPSILON));
	invDistSum += invDist[0];

	for (int i = 0; i < nneigh; ++i) {
		const int vNeigh = neighbors[i];
		const int nn = nearestNeighbors[vNeigh];
		const Vec3f pNeighbor_world = vertexPointsInit[nn];

		invDist[i + 1] = (1.f / ((pNeighbor_world - p_world).squaredNorm() + EPSILON));
		invDistSum += invDist[i + 1];

		localSurface localSurfNeighbor;

		if (mesh->isFeature(nn)) {
			// new
			const int nnOffset = nn - featureStartOffset;
			assert(nnOffset >= 0);
			Halfedge halfedge;
			float minDist = FLT_MAX;
			int minDistArg = 0;
			const Vec3f vidPos(mesh->vertexPoints[vid]);
			int heIndex = 0;
			for (auto it = mesh->he_ccw_order_begin(nn); it != mesh->he_ccw_order_end(nn); ++it) {
				halfedge = *it;
				const Vec3f targetPoint = mesh->vertexPoints[halfedge.targetVertex];
				const float dist = (targetPoint - vidPos).squaredNorm();
				if (dist < minDist) {
					minDist = dist;
					minDistArg = heIndex;
				}

				++heIndex;
			}
			const int sid = table[TABLE_SIZE * nnOffset + minDistArg];
			assert(sid != -1);
			localSurfNeighbor = localSurfacesFeatureInit[rowPtr[nnOffset] + sid];


			//localSurfNeighbor = surfaceHack(mesh, vid, vNeigh);

		}
		else {
			localSurfNeighbor = getNearestNeighborSurf(localSurfacesInit, nearestNeighbors, neighbors[i]);
		}

		Vec3f p_localOfNeighbor = localSurfNeighbor.worldToLocalCoords(p_world);
		p_localOfNeighbor[2] = localSurfNeighbor.polynomialFunction(p_localOfNeighbor[0], p_localOfNeighbor[1]);

		const Vec3f p_world2 = localSurfNeighbor.localToWorldCoords(p_localOfNeighbor[0], p_localOfNeighbor[1]);

		p_localOfNeighbors[i + 1] = p_world2;
	}

	Vec3f projectedNeighborsSum = { 0,0,0 };
	float invInvDistSum = 1.f / invDistSum;
	for (int i = 0; i < nneigh + 1; ++i) {
		projectedNeighborsSum += p_localOfNeighbors[i] * invDist[i] * invInvDistSum;
	}
	return projectedNeighborsSum;
}




// expects oneRing of nearest neighbor of vid in neighbors
__device__ inline Vec3f localSurfaceEstimationWithFeatureNew(Vec3f* vertexPointsInit, localSurface* localSurfacesInit, int* nearestNeighbors, MeshBaseDevice* mesh, const int vid,
	int* neighbors, const int nneigh, const float u, const float v, const int* table, const int* rowPtr, const localSurface* localSurfacesFeatureInit)
{
	const int featureStartOffset = mesh->nVerticesSurfFree;
	const int nnvid = nearestNeighbors[vid];
	const localSurface& localSurf = localSurfacesInit[nnvid];
	const float w = localSurf.polynomialFunction(u, v);

	const Vec3f p_world = localSurf.localToWorldCoords(u, v, w);		// position of (u,v) in world coordinates

	Vec3f p_localOfNeighbors[MAX_NEIGHBORS + 1];
	float invDist[MAX_NEIGHBORS + 1];
	float invDistSum = 0;

	// add vh to estimation
	p_localOfNeighbors[0] = p_world;
	invDist[0] = (1.f / ((vertexPointsInit[nnvid] - p_world).squaredNorm() + EPSILON));
	invDistSum += invDist[0];

	for (int i = 0; i < nneigh; ++i) {
		const int neighvid = neighbors[i];
		const Vec3f pNeighbor_world = vertexPointsInit[neighvid];

		invDist[i + 1] = (1.f / ((pNeighbor_world - p_world).squaredNorm() + EPSILON));
		invDistSum += invDist[i + 1];

		localSurface localSurfNeighbor;

		if (mesh->isFeature(neighvid)) {
			int heIndex = 0;

			for (auto it = mesh->he_ccw_order_begin(neighvid); it != mesh->he_ccw_order_end(neighvid); ++it) {
				const Halfedge& halfedge = *it;
				if (halfedge.targetVertex == nnvid) {
					break;
				}
				++heIndex;
			}

			const int nnOffset = neighvid - featureStartOffset;
			assert(nnOffset >= 0);
			const int sid = table[TABLE_SIZE * nnOffset + heIndex];
			assert(sid != -1);
			localSurfNeighbor = localSurfacesFeatureInit[rowPtr[nnOffset] + sid];

			//localSurfNeighbor = surfaceHack(mesh, vid, neighvid);
		}
		else {
			localSurfNeighbor = localSurfacesInit[neighvid];
		}

		Vec3f p_localOfNeighbor = localSurfNeighbor.worldToLocalCoords(p_world);
		p_localOfNeighbor[2] = localSurfNeighbor.polynomialFunction(p_localOfNeighbor[0], p_localOfNeighbor[1]);

		const Vec3f p_world2 = localSurfNeighbor.localToWorldCoords(p_localOfNeighbor[0], p_localOfNeighbor[1]);

		p_localOfNeighbors[i + 1] = p_world2;
	}

	Vec3f projectedNeighborsSum = { 0,0,0 };
	float invInvDistSum = 1.f / invDistSum;
	for (int i = 0; i < nneigh + 1; ++i) {
		projectedNeighborsSum += p_localOfNeighbors[i] * invDist[i] * invInvDistSum;
	}
	return projectedNeighborsSum;
}




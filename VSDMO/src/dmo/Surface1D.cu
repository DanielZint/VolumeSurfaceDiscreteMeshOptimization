#include "Surface1D.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "CudaUtil.h"

namespace Surf1D {

__device__ inline void computeLocalCoordinates1d(const int vid, const int vidFeature, const MeshBaseDevice* mesh, Vec3f* xi1, Vec3f* xi2) {
	Vec3f normal = mesh->vertexNormals[vid];
	if (mesh->isFlat) {
		normal = Vec3f(0, 0, 1);
	}

	// construct a random vector
	Vec3f randVec = mesh->vertexPoints[vidFeature] - mesh->vertexPoints[vid];

	if (normal.cross(randVec).norm() < 10.f * EPSILON) {
		printf("computeLocalCoordinates: normal vector needs to be recalculated. normal: %.10g %.10g %.10g randVec: %.10g %.10g %.10g\n", normal.x, normal.y, normal.z, randVec.x, randVec.y, randVec.z);
		//assert(0);
		randVec = { normal[0], normal[2], -normal[1] };
	}

	// compute tanget vectors
	Vec3f t1 = normal.cross(randVec);
	Vec3f t2 = normal.cross(t1);

	// normalize
	normal.normalize();
	t1.normalize();
	t2.normalize();

	if (std::abs(t1.cross(t2).dot(normal) - 1) > 10.f * EPSILON) {
		printf("computeLocalCoordinates: computation of local coordinates failed for vertex %i\n", vid);
		assert(0);
	}

	*xi1 = t2;
	*xi2 = normal;
}

__device__ localSurface1d computeLocalSurface1d(const int vid, MeshBaseDevice* mesh) {
	Vec3f p_center = mesh->vertexPoints[vid];

	// neighbors (also contains vh itself)
	int neighbors[MAX_NEIGHBORS];
	int pos = 0;
	neighbors[pos++] = vid;
	int vidFeature = -1; // get a neighbor vertex of vid which is a feature vertex
	// collect one-ring
	for (auto it = mesh->he_ccw_order_begin(vid); it != mesh->he_ccw_order_end(vid); ++it) {
		int dst = it->targetVertex;
		if (mesh->isFeature(dst)) {
			neighbors[pos++] = dst;
			vidFeature = dst;
		}
	}
	// neighbors contains all feature neighbors and us
	if (vidFeature == -1) {
		printf("vidFeature is -1, vid %i\n", vid);
		assert(0);
	}

	Vec3f ei[2];

	computeLocalCoordinates1d(vid, vidFeature, mesh, &ei[0], &ei[1]);

	//convert points to frenet basis & fitting function f(x,y) = ax -> (x)(a)T = z =^= Ax=b -> ATA x = ATb
	float ATA = 0;
	float ATb = 0;
	for (auto i = 0; i < pos; ++i) {
		Vec3f p_neighbor = mesh->vertexPoints[neighbors[i]];
		Vec3f p = p_neighbor - p_center;
		const float u = ei[0].dot(p);
		const float v = ei[1].dot(p);

		ATb += u * u * v;
		ATA += u * u * u * u;
	}

	float abc = ATb / ATA;

	localSurface1d para;
	para.p0 = p_center;
	para.a = abc;
	para.ei[0] = ei[0];
	para.ei[1] = ei[1];
	//printf("vid %i p0 %f %f %f a %f e0 %f %f %f e1 %f %f %f\n", vid, XYZ(p_center),abc,XYZ(ei[0]), XYZ(ei[1]));

	return para;
}

__global__ void k_computeLocalSurfaces1d(MeshBaseDevice* mesh, ArrayView<localSurface1d> localSurfaces)
{
	for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < mesh->nVerticesSurf; idx += blockDim.x * gridDim.x) {
		if (mesh->isFeature(idx)) {
			localSurfaces[idx] = computeLocalSurface1d(idx, mesh);
		}
	}
}



}

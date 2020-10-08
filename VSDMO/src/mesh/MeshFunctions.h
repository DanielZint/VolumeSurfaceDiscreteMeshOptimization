#pragma once

#include "MeshTriGPU.h"
#include "DMOMeshTri.h"
#include "DMOMeshTet.h"
#include <thrust/extrema.h>

struct AABB {
	__host__ __device__ AABB() {}
	__host__ __device__ AABB(const Vec3f& point) : minPos(point), maxPos(point) {}
	__host__ __device__ AABB& operator=(const Vec3f& point) {
		minPos = point;
		maxPos = point;
		return *this;
	}
	__host__ __device__ AABB(const Vec3f& pointMin, const Vec3f& pointMax) : minPos(pointMin), maxPos(pointMax) {}
	Vec3f minPos;
	Vec3f maxPos;
};

int findClosestVertex(DMOMeshBase& mesh, Vec3f pos);

AABB findAABB(DMOMeshBase& mesh);



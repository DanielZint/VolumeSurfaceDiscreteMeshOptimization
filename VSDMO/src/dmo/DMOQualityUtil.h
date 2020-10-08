#pragma once

#include "ConfigUsing.h"
#include "DMOConfig.h"
#include "Vec3.h"
#include "SurfaceCommon.h"

namespace DMO {

	enum class QualityCriterium {
		MEAN_RATIO,
		AREA,
		RIGHT_ANGLE,
		JACOBIAN,
		MIN_ANGLE,
		RADIUS_RATIO,
		MAX_ANGLE
	};

	__device__ __forceinline__ float sign(float a) {
		if (a > 0.f) {
			return 1.f;
		}
		else if (a < 0.f) {
			return -1.f;
		}
		return 0.f;
	}

	__host__ __device__ __forceinline__ float calcAngle(const Vec2f p1, const Vec2f p2, const Vec2f p3) {
		const Vec2f e0(p1 - p2);
		const Vec2f e1(p3 - p2);
		float dot = e0.dot(e1);
		float len = sqrtf(e0.squaredNorm() * e1.squaredNorm());
		return acosf(dot / len);
	}

	__host__ __device__ __forceinline__ float calcAngle(const Vec3f p1, const Vec3f p2, const Vec3f p3) {
		const Vec3f e0(p1 - p2);
		const Vec3f e1(p3 - p2);
		float dot = e0.dot(e1);
		float len = sqrtf(e0.squaredNorm() * e1.squaredNorm());
		return acosf(dot / len);
	}

	__host__ __device__ __forceinline__ float calcAngle(const Vec3f n1, const Vec3f n2) {
		float dot = n1.dot(n2);
		float len = sqrtf(n1.squaredNorm() * n2.squaredNorm());
		return acosf(dot / len);
	}

	__host__ __device__ __forceinline__ float calcVolume(const Vec3f p[4]) {
		Vec3f e[3];
		e[0] = p[1] - p[0];
		e[1] = p[2] - p[0];
		e[2] = p[3] - p[0];
		float volume = (1.f / 6.f) * e[0].dot(e[1].cross(e[2]));
		return volume;
	}

	__host__ __device__ __forceinline__ float calcVolume(const Vec3f p0, const Vec3f p1, const Vec3f p2, const Vec3f p3) {
		Vec3f e[3];
		e[0] = p1 - p0;
		e[1] = p2 - p0;
		e[2] = p3 - p0;
		float volume = (1.f / 6.f) * e[0].dot(e[1].cross(e[2]));
		return volume;
	}

	__host__ __device__ __forceinline__ float determinant(const Vec3f p0, const Vec3f p1, const Vec3f p2, const Vec3f p3) {
		Vec3f e[3];
		e[0] = p1 - p0;
		e[1] = p2 - p0;
		e[2] = p3 - p0;
		return e[0].dot(e[1].cross(e[2]));
	}

	__host__ __device__ __forceinline__ float determinant(const Vec3f e0, const Vec3f e1, const Vec3f e2) {
		return e0.dot(e1.cross(e2));
	}

	__device__ __forceinline__ float volumeHex(const Vec3f p[8]) {
		float volume = 0.f;
		volume += determinant(p[0], p[1], p[4], p[3]);
		volume += determinant(p[5], p[4], p[1], p[6]);
		volume += determinant(p[2], p[3], p[6], p[1]);
		volume += determinant(p[7], p[6], p[3], p[4]);
		volume += determinant(p[3], p[1], p[4], p[6]);
		return volume * (1.f / 6.f);
	}

	__device__ __forceinline__ float volumeHex2(const Vec3f p[8]) {
		float volume = 0.f;
		volume += determinant(p[6] - p[0], p[1] - p[0], p[5] - p[2]);
		volume += determinant(p[6] - p[0], p[3] - p[0], p[2] - p[7]);
		volume += determinant(p[6] - p[0], p[4] - p[0], p[7] - p[5]);
		return volume * (1.f / 6.f);
	}

	__device__ __forceinline__ float volumeHex3(const Vec3f p[8]) {
		float d1 = determinant((p[6] - p[1]) + (p[7] - p[0]), p[6] - p[4], p[5] - p[0]);
		float d2 = determinant(p[7] - p[0], (p[6] - p[4]) + (p[2] - p[0]), p[6] - p[3]);
		float d3 = determinant(p[6] - p[1], p[2] - p[0], (p[6] - p[3]) + (p[5] - p[0]));
		float volume = d1 + d2 + d3;
		return volume * (1.f / 6.f);
	}

	__device__ __forceinline__ void hex_edges(const Vec3f& p0, const Vec3f& p1, const Vec3f& p2, const Vec3f& p3,
		const Vec3f& p4, const Vec3f& p5, const Vec3f& p6, const Vec3f& p7,
		Vec3f L[], const bool normalized)
	{
		L[0] = p1 - p0;    L[4] = p4 - p0;    L[8] = p5 - p4;
		L[1] = p2 - p1;    L[5] = p5 - p1;    L[9] = p6 - p5;
		L[2] = p3 - p2;    L[6] = p6 - p2;    L[10] = p7 - p6;
		L[3] = p3 - p0;    L[7] = p7 - p3;    L[11] = p7 - p4;

		if (normalized) for (int i = 0; i < 12; ++i) L[i].normalize();
	}

	__device__ __forceinline__ void hex_principal_axes(const Vec3f& p0, const Vec3f& p1, const Vec3f& p2, const Vec3f& p3,
		const Vec3f& p4, const Vec3f& p5, const Vec3f& p6, const Vec3f& p7,
		Vec3f X[], const bool normalized)
	{
		X[0] = (p1 - p0) + (p2 - p3) + (p5 - p4) + (p6 - p7);
		X[1] = (p3 - p0) + (p2 - p1) + (p7 - p4) + (p6 - p5);
		X[2] = (p4 - p0) + (p5 - p1) + (p6 - p2) + (p7 - p3);

		if (normalized) for (int i = 0; i < 3; ++i) X[i].normalize();
	}

	__device__ __forceinline__ void hex_subtets(const Vec3f L[], const Vec3f X[], const int id, Vec3f tet[])
	{
		switch (id)
		{
		case 0: tet[0] = L[0];  tet[1] = L[3];  tet[2] = L[4]; break;
		case 1: tet[0] = L[1];  tet[1] = -L[0];  tet[2] = L[5]; break;
		case 2: tet[0] = L[2];  tet[1] = -L[1];  tet[2] = L[6]; break;
		case 3: tet[0] = -L[3];  tet[1] = -L[2];  tet[2] = L[7]; break;
		case 4: tet[0] = L[11]; tet[1] = L[8];  tet[2] = -L[4]; break;
		case 5: tet[0] = -L[8];  tet[1] = L[9];  tet[2] = -L[5]; break;
		case 6: tet[0] = -L[9];  tet[1] = L[10]; tet[2] = -L[6]; break;
		case 7: tet[0] = -L[10]; tet[1] = -L[11]; tet[2] = -L[7]; break;
		case 8: tet[0] = X[0];  tet[1] = X[1];  tet[2] = X[2]; break;
		}
	}

	__host__ __device__ __forceinline__ float jacobianNormSquared(const Vec3f e0, const Vec3f e1, const Vec3f e2) {
		return e0.squaredNorm() + e1.squaredNorm() + e2.squaredNorm();
	}



}


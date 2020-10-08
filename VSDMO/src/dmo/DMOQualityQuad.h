#pragma once

#include "DMOQualityUtil.h"

namespace DMO {



	///////// Mean Ratio



	__device__ __forceinline__ float meanRatioMetricQuad(const Vec3f p[4]) {
		Vec3f e[4];
		float e_length_squared[4];
		e[0] = p[1] - p[0];
		e[1] = p[2] - p[1];
		e[2] = p[3] - p[2];
		e[3] = p[0] - p[3];
		e_length_squared[0] = e[0].dot(e[0]);
		e_length_squared[1] = e[1].dot(e[1]);
		e_length_squared[2] = e[2].dot(e[2]);
		e_length_squared[3] = e[3].dot(e[3]);
		float l = e_length_squared[0] + e_length_squared[1] + e_length_squared[2] + e_length_squared[3];
		float area1 = ((e[1]).cross(-e[0])).norm();
		float area2 = ((e[3]).cross(-e[2])).norm();
		return 2.f * (area1 + area2) / l;
	}

	__device__ __forceinline__ float meanRatioMetricQuad(const Vec3f p[4], const Vec3f n) {
		Vec3f e[4];
		float e_length_squared[4];
		e[0] = p[1] - p[0];
		e[1] = p[2] - p[1];
		e[2] = p[3] - p[2];
		e[3] = p[0] - p[3];
		e_length_squared[0] = e[0].dot(e[0]);
		e_length_squared[1] = e[1].dot(e[1]);
		e_length_squared[2] = e[2].dot(e[2]);
		e_length_squared[3] = e[3].dot(e[3]);
		float l = e_length_squared[0] + e_length_squared[1] + e_length_squared[2] + e_length_squared[3];
		Vec3f normalTri1 = (e[1]).cross(-e[0]);
		Vec3f normalTri2 = (e[3]).cross(-e[2]);
		float area1 = normalTri1.norm();
		float area2 = normalTri2.norm();
		float ndotn1 = normalTri1.normalized().dot(n);
		float ndotn2 = normalTri2.normalized().dot(n);
		if (ndotn1 < 0.7f || ndotn2 < 0.7f) return -1.f; // TODO threshold variable?
		return 2.f * (area1 + area2) / l;
		//float area = normalTri.norm() * sign(ndotn);
	}


	__device__ __forceinline__ float qualityShapeMRMQuad(const int n_oneRing, const Vec3f* oneRing, const Vec3f p) {
		float q = FLT_MAX;
		if (n_oneRing <= 0) {
			printf("n_oneRing <= 0\n");
			assert(0);
		}
		for (int k = 0; k < n_oneRing; ++k) {
			Vec3f v[4];
			v[0] = p;
			v[1] = oneRing[2 * k + 0];
			v[2] = oneRing[2 * k + 1];
			v[3] = oneRing[2 * k + 2];
			
			float currq = meanRatioMetricQuad(v);
			q = fminf(q, currq);
		}
		return q;
	}

	__device__ __forceinline__ float qualityShapeMRMQuad(const int n_oneRing, const Vec3f* oneRing, const Vec3f p, const Vec3f n) {
		float q = FLT_MAX;
		if (n_oneRing <= 0) {
			printf("n_oneRing <= 0\n");
			assert(0);
		}
		for (int k = 0; k < n_oneRing; ++k) {
			Vec3f v[4];
			v[0] = p;
			v[1] = oneRing[2 * k + 0];
			v[2] = oneRing[2 * k + 1];
			v[3] = oneRing[2 * k + 2];

			float currq = meanRatioMetricQuad(v, n);
			q = fminf(q, currq);
		}
		return q;
	}


	__device__ __forceinline__ float qualityQuadRing(const int n_oneRing, const Vec3f* oneRing, const Vec3f p, QualityCriterium q_crit) {
		if (q_crit == QualityCriterium::MEAN_RATIO) {
			return qualityShapeMRMQuad(n_oneRing, oneRing, p);
		}
		else if (q_crit == QualityCriterium::MIN_ANGLE) {
			return 1.f;
		}
		else if (q_crit == QualityCriterium::MAX_ANGLE) {
			return 1.f;
		}
		else {
			return 1.f;
		}
	}

	__device__ __forceinline__ float qualityQuadRing(const int n_oneRing, const Vec3f* oneRing, const Vec3f p, const Vec3f n, QualityCriterium q_crit) {
		if (q_crit == QualityCriterium::MEAN_RATIO) {
			return qualityShapeMRMQuad(n_oneRing, oneRing, p, n);
		}
		else if (q_crit == QualityCriterium::MIN_ANGLE) {
			return 1.f;
		}
		else if (q_crit == QualityCriterium::MAX_ANGLE) {
			return 1.f;
		}
		else {
			return 1.f;
		}
	}


	__device__ __forceinline__ float qualityQuad(const Vec3f p[4], QualityCriterium q_crit) {
		if (q_crit == QualityCriterium::MEAN_RATIO) {
			return meanRatioMetricQuad(p);
		}
		else if (q_crit == QualityCriterium::MIN_ANGLE) {
			return 1.f;
		}
		else if (q_crit == QualityCriterium::MAX_ANGLE) {
			return 1.f;
		}
		else {
			return 1.f;
		}
	}



}


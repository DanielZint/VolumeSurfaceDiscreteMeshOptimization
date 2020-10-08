#pragma once

#include "DMOQualityUtil.h"

namespace DMO {

	/////// Min Angle


	__device__ __forceinline__ float minAngleMetricTet(const Vec3f p[4]) {
		Vec3f e[6];
		e[0] = p[1] - p[0];
		e[1] = p[2] - p[0];
		e[2] = p[3] - p[0];
		e[3] = p[2] - p[1];
		e[4] = p[3] - p[1];
		e[5] = p[3] - p[2];
		Vec3f n[4];
		n[0] = e[0].cross(e[2]).normalized();
		n[1] = e[3].cross(e[4]).normalized();
		n[2] = e[1].cross(e[0]).normalized();
		n[3] = e[2].cross(e[1]).normalized();
		float a1 = calcAngle(-n[0], n[1]);
		float a2 = calcAngle(-n[0], n[2]);
		float a3 = calcAngle(-n[0], n[3]);
		float a4 = calcAngle(-n[1], n[2]);
		float a5 = calcAngle(-n[1], n[3]);
		float a6 = calcAngle(-n[2], n[3]);
		float min_angle = fminf(fminf(a1, a2), fminf(fminf(a3, a4), fminf(a5, a6)));

		float volume = (1.f / 6.f) * e[0].dot(e[1].cross(e[2]));
		if (volume < 0.f) {
			return volume;
		}
		float q = 3.f * min_angle / M_PI;
		return q;
	}



	__device__ __forceinline__ float qualityShapeMinAMTet(const int n_oneRing, const Vec3f* oneRing, const Vec3f p) {
		float q = FLT_MAX;
		if (n_oneRing <= 0) {
			printf("n_oneRing <= 0\n");
			assert(0);
		}
		for (int k = 0; k < n_oneRing; ++k) {
			Vec3f v[4];
			v[0] = p;
			v[1] = oneRing[3 * k + 0];
			v[2] = oneRing[3 * k + 1];
			v[3] = oneRing[3 * k + 2];
			float currq = minAngleMetricTet(v);
			q = fminf(q, currq);
		}
		return q;
	}


	/////// Max Angle



	__device__ __forceinline__ float maxAngleMetricTet(const Vec3f p[4]) {
		Vec3f e[6];
		e[0] = p[1] - p[0];
		e[1] = p[2] - p[0];
		e[2] = p[3] - p[0];
		e[3] = p[2] - p[1];
		e[4] = p[3] - p[1];
		e[5] = p[3] - p[2];
		Vec3f n[4];
		n[0] = e[0].cross(e[2]).normalized();
		n[1] = e[3].cross(e[4]).normalized();
		n[2] = e[1].cross(e[0]).normalized();
		n[3] = e[2].cross(e[1]).normalized();
		float a1 = calcAngle(-n[0], n[1]);
		float a2 = calcAngle(-n[0], n[2]);
		float a3 = calcAngle(-n[0], n[3]);
		float a4 = calcAngle(-n[1], n[2]);
		float a5 = calcAngle(-n[1], n[3]);
		float a6 = calcAngle(-n[2], n[3]);
		float max_angle = fmaxf(fmaxf(a1, a2), fmaxf(fmaxf(a3, a4), fmaxf(a5, a6)));

		float volume = (1.f / 6.f) * e[0].dot(e[1].cross(e[2]));
		if (volume < 0.f) {
			return volume;
		}
		//float q = 1.f - (3.f * max_angle - M_PI) / (2.f * M_PI);
		float q = 1.f - (max_angle / M_PI);
		return q;
	}



	__device__ __forceinline__ float qualityShapeMaxAMTet(const int n_oneRing, const Vec3f* oneRing, const Vec3f p) {
		float q = FLT_MAX;
		if (n_oneRing <= 0) {
			printf("n_oneRing <= 0\n");
			assert(0);
		}
		for (int k = 0; k < n_oneRing; ++k) {
			Vec3f v[4];
			v[0] = p;
			v[1] = oneRing[3 * k + 0];
			v[2] = oneRing[3 * k + 1];
			v[3] = oneRing[3 * k + 2];
			float currq = maxAngleMetricTet(v);
			q = fminf(q, currq);
		}
		return q;
	}


	///////// Mean Ratio



	__device__ __forceinline__ float meanRatioMetricTet(const Vec3f p[4]) {
		Vec3f e[6];
		e[0] = p[1] - p[0];
		e[1] = p[2] - p[0];
		e[2] = p[3] - p[0];

		//float volume = (1.f / 6.f) * (e[0].cross(e[1])).dot(e[2]);
		float volume = (1.f / 6.f) * e[0].dot(e[1].cross(e[2]));
		if (volume < 0.f) {
			return volume;
		}
		float mu = cbrtf(volume * volume);

		e[3] = p[2] - p[1];
		e[4] = p[3] - p[1];
		e[5] = p[3] - p[2];

		float l_sum = 0.f;
		for (int i = 0; i < 6; ++i) {
			l_sum += e[i].dot(e[i]);
		}

		const float cbrtf9 = 2.0800838230519f;
		return 12.f * cbrtf9 * mu / l_sum;
	}


	__device__ __forceinline__ float qualityShapeMRMTet(const int n_oneRing, const Vec3f* oneRing, const Vec3f p) {
		float q = FLT_MAX;
		if (n_oneRing <= 0) {
			printf("n_oneRing <= 0\n");
			assert(0);
		}
		for (int k = 0; k < n_oneRing; ++k) {
			Vec3f v[4];
			v[0] = p;
			v[1] = oneRing[3 * k + 0];
			v[2] = oneRing[3 * k + 1];
			v[3] = oneRing[3 * k + 2];
			float currq = meanRatioMetricTet(v);
			q = fminf(q, currq);
		}
		return q;
	}


	__device__ __forceinline__ float qualityTetRing(const int n_oneRing, const Vec3f* oneRing, const Vec3f p, QualityCriterium q_crit) {
		if (q_crit == QualityCriterium::MEAN_RATIO) {
			return qualityShapeMRMTet(n_oneRing, oneRing, p);
		}
		else if (q_crit == QualityCriterium::MIN_ANGLE) {
			return qualityShapeMinAMTet(n_oneRing, oneRing, p);
		}
		else if (q_crit == QualityCriterium::MAX_ANGLE) {
			return qualityShapeMaxAMTet(n_oneRing, oneRing, p);
		}
		else {
			return 1.f;
		}
	}


	__device__ __forceinline__ float qualityTet(const Vec3f p[4], QualityCriterium q_crit) {
		if (q_crit == QualityCriterium::MEAN_RATIO) {
			return meanRatioMetricTet(p);
		}
		else if (q_crit == QualityCriterium::MIN_ANGLE) {
			return minAngleMetricTet(p);
		}
		else if (q_crit == QualityCriterium::MAX_ANGLE) {
			return maxAngleMetricTet(p);
		}
		else {
			return 1.f;
		}
	}



}


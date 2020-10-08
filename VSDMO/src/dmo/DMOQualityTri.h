#pragma once

#include "DMOQualityUtil.h"

namespace DMO {


	/////// Min Angle

	__device__ __forceinline__ float minAngleMetricTri2D(const Vec2f p[3]) {
		Vec2f e[3];
		for (int i = 0; i < 3; ++i) {
			int j = (i + 1) % 3;
			e[i] = p[j] - p[i];
		}
		float a = calcAngle(p[0], p[1], p[2]);
		float b = calcAngle(p[1], p[2], p[0]);
		float c = calcAngle(p[2], p[0], p[1]);
		float min_angle = fminf(a, b);
		min_angle = fminf(min_angle, c);
		float area = e[0][0] * e[1][1] - e[0][1] * e[1][0];
		float q = 3.f * min_angle / M_PI;
		// if triangle is flipped, make value negative
		if (area < 0)
			return area;
		else
			return q;
	}


	__device__ __forceinline__ float minAngleMetricTri(const Vec3f p[3]) {
		Vec3f e[3];
		for (int i = 0; i < 3; ++i) {
			int j = (i + 1) % 3;
			e[i] = p[j] - p[i];
		}
		float a = calcAngle(p[0], p[1], p[2]);
		float b = calcAngle(p[1], p[2], p[0]);
		float c = calcAngle(p[2], p[0], p[1]);
		//printf("angles %f %f %f\n", a, b, c);
		float min_angle = fminf(a, b);
		min_angle = fminf(min_angle, c);
		//float area = (e[0].cross(e[1])).norm();
		float q = 3.f * min_angle / M_PI;
		return q;
	}



	__device__ __forceinline__ float minAngleMetricTri(const Vec3f p[3], const Vec3f n) {
		Vec3f e[3];
		for (int i = 0; i < 3; ++i) {
			int j = (i + 1) % 3;
			e[i] = p[j] - p[i];
		}
		float a = calcAngle(p[0], p[1], p[2]);
		float b = calcAngle(p[1], p[2], p[0]);
		float c = calcAngle(p[2], p[0], p[1]);
		//printf("angles %f %f %f\n", a, b, c);
		float min_angle = fminf(a, b);
		min_angle = fminf(min_angle, c);
		Vec3f normalTri = (e[0].cross(e[1]));
		float ndotn = normalTri.normalized().dot(n);
		float area = normalTri.norm() * sign(ndotn);
		float q = 3.f * min_angle / M_PI;
		if (area < 0.f) {
			q *= -1.f;
		}
		return q;
	}


	__device__ __forceinline__ float qualityShapeMinAMTri2D(const int n_oneRing, const Vec3f* oneRing, const Vec3f p) {
		float q = FLT_MAX;
		if (n_oneRing <= 1) {
			printf("n_oneRing <= 1\n");
			assert(0);
		}
		for (int k = 0; k < n_oneRing - 1; ++k) {
			Vec2f v[3];
			v[0] = p;
			v[1] = oneRing[k];
			v[2] = oneRing[k + 1];
			q = fminf(q, minAngleMetricTri2D(v));
		}
		return q;
	}

	__device__ __forceinline__ float qualityShapeMinAMTri(const int n_oneRing, const Vec3f* oneRing, const Vec3f p) {
		float q = FLT_MAX;
		if (n_oneRing <= 1) {
			printf("n_oneRing <= 1\n");
			assert(0);
		}
		for (int k = 0; k < n_oneRing - 1; ++k) {
			Vec3f v[3];
			v[0] = p;
			v[1] = oneRing[k];
			v[2] = oneRing[k + 1];
			q = fminf(q, minAngleMetricTri(v));
		}
		return q;
	}

	__device__ __forceinline__ float qualityShapeMinAMTri(const int n_oneRing, const Vec3f* oneRing, const Vec3f p, const Vec3f n) {
		float q = FLT_MAX;
		if (n_oneRing <= 1) {
			printf("n_oneRing <= 1\n");
			assert(0);
		}
		for (int k = 0; k < n_oneRing - 1; ++k) {
			Vec3f v[3];
			v[0] = p;
			v[1] = oneRing[k];
			v[2] = oneRing[k + 1];
			q = fminf(q, minAngleMetricTri(v, n));
		}
		return q;
	}


	/////// Max Angle

	__device__ __forceinline__ float maxAngleMetricTri2D(const Vec2f p[3]) {
		Vec2f e[3];
		for (int i = 0; i < 3; ++i) {
			int j = (i + 1) % 3;
			e[i] = p[j] - p[i];
		}
		float a = calcAngle(p[0], p[1], p[2]);
		float b = calcAngle(p[1], p[2], p[0]);
		float c = calcAngle(p[2], p[0], p[1]);
		float max_angle = fmaxf(a, b);
		max_angle = fmaxf(max_angle, c);
		float area = e[0][0] * e[1][1] - e[0][1] * e[1][0];
		//float q = 1.f - (3.f * max_angle - M_PI) / (2.f * M_PI);
		float q = 1.f - (max_angle / M_PI);
		// if triangle is flipped, make value negative
		if (area < 0)
			return area;
		else
			return q;
	}


	__device__ __forceinline__ float maxAngleMetricTri(const Vec3f p[3]) {
		Vec3f e[3];
		for (int i = 0; i < 3; ++i) {
			int j = (i + 1) % 3;
			e[i] = p[j] - p[i];
		}
		float a = calcAngle(p[0], p[1], p[2]);
		float b = calcAngle(p[1], p[2], p[0]);
		float c = calcAngle(p[2], p[0], p[1]);
		//printf("angles %f %f %f\n", a, b, c);
		float max_angle = fmaxf(a, b);
		max_angle = fmaxf(max_angle, c);
		//float area = (e[0].cross(e[1])).norm();
		//float q = 1.f - (3.f * max_angle - M_PI) / (2.f * M_PI);
		float q = 1.f - (max_angle / M_PI);
		return q;
	}



	__device__ __forceinline__ float maxAngleMetricTri(const Vec3f p[3], const Vec3f n) {
		Vec3f e[3];
		for (int i = 0; i < 3; ++i) {
			int j = (i + 1) % 3;
			e[i] = p[j] - p[i];
		}
		float a = calcAngle(p[0], p[1], p[2]);
		float b = calcAngle(p[1], p[2], p[0]);
		float c = calcAngle(p[2], p[0], p[1]);
		//printf("angles %f %f %f\n", a, b, c);
		float max_angle = fmaxf(a, b);
		max_angle = fmaxf(max_angle, c);
		Vec3f normalTri = (e[0].cross(e[1]));
		float ndotn = normalTri.normalized().dot(n);
		float area = normalTri.norm() * sign(ndotn);
		//float q = 1.f - (3.f * max_angle - M_PI) / (2.f * M_PI);
		float q = 1.f - (max_angle / M_PI);
		if (area < 0.f) {
			q *= -1.f;
		}
		return q;
	}



	__device__ __forceinline__ float qualityShapeMaxAMTri2D(const int n_oneRing, const Vec3f* oneRing, const Vec3f p) {
		float q = FLT_MAX;
		if (n_oneRing <= 1) {
			printf("n_oneRing <= 1\n");
			assert(0);
		}
		for (int k = 0; k < n_oneRing - 1; ++k) {
			Vec2f v[3];
			v[0] = p;
			v[1] = oneRing[k];
			v[2] = oneRing[k + 1];
			q = fminf(q, maxAngleMetricTri2D(v));
		}
		return q;
	}

	__device__ __forceinline__ float qualityShapeMaxAMTri(const int n_oneRing, const Vec3f* oneRing, const Vec3f p) {
		float q = FLT_MAX;
		if (n_oneRing <= 1) {
			printf("n_oneRing <= 1\n");
			assert(0);
		}
		for (int k = 0; k < n_oneRing - 1; ++k) {
			Vec3f v[3];
			v[0] = p;
			v[1] = oneRing[k];
			v[2] = oneRing[k + 1];
			q = fminf(q, maxAngleMetricTri(v));
		}
		return q;
	}

	__device__ __forceinline__ float qualityShapeMaxAMTri(const int n_oneRing, const Vec3f* oneRing, const Vec3f p, const Vec3f n) {
		float q = FLT_MAX;
		if (n_oneRing <= 1) {
			printf("n_oneRing <= 1\n");
			assert(0);
		}
		for (int k = 0; k < n_oneRing - 1; ++k) {
			Vec3f v[3];
			v[0] = p;
			v[1] = oneRing[k];
			v[2] = oneRing[k + 1];
			q = fminf(q, maxAngleMetricTri(v, n));
		}
		return q;
	}


	///////// Mean Ratio


	__device__ __forceinline__ float meanRatioMetricTri2D(const Vec2f p[3]) {
		Vec2f e[3];
		float e_length_squared[3];
		for (int i = 0; i < 3; ++i) {
			int j = (i + 1) % 3;
			e[i] = p[j] - p[i];
			e_length_squared[i] = e[i].x * e[i].x + e[i].y * e[i].y;
		}
		float l = e_length_squared[0] + e_length_squared[1] + e_length_squared[2];
		float area = e[0].x * e[1].y - e[0].y * e[1].x;
		if (area < 0.f) {
			return area;
		}
		return 2.f * sqrtf(3.f) * area / l;
	}


	__device__ __forceinline__ float meanRatioMetricTri(const Vec3f p[3]) {
		Vec3f e[3];
		float e_length_squared[3];
		e[0] = p[1] - p[0];
		e[1] = p[2] - p[0];
		e[2] = p[2] - p[1];
		e_length_squared[0] = e[0].dot(e[0]);
		e_length_squared[1] = e[1].dot(e[1]);
		e_length_squared[2] = e[2].dot(e[2]);
		float l = e_length_squared[0] + e_length_squared[1] + e_length_squared[2];
		float area = (e[0].cross(e[1])).norm();
		return 2.f * sqrt(3.f) * area / l;
	}



	__device__ __forceinline__ float meanRatioMetricTri(const Vec3f p[3], const Vec3f n) {
		Vec3f e[3];
		float e_length_squared[3];
		e[0] = p[1] - p[0];
		e[1] = p[2] - p[0];
		e[2] = p[2] - p[1];
		e_length_squared[0] = e[0].dot(e[0]);
		e_length_squared[1] = e[1].dot(e[1]);
		e_length_squared[2] = e[2].dot(e[2]);
		float l = e_length_squared[0] + e_length_squared[1] + e_length_squared[2];
		Vec3f normalTri = (e[0].cross(e[1]));
		float ndotn = normalTri.normalized().dot(n);
		//if (ndotn < 0.7f) {
		//	return -1.f;
		//}
		float area = normalTri.norm() * sign(ndotn);
		return 2.f * sqrt(3.f) * area / l;
	}



	__device__ __forceinline__ float qualityShapeMRMTri2D(const int n_oneRing, const Vec3f* oneRing, const Vec3f p) {
		float q = FLT_MAX;
		if (n_oneRing <= 1) {
			printf("n_oneRing <= 1\n");
			assert(0);
		}
		for (int k = 0; k < n_oneRing - 1; ++k) {
			Vec2f v[3];
			v[0] = p;
			v[1] = oneRing[k];
			v[2] = oneRing[k + 1];
			q = fminf(q, meanRatioMetricTri2D(v));
		}
		return q;
	}

	__device__ __forceinline__ float qualityShapeMRMTri(const int n_oneRing, const Vec3f* oneRing, const Vec3f p) {
		float q = FLT_MAX;
		if (n_oneRing <= 1) {
			printf("n_oneRing <= 1\n");
			assert(0);
		}
		for (int k = 0; k < n_oneRing - 1; ++k) {
			Vec3f v[3];
			v[0] = p;
			v[1] = oneRing[k];
			v[2] = oneRing[k + 1];
			q = fminf(q, meanRatioMetricTri(v));
		}
		return q;
	}

	__device__ __forceinline__ float qualityShapeMRMTri(const int n_oneRing, const Vec3f* oneRing, const Vec3f p, const Vec3f n) {
		float q = FLT_MAX;
		if (n_oneRing <= 1) {
			printf("n_oneRing <= 1\n");
			assert(0);
		}
		for (int k = 0; k < n_oneRing - 1; ++k) {
			Vec3f v[3];
			v[0] = p;
			v[1] = oneRing[k];
			v[2] = oneRing[k + 1];
			q = fminf(q, meanRatioMetricTri(v, n));
		}
		return q;
	}



	__device__ __forceinline__ float qualityTri2DRing(const int n_oneRing, const Vec3f* oneRing, const Vec3f p, QualityCriterium q_crit) {
		if (q_crit == QualityCriterium::MEAN_RATIO) {
			return qualityShapeMRMTri2D(n_oneRing, oneRing, p);
		}
		else if (q_crit == QualityCriterium::MIN_ANGLE) {
			return qualityShapeMinAMTri2D(n_oneRing, oneRing, p);
		}
		else if (q_crit == QualityCriterium::MAX_ANGLE) {
			return qualityShapeMaxAMTri2D(n_oneRing, oneRing, p);
		}
		else {
			return 1.f;
		}
	}

	__device__ __forceinline__ float qualityTriRing(const int n_oneRing, const Vec3f* oneRing, const Vec3f p, QualityCriterium q_crit) {
		if (q_crit == QualityCriterium::MEAN_RATIO) {
			return qualityShapeMRMTri(n_oneRing, oneRing, p);
		}
		else if (q_crit == QualityCriterium::MIN_ANGLE) {
			return qualityShapeMinAMTri(n_oneRing, oneRing, p);
		}
		else if (q_crit == QualityCriterium::MAX_ANGLE) {
			return qualityShapeMaxAMTri(n_oneRing, oneRing, p);
		}
		else {
			return 1.f;
		}
	}

	__device__ __forceinline__ float qualityTriRing(const int n_oneRing, const Vec3f* oneRing, const Vec3f p, const Vec3f n, QualityCriterium q_crit) {
		if (q_crit == QualityCriterium::MEAN_RATIO) {
			return qualityShapeMRMTri(n_oneRing, oneRing, p, n);
		}
		else if (q_crit == QualityCriterium::MIN_ANGLE) {
			return qualityShapeMinAMTri(n_oneRing, oneRing, p, n);
		}
		else if (q_crit == QualityCriterium::MAX_ANGLE) {
			return qualityShapeMaxAMTri(n_oneRing, oneRing, p, n);
		}
		else {
			return 1.f;
		}
	}




	__device__ __forceinline__ float qualityTri2D(const Vec2f p[3], QualityCriterium q_crit) {
		if (q_crit == QualityCriterium::MEAN_RATIO) {
			return meanRatioMetricTri2D(p);
		}
		else if (q_crit == QualityCriterium::MIN_ANGLE) {
			return minAngleMetricTri2D(p);
		}
		else if (q_crit == QualityCriterium::MAX_ANGLE) {
			return maxAngleMetricTri2D(p);
		}
		else {
			return 1.f;
		}
	}

	__device__ __forceinline__ float qualityTri(const Vec3f p[3], QualityCriterium q_crit) {
		if (q_crit == QualityCriterium::MEAN_RATIO) {
			return meanRatioMetricTri(p);
		}
		else if (q_crit == QualityCriterium::MIN_ANGLE) {
			return minAngleMetricTri(p);
		}
		else if (q_crit == QualityCriterium::MAX_ANGLE) {
			return maxAngleMetricTri(p);
		}
		else {
			return 1.f;
		}
	}



}


#pragma once

#include "DMOQualityUtil.h"

namespace DMO {

	///////// Mean Ratio



	__device__ __forceinline__ float meanRatioMetricHex(const Vec3f p[8]) {
		float volume = volumeHex3(p);
		//printf("volume %.10g\n", volume);
		if (volume <= 0.f) {
			return volume;
		}
		//return volume;

		// TODO
		Vec3f e[12];
		e[0] = p[1] - p[0];
		e[1] = p[2] - p[1];
		e[2] = p[3] - p[2];
		e[3] = p[0] - p[3];
		e[4] = p[4] - p[0];
		e[5] = p[5] - p[1];
		e[6] = p[6] - p[2];
		e[7] = p[7] - p[3];
		e[8] = p[5] - p[4];
		e[9] = p[6] - p[5];
		e[10] = p[7] - p[6];
		e[11] = p[4] - p[7];
		float e_length_squared[12];

		float mu = cbrtf(volume * volume);

		float l_sum = 0.f;
		float maxl = 0.f;
		float minl = 1e90;
		for (int i = 0; i < 12; ++i) {
			e_length_squared[i] = e[i].dot(e[i]);
			l_sum += e[i].dot(e[i]);
			maxl = fmaxf(maxl, e_length_squared[i]);
			minl = fminf(minl, e_length_squared[i]);
		}

		return sqrtf(minl)/sqrtf(maxl);

		//const float cbrtf9 = 2.0800838230519f;
		//return 12.f * cbrtf9 * mu / l_sum;
		//return mu / l_sum;
	}

	__device__ __forceinline__ float shapeMetricHex(const Vec3f p[8]) {
		Vec3f L[12];
		Vec3f X[3];
		hex_edges(p[4], p[5], p[6], p[7], p[0], p[1], p[2], p[3], L, false);
		hex_principal_axes(p[4], p[5], p[6], p[7], p[0], p[1], p[2], p[3], X, false);

		float retval = 1e30;
		for (int i = 0; i < 9; ++i)
		{
			Vec3f tet[3];
			hex_subtets(L, X, i, tet);
			float sj = determinant(tet[0], tet[1], tet[2]);
			float Aj = jacobianNormSquared(tet[0], tet[1], tet[2]);
			if (sj < EPSILON || Aj < EPSILON) return 0.f;
			retval = fminf(retval, cbrtf(sj * sj) / Aj);
		}
		return 3.f*retval;
	}



	__device__ __forceinline__ float qualityShapeMRMHex(const int n_oneRing, const Vec3f* oneRing, const Vec3f p) {
		float q = FLT_MAX;
		if (n_oneRing <= 0) {
			printf("n_oneRing <= 0\n");
			assert(0);
		}
		for (int k = 0; k < n_oneRing; ++k) {
			Vec3f v[8];
			v[0] = p;
			v[1] = oneRing[7 * k + 0];
			v[2] = oneRing[7 * k + 1];
			v[3] = oneRing[7 * k + 2];
			v[4] = oneRing[7 * k + 3];
			v[5] = oneRing[7 * k + 4];
			v[6] = oneRing[7 * k + 5];
			v[7] = oneRing[7 * k + 6];
			float currq = shapeMetricHex(v);// meanRatioMetricHex(v);
			q = fminf(q, currq);
		}
		return q;
	}



	__device__ __forceinline__ float qualityHexRing(const int n_oneRing, const Vec3f* oneRing, const Vec3f p, QualityCriterium q_crit) {
		if (q_crit == QualityCriterium::MEAN_RATIO) {
			return qualityShapeMRMHex(n_oneRing, oneRing, p);
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







	__device__ __forceinline__ float qualityHex(const Vec3f p[8], QualityCriterium q_crit) {
		if (q_crit == QualityCriterium::MEAN_RATIO) {
			return shapeMetricHex(p);//meanRatioMetricHex(p);
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


#pragma once

#include "cuda_runtime.h"


__device__ inline static float myAtomicMin(float* address, float val)
{
	int* address_as_i = (int*)address;
	int old = *address_as_i, assumed;
	do {
		assumed = old;
		old = atomicCAS(address_as_i, assumed,
			__float_as_int(fminf(val, __int_as_float(assumed))));
	} while (assumed != old);
	return __int_as_float(old);
}

__device__ inline float myAtomicMax(float* address, float val)
{
	int* address_as_i = (int*)address;
	int old = *address_as_i, assumed;
	do {
		assumed = old;
		old = atomicCAS(address_as_i, assumed,
			__float_as_int(fmaxf(val, __int_as_float(assumed))));
	} while (assumed != old);
	return __int_as_float(old);
}

#pragma once

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/device_allocator.h>

#include <vector>
#include <iostream>
#include "Vec3.h"
#include "Vec2.h"

using thrust::device_vector;
using thrust::host_vector;
using thrust::raw_pointer_cast;
using std::vector;
using std::cout;
using std::endl;

//// https://github.com/thrust/thrust/blob/master/examples/uninitialized_vector.cu
//template<typename T>
//struct uninitialized_allocator
//    : thrust::device_malloc_allocator<T>
//{
//    // note that construct is annotated as
//    // a __host__ __device__ function
//    __host__ __device__
//        void construct(T* p)
//    {
//        // no-op
//    }
//};
//
//typedef thrust::device_vector<float, uninitialized_allocator<float>> uninitialized_vector;

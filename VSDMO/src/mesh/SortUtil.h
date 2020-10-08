#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math_constants.h"
#include <thrust/count.h>
#include <thrust/for_each.h>
#include <thrust/device_vector.h>
#include <algorithm>
#include "CudaUtil.h"
#include "mesh/MeshTypes.h"
#include <iostream>
#include "ConfigUsing.h"

// creates a "rowPtr" array in offsets
__global__ void k_find_first_n(int N, int* arr, int* offsets, int lastIndexOffsets);

__global__ void k_map_each(int nV, int nE, int* arr, ArrayView<int> sortMap);
__global__ void k_map_each(int nV, int nE, Triangle* arr, ArrayView<int> sortMap);
__global__ void k_map_each(int nV, int nE, Tetrahedron* arr, ArrayView<int> sortMap);
__global__ void k_map_each(int nV, int nE, Quad* arr, ArrayView<int> sortMap);
__global__ void k_map_each(int nV, int nE, Hexahedron* arr, ArrayView<int> sortMap);

__global__ void k_populateSortMapInverse(int nV, ArrayView<int> sortMap, ArrayView<int> sortMapInverse);

__global__ void k_fill_exists(ArrayView<int> vec, ArrayView<int> exists);
__global__ void k_map_each(ArrayView<int> vec, ArrayView<int> mapping);

template<class T>
__host__ void remapElementIndices(int n, T* elems, device_vector<int>& sortMapInverse) {
    const int BLOCK_SIZE = 128;
    k_map_each << <getBlockCount(n, BLOCK_SIZE), BLOCK_SIZE >> > ((int)sortMapInverse.size(), n, elems, sortMapInverse);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
}

__host__ void reindex(device_vector<int>& vec);

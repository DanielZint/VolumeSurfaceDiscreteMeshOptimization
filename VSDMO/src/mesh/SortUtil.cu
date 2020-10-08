#include "SortUtil.h"

__global__ void k_find_first_n(int N, int* arr, int* offsets, int lastIndexOffsets) {
    for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < N; idx += blockDim.x * gridDim.x) {
        if (idx == 0) {
            offsets[0] = 0;
            //offsets[lastIndexOffsets] = N;
            if (N == 1) {
                for (int i = 0; i <= arr[0]; ++i) {
                    offsets[i] = 0;
                }
                offsets[lastIndexOffsets] = 1;
                return;
            }
            int endZeros = arr[1];
            if (lastIndexOffsets == 1) {
                endZeros = 1;
            }
            for (int i = 0; i < endZeros; ++i) {
                offsets[i] = 0;
            }
            for (int i = arr[N-1]+1; i <= lastIndexOffsets; ++i) {
                offsets[i] = N;
            }
        }
        else {
            if (arr[idx - 1] != arr[idx]) {
                for (int i = arr[idx - 1] + 1; i <= arr[idx]; ++i) {
                    offsets[i] = idx;
                }
                //offsets[arr[idx]] = idx;
            }
        }
    }
}

//__global__ void k_find_first_n(int N, int* arr, int* offsets, int lastIndexOffsets) {
//    for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < N; idx += blockDim.x * gridDim.x) {
//        if (idx == 0) {
//            offsets[0] = 0;
//            offsets[lastIndexOffsets] = N;
//        }
//        else {
//            if (arr[idx - 1] != arr[idx]) {
//                offsets[arr[idx]] = idx;
//            }
//        }
//    }
//}

__global__ void k_map_each(int nV, int nE, int* arr, ArrayView<int> sortMap) {
    for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < nE; idx += blockDim.x * gridDim.x) {
        if (arr[idx] < nV) {
            arr[idx] = sortMap[arr[idx]];
        }
    }
}

__global__ void k_map_each(int nV, int nE, Triangle* arr, ArrayView<int> sortMap) {
    for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < nE; idx += blockDim.x * gridDim.x) {
        Triangle& t = arr[idx];
        for (int i = 0; i < 3; ++i) {
            if (t.data[i] < nV) {
                t.data[i] = sortMap[t.data[i]];
            }
        }
    }
}

__global__ void k_map_each(int nV, int nE, Tetrahedron* arr, ArrayView<int> sortMap) {
    for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < nE; idx += blockDim.x * gridDim.x) {
        Tetrahedron& t = arr[idx];
        for (int i = 0; i < 4; ++i) {
            if (t.data[i] < nV) {
                t.data[i] = sortMap[t.data[i]];
            }
        }
    }
}

__global__ void k_map_each(int nV, int nE, Quad* arr, ArrayView<int> sortMap) {
    for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < nE; idx += blockDim.x * gridDim.x) {
        Quad& t = arr[idx];
        for (int i = 0; i < 4; ++i) {
            if (t.data[i] < nV) {
                t.data[i] = sortMap[t.data[i]];
            }
        }
    }
}

__global__ void k_map_each(int nV, int nE, Hexahedron* arr, ArrayView<int> sortMap) {
    for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < nE; idx += blockDim.x * gridDim.x) {
        Hexahedron& t = arr[idx];
        for (int i = 0; i < 8; ++i) {
            if (t.data[i] < nV) {
                t.data[i] = sortMap[t.data[i]];
            }
        }
    }
}

__global__ void k_populateSortMapInverse(int nV, ArrayView<int> sortMap, ArrayView<int> sortMapInverse) {
    for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < nV; idx += blockDim.x * gridDim.x) {
        sortMapInverse[sortMap[idx]] = idx;
    }
}

__global__ void k_fill_exists(ArrayView<int> vec, ArrayView<int> exists) {
    for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < vec.size(); idx += blockDim.x * gridDim.x) {
        exists[vec[idx]] = 1;
    }
}

__global__ void k_map_each(ArrayView<int> vec, ArrayView<int> mapping) {
    for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < vec.size(); idx += blockDim.x * gridDim.x) {
        vec[idx] = mapping[vec[idx]];
    }
}

__host__ void reindex(device_vector<int>& vec) {
    // vec is sorted, eg. 0002446. turns to 0001223
    int largest_val;
    thrust::copy(vec.begin() + vec.size() - 1, vec.end(), &largest_val);
    device_vector<int> exists(largest_val + 1, 0);
    const int BLOCK_SIZE = 128;
    k_fill_exists << <getBlockCount(vec.size(), BLOCK_SIZE), BLOCK_SIZE >> > (vec, exists);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    host_vector<int> h_exists(exists);
    host_vector<int> h_mapping(exists.size());
    int curr = 0;
    for (int i = 0; i < exists.size(); ++i) {
        if (exists[i]) {
            h_mapping[i] = curr++;
        }
    }
    device_vector<int> d_mapping(h_mapping);

    k_map_each << <getBlockCount(vec.size(), BLOCK_SIZE), BLOCK_SIZE >> > (vec, d_mapping);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
}




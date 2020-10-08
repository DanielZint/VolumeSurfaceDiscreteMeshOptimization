#include "VertexColoring.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math_constants.h"
//#include <curand.h>
#include <thrust/count.h>
#include <thrust/for_each.h>
#include <thrust/device_vector.h>
#include <algorithm>
#include "CudaUtil.h"
//#include "mesh/MeshTypes.h"
#include <iostream>

#include "cusparse.h"
#include "cuda.h"

#define CUDA_MAX_BLOCKS (1 << 30)

// old, no offset
// https://devblogs.nvidia.com/graph-coloring-more-parallelism-for-incomplete-lu-factorization/
//n: number of vertices
//Av: values
//Ao: row/col_ptr
//Ac: indices
//colors: output color per vertex
__global__ void k_color_jpl_kernel(int nVertices, int c, int* rowPtr, int* colInd, ArrayView<int> randoms, ArrayView<int> colors) {
    for (int i = threadIdx.x + blockIdx.x * blockDim.x;
        i < nVertices;
        i += blockDim.x * gridDim.x)
    {
        bool f = true; // true iff you have max random

        // ignore nodes colored earlier
        if ((colors[i] != -1)) continue;

        int ir = randoms[i];

        // look at neighbors to check their random number
        for (int k = rowPtr[i]; k < rowPtr[i + 1]; k++) {
            // ignore nodes colored earlier (and yourself)
            int j = colInd[k];
            int jc = colors[j];
            if (((jc != -1) && (jc != c)) || (i == j)) continue;
            int jr = randoms[j];
            if (ir <= jr) f = false;
        }

        // assign color if you have the maximum random number
        if (f) colors[i] = c;
    }
}

void color_jpl(int nVertices, int* rowPtr, int* colInd, device_vector<int>& colors) {
    //int* randoms; // allocate and init random array
    vector<int> randomsHost(nVertices);
    srand(1);
    for (int i = 0; i < nVertices; ++i) {
        randomsHost[i] = i;
    }
    device_vector<int> randoms(randomsHost);
    thrust::fill(colors.begin(), colors.end(), -1); // init colors to -1

    for (int c = 0; c < nVertices; c++) {
        int nt = 256;
        k_color_jpl_kernel << <getBlockCount(nVertices, nt), nt >> > (nVertices, c, rowPtr, colInd, randoms, colors);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
        int left = (int)thrust::count(colors.begin(), colors.end(), -1);
        if (left == 0) break;
    }
}



__global__ void k_color_jpl_kernel(int start, int nVertices, int c, int* rowPtr, int* colInd, ArrayView<int> randoms, ArrayView<int> colors) {
    for (int idx = threadIdx.x + blockIdx.x * blockDim.x;
        idx < nVertices;
        idx += blockDim.x * gridDim.x)
    {
        int i = idx + start;
        bool f = true; // true iff you have max random

        // ignore nodes colored earlier
        if ((colors[i - start] != -1)) continue;

        int ir = randoms[i - start];

        // look at neighbors to check their random number
        for (int k = rowPtr[i]; k < rowPtr[i + 1]; k++) {
            // ignore nodes colored earlier (and yourself)
            int j = colInd[k];
            if (j < start || j >= (nVertices + start)) {
                continue;
            }
            int jc = colors[j - start];
            if (((jc != -1) && (jc != c)) || (i == j)) continue;
            int jr = randoms[j - start];
            if (ir <= jr) f = false;
        }

        // assign color if you have the maximum random number
        if (f) colors[i - start] = c;
    }
}

void color_jpl(int start, int n, int* rowPtr, int* colInd, device_vector<int>& colors) {
    //int* randoms; // allocate and init random array
    vector<int> randomsHost(n);
    srand(1);
    for (int i = 0; i < n; ++i) {
        randomsHost[i] = n-i;// rand();
    }
    device_vector<int> randoms(randomsHost);
    thrust::fill(colors.begin(), colors.end(), -1); // init colors to -1

    for (int c = 0; c < n; c++) {
        int nt = 256;
        k_color_jpl_kernel << <getBlockCount(n, nt), nt >> > (start, n, c, rowPtr, colInd, randoms, colors);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
        int left = (int)thrust::count(colors.begin(), colors.end(), -1);
        if (left == 0) break;
    }
}


void color_cuSPARSE(int nVertices, int* rowPtr, int* colInd, device_vector<int>& colors, int nnz) {
    cusparseStatus_t status;
    cusparseHandle_t handle;
    status = cusparseCreate(&handle);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        printf("error!");
        exit(1);
    }
    cusparseMatDescr_t descr;
    status = cusparseCreateMatDescr(&descr);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        printf("error!");
        exit(1);
    }
    cusparseColorInfo_t info;
    status = cusparseCreateColorInfo(&info);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        printf("error!");
        exit(1);
    }

    int ncolors = 0;
    float fraction = 1.0;
    status = cusparseScsrcolor(handle, nVertices, nnz, descr, NULL, rowPtr, colInd, &fraction, &ncolors, raw(colors), NULL, info);
    status = cusparseDestroy(handle);
}


//
//template<class T>
//int sortVerticesByColor(int nV, device_vector<int>& colors, device_vector<Vec3f>& vertexPoints, device_vector<bool>& vertexIsBoundary1D,
//    device_vector<bool>& vertexIsFeature, device_vector<Vec3f>& vertexNormals, device_vector<int>& col_offsets, device_vector<T>& elems)
//{
//    // sort colors ascending
//    device_vector<int> sortMap(nV);
//    thrust::sequence(sortMap.begin(), sortMap.end());
//    thrust::sort_by_key(colors.begin(), colors.end(), sortMap.begin());
//    // get number of colors by looking at last color value
//    int numColors;
//    thrust::copy(colors.begin() + colors.size() - 1, colors.end(), &numColors);
//    numColors++;
//    // start index of each new color
//    col_offsets = device_vector<int>(numColors + 1);
//    const int BLOCK_SIZE = 128;
//    k_find_first_n << <getBlockCount((int)colors.size(), BLOCK_SIZE), BLOCK_SIZE >> > (colors.size(), colors, col_offsets, numColors);
//    gpuErrchk(cudaPeekAtLastError());
//    gpuErrchk(cudaDeviceSynchronize());
//
//    // sort vertex data the same way we sorted their colors
//    device_vector<Vec3f> vertexPointsSorted(nV);
//    thrust::gather(sortMap.begin(), sortMap.end(), vertexPoints.begin(), vertexPointsSorted.begin());
//    thrust::copy(vertexPointsSorted.begin(), vertexPointsSorted.end(), vertexPoints.begin());
//
//    if (vertexIsBoundary1D.size() > 0) {
//        device_vector<bool> vertexIsBoundary1DSorted(nV);
//        thrust::gather(sortMap.begin(), sortMap.end(), vertexIsBoundary1D.begin(), vertexIsBoundary1DSorted.begin());
//        thrust::copy(vertexIsBoundary1DSorted.begin(), vertexIsBoundary1DSorted.end(), vertexIsBoundary1D.begin());
//    }
//    if (vertexIsFeature.size() > 0) {
//        device_vector<bool> vertexIsFeatureSorted(nV);
//        thrust::gather(sortMap.begin(), sortMap.end(), vertexIsFeature.begin(), vertexIsFeatureSorted.begin());
//        thrust::copy(vertexIsFeatureSorted.begin(), vertexIsFeatureSorted.end(), vertexIsFeature.begin());
//    }
//    if (vertexNormals.size() > 0) {
//        device_vector<Vec3f> vertexNormalsSorted(nV);
//        thrust::gather(sortMap.begin(), sortMap.end(), vertexNormals.begin(), vertexNormalsSorted.begin());
//        thrust::copy(vertexNormalsSorted.begin(), vertexNormalsSorted.end(), vertexNormals.begin());
//    }
//
//    // remap tetrahedra indices, since vertices were shuffled
//
//    device_vector<int> sortMapInverse(nV);
//    k_populateSortMapInverse << <getBlockCount(nV, BLOCK_SIZE), BLOCK_SIZE >> > (nV, sortMap, sortMapInverse);
//    gpuErrchk(cudaPeekAtLastError());
//    gpuErrchk(cudaDeviceSynchronize());
//
//    // TODO elems.size ... identity map non surface vertices
//    k_map_each << <getBlockCount((int)elems.size() * sizeof(T)/4, BLOCK_SIZE), BLOCK_SIZE >> > (nV, (int)elems.size() * sizeof(T)/4, (int *)raw_pointer_cast(elems.data()), sortMapInverse);
//    gpuErrchk(cudaPeekAtLastError());
//    gpuErrchk(cudaDeviceSynchronize());
//    return numColors;
//}

//template int sortVerticesByColor<Triangle>(int nV, device_vector<int>& colors, device_vector<Vec3f>& vertexPoints, device_vector<bool>& vertexIsBoundary1D,
//    device_vector<bool>& vertexIsFeature, device_vector<Vec3f>& vertexNormals, device_vector<int>& col_offsets, device_vector<Triangle>& elems);
//template int sortVerticesByColor<Tetrahedron>(int nV, device_vector<int>& colors, device_vector<Vec3f>& vertexPoints, device_vector<bool>& vertexIsBoundary1D,
//    device_vector<bool>& vertexIsFeature, device_vector<Vec3f>& vertexNormals, device_vector<int>& col_offsets, device_vector<Tetrahedron>& elems);
//


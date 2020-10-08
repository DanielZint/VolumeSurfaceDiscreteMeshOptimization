#include "CopyUtil.h"
#include "CudaUtil.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math_constants.h"
#include "MeshTypes.h"


__global__ void k_copyHexahedraToQuads(Hexahedron* srcData, Quad* dstPtr, int nHexahedra) {
    for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < nHexahedra; idx += blockDim.x * gridDim.x) {
        Hexahedron hex = srcData[idx];
        dstPtr[6 * idx + 0] = { hex.v0, hex.v1, hex.v2, hex.v3 };
        dstPtr[6 * idx + 1] = { hex.v1, hex.v5, hex.v6, hex.v2 };
        dstPtr[6 * idx + 2] = { hex.v3, hex.v2, hex.v6, hex.v7 };
        dstPtr[6 * idx + 3] = { hex.v4, hex.v0, hex.v3, hex.v7 };
        dstPtr[6 * idx + 4] = { hex.v1, hex.v0, hex.v4, hex.v5 };
        dstPtr[6 * idx + 5] = { hex.v5, hex.v4, hex.v7, hex.v6 };
    }
}

void copyHexahedraToQuads(void* srcData, void* dstPtr, int sizeBytes) {
    //nHexahedra * 8 * sizeof(int)
    int nHexahedra = sizeBytes / (8 * sizeof(int));
    const int BLOCK_SIZE = 128;
    k_copyHexahedraToQuads << <getBlockCount(nHexahedra, BLOCK_SIZE), BLOCK_SIZE >> > ((Hexahedron*)srcData, (Quad*)dstPtr, nHexahedra);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
}

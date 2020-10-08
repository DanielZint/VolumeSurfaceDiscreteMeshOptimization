#include "Laplace.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "CudaUtil.h"
#include "CudaAtomic.h"

__global__ void k_laplaceSmoothing(int cStart, int cEnd, MeshTriDevice* mesh) {
	if (blockIdx.x >= cEnd - cStart) {
		return;
	}

	const int vid = cStart + blockIdx.x;

	const int heStart = mesh->halfedges.rowPtr_[vid];
	const int nHalfedges = mesh->halfedges.rowPtr_[vid + 1] - heStart;

	Vec3f newPos = mesh->vertexPoints[vid];

	for (int i = 0; i < nHalfedges; ++i) {
		const Halfedge& halfedge = mesh->halfedges.values_[heStart + i];
		const int vdst = halfedge.targetVertex;
		const Vec3f p = mesh->vertexPoints[vdst];
		newPos += p;
	}

	newPos = newPos * (1.f / (1 + nHalfedges));

	mesh->vertexPoints[vid] = newPos;
}

void laplaceSmoothing(DMOMeshTri& dmo_mesh, int n_iter) {
	for (int i = 0; i < n_iter; ++i) {
		for (int cid = 0; cid < dmo_mesh.nColorsFree; ++cid) {
			k_laplaceSmoothing << <dmo_mesh.colorOffsetsFree[cid + 1] - dmo_mesh.colorOffsetsFree[cid], 1 >> >
				(dmo_mesh.colorOffsetsFree[cid], dmo_mesh.colorOffsetsFree[cid + 1], dmo_mesh.d_mesh);
			gpuErrchk(cudaPeekAtLastError());
			gpuErrchk(cudaDeviceSynchronize());
		}
	}
}

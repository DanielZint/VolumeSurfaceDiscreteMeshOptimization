#include "MeshFunctions.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "CudaUtil.h"
#include "CudaAtomic.h"

__global__ void k_findClosestVertex(ArrayView<float> dists, Vec3f* points, Vec3f pos) {
	for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < dists.size(); idx += blockDim.x * gridDim.x) {
		Vec3f diff = points[idx] - pos;
		dists[idx] = diff.dot(diff);
	}
}

int findClosestVertex(DMOMeshBase& mesh, Vec3f pos) {
	device_vector<float> dists(mesh.nVerticesSurf);

	const int BLOCK_SIZE = 128;
	k_findClosestVertex << <getBlockCount(mesh.nVerticesSurf, BLOCK_SIZE), BLOCK_SIZE >> > (dists, mesh.getVertexPoints(), pos);

	auto it = thrust::min_element(dists.begin(), dists.end());
	return (int)(it - dists.begin());
}

// from https://github.com/thrust/thrust/blob/master/examples/bounding_box.cu
struct bbox_reduction : public thrust::binary_function<AABB, AABB, AABB>
{
	__host__ __device__
		AABB operator()(AABB a, AABB b)
	{
		Vec3f ll(thrust::min(a.minPos.x, b.minPos.x), thrust::min(a.minPos.y, b.minPos.y), thrust::min(a.minPos.z, b.minPos.z));
		Vec3f ur(thrust::max(a.maxPos.x, b.maxPos.x), thrust::max(a.maxPos.y, b.maxPos.y), thrust::max(a.maxPos.z, b.maxPos.z));
		return AABB{ ll, ur };
	}
};

AABB findAABB(DMOMeshBase& mesh) {
	bbox_reduction binary_op;
	AABB init{ Vec3f(FLT_MAX, FLT_MAX, FLT_MAX), Vec3f(-FLT_MAX, -FLT_MAX, -FLT_MAX) };
	AABB result = thrust::reduce(device_ptr<Vec3f>(mesh.getVertexPoints()), device_ptr<Vec3f>(mesh.getVertexPoints() + mesh.nVertices), init, binary_op);
	return result;
}

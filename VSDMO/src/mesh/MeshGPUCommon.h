#pragma once

#include "cuda_runtime.h"
#include <vector>
#include <array>
#include <thrust/device_vector.h>
#include "MeshTypes.h"
#include "CudaUtil.h"
#include "ConfigUsing.h"

//extern __constant__ int c_MAX_HE_PER_VERTEX;

struct triangleSort {
	__host__ __device__ bool operator()(Triangle& t1, Triangle& t2) {
		if (t1.v0 < t2.v0) {
			return true;
		}
		else if (t1.v0 > t2.v0) {
			return false;
		}
		else {
			if (t1.v1 < t2.v1) {
				return true;
			}
			else if (t1.v1 > t2.v1) {
				return false;
			}
			else {
				return t1.v2 < t2.v2;
			}
		}
	}
};

struct quadSort {
	__host__ __device__ bool operator()(Quad& t1, Quad& t2) {
		if (t1.v0 < t2.v0) {
			return true;
		}
		else if (t1.v0 > t2.v0) {
			return false;
		}
		else {
			if (t1.v1 < t2.v1) {
				return true;
			}
			else if (t1.v1 > t2.v1) {
				return false;
			}
			else {
				if (t1.v2 < t2.v2) {
					return true;
				}
				else if (t1.v2 > t2.v2) {
					return false;
				}
				else {
					return t1.v3 < t2.v3;
				}
			}
		}
	}
};

struct isTriangleMarked {
	__host__ __device__ bool operator()(Triangle t) {
		return t.v0 == -1;
	}
};

struct isTriangleUnmarked {
	__host__ __device__ bool operator()(Triangle t) {
		return t.v0 != -1;
	}
};

struct isQuadMarked {
	__host__ __device__ bool operator()(Quad t) {
		return t.v0 == -1;
	}
};

struct isQuadUnmarked {
	__host__ __device__ bool operator()(Quad t) {
		return t.v0 != -1;
	}
};

struct isEmptyHalfedgeEntry {
	__host__ __device__ bool operator()(Halfedge he) {
		return he.targetVertex == -1;
	}
};

struct halfedgeSort {
	__host__ __device__
		bool operator()(const Halfedge& h1, const Halfedge& h2)
	{
		return h1.targetVertex < h2.targetVertex;
	}
};

struct isEmptyVertexTetrahedraEntry {
	__host__ __device__ bool operator()(Triangle t) {
		return t.v0 == -1;
	}
};


struct isEmptySimpleHalfedgeEntry {
	__host__ __device__ bool operator()(int he) {
		return he == -1;
	}
};

struct isNegative {
	__host__ __device__ bool operator()(int i) {
		return i < 0;
	}
};

struct simpleHalfedgeSort {
	__host__ __device__ bool operator()(int h1, int h2) {
		return h1 < h2;
	}
};

struct TupleCompare {
	__host__ __device__ bool operator()(const thrust::tuple<int, int>& t1, const thrust::tuple<int, int>& t2)
	{
		if (t1.get<0>() < t2.get<0>())
			return true;
		if (t1.get<0>() > t2.get<0>())
			return false;
		return t1.get<1>() < t2.get<1>();
	}
};


__global__ void k_reorderVertices(int nT, Triangle* triangles);
__global__ void k_reorderVertices(int nT, Quad* quads);

__global__ void k_markDuplicateTris(int nT, Triangle* triangles);
__global__ void k_markDuplicateQuads(int nT, Quad* quads);

__global__ void k_fillHelperArray(int n, ArrayView<int> csrHalfedgeRowPtr, ArrayView<int> indexToCol);


__global__ void k_setOppositeHalfedgeHelper(int nHalfedges, ArrayView<Halfedge> csrHalfedgesValues, ArrayView<int> csrHalfedgeRowPtr, ArrayView<int> indexToCol);

__global__ void k_setNextHalfedgeHelper(int nHalfedges, ArrayView<Halfedge> csrHalfedgesValues, ArrayView<int> csrHalfedgeRowPtr, ArrayView<int> indexToCol, int nVerticesSurf);

__global__ void k_sortCCW(int nVerticesSurf, ArrayView<int> csrHalfedgeRowPtr, ArrayView<Halfedge> halfedgesDev, ArrayView<Halfedge> csrHalfedgesValuesOut,
	bool* vertexIsBoundary1D, Triangle* triangles);

__global__ void k_sortCCWDummyOpposite(int nVerticesSurf, ArrayView<int> csrHalfedgeRowPtr, ArrayView<Halfedge> halfedgesDev, ArrayView<Halfedge> csrHalfedgesValuesOut,
	bool* vertexIsBoundary1D, bool* vertexIsFeature, Triangle* triangles);

__global__ void k_sortCCWDummyOpposite(int nVerticesSurf, ArrayView<int> csrHalfedgeRowPtr, ArrayView<Halfedge> halfedgesDev, ArrayView<Halfedge> csrHalfedgesValuesOut,
	bool* vertexIsBoundary1D, bool* vertexIsFeature, Quad* quads);

__global__ void k_calcFaceNormals(int nT, Vec3f* vertexPoints, Vec3f* faceNormals, Triangle* triangles);
__global__ void k_calcFaceNormals(int nFaces, Vec3f* vertexPoints, Vec3f* faceNormals, Quad* quads);

__global__ void k_calcVertexNormals(int nV, Vec3f* vertexNormals, Halfedge* csrHalfedgesValues,
	int* csrHalfedgeRowPtr, Vec3f* faceNormals);

__global__ void k_initFeatures(int nV, ArrayView<bool> vertexIsFeature, ArrayView<int> vertexNumHalfedges,
	ArrayView<Halfedge> halfedges, Vec3f* faceNormals, const int MAX_HE_PER_VERTEX, float maxAngle);

__global__ void k_initFeaturesRowPtr(int nV, ArrayView<bool> vertexIsFeature, ArrayView<int> rowPtr,
	ArrayView<Halfedge> halfedges, Vec3f* faceNormals, float maxAngle);

__global__ void k_setBoundary2DVertices(int nT, Triangle* triangles, ArrayView<bool> vertexIsBoundary2D);
__global__ void k_setBoundary2DVertices(int nT, Quad* quads, ArrayView<bool> vertexIsBoundary2D);





// adds Halfedge (dst, incidentTri) to buffer
__global__ void k_constructHalfedges1(int nTriangles, Triangle* triangles, int* vertexNumHalfedges, Halfedge* halfedges);
__global__ void k_addBoundaryHalfedges(int nHalfedges, int* keys, Halfedge* halfedges);
__global__ void k_constructHalfedges1New(int nTriangles, Triangle* triangles, Halfedge* halfedges, int* keys);
__global__ void k_constructHalfedges1New(int nQuads, Quad* quads, Halfedge* halfedges, int* keys);

// sets oppositeHE
__global__ void k_constructHalfedges2(int nHalfedgeEntries, int* vertexNumHalfedges, Halfedge* halfedges);

// adds simple Fulledge to buffer
__global__ void k_constructFulledgesFromTriangles1Pass1(int nTriangles, Triangle* triangles, int* vertexNumSimpleFulledges, int* simpleFulledges, bool* vertexIsBoundary1D);
__global__ void k_constructFulledgesFromTriangles1Pass2(int nTriangles, Triangle* triangles, int* vertexNumSimpleFulledges, int* simpleFulledges, bool* vertexIsBoundary1D);

// only considers edges from free 2d verts to other free 2d verts
__global__ void k_constructFulledgesFromTriangles1OnlyFree(int nTriangles, Triangle* triangles, int* vertexNumSimpleFulledges, int* simpleFulledges, bool* isFeature);
// only considers edges from feature verts to other feature verts
__global__ void k_constructFulledgesFromTriangles1Pass1OnlyFeature(int nTriangles, Triangle* triangles, int* vertexNumSimpleFulledges, int* simpleFulledges, bool* isFeature, int vOffset);
__global__ void k_constructFulledgesFromTriangles1Pass2OnlyFeature(int nTriangles, Triangle* triangles, int* vertexNumSimpleFulledges, int* simpleFulledges, bool* isFeature, int vOffset);


// adds simple Fulledge to buffer
__global__ void k_constructFulledgesFromTetrahedraInnerNew(int nTetrahedra, Tetrahedron* tetrahedra, int* simpleFulledges, int* keys, bool* isBoundary2D, int vOffset);
__global__ void k_constructFulledgesFromTetrahedraFreeNew(int nTetrahedra, Tetrahedron* tetrahedra, int* simpleFulledges, int* keys, bool* isBoundary2D, int vOffset, int vMax);
__global__ void k_constructFulledgesFromTetrahedraFeatureNew(int nTetrahedra, Tetrahedron* tetrahedra, int* simpleFulledges, int* keys, bool* isBoundary2D, int vOffset, int vMax);

__global__ void k_constructFulledgesFromHexahedraInnerNew(int nTetrahedra, Hexahedron* hexahedra, int* simpleFulledges, int* keys, bool* isBoundary2D, int vOffset);
__global__ void k_constructFulledgesFromHexahedraFreeNew(int nTetrahedra, Hexahedron* hexahedra, int* simpleFulledges, int* keys, bool* isBoundary2D, int vOffset, int vMax);
__global__ void k_constructFulledgesFromHexahedraFeatureNew(int nTetrahedra, Hexahedron* hexahedra, int* simpleFulledges, int* keys, bool* isBoundary2D, int vOffset, int vMax);

__global__ void k_constructFulledgesFromHexahedraInnerNewFixed(int nTetrahedra, Hexahedron* hexahedra, int* simpleFulledges, int* keys, bool* isBoundary2D, int vOffset);
__global__ void k_constructFulledgesFromHexahedraFreeNewFixed(int nTetrahedra, Hexahedron* hexahedra, int* simpleFulledges, int* keys, bool* isBoundary2D, int vOffset, int vMax);
__global__ void k_constructFulledgesFromHexahedraFeatureNewFixed(int nTetrahedra, Hexahedron* hexahedra, int* simpleFulledges, int* keys, bool* isBoundary2D, int vOffset, int vMax);

__global__ void k_constructFulledgesFromTrianglesFreeNew(int nTriangles, Triangle* triangles, int* simpleFulledges, int* keys, int vOffset, int vMax);
__global__ void k_constructFulledgesFromTrianglesFeatureNew(int nTriangles, Triangle* triangles, int* simpleFulledges, int* keys, int vOffset, int vMax);

__global__ void k_constructFulledgesFromQuadsFreeNew(int nQuads, Quad* quads, int* simpleFulledges, int* keys, int vOffset, int vMax);
__global__ void k_constructFulledgesFromQuadsFeatureNew(int nQuads, Quad* quads, int* simpleFulledges, int* keys, int vOffset, int vMax);

__global__ void k_constructFulledgesFromTetrahedra1(int nTetrahedra, Tetrahedron* tetrahedra, int* vertexNumSimpleFulledges, int* simpleFulledges, int* ticket, int* serving);
__global__ void k_constructFulledgesFromTetrahedra1Counters(int nTetrahedra, Tetrahedron* tetrahedra, int* vertexNumSimpleFulledges, int* simpleFulledges,
	int* counters, int* added, int* addedCounter, int iter);
__global__ void k_constructFulledgesFromTetrahedra1CountersOnlyInner(int nTetrahedra, Tetrahedron* tetrahedra, int* vertexNumSimpleFulledges, int* simpleFulledges, bool* isBoundary2D,
	int* counters, int* added, int* addedCounter, int iter, int vOffset);
__global__ void k_constructFulledgesFromTetrahedra1CountersOnlyFree(int nTetrahedra, Tetrahedron* tetrahedra, int* vertexNumSimpleFulledges, int* simpleFulledges, bool* isBoundary2D,
	int* counters, int* added, int* addedCounter, int iter, int vOffset, int vMax);
__global__ void k_constructFulledgesFromTetrahedra1CountersOnlyFeature(int nTetrahedra, Tetrahedron* tetrahedra, int* vertexNumSimpleFulledges, int* simpleFulledges, bool* isFeature,
	int* counters, int* added, int* addedCounter, int iter, int vOffset, int vMax);


__global__ void k_initBoundary1D(int nVerticesSurf, int* vertexNumHalfedges, Halfedge* halfedges, bool* vertexIsBoundary1D);
__global__ void k_initBoundary1DRowPtr(int nVerticesSurf, int* rowPtr, Halfedge* halfedges, bool* vertexIsBoundary1D);


__global__ void k_constructFulledgesFromTriangles1Pass1(int nTriangles, Triangle* triangles, int* vertexNumHalfedges, Halfedge* halfedges, bool* vertexIsBoundary1D);
__global__ void k_constructFulledgesFromTriangles1Pass2(int nTriangles, Triangle* triangles, int* vertexNumHalfedges, Halfedge* halfedges, bool* vertexIsBoundary1D);

__global__ void k_constructHalffaces(int nTetrahedra, Tetrahedron* tetrahedra, int* vertexNumTetrahedra, Halfface* vertexTetrahedra);
__global__ void k_constructHalffacesNew(int nTetrahedra, Tetrahedron* tetrahedra, Halfface* halffaces, int* keys);
__global__ void k_constructHalfhexesNew(int nHexahedra, Hexahedron* hexahedra, Halfhex* halfhexes, int* keys);

__global__ void k_constructTrisFromTetrahedra(int nTetrahedra, Tetrahedron* tetrahedra, Triangle* simpleTris);
__global__ void k_constructQuadsFromHexahedra(int nHexahedra, Hexahedron* hexahedra, Quad* simpleQuads);

__global__ void k_constructSurfaceTrisFromTetrahedra(int nTetrahedra, Tetrahedron* tetrahedra, Triangle* trianglesOut, bool* vertexIsBoundary2D, int* pos_counter);


__global__ void k_checkColoring(int nTetrahedra, Tetrahedron* tetrahedra, int startNode, int nNodes, int* coloring);
__global__ void k_checkColoring(int nTriangles, Triangle* triangles, int startNode, int nNodes, int* coloring);
__global__ void k_checkColoring(int nTetrahedra, Quad* quads, int startNode, int nNodes, int* coloring);
__global__ void k_checkColoring(int nTriangles, Hexahedron* hexahedra, int startNode, int nNodes, int* coloring);



template<class T>
__global__ void k_streamCompaction(int buckets, int inBlksize, int* counts, int* startIndices, T* inBuffer, T* outBuffer, int outBufOffset) {
	for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < buckets; idx += blockDim.x * gridDim.x) {
		const int bucket = idx;
		const int startIdx = startIndices[bucket];
		for (int i = 0; i < counts[bucket]; ++i) {
			outBuffer[outBufOffset + startIdx + i] = inBuffer[bucket * inBlksize + i];
		}
	}
}

template<class K, class T>
__global__ void k_streamCompactionWithKeys(int buckets, int inBlksize, int* counts, int* startIndices, K* inKeys, K* outKeys, T* inBuffer, T* outBuffer, int outBufOffset) {
	for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < buckets; idx += blockDim.x * gridDim.x) {
		const int bucket = idx;
		const int startIdx = startIndices[bucket];
		for (int i = 0; i < counts[bucket]; ++i) {
			outBuffer[outBufOffset + startIdx + i] = inBuffer[bucket * inBlksize + i];
			outKeys[outBufOffset + startIdx + i] = inKeys[bucket * inBlksize + i];
		}
	}
}

template<class T, class FunctorRemove>
int compress(device_vector<T>& vals, const device_vector<int>& nnz, device_vector<int>& rowPtr, int rows) {
	rowPtr.resize(rows+1);
	rowPtr[0] = 0;
	thrust::inclusive_scan(nnz.begin(), nnz.begin() + rows, rowPtr.begin() + 1);

	auto endit = thrust::remove_if(vals.begin(), vals.end(), FunctorRemove());
	int enditPos = static_cast<int>(endit - vals.begin());
	vals.resize(enditPos);
	return enditPos;
}

template<class T>
int compressBatch(device_vector<T>& inVals, device_vector<T>& outVals, const device_vector<int>& nnz, int rows, int blkSize, int outBufOffset) {
	device_vector<int> rowPtr(rows + 1);
	rowPtr[0] = 0;
	thrust::inclusive_scan(nnz.begin(), nnz.begin() + rows, rowPtr.begin() + 1);

	//outVals.resize(rowPtr[rows]);
	const int BLOCK_SIZE = 128;
	k_streamCompaction << <getBlockCount(rows, BLOCK_SIZE), BLOCK_SIZE>> > (rows, blkSize, raw(nnz), raw(rowPtr), raw(inVals), raw(outVals), outBufOffset);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	return rowPtr[rows];
}

template<class KeyType, class ValueType>
int compressBatch(device_vector<KeyType>& inKeys, device_vector<KeyType>& outKeys, device_vector<ValueType>& inVals, device_vector<ValueType>& outVals, device_vector<int>& nnz,
	int rows, int blkSize, int outBufOffset)
{
	device_vector<int> rowPtr(rows + 1);
	rowPtr[0] = 0;
	thrust::inclusive_scan(nnz.begin(), nnz.begin() + rows, rowPtr.begin() + 1);

	//outVals.resize(rowPtr[rows]);
	const int BLOCK_SIZE = 128;
	k_streamCompactionWithKeys << <getBlockCount(rows, BLOCK_SIZE), BLOCK_SIZE >> > (rows, blkSize, raw(nnz), raw(rowPtr), raw(inKeys), raw(outKeys), raw(inVals), raw(outVals), outBufOffset);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	return rowPtr[rows];
}


__device__ static void addHalfedgeBatch(int* countBuffers, int* keyBuffer, Halfedge* valBuffer, int vsrc, int vdst, int incidentFace, const int nBuckets, const int sizeBucket) {
	const int bucket = vsrc % nBuckets;
	int srcCount = atomicAdd(&countBuffers[bucket], 1);
	if (srcCount >= sizeBucket) {
		printf("exceed bucketsize\n");
		assert(0);
	}
	size_t indexInBucket = (size_t)sizeBucket * bucket + srcCount;
	Halfedge& halfedge = valBuffer[indexInBucket];
	halfedge.incidentFace = incidentFace;
	halfedge.targetVertex = vdst;
	int& key = keyBuffer[indexInBucket];
	key = vsrc;
}

// adds Halfedge (dst, incidentTri) to buffer
__global__ static void k_constructHalfedges1Batch(int nTriangles, Triangle* triangles, int* countBuffers, int* keyBuffer, Halfedge* valBuffer, int nBuckets, int sizeBucket, int batch, int batchSize) {
	for (int index = threadIdx.x + blockIdx.x * blockDim.x; index < nTriangles; index += blockDim.x * gridDim.x) {
		int idx = batch * batchSize + index;
		const Triangle& tri = triangles[idx];
		int v0 = tri.v0;
		int v1 = tri.v1;
		int v2 = tri.v2;
		addHalfedgeBatch(countBuffers, keyBuffer, valBuffer, v0, v1, idx, nBuckets, sizeBucket);
		addHalfedgeBatch(countBuffers, keyBuffer, valBuffer, v1, v2, idx, nBuckets, sizeBucket);
		addHalfedgeBatch(countBuffers, keyBuffer, valBuffer, v2, v0, idx, nBuckets, sizeBucket);
	}
}

#include "MeshGPUCommon.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <utility>
#include <numeric>
#include <algorithm>
#include <thrust/remove.h>
#include <thrust/sort.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>
#include <thrust/find.h>
#include <thrust/execution_policy.h>
//#include "VertexColoring.h"
#include "CudaUtil.h"
#include <iostream>
#include <stdio.h>

#include "ConfigUsing.h"

//__constant__ int c_MAX_HE_PER_VERTEX;
const int MAX_HE_PER_VERTEX = 128;
const int MAX_TETS_PER_VERTEX = 256;

__global__ void k_reorderVertices(int nT, Triangle* triangles) {
	for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < nT; idx += blockDim.x * gridDim.x) {
		Triangle& tri = triangles[idx];
		if (tri.v1 < tri.v0 && tri.v1 < tri.v2) {
			tri = { tri.v1, tri.v2, tri.v0 };
		}
		else if (tri.v2 < tri.v0 && tri.v2 < tri.v1) {
			tri = { tri.v2, tri.v0, tri.v1 };
		}
	}
}

__global__ void k_reorderVertices(int nT, Quad* quads) {
	for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < nT; idx += blockDim.x * gridDim.x) {
		Quad& quad = quads[idx];
		const int v0 = quad.v0;
		const int v1 = quad.v1;
		const int v2 = quad.v2;
		const int v3 = quad.v3;
		if (v1 < v0 && v1 < v2 && v1 < v3) {
			quad = { v1, v2, v3, v0 };
		}
		else if (v2 < v0 && v2 < v1 && v2 < v3) {
			quad = { v2, v3, v0, v1 };
		}
		else if (v3 < v0 && v3 < v1 && v3 < v2) {
			quad = { v3, v0, v1, v2 };
		}
	}
}

__global__ void k_markDuplicateTris(int nT, Triangle* triangles) {
	for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < nT; idx += blockDim.x * gridDim.x) {
		//only the threads whose tri is in order abc with a<b<c search if there is a tri acb, triangles is fully sorted
		Triangle& tri = triangles[idx];
		if (tri.v1 < tri.v2) {
			for (int i = idx + 1; i < nT; ++i) {
				Triangle& it = triangles[i];
				if (it.v0 > tri.v0) {
					break;
				}
				if ((tri.v0 == it.v0) && (tri.v1 == it.v2) && (tri.v2 == it.v1)) {
					tri.v0 = -1;
					it.v0 = -1;
					break;
				}
			}
		}
	}
}

__global__ void k_markDuplicateQuads(int nT, Quad* quads) {
	for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < nT; idx += blockDim.x * gridDim.x) {
		//only the threads whose tri is in order abc with a<b<c search if there is a tri acb, triangles is fully sorted
		Quad& q = quads[idx];
		if (q.v1 < q.v3) {
			for (int i = idx + 1; i < nT; ++i) {
				Quad& it = quads[i];
				if (it.v0 > q.v0) {
					break;
				}
				if ((q.v0 == it.v0) && (q.v1 == it.v3) && (q.v2 == it.v2) && (q.v3 == it.v1)) {
					q.v0 = -1;
					it.v0 = -1;
					break;
				}
			}
		}
	}
}

__global__ void k_fillHelperArray(int nV, ArrayView<int> csrHalfedgeRowPtr, ArrayView<int> indexToCol) {
	for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < nV; idx += blockDim.x * gridDim.x) {
		int startIndex = csrHalfedgeRowPtr[idx];
		int endIndex = csrHalfedgeRowPtr[idx + 1];
		for (int j = startIndex; j < endIndex; ++j) {
			indexToCol[j] = idx;
		}
	}
}


__global__ void k_setOppositeHalfedgeHelper(int nHalfedges, ArrayView<Halfedge> csrHalfedgesValues, ArrayView<int> csrHalfedgeRowPtr, ArrayView<int> indexToCol)
{
	for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < nHalfedges; idx += blockDim.x * gridDim.x) {
		int ourRow = csrHalfedgesValues[idx].targetVertex;
		int ourCol = indexToCol[idx];
		int opposite = -1;
		for (int index = csrHalfedgeRowPtr[ourRow]; index < csrHalfedgeRowPtr[ourRow + 1]; ++index) {
			if (csrHalfedgesValues[index].targetVertex == ourCol) {
				//found
				opposite = index;
			}
		}
		csrHalfedgesValues[idx].oppositeHE = opposite;
	}
}

__global__ void k_setNextHalfedgeHelper(int nHalfedges, ArrayView<Halfedge> csrHalfedgesValues, ArrayView<int> csrHalfedgeRowPtr, ArrayView<int> indexToCol, int nVerticesSurf)
{
	for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < nHalfedges; idx += blockDim.x * gridDim.x) {
		Halfedge& us = csrHalfedgesValues[idx];
		int dst = us.targetVertex;
		int src = indexToCol[idx];
		for (int i = csrHalfedgeRowPtr[dst]; i < csrHalfedgeRowPtr[dst + 1]; ++i) {
			if (csrHalfedgesValues[i].incidentFace == us.incidentFace) {
				us.nextEdge = i;
			}
		}
	}
}

__device__ int getThirdVertex(const Triangle& tri, int src, int currDst) {
	if (tri.v0 != src && tri.v0 != currDst) {
		return tri.v0;
	}
	else if (tri.v1 != src && tri.v1 != currDst) {
		return tri.v1;
	}
	else {
		return tri.v2;
	}
}

__device__ int getThirdVertex(const Quad& q, int src, int currDst) {
	if (currDst == q.v1) {
		return q.v3;
	}
	else if (currDst == q.v2) {
		return q.v0;
	}
	else if (currDst == q.v3) {
		return q.v1;
	}
	else if (currDst == q.v0) {
		return q.v2;
	}
}


__global__ void k_sortCCW(int nVerticesSurf, ArrayView<int> csrHalfedgeRowPtr, ArrayView<Halfedge> halfedgesDev, ArrayView<Halfedge> csrHalfedgesValuesOut,
	bool* vertexIsBoundary1D, Triangle* triangles)
{
	for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < nVerticesSurf; idx += blockDim.x * gridDim.x) {
		int src = idx;
		int start = csrHalfedgeRowPtr[src];
		int nHalfedgesVertex = csrHalfedgeRowPtr[src + 1] - csrHalfedgeRowPtr[src];
		int outPos = 0;
		Halfedge h = halfedgesDev[start];
		if (vertexIsBoundary1D[src]) {
			// find he with opp == -1
			int j = 0;
			while (h.oppositeHE != -1) {
				++j;
				if (j == nHalfedgesVertex) {
					printf("Fehler k_sortCCW\n");
				}
				h = halfedgesDev[start + j];
			}
		}
		// add h to outlist
		csrHalfedgesValuesOut[start + outPos] = h;
		//printf("%d at %d = %d\n", src, start+outPos, h.incidentFace, h.oppositeHE, h.targetVertex);
		++outPos;
		
		bool found;
		do {
			int currDst = h.targetVertex;
			int currTri = h.incidentFace;
			int nextVert = getThirdVertex(triangles[currTri], src, currDst);

			// find he with dst == nextVert
			found = false;
			for (int j = 0; j < nHalfedgesVertex; ++j) {
				if (halfedgesDev[start + j].targetVertex == nextVert) {
					// found
					found = true;
					h = halfedgesDev[start + j];
					csrHalfedgesValuesOut[start + outPos] = h;
					++outPos;
				}
			}
		} while (found && outPos < nHalfedgesVertex);
	}
}

__global__ void k_sortCCWDummyOpposite(int nVerticesSurf, ArrayView<int> csrHalfedgeRowPtr, ArrayView<Halfedge> halfedgesDev, ArrayView<Halfedge> csrHalfedgesValuesOut,
	bool* vertexIsBoundary1D, bool* vertexIsFeature, Triangle* triangles)
{
	for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < nVerticesSurf; idx += blockDim.x * gridDim.x) {
		int src = idx;
		int start = csrHalfedgeRowPtr[src];
		int nHalfedgesVertex = csrHalfedgeRowPtr[src + 1] - csrHalfedgeRowPtr[src];
		int outPos = 0;
		// starting halfedge in fan
		Halfedge h = halfedgesDev[start];
		if (vertexIsBoundary1D[src]) {
			// find he with opp == -1
			int j = 0;
			while (halfedgesDev[h.oppositeHE].incidentFace != -1) {
				++j;
				if (j == nHalfedgesVertex) {
					printf("Fehler1 k_sortCCWDummyOpposite src=%i\n", src);
				}
				h = halfedgesDev[start + j];
			}
		}
		else if (vertexIsFeature[src]) {
			h.targetVertex = INT_MAX;
			// find he which leads to another feature vertex
			for (int j = 0; j < nHalfedgesVertex; ++j) {
				const Halfedge& heTest = halfedgesDev[start + j];
				if (vertexIsFeature[heTest.targetVertex] && heTest.targetVertex < h.targetVertex) {
					h = heTest;
				}
			}
			if (h.targetVertex == INT_MAX) {
				printf("Fehler2 k_sortCCWDummyOpposite src=%i\n", src);
				assert(0);
			}
		}
		else {
			for (int i = 1; i < nHalfedgesVertex; ++i) {
				const Halfedge& heTest = halfedgesDev[start + i];
				if (heTest.targetVertex < h.targetVertex) {
					h = heTest;
				}
			}
		}
		// add h to outlist
		csrHalfedgesValuesOut[start + outPos] = h;
		
		++outPos;

		bool found;
		do {
			int currDst = h.targetVertex;
			int currTri = h.incidentFace;
			if (currTri == -1) {
				printf("currTri = -1 in MeshGPUCommon, idx=%i\n", src);
				assert(0);
			}
			int nextVert = getThirdVertex(triangles[currTri], src, currDst);

			// find he with dst == nextVert
			found = false;
			for (int j = 0; j < nHalfedgesVertex; ++j) {
				if (halfedgesDev[start + j].targetVertex == nextVert) {
					// found
					found = true;
					h = halfedgesDev[start + j];
					csrHalfedgesValuesOut[start + outPos] = h;
					/*if (src == 0) {
						printf("face=%i opp=%i dst=%i\n", h.incidentFace, h.oppositeHE, h.targetVertex);
					}*/
					++outPos;
				}
			}
		} while (found && outPos < nHalfedgesVertex);
	}
}

__global__ void k_sortCCWDummyOpposite(int nVerticesSurf, ArrayView<int> csrHalfedgeRowPtr, ArrayView<Halfedge> halfedgesDev, ArrayView<Halfedge> csrHalfedgesValuesOut,
	bool* vertexIsBoundary1D, bool* vertexIsFeature, Quad* quads)
{
	for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < nVerticesSurf; idx += blockDim.x * gridDim.x) {
		int src = idx;
		int start = csrHalfedgeRowPtr[src];
		int nHalfedgesVertex = csrHalfedgeRowPtr[src + 1] - csrHalfedgeRowPtr[src];
		int outPos = 0;
		// starting halfedge in fan
		Halfedge h = halfedgesDev[start];
		if (vertexIsBoundary1D[src]) {
			// find he with opp == -1
			int j = 0;
			while (halfedgesDev[h.oppositeHE].incidentFace != -1) {
				++j;
				if (j == nHalfedgesVertex) {
					printf("Fehler1 k_sortCCWDummyOpposite src=%i\n", src);
				}
				h = halfedgesDev[start + j];
			}
		}
		else if (vertexIsFeature[src]) {
			h.targetVertex = INT_MAX;
			// find he which leads to another feature vertex
			for (int j = 0; j < nHalfedgesVertex; ++j) {
				const Halfedge& heTest = halfedgesDev[start + j];
				if (vertexIsFeature[heTest.targetVertex] && heTest.targetVertex < h.targetVertex) {
					h = heTest;
				}
			}
			if (h.targetVertex == INT_MAX) {
				printf("Fehler2 k_sortCCWDummyOpposite src=%i\n", src);
				assert(0);
			}
		}
		else {
			for (int i = 1; i < nHalfedgesVertex; ++i) {
				const Halfedge& heTest = halfedgesDev[start + i];
				if (heTest.targetVertex < h.targetVertex) {
					h = heTest;
				}
			}
		}
		// add h to outlist
		csrHalfedgesValuesOut[start + outPos] = h;

		++outPos;

		bool found;
		do {
			int currDst = h.targetVertex;
			int currTri = h.incidentFace;
			if (currTri == -1) {
				printf("currTri = -1 in MeshGPUCommon, idx=%i\n", src);
				assert(0);
			}
			int nextVert = getThirdVertex(quads[currTri], src, currDst);

			// find he with dst == nextVert
			found = false;
			for (int j = 0; j < nHalfedgesVertex; ++j) {
				if (halfedgesDev[start + j].targetVertex == nextVert) {
					// found
					found = true;
					h = halfedgesDev[start + j];
					csrHalfedgesValuesOut[start + outPos] = h;
					/*if (src == 0) {
						printf("face=%i opp=%i dst=%i\n", h.incidentFace, h.oppositeHE, h.targetVertex);
					}*/
					++outPos;
				}
			}
		} while (found && outPos < nHalfedgesVertex);
	}
}

__global__ void k_calcFaceNormals(int nT, Vec3f* vertexPoints, Vec3f* faceNormals, Triangle* triangles)
{
	for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < nT; idx += blockDim.x * gridDim.x) {
		Triangle& tri = triangles[idx];
		Vec3f points[3] = { vertexPoints[tri.v0], vertexPoints[tri.v1], vertexPoints[tri.v2] };
		Vec3f e0 = points[1] - points[0];
		Vec3f e1 = points[2] - points[0];
		if (e0.norm() < 1e-6) {
			printf("e0 is zerovec, tri: %i\n", idx);
			assert(0);
		}
		if (e1.norm() < 1e-6) {
			printf("e1 is zerovec, tri: %i\n", idx);
			assert(0);
		}
		Vec3f normal = e0.cross(e1);
		normal.normalize();
		if (normal.norm() < 1e-6) {
			printf("normal is zerovec, tri: %i\n", idx);
			assert(0);
		}
		faceNormals[idx] = normal;
	}
}

__global__ void k_calcFaceNormals(int nFaces, Vec3f* vertexPoints, Vec3f* faceNormals, Quad* quads)
{
	for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < nFaces; idx += blockDim.x * gridDim.x) {
		Quad& quad = quads[idx];
		Vec3f points[3] = { vertexPoints[quad.v0], vertexPoints[quad.v1], vertexPoints[quad.v3] };
		Vec3f e0 = points[1] - points[0];
		Vec3f e1 = points[2] - points[0];
		if (e0.norm() < 1e-6) {
			printf("e0 is zerovec, quad: %i\n", idx);
			assert(0);
		}
		if (e1.norm() < 1e-6) {
			printf("e1 is zerovec, quad: %i\n", idx);
			assert(0);
		}
		Vec3f normal = e0.cross(e1);
		normal.normalize();
		if (normal.norm() < 1e-6) {
			printf("normal is zerovec, quad: %i\n", idx);
			assert(0);
		}
		faceNormals[idx] = normal;
	}
}

__global__ void k_calcVertexNormals(int nV, Vec3f* vertexNormals, Halfedge* csrHalfedgesValues,
	int* csrHalfedgeRowPtr, Vec3f* faceNormals)
{
	for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < nV; idx += blockDim.x * gridDim.x) {
		Vec3f normal(0, 0, 0);
		int start = csrHalfedgeRowPtr[idx];
		int end = csrHalfedgeRowPtr[idx + 1];
		if (start == end) {
			printf("0 halfedges, vertex: %i\n", idx);
			assert(0);
		}
		for (int i = start; i < end; ++i) {
			Halfedge& halfedge = csrHalfedgesValues[i];
			if (halfedge.incidentFace != -1) {
				Vec3f faceNormal = faceNormals[halfedge.incidentFace];
				normal += faceNormal;
			}
		}
		normal.normalize();
		vertexNormals[idx] = normal;
	}
}

__global__ void k_initFeatures(int nV, ArrayView<bool> vertexIsFeature, ArrayView<int> vertexNumHalfedges,
	ArrayView<Halfedge> halfedges, Vec3f* faceNormals, const int MAX_HE_PER_VERTEX, float maxAngle)
{
	for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < nV; idx += blockDim.x * gridDim.x) {
		int src = idx;
		for (int j = 0; j < vertexNumHalfedges[src]; ++j) {
			// for each halfedge: get src and dst vertex
			const Halfedge& halfedge = halfedges[src * MAX_HE_PER_VERTEX + j];
			int dst = halfedge.targetVertex;
			if (halfedge.oppositeHE == -1) {
				//boundary edge
				vertexIsFeature[src] = true;
				vertexIsFeature[dst] = true; // kernel is executed per vertex, so this might be unnecessary
				continue;
			}
			Vec3f fn1 = faceNormals[halfedge.incidentFace];
			const Halfedge& halfedgeOpp = halfedges[halfedge.oppositeHE];
			Vec3f fn2 = faceNormals[halfedgeOpp.incidentFace];
			float dot = fn1.dot(fn2);
			float len = sqrtf(fn1.squaredNorm() * fn2.squaredNorm());
			float angle = 180.f / M_PI * acosf(dot / len);

			if (angle > maxAngle) {
				vertexIsFeature[src] = true;
				vertexIsFeature[dst] = true;
			}
		}
	}
}

__global__ void k_initFeaturesRowPtr(int nV, ArrayView<bool> vertexIsFeature, ArrayView<int> rowPtr,
	ArrayView<Halfedge> halfedges, Vec3f* faceNormals, float maxAngle)
{
	for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < nV; idx += blockDim.x * gridDim.x) {
		int src = idx;
		int num = rowPtr[src + 1] - rowPtr[src];
		for (int j = 0; j < num; ++j) {
			// for each halfedge: get src and dst vertex
			const Halfedge& halfedge = halfedges[rowPtr[src] + j];
			int dst = halfedge.targetVertex;
			if (halfedge.oppositeHE == -1) {
				//boundary edge
				vertexIsFeature[src] = true;
				vertexIsFeature[dst] = true; // kernel is executed per vertex, so this might be unnecessary
				continue;
			}
			Vec3f fn1 = faceNormals[halfedge.incidentFace];
			const Halfedge& halfedgeOpp = halfedges[halfedge.oppositeHE];
			Vec3f fn2 = faceNormals[halfedgeOpp.incidentFace];
			float dot = fn1.dot(fn2);
			float len = sqrtf(fn1.squaredNorm() * fn2.squaredNorm());
			float angle = 180.f / M_PI * acosf(dot / len);

			if (angle > maxAngle) {
				vertexIsFeature[src] = true;
				vertexIsFeature[dst] = true;
			}
		}
	}
}

__global__ void k_setBoundary2DVertices(int nT, Triangle* triangles, ArrayView<bool> vertexIsBoundary2D) {
	for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < nT; idx += blockDim.x * gridDim.x) {
		const Triangle& t = triangles[idx];
		vertexIsBoundary2D[t.v0] = true;
		vertexIsBoundary2D[t.v1] = true;
		vertexIsBoundary2D[t.v2] = true;
	}
}

__global__ void k_setBoundary2DVertices(int nT, Quad* quads, ArrayView<bool> vertexIsBoundary2D) {
	for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < nT; idx += blockDim.x * gridDim.x) {
		const Quad& t = quads[idx];
		vertexIsBoundary2D[t.v0] = true;
		vertexIsBoundary2D[t.v1] = true;
		vertexIsBoundary2D[t.v2] = true;
		vertexIsBoundary2D[t.v3] = true;
	}
}



__device__ inline void addHalfedge(int* vertexNumHalfedges, Halfedge* halfedges, int vsrc, int vdst, int incidentFace) {
	//const int MAX_HE_PER_VERTEX = 128; // TODO
	int numSrcHalfedges = atomicAdd(&vertexNumHalfedges[vsrc], 1);
	if (numSrcHalfedges >= MAX_HE_PER_VERTEX) {
		printf("temp buffer too small for halfedges\n");
		assert(0);
	}
	//int numSrcHalfedges = vertexNumHalfedges[vsrc];
	Halfedge& halfedge = halfedges[(size_t)MAX_HE_PER_VERTEX * vsrc + numSrcHalfedges];
	halfedge.incidentFace = incidentFace;
	halfedge.targetVertex = vdst;
}


__device__ inline void addHalfedgeCheckExisting(int* vertexNumHalfedges, Halfedge* halfedges, int vsrc, int vdst, int incidentFace) {
	//const int MAX_HE_PER_VERTEX = 128; // TODO
	int numSrcHalfedgesBefore = vertexNumHalfedges[vsrc];
	for (int i = 0; i < numSrcHalfedgesBefore; ++i) {
		if (halfedges[(size_t)MAX_HE_PER_VERTEX * vsrc + i].targetVertex == vdst) {
			return;
		}
	}

	int numSrcHalfedges = atomicAdd(&vertexNumHalfedges[vsrc], 1);
	if (numSrcHalfedges >= MAX_HE_PER_VERTEX) {
		printf("temp buffer too small for halfedges\n");
		assert(0);
	}
	//int numSrcHalfedges = vertexNumHalfedges[vsrc];
	Halfedge& halfedge = halfedges[(size_t)MAX_HE_PER_VERTEX * vsrc + numSrcHalfedges];
	halfedge.incidentFace = incidentFace;
	halfedge.targetVertex = vdst;
}

// adds Halfedge (dst, incidentTri) to buffer
__global__ void k_constructHalfedges1(int nTriangles, Triangle* triangles, int* vertexNumHalfedges, Halfedge* halfedges) {
	for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < nTriangles; idx += blockDim.x * gridDim.x) {
		const Triangle& tri = triangles[idx];
		int v0 = tri.v0;
		int v1 = tri.v1;
		int v2 = tri.v2;
		addHalfedge(vertexNumHalfedges, halfedges, v0, v1, idx);
		addHalfedge(vertexNumHalfedges, halfedges, v1, v2, idx);
		addHalfedge(vertexNumHalfedges, halfedges, v2, v0, idx);
	}
}

__global__ void k_addBoundaryHalfedges(int nHalfedges, int* keys, Halfedge* halfedges) {
	for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < nHalfedges; idx += blockDim.x * gridDim.x) {
		const int src = keys[idx];
		const Halfedge& he = halfedges[idx];
		if (he.oppositeHE == -1) {
			keys[nHalfedges + idx] = he.targetVertex;
			halfedges[nHalfedges + idx].targetVertex = src;
			halfedges[nHalfedges + idx].incidentFace = -1;
		}
	}
}

__global__ void k_constructHalfedges1New(int nTriangles, Triangle* triangles, Halfedge* halfedges, int* keys) {
	for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < nTriangles; idx += blockDim.x * gridDim.x) {
		const Triangle& tri = triangles[idx];
		int v0 = tri.v0;
		int v1 = tri.v1;
		int v2 = tri.v2;
		keys[3 * idx + 0] = v0;
		halfedges[3 * idx + 0].targetVertex = v1;
		halfedges[3 * idx + 0].incidentFace = idx;

		keys[3 * idx + 1] = v1;
		halfedges[3 * idx + 1].targetVertex = v2;
		halfedges[3 * idx + 1].incidentFace = idx;

		keys[3 * idx + 2] = v2;
		halfedges[3 * idx + 2].targetVertex = v0;
		halfedges[3 * idx + 2].incidentFace = idx;
	}
}

__global__ void k_constructHalfedges1New(int nQuads, Quad* quads, Halfedge* halfedges, int* keys) {
	for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < nQuads; idx += blockDim.x * gridDim.x) {
		const Quad& quad = quads[idx];
		int v0 = quad.v0;
		int v1 = quad.v1;
		int v2 = quad.v2;
		int v3 = quad.v3;
		keys[4 * idx + 0] = v0;
		halfedges[4 * idx + 0].targetVertex = v1;
		halfedges[4 * idx + 0].incidentFace = idx;

		keys[4 * idx + 1] = v1;
		halfedges[4 * idx + 1].targetVertex = v2;
		halfedges[4 * idx + 1].incidentFace = idx;

		keys[4 * idx + 2] = v2;
		halfedges[4 * idx + 2].targetVertex = v3;
		halfedges[4 * idx + 2].incidentFace = idx;

		keys[4 * idx + 3] = v3;
		halfedges[4 * idx + 3].targetVertex = v0;
		halfedges[4 * idx + 3].incidentFace = idx;
	}
}

// sets oppositeHE
__global__ void k_constructHalfedges2(int nHalfedgeEntries, int* vertexNumHalfedges, Halfedge* halfedges) {
	//const int MAX_HE_PER_VERTEX = 128; // TODO
	for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < nHalfedgeEntries; idx += blockDim.x * gridDim.x) {
		// if invalid entry
		Halfedge& he = halfedges[idx];
		int vdst = he.targetVertex;
		if (vdst == -1) {
			continue;
		}
		int vsrc = idx / MAX_HE_PER_VERTEX;

		int numDstHalfedges = vertexNumHalfedges[vdst];
		for (int i = 0; i < numDstHalfedges; ++i) {
			Halfedge& other = halfedges[(size_t)MAX_HE_PER_VERTEX * vdst + i];
			if (other.targetVertex == vsrc) {
				he.oppositeHE = MAX_HE_PER_VERTEX * vdst + i;
				other.oppositeHE = idx;
				break;
			}
		}
	}
}

__device__ inline void addFulledge(int* vertexNumSimpleFulledges, int* simpleFulledges, int vsrc, int vdst) {
	//const int MAX_HE_PER_VERTEX = 128; // TODO
	int numSrcHalfedges = atomicAdd(&vertexNumSimpleFulledges[vsrc], 1);
	if (numSrcHalfedges >= MAX_HE_PER_VERTEX) {
		printf("temp buffer too small for halfedges src %i dst %i num %i\n", vsrc, vdst, numSrcHalfedges);
		assert(0);
	}
	simpleFulledges[(size_t)MAX_HE_PER_VERTEX * vsrc + numSrcHalfedges] = vdst;
}

__device__ inline void addFulledgeCheckExisting(int* vertexNumSimpleFulledges, int* simpleFulledges, int vsrc, int vdst) {
	//const int MAX_HE_PER_VERTEX = 128; // TODO
	int numSrcHalfedgesBefore = vertexNumSimpleFulledges[vsrc];
	for (int i = 0; i < numSrcHalfedgesBefore; ++i) {
		if (simpleFulledges[(size_t)MAX_HE_PER_VERTEX * vsrc + i] == vdst) {
			return;
		}
	}

	int numSrcHalfedges = atomicAdd(&vertexNumSimpleFulledges[vsrc], 1);
	if (numSrcHalfedges >= MAX_HE_PER_VERTEX) {
		printf("temp buffer too small for halfedges\n");
		assert(0);
	}
	simpleFulledges[(size_t)MAX_HE_PER_VERTEX * vsrc + numSrcHalfedges] = vdst;
}

// adds simple Fulledge to buffer
__global__ void k_constructFulledgesFromTriangles1Pass1(int nTriangles, Triangle* triangles, int* vertexNumSimpleFulledges, int* simpleFulledges, bool* vertexIsBoundary1D) {
	for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < nTriangles; idx += blockDim.x * gridDim.x) {
		const Triangle& tri = triangles[idx];
		int v0 = tri.v0;
		int v1 = tri.v1;
		int v2 = tri.v2;
		addFulledge(vertexNumSimpleFulledges, simpleFulledges, v0, v1);
		addFulledge(vertexNumSimpleFulledges, simpleFulledges, v1, v2);
		addFulledge(vertexNumSimpleFulledges, simpleFulledges, v2, v0);
	}
}

// adds simple Fulledge to buffer
__global__ void k_constructFulledgesFromTriangles1Pass2(int nTriangles, Triangle* triangles, int* vertexNumSimpleFulledges, int* simpleFulledges, bool* vertexIsBoundary1D) {
	for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < nTriangles; idx += blockDim.x * gridDim.x) {
		const Triangle& tri = triangles[idx];
		int v0 = tri.v0;
		int v1 = tri.v1;
		int v2 = tri.v2;
		if (vertexIsBoundary1D[v0] && vertexIsBoundary1D[v1]) {
			addFulledgeCheckExisting(vertexNumSimpleFulledges, simpleFulledges, v1, v0);
		}
		if (vertexIsBoundary1D[v1] && vertexIsBoundary1D[v2]) {
			addFulledgeCheckExisting(vertexNumSimpleFulledges, simpleFulledges, v2, v1);
		}
		if (vertexIsBoundary1D[v2] && vertexIsBoundary1D[v0]) {
			addFulledgeCheckExisting(vertexNumSimpleFulledges, simpleFulledges, v0, v2);
		}
	}
}


// only considers edges from free 2d verts to other free 2d verts
__global__ void k_constructFulledgesFromTriangles1OnlyFree(int nTriangles, Triangle* triangles, int* vertexNumSimpleFulledges, int* simpleFulledges, bool* isFeature) {
	for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < nTriangles; idx += blockDim.x * gridDim.x) {
		const Triangle& tri = triangles[idx];
		int v0 = tri.v0;
		int v1 = tri.v1;
		int v2 = tri.v2;
		if (!isFeature[v0] && !isFeature[v1]) addFulledge(vertexNumSimpleFulledges, simpleFulledges, v0, v1);
		if (!isFeature[v1] && !isFeature[v2]) addFulledge(vertexNumSimpleFulledges, simpleFulledges, v1, v2);
		if (!isFeature[v2] && !isFeature[v0]) addFulledge(vertexNumSimpleFulledges, simpleFulledges, v2, v0);
	}
}

// only considers edges from feature verts to other feature verts
__global__ void k_constructFulledgesFromTriangles1Pass1OnlyFeature(int nTriangles, Triangle* triangles, int* vertexNumSimpleFulledges, int* simpleFulledges, bool* isFeature, int vOffset) {
	for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < nTriangles; idx += blockDim.x * gridDim.x) {
		const Triangle& tri = triangles[idx];
		int v0 = tri.v0 - vOffset;
		int v1 = tri.v1 - vOffset;
		int v2 = tri.v2 - vOffset;
		if (isFeature[tri.v0] && isFeature[tri.v1]) addFulledge(vertexNumSimpleFulledges, simpleFulledges, v0, v1);
		if (isFeature[tri.v1] && isFeature[tri.v2]) addFulledge(vertexNumSimpleFulledges, simpleFulledges, v1, v2);
		if (isFeature[tri.v2] && isFeature[tri.v0]) addFulledge(vertexNumSimpleFulledges, simpleFulledges, v2, v0);
	}
}

__global__ void k_constructFulledgesFromTriangles1Pass2OnlyFeature(int nTriangles, Triangle* triangles, int* vertexNumSimpleFulledges, int* simpleFulledges, bool* isFeature, int vOffset) {
	for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < nTriangles; idx += blockDim.x * gridDim.x) {
		const Triangle& tri = triangles[idx];
		int v0 = tri.v0 - vOffset;
		int v1 = tri.v1 - vOffset;
		int v2 = tri.v2 - vOffset;
		if (isFeature[tri.v0] && isFeature[tri.v1]) addFulledgeCheckExisting(vertexNumSimpleFulledges, simpleFulledges, v1, v0);
		if (isFeature[tri.v1] && isFeature[tri.v2]) addFulledgeCheckExisting(vertexNumSimpleFulledges, simpleFulledges, v2, v1);
		if (isFeature[tri.v2] && isFeature[tri.v0]) addFulledgeCheckExisting(vertexNumSimpleFulledges, simpleFulledges, v0, v2);
	}
}


__device__ inline void addFulledgeSync(int* vertexNumSimpleFulledges, int* simpleFulledges, int vsrc, int vdst, int* ticket, int* serving) {
	//const int MAX_HE_PER_VERTEX = 128; // TODO

	// lock
	int& ticketi = ticket[vsrc];
	volatile int& servingi = serving[vsrc];
	int myticket = atomicAdd(&ticketi, 1);
	while (myticket != servingi) {

	}

	int numSrcHalfedges = vertexNumSimpleFulledges[vsrc];
	if (numSrcHalfedges >= MAX_HE_PER_VERTEX) {
		printf("temp buffer too small for halfedges, vsrc %i\n", vsrc);
		assert(0);
	}
	// check if dst vertex was already added
	for (int i = 0; i < numSrcHalfedges; ++i) {
		if (simpleFulledges[(size_t)MAX_HE_PER_VERTEX * vsrc + i] == vdst) {
			serving[vsrc]++; // unlock and early exit
			return;
		}
	}
	simpleFulledges[(size_t)MAX_HE_PER_VERTEX * vsrc + numSrcHalfedges] = vdst;
	vertexNumSimpleFulledges[vsrc]++;

	serving[vsrc]++;
}

__global__ void k_constructFulledgesFromTetrahedra1(int nTetrahedra, Tetrahedron* tetrahedra, int* vertexNumSimpleFulledges, int* simpleFulledges, int* ticket, int* serving) {
	for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < nTetrahedra; idx += blockDim.x * gridDim.x) {
		const Tetrahedron& tet = tetrahedra[idx];
		int v0 = tet.v0;
		int v1 = tet.v1;
		int v2 = tet.v2;
		int v3 = tet.v3;
		addFulledgeSync(vertexNumSimpleFulledges, simpleFulledges, v0, v1, ticket, serving);
		addFulledgeSync(vertexNumSimpleFulledges, simpleFulledges, v0, v2, ticket, serving);
		addFulledgeSync(vertexNumSimpleFulledges, simpleFulledges, v0, v3, ticket, serving);
		addFulledgeSync(vertexNumSimpleFulledges, simpleFulledges, v1, v2, ticket, serving);
		addFulledgeSync(vertexNumSimpleFulledges, simpleFulledges, v1, v3, ticket, serving);
		addFulledgeSync(vertexNumSimpleFulledges, simpleFulledges, v2, v3, ticket, serving);

		addFulledgeSync(vertexNumSimpleFulledges, simpleFulledges, v1, v0, ticket, serving);
		addFulledgeSync(vertexNumSimpleFulledges, simpleFulledges, v2, v0, ticket, serving);
		addFulledgeSync(vertexNumSimpleFulledges, simpleFulledges, v3, v0, ticket, serving);
		addFulledgeSync(vertexNumSimpleFulledges, simpleFulledges, v2, v1, ticket, serving);
		addFulledgeSync(vertexNumSimpleFulledges, simpleFulledges, v3, v1, ticket, serving);
		addFulledgeSync(vertexNumSimpleFulledges, simpleFulledges, v3, v2, ticket, serving);
	}
}


__device__ inline int addFulledgeCounter(int* vertexNumSimpleFulledges, int* simpleFulledges, int vsrc, int vdst, int* counters, int iter) {
	//const int MAX_HE_PER_VERTEX = 128; // TODO

	int old = atomicCAS(&counters[vsrc], iter, iter + 1); // doesnt require reset of counters to value iter at each iteration
	//int old = atomicAdd(&counters[vsrc], 1);
	if (old == iter) {
		if (vertexNumSimpleFulledges[vsrc] >= MAX_HE_PER_VERTEX) {
			printf("temp buffer too small for edges, vsrc %i\n", vsrc);
			assert(0);
		}
		// check if exists
		for (int i = 0; i < vertexNumSimpleFulledges[vsrc]; ++i) {
			if (simpleFulledges[(size_t)MAX_HE_PER_VERTEX * vsrc + i] == vdst) {
				return 1;
			}
		}
		simpleFulledges[(size_t)MAX_HE_PER_VERTEX * vsrc + vertexNumSimpleFulledges[vsrc]] = vdst;
		++vertexNumSimpleFulledges[vsrc];
		return 1;
	}
	else {
		return 0;
	}
}

__global__ void k_constructFulledgesFromTetrahedra1Counters(int nTetrahedra, Tetrahedron* tetrahedra, int* vertexNumSimpleFulledges, int* simpleFulledges,
	int* counters, int* added, int* addedCounter, int iter) {
	for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < nTetrahedra; idx += blockDim.x * gridDim.x) {
		if (added[idx] == 0xfff) {
			continue;
		}
		const Tetrahedron& tet = tetrahedra[idx];
		int v0 = tet.v0;
		int v1 = tet.v1;
		int v2 = tet.v2;
		int v3 = tet.v3;
		if (!(added[idx] & (1 << 0))) added[idx] |= (addFulledgeCounter(vertexNumSimpleFulledges, simpleFulledges, v0, v1, counters, iter) << 0);
		if (!(added[idx] & (1 << 1))) added[idx] |= (addFulledgeCounter(vertexNumSimpleFulledges, simpleFulledges, v0, v2, counters, iter) << 1);
		if (!(added[idx] & (1 << 2))) added[idx] |= (addFulledgeCounter(vertexNumSimpleFulledges, simpleFulledges, v0, v3, counters, iter) << 2);
		if (!(added[idx] & (1 << 3))) added[idx] |= (addFulledgeCounter(vertexNumSimpleFulledges, simpleFulledges, v1, v2, counters, iter) << 3);
		if (!(added[idx] & (1 << 4))) added[idx] |= (addFulledgeCounter(vertexNumSimpleFulledges, simpleFulledges, v1, v3, counters, iter) << 4);
		if (!(added[idx] & (1 << 5))) added[idx] |= (addFulledgeCounter(vertexNumSimpleFulledges, simpleFulledges, v2, v3, counters, iter) << 5);

		if (!(added[idx] & (1 << 6))) added[idx] |= (addFulledgeCounter(vertexNumSimpleFulledges, simpleFulledges, v1, v0, counters, iter) << 6);
		if (!(added[idx] & (1 << 7))) added[idx] |= (addFulledgeCounter(vertexNumSimpleFulledges, simpleFulledges, v2, v0, counters, iter) << 7);
		if (!(added[idx] & (1 << 8))) added[idx] |= (addFulledgeCounter(vertexNumSimpleFulledges, simpleFulledges, v3, v0, counters, iter) << 8);
		if (!(added[idx] & (1 << 9))) added[idx] |= (addFulledgeCounter(vertexNumSimpleFulledges, simpleFulledges, v2, v1, counters, iter) << 9);
		if (!(added[idx] & (1 << 10))) added[idx] |= (addFulledgeCounter(vertexNumSimpleFulledges, simpleFulledges, v3, v1, counters, iter) << 10);
		if (!(added[idx] & (1 << 11))) added[idx] |= (addFulledgeCounter(vertexNumSimpleFulledges, simpleFulledges, v3, v2, counters, iter) << 11);
		if (added[idx] == 0xfff) {
			atomicAdd(addedCounter, 1);
		}
	}
}

__device__ void addFulledgeKV(int* simpleFulledges, int* keys, int mul, int idx, int pos, int src, int dst) {
	keys[idx * mul + pos] = src;
	simpleFulledges[idx * mul + pos] = dst;
}

__global__ void k_constructFulledgesFromTetrahedraInnerNew(int nTetrahedra, Tetrahedron* tetrahedra, int* simpleFulledges, int* keys, bool* isBoundary2D, int vOffset) {
	for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < nTetrahedra; idx += blockDim.x * gridDim.x) {
		const Tetrahedron& tet = tetrahedra[idx];
		int v0 = tet.v0 - vOffset;
		int v1 = tet.v1 - vOffset;
		int v2 = tet.v2 - vOffset;
		int v3 = tet.v3 - vOffset;
		if (!isBoundary2D[tet.v0] && !isBoundary2D[tet.v1]) { addFulledgeKV(simpleFulledges, keys, 12, idx, 0, v0, v1); addFulledgeKV(simpleFulledges, keys, 12, idx, 6, v1, v0); }
		if (!isBoundary2D[tet.v0] && !isBoundary2D[tet.v2]) { addFulledgeKV(simpleFulledges, keys, 12, idx, 1, v0, v2); addFulledgeKV(simpleFulledges, keys, 12, idx, 7, v2, v0); }
		if (!isBoundary2D[tet.v0] && !isBoundary2D[tet.v3]) { addFulledgeKV(simpleFulledges, keys, 12, idx, 2, v0, v3); addFulledgeKV(simpleFulledges, keys, 12, idx, 8, v3, v0); }
		if (!isBoundary2D[tet.v1] && !isBoundary2D[tet.v2]) { addFulledgeKV(simpleFulledges, keys, 12, idx, 3, v1, v2); addFulledgeKV(simpleFulledges, keys, 12, idx, 9, v2, v1); }
		if (!isBoundary2D[tet.v1] && !isBoundary2D[tet.v3]) { addFulledgeKV(simpleFulledges, keys, 12, idx, 4, v1, v3); addFulledgeKV(simpleFulledges, keys, 12, idx, 10, v3, v1); }
		if (!isBoundary2D[tet.v2] && !isBoundary2D[tet.v3]) { addFulledgeKV(simpleFulledges, keys, 12, idx, 5, v2, v3); addFulledgeKV(simpleFulledges, keys, 12, idx, 11, v3, v2); }
	}
}

__global__ void k_constructFulledgesFromTetrahedraFreeNew(int nTetrahedra, Tetrahedron* tetrahedra, int* simpleFulledges, int* keys, bool* isBoundary2D, int vOffset, int vMax) {
	for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < nTetrahedra; idx += blockDim.x * gridDim.x) {
		const Tetrahedron& tet = tetrahedra[idx];
		int v0 = tet.v0;
		int v1 = tet.v1;
		int v2 = tet.v2;
		int v3 = tet.v3;
		if (v0 < vMax && v1 < vMax) { addFulledgeKV(simpleFulledges, keys, 12, idx, 0, v0, v1); addFulledgeKV(simpleFulledges, keys, 12, idx, 6, v1, v0); }
		if (v0 < vMax && v2 < vMax) { addFulledgeKV(simpleFulledges, keys, 12, idx, 1, v0, v2); addFulledgeKV(simpleFulledges, keys, 12, idx, 7, v2, v0); }
		if (v0 < vMax && v3 < vMax) { addFulledgeKV(simpleFulledges, keys, 12, idx, 2, v0, v3); addFulledgeKV(simpleFulledges, keys, 12, idx, 8, v3, v0); }
		if (v1 < vMax && v2 < vMax) { addFulledgeKV(simpleFulledges, keys, 12, idx, 3, v1, v2); addFulledgeKV(simpleFulledges, keys, 12, idx, 9, v2, v1); }
		if (v1 < vMax && v3 < vMax) { addFulledgeKV(simpleFulledges, keys, 12, idx, 4, v1, v3); addFulledgeKV(simpleFulledges, keys, 12, idx, 10, v3, v1); }
		if (v2 < vMax && v3 < vMax) { addFulledgeKV(simpleFulledges, keys, 12, idx, 5, v2, v3); addFulledgeKV(simpleFulledges, keys, 12, idx, 11, v3, v2); }
	}
}

__global__ void k_constructFulledgesFromTetrahedraFeatureNew(int nTetrahedra, Tetrahedron* tetrahedra, int* simpleFulledges, int* keys, bool* isBoundary2D, int vOffset, int vMax) {
	for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < nTetrahedra; idx += blockDim.x * gridDim.x) {
		const Tetrahedron& tet = tetrahedra[idx];
		int v0 = tet.v0 - vOffset;
		int v1 = tet.v1 - vOffset;
		int v2 = tet.v2 - vOffset;
		int v3 = tet.v3 - vOffset;
		if ((v0 >= 0 && v0 < vMax) && (v1 >= 0 && v1 < vMax)) { addFulledgeKV(simpleFulledges, keys, 12, idx, 0, v0, v1); addFulledgeKV(simpleFulledges, keys, 12, idx, 6, v1, v0); }
		if ((v0 >= 0 && v0 < vMax) && (v2 >= 0 && v2 < vMax)) { addFulledgeKV(simpleFulledges, keys, 12, idx, 1, v0, v2); addFulledgeKV(simpleFulledges, keys, 12, idx, 7, v2, v0); }
		if ((v0 >= 0 && v0 < vMax) && (v3 >= 0 && v3 < vMax)) { addFulledgeKV(simpleFulledges, keys, 12, idx, 2, v0, v3); addFulledgeKV(simpleFulledges, keys, 12, idx, 8, v3, v0); }
		if ((v1 >= 0 && v1 < vMax) && (v2 >= 0 && v2 < vMax)) { addFulledgeKV(simpleFulledges, keys, 12, idx, 3, v1, v2); addFulledgeKV(simpleFulledges, keys, 12, idx, 9, v2, v1); }
		if ((v1 >= 0 && v1 < vMax) && (v3 >= 0 && v3 < vMax)) { addFulledgeKV(simpleFulledges, keys, 12, idx, 4, v1, v3); addFulledgeKV(simpleFulledges, keys, 12, idx, 10, v3, v1); }
		if ((v2 >= 0 && v2 < vMax) && (v3 >= 0 && v3 < vMax)) { addFulledgeKV(simpleFulledges, keys, 12, idx, 5, v2, v3); addFulledgeKV(simpleFulledges, keys, 12, idx, 11, v3, v2); }
	}
}

//hexes
__global__ void k_constructFulledgesFromHexahedraInnerNew(int nTetrahedra, Hexahedron* hexahedra, int* simpleFulledges, int* keys, bool* isBoundary2D, int vOffset) {
	for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < nTetrahedra; idx += blockDim.x * gridDim.x) {
		const Hexahedron& hex = hexahedra[idx];
		int v0 = hex.v0 - vOffset;
		int v1 = hex.v1 - vOffset;
		int v2 = hex.v2 - vOffset;
		int v3 = hex.v3 - vOffset;
		int v4 = hex.v4 - vOffset;
		int v5 = hex.v5 - vOffset;
		int v6 = hex.v6 - vOffset;
		int v7 = hex.v7 - vOffset;
		if (!isBoundary2D[hex.v0] && !isBoundary2D[hex.v1]) { addFulledgeKV(simpleFulledges, keys, 24, idx, 0, v0, v1); addFulledgeKV(simpleFulledges, keys, 24, idx, 12, v1, v0); }
		if (!isBoundary2D[hex.v1] && !isBoundary2D[hex.v2]) { addFulledgeKV(simpleFulledges, keys, 24, idx, 1, v1, v2); addFulledgeKV(simpleFulledges, keys, 24, idx, 13, v2, v1); }
		if (!isBoundary2D[hex.v2] && !isBoundary2D[hex.v3]) { addFulledgeKV(simpleFulledges, keys, 24, idx, 2, v2, v3); addFulledgeKV(simpleFulledges, keys, 24, idx, 14, v3, v2); }
		if (!isBoundary2D[hex.v3] && !isBoundary2D[hex.v0]) { addFulledgeKV(simpleFulledges, keys, 24, idx, 3, v3, v0); addFulledgeKV(simpleFulledges, keys, 24, idx, 15, v0, v3); }

		if (!isBoundary2D[hex.v0] && !isBoundary2D[hex.v4]) { addFulledgeKV(simpleFulledges, keys, 24, idx, 4, v0, v4); addFulledgeKV(simpleFulledges, keys, 24, idx, 16, v4, v0); }
		if (!isBoundary2D[hex.v1] && !isBoundary2D[hex.v5]) { addFulledgeKV(simpleFulledges, keys, 24, idx, 5, v1, v5); addFulledgeKV(simpleFulledges, keys, 24, idx, 17, v5, v1); }
		if (!isBoundary2D[hex.v2] && !isBoundary2D[hex.v6]) { addFulledgeKV(simpleFulledges, keys, 24, idx, 6, v2, v6); addFulledgeKV(simpleFulledges, keys, 24, idx, 18, v6, v2); }
		if (!isBoundary2D[hex.v3] && !isBoundary2D[hex.v7]) { addFulledgeKV(simpleFulledges, keys, 24, idx, 7, v3, v7); addFulledgeKV(simpleFulledges, keys, 24, idx, 19, v7, v3); }

		if (!isBoundary2D[hex.v4] && !isBoundary2D[hex.v5]) { addFulledgeKV(simpleFulledges, keys, 24, idx, 8, v4, v5); addFulledgeKV(simpleFulledges, keys, 24, idx, 20, v5, v4); }
		if (!isBoundary2D[hex.v5] && !isBoundary2D[hex.v6]) { addFulledgeKV(simpleFulledges, keys, 24, idx, 9, v5, v6); addFulledgeKV(simpleFulledges, keys, 24, idx, 21, v6, v5); }
		if (!isBoundary2D[hex.v6] && !isBoundary2D[hex.v7]) { addFulledgeKV(simpleFulledges, keys, 24, idx, 10, v6, v7); addFulledgeKV(simpleFulledges, keys, 24, idx, 22, v7, v6); }
		if (!isBoundary2D[hex.v7] && !isBoundary2D[hex.v4]) { addFulledgeKV(simpleFulledges, keys, 24, idx, 11, v7, v4); addFulledgeKV(simpleFulledges, keys, 24, idx, 23, v4, v7); }
	}
}

//hexes fixed
__global__ void k_constructFulledgesFromHexahedraInnerNewFixed(int nTetrahedra, Hexahedron* hexahedra, int* simpleFulledges, int* keys, bool* isBoundary2D, int vOffset) {
	for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < nTetrahedra; idx += blockDim.x * gridDim.x) {
		const Hexahedron& hex = hexahedra[idx];
		const int* v = hex.data;
		int vOff[8];
		for (int i = 0; i < 8; ++i) {
			vOff[i] = v[i] - vOffset;
		}
		int pos = 0;
		for (int i = 0; i < 7; ++i) {
			for (int j = i + 1; j < 8; ++j) {
				if (!isBoundary2D[v[i]] && !isBoundary2D[v[j]]) { addFulledgeKV(simpleFulledges, keys, 56, idx, pos++, vOff[i], vOff[j]); addFulledgeKV(simpleFulledges, keys, 56, idx, pos++, vOff[j], vOff[i]); }
			}
		}
	}
}

__global__ void k_constructFulledgesFromHexahedraFreeNew(int nTetrahedra, Hexahedron* hexahedra, int* simpleFulledges, int* keys, bool* isBoundary2D, int vOffset, int vMax) {
	for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < nTetrahedra; idx += blockDim.x * gridDim.x) {
		const Hexahedron& hex = hexahedra[idx];
		int v0 = hex.v0;
		int v1 = hex.v1;
		int v2 = hex.v2;
		int v3 = hex.v3;
		int v4 = hex.v4;
		int v5 = hex.v5;
		int v6 = hex.v6;
		int v7 = hex.v7;
		if (v0 < vMax && v1 < vMax) { addFulledgeKV(simpleFulledges, keys, 24, idx, 0, v0, v1); addFulledgeKV(simpleFulledges, keys, 24, idx, 12, v1, v0); }
		if (v1 < vMax && v2 < vMax) { addFulledgeKV(simpleFulledges, keys, 24, idx, 1, v1, v2); addFulledgeKV(simpleFulledges, keys, 24, idx, 13, v2, v1); }
		if (v2 < vMax && v3 < vMax) { addFulledgeKV(simpleFulledges, keys, 24, idx, 2, v2, v3); addFulledgeKV(simpleFulledges, keys, 24, idx, 14, v3, v2); }
		if (v3 < vMax && v0 < vMax) { addFulledgeKV(simpleFulledges, keys, 24, idx, 3, v3, v0); addFulledgeKV(simpleFulledges, keys, 24, idx, 15, v0, v3); }

		if (v0 < vMax && v4 < vMax) { addFulledgeKV(simpleFulledges, keys, 24, idx, 4, v0, v4); addFulledgeKV(simpleFulledges, keys, 24, idx, 16, v4, v0); }
		if (v1 < vMax && v5 < vMax) { addFulledgeKV(simpleFulledges, keys, 24, idx, 5, v1, v5); addFulledgeKV(simpleFulledges, keys, 24, idx, 17, v5, v1); }
		if (v2 < vMax && v6 < vMax) { addFulledgeKV(simpleFulledges, keys, 24, idx, 6, v2, v6); addFulledgeKV(simpleFulledges, keys, 24, idx, 18, v6, v2); }
		if (v3 < vMax && v7 < vMax) { addFulledgeKV(simpleFulledges, keys, 24, idx, 7, v3, v7); addFulledgeKV(simpleFulledges, keys, 24, idx, 19, v7, v3); }

		if (v4 < vMax && v5 < vMax) { addFulledgeKV(simpleFulledges, keys, 24, idx, 8, v4, v5); addFulledgeKV(simpleFulledges, keys, 24, idx, 20, v5, v4); }
		if (v5 < vMax && v6 < vMax) { addFulledgeKV(simpleFulledges, keys, 24, idx, 9, v5, v6); addFulledgeKV(simpleFulledges, keys, 24, idx, 21, v6, v5); }
		if (v6 < vMax && v7 < vMax) { addFulledgeKV(simpleFulledges, keys, 24, idx, 10, v6, v7); addFulledgeKV(simpleFulledges, keys, 24, idx, 22, v7, v6); }
		if (v7 < vMax && v4 < vMax) { addFulledgeKV(simpleFulledges, keys, 24, idx, 11, v7, v4); addFulledgeKV(simpleFulledges, keys, 24, idx, 23, v4, v7); }
	}
}

//hexes fixed
__global__ void k_constructFulledgesFromHexahedraFreeNewFixed(int nTetrahedra, Hexahedron* hexahedra, int* simpleFulledges, int* keys, bool* isBoundary2D, int vOffset, int vMax) {
	for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < nTetrahedra; idx += blockDim.x * gridDim.x) {
		const Hexahedron& hex = hexahedra[idx];
		const int* v = hex.data;
		int pos = 0;
		for (int i = 0; i < 7; ++i) {
			for (int j = i + 1; j < 8; ++j) {
				if (v[i] < vMax && v[j] < vMax) { addFulledgeKV(simpleFulledges, keys, 56, idx, pos++, v[i], v[j]); addFulledgeKV(simpleFulledges, keys, 56, idx, pos++, v[j], v[i]); }
			}
		}
	}
}

__global__ void k_constructFulledgesFromHexahedraFeatureNew(int nTetrahedra, Hexahedron* hexahedra, int* simpleFulledges, int* keys, bool* isBoundary2D, int vOffset, int vMax) {
	for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < nTetrahedra; idx += blockDim.x * gridDim.x) {
		const Hexahedron& hex = hexahedra[idx];
		int v0 = hex.v0 - vOffset;
		int v1 = hex.v1 - vOffset;
		int v2 = hex.v2 - vOffset;
		int v3 = hex.v3 - vOffset;
		int v4 = hex.v4 - vOffset;
		int v5 = hex.v5 - vOffset;
		int v6 = hex.v6 - vOffset;
		int v7 = hex.v7 - vOffset;
		if ((v0 >= 0 && v0 < vMax) && (v1 >= 0 && v1 < vMax)) { addFulledgeKV(simpleFulledges, keys, 24, idx, 0, v0, v1); addFulledgeKV(simpleFulledges, keys, 24, idx, 12, v1, v0); }
		if ((v1 >= 0 && v1 < vMax) && (v2 >= 0 && v2 < vMax)) { addFulledgeKV(simpleFulledges, keys, 24, idx, 1, v1, v2); addFulledgeKV(simpleFulledges, keys, 24, idx, 13, v2, v1); }
		if ((v2 >= 0 && v2 < vMax) && (v3 >= 0 && v3 < vMax)) { addFulledgeKV(simpleFulledges, keys, 24, idx, 2, v2, v3); addFulledgeKV(simpleFulledges, keys, 24, idx, 14, v3, v2); }
		if ((v3 >= 0 && v3 < vMax) && (v0 >= 0 && v0 < vMax)) { addFulledgeKV(simpleFulledges, keys, 24, idx, 3, v3, v0); addFulledgeKV(simpleFulledges, keys, 24, idx, 15, v0, v3); }

		if ((v0 >= 0 && v0 < vMax) && (v4 >= 0 && v4 < vMax)) { addFulledgeKV(simpleFulledges, keys, 24, idx, 4, v0, v4); addFulledgeKV(simpleFulledges, keys, 24, idx, 16, v4, v0); }
		if ((v1 >= 0 && v1 < vMax) && (v5 >= 0 && v5 < vMax)) { addFulledgeKV(simpleFulledges, keys, 24, idx, 5, v1, v5); addFulledgeKV(simpleFulledges, keys, 24, idx, 17, v5, v1); }
		if ((v2 >= 0 && v2 < vMax) && (v6 >= 0 && v6 < vMax)) { addFulledgeKV(simpleFulledges, keys, 24, idx, 6, v2, v6); addFulledgeKV(simpleFulledges, keys, 24, idx, 18, v6, v2); }
		if ((v3 >= 0 && v3 < vMax) && (v7 >= 0 && v7 < vMax)) { addFulledgeKV(simpleFulledges, keys, 24, idx, 7, v3, v7); addFulledgeKV(simpleFulledges, keys, 24, idx, 19, v7, v3); }

		if ((v4 >= 0 && v4 < vMax) && (v5 >= 0 && v5 < vMax)) { addFulledgeKV(simpleFulledges, keys, 24, idx, 8, v4, v5); addFulledgeKV(simpleFulledges, keys, 24, idx, 20, v5, v4); }
		if ((v5 >= 0 && v5 < vMax) && (v6 >= 0 && v6 < vMax)) { addFulledgeKV(simpleFulledges, keys, 24, idx, 9, v5, v6); addFulledgeKV(simpleFulledges, keys, 24, idx, 21, v6, v5); }
		if ((v6 >= 0 && v6 < vMax) && (v7 >= 0 && v7 < vMax)) { addFulledgeKV(simpleFulledges, keys, 24, idx, 10, v6, v7); addFulledgeKV(simpleFulledges, keys, 24, idx, 22, v7, v6); }
		if ((v7 >= 0 && v7 < vMax) && (v4 >= 0 && v4 < vMax)) { addFulledgeKV(simpleFulledges, keys, 24, idx, 11, v7, v4); addFulledgeKV(simpleFulledges, keys, 24, idx, 23, v4, v7); }
	}
}

//hexes fixed
__global__ void k_constructFulledgesFromHexahedraFeatureNewFixed(int nTetrahedra, Hexahedron* hexahedra, int* simpleFulledges, int* keys, bool* isBoundary2D, int vOffset, int vMax) {
	for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < nTetrahedra; idx += blockDim.x * gridDim.x) {
		const Hexahedron& hex = hexahedra[idx];
		const int* v = hex.data;
		int vOff[8];
		for (int i = 0; i < 8; ++i) {
			vOff[i] = v[i] - vOffset;
		}
		int pos = 0;
		for (int i = 0; i < 7; ++i) {
			for (int j = i + 1; j < 8; ++j) {
				if ((vOff[i] >= 0 && vOff[i] < vMax) && (vOff[j] >= 0 && vOff[j] < vMax)) { addFulledgeKV(simpleFulledges, keys, 56, idx, pos++, vOff[i], vOff[j]); addFulledgeKV(simpleFulledges, keys, 56, idx, pos++, vOff[j], vOff[i]); }
			}
		}
	}
}

//tris
__global__ void k_constructFulledgesFromTrianglesFreeNew(int nTriangles, Triangle* triangles, int* simpleFulledges, int* keys, int vOffset, int vMax) {
	for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < nTriangles; idx += blockDim.x * gridDim.x) {
		const Triangle& tri = triangles[idx];
		int v0 = tri.v0;
		int v1 = tri.v1;
		int v2 = tri.v2;
		if (v0 < vMax && v1 < vMax) { addFulledgeKV(simpleFulledges, keys, 6, idx, 0, v0, v1); addFulledgeKV(simpleFulledges, keys, 6, idx, 3, v1, v0); }
		if (v0 < vMax && v2 < vMax) { addFulledgeKV(simpleFulledges, keys, 6, idx, 1, v0, v2); addFulledgeKV(simpleFulledges, keys, 6, idx, 4, v2, v0); }
		if (v1 < vMax && v2 < vMax) { addFulledgeKV(simpleFulledges, keys, 6, idx, 2, v1, v2); addFulledgeKV(simpleFulledges, keys, 6, idx, 5, v2, v1); }
	}
}

__global__ void k_constructFulledgesFromTrianglesFeatureNew(int nTriangles, Triangle* triangles, int* simpleFulledges, int* keys, int vOffset, int vMax) {
	for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < nTriangles; idx += blockDim.x * gridDim.x) {
		const Triangle& tri = triangles[idx];
		int v0 = tri.v0 - vOffset;
		int v1 = tri.v1 - vOffset;
		int v2 = tri.v2 - vOffset;
		if ((v0 >= 0 && v0 < vMax) && (v1 >= 0 && v1 < vMax)) { addFulledgeKV(simpleFulledges, keys, 6, idx, 0, v0, v1); addFulledgeKV(simpleFulledges, keys, 6, idx, 3, v1, v0); }
		if ((v0 >= 0 && v0 < vMax) && (v2 >= 0 && v2 < vMax)) { addFulledgeKV(simpleFulledges, keys, 6, idx, 1, v0, v2); addFulledgeKV(simpleFulledges, keys, 6, idx, 4, v2, v0); }
		if ((v1 >= 0 && v1 < vMax) && (v2 >= 0 && v2 < vMax)) { addFulledgeKV(simpleFulledges, keys, 6, idx, 2, v1, v2); addFulledgeKV(simpleFulledges, keys, 6, idx, 5, v2, v1); }
	}
}

//quads
__global__ void k_constructFulledgesFromQuadsFreeNew(int nQuads, Quad* quads, int* simpleFulledges, int* keys, int vOffset, int vMax) {
	for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < nQuads; idx += blockDim.x * gridDim.x) {
		const Quad& quad = quads[idx];
		int v0 = quad.v0;
		int v1 = quad.v1;
		int v2 = quad.v2;
		int v3 = quad.v3;
		if (v0 < vMax && v1 < vMax) { addFulledgeKV(simpleFulledges, keys, 12, idx, 0, v0, v1); addFulledgeKV(simpleFulledges, keys, 12, idx, 6, v1, v0); }
		if (v1 < vMax && v2 < vMax) { addFulledgeKV(simpleFulledges, keys, 12, idx, 1, v1, v2); addFulledgeKV(simpleFulledges, keys, 12, idx, 7, v2, v1); }
		if (v2 < vMax && v3 < vMax) { addFulledgeKV(simpleFulledges, keys, 12, idx, 2, v2, v3); addFulledgeKV(simpleFulledges, keys, 12, idx, 8, v3, v2); }
		if (v3 < vMax && v0 < vMax) { addFulledgeKV(simpleFulledges, keys, 12, idx, 3, v3, v0); addFulledgeKV(simpleFulledges, keys, 12, idx, 9, v0, v3); }
		if (v0 < vMax && v2 < vMax) { addFulledgeKV(simpleFulledges, keys, 12, idx, 4, v0, v2); addFulledgeKV(simpleFulledges, keys, 12, idx, 10, v2, v0); }
		if (v1 < vMax && v3 < vMax) { addFulledgeKV(simpleFulledges, keys, 12, idx, 5, v1, v3); addFulledgeKV(simpleFulledges, keys, 12, idx, 11, v3, v1); }
	}
}

__global__ void k_constructFulledgesFromQuadsFeatureNew(int nQuads, Quad* quads, int* simpleFulledges, int* keys, int vOffset, int vMax) {
	for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < nQuads; idx += blockDim.x * gridDim.x) {
		const Quad& quad = quads[idx];
		int v0 = quad.v0 - vOffset;
		int v1 = quad.v1 - vOffset;
		int v2 = quad.v2 - vOffset;
		int v3 = quad.v3 - vOffset;
		if ((v0 >= 0 && v0 < vMax) && (v1 >= 0 && v1 < vMax)) { addFulledgeKV(simpleFulledges, keys, 12, idx, 0, v0, v1); addFulledgeKV(simpleFulledges, keys, 12, idx, 6, v1, v0); }
		if ((v1 >= 0 && v1 < vMax) && (v2 >= 0 && v2 < vMax)) { addFulledgeKV(simpleFulledges, keys, 12, idx, 1, v1, v2); addFulledgeKV(simpleFulledges, keys, 12, idx, 7, v2, v1); }
		if ((v2 >= 0 && v2 < vMax) && (v3 >= 0 && v3 < vMax)) { addFulledgeKV(simpleFulledges, keys, 12, idx, 2, v2, v3); addFulledgeKV(simpleFulledges, keys, 12, idx, 8, v3, v2); }
		if ((v3 >= 0 && v3 < vMax) && (v0 >= 0 && v0 < vMax)) { addFulledgeKV(simpleFulledges, keys, 12, idx, 3, v3, v0); addFulledgeKV(simpleFulledges, keys, 12, idx, 9, v0, v3); }
		if ((v0 >= 0 && v0 < vMax) && (v2 >= 0 && v2 < vMax)) { addFulledgeKV(simpleFulledges, keys, 12, idx, 4, v0, v2); addFulledgeKV(simpleFulledges, keys, 12, idx, 10, v2, v0); }
		if ((v1 >= 0 && v1 < vMax) && (v3 >= 0 && v3 < vMax)) { addFulledgeKV(simpleFulledges, keys, 12, idx, 5, v1, v3); addFulledgeKV(simpleFulledges, keys, 12, idx, 11, v3, v1); }
	}
}

__global__ void k_constructFulledgesFromTetrahedra1CountersOnlyInner(int nTetrahedra, Tetrahedron* tetrahedra, int* vertexNumSimpleFulledges, int* simpleFulledges, bool* isBoundary2D,
	int* counters, int* added, int* addedCounter, int iter, int vOffset) {
	for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < nTetrahedra; idx += blockDim.x * gridDim.x) {
		// there are 12 directed edges, so bitset of 12 bits
		if (added[idx] == 0xfff) {
			continue;
		}
		const Tetrahedron& tet = tetrahedra[idx];
		int v0 = tet.v0 - vOffset;
		int v1 = tet.v1 - vOffset;
		int v2 = tet.v2 - vOffset;
		int v3 = tet.v3 - vOffset;
		// todo move this to a init step, dont do in every iter.
		if (isBoundary2D[tet.v0]) added[idx] |= (0b000111000111); // every edge where v0 appears
		if (isBoundary2D[tet.v1]) added[idx] |= (0b011001011001);
		if (isBoundary2D[tet.v2]) added[idx] |= (0b101010101010);
		if (isBoundary2D[tet.v3]) added[idx] |= (0b110100110100);
		if (!(added[idx] & (1 << 0))) added[idx] |= (addFulledgeCounter(vertexNumSimpleFulledges, simpleFulledges, v0, v1, counters, iter) << 0);
		if (!(added[idx] & (1 << 1))) added[idx] |= (addFulledgeCounter(vertexNumSimpleFulledges, simpleFulledges, v0, v2, counters, iter) << 1);
		if (!(added[idx] & (1 << 2))) added[idx] |= (addFulledgeCounter(vertexNumSimpleFulledges, simpleFulledges, v0, v3, counters, iter) << 2);
		if (!(added[idx] & (1 << 3))) added[idx] |= (addFulledgeCounter(vertexNumSimpleFulledges, simpleFulledges, v1, v2, counters, iter) << 3);
		if (!(added[idx] & (1 << 4))) added[idx] |= (addFulledgeCounter(vertexNumSimpleFulledges, simpleFulledges, v1, v3, counters, iter) << 4);
		if (!(added[idx] & (1 << 5))) added[idx] |= (addFulledgeCounter(vertexNumSimpleFulledges, simpleFulledges, v2, v3, counters, iter) << 5);

		if (!(added[idx] & (1 << 6))) added[idx] |= (addFulledgeCounter(vertexNumSimpleFulledges, simpleFulledges, v1, v0, counters, iter) << 6);
		if (!(added[idx] & (1 << 7))) added[idx] |= (addFulledgeCounter(vertexNumSimpleFulledges, simpleFulledges, v2, v0, counters, iter) << 7);
		if (!(added[idx] & (1 << 8))) added[idx] |= (addFulledgeCounter(vertexNumSimpleFulledges, simpleFulledges, v3, v0, counters, iter) << 8);
		if (!(added[idx] & (1 << 9))) added[idx] |= (addFulledgeCounter(vertexNumSimpleFulledges, simpleFulledges, v2, v1, counters, iter) << 9);
		if (!(added[idx] & (1 << 10))) added[idx] |= (addFulledgeCounter(vertexNumSimpleFulledges, simpleFulledges, v3, v1, counters, iter) << 10);
		if (!(added[idx] & (1 << 11))) added[idx] |= (addFulledgeCounter(vertexNumSimpleFulledges, simpleFulledges, v3, v2, counters, iter) << 11);
		if (added[idx] == 0xfff) {
			atomicAdd(addedCounter, 1);
		}
	}
}

// only difference is these 4 lines: if (!isBoundary2D[tet.v0]), TODO refactor
__global__ void k_constructFulledgesFromTetrahedra1CountersOnlyFree(int nTetrahedra, Tetrahedron* tetrahedra, int* vertexNumSimpleFulledges, int* simpleFulledges, bool* isBoundary2D,
	int* counters, int* added, int* addedCounter, int iter, int vOffset, int vMax) {
	for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < nTetrahedra; idx += blockDim.x * gridDim.x) {
		// there are 12 directed edges, so bitset of 12 bits
		if (added[idx] == 0xfff) {
			continue;
		}
		const Tetrahedron& tet = tetrahedra[idx];
		int v0 = tet.v0 - vOffset;
		int v1 = tet.v1 - vOffset;
		int v2 = tet.v2 - vOffset;
		int v3 = tet.v3 - vOffset;
		// todo move this to a init step, dont do in every iter.
		if (v0 >= vMax) added[idx] |= (0b000111000111); else if (!isBoundary2D[tet.v0]) added[idx] |= (0b000111000111); // every edge where v0 appears
		if (v1 >= vMax) added[idx] |= (0b011001011001); else if (!isBoundary2D[tet.v1]) added[idx] |= (0b011001011001);
		if (v2 >= vMax) added[idx] |= (0b101010101010); else if (!isBoundary2D[tet.v2]) added[idx] |= (0b101010101010);
		if (v3 >= vMax) added[idx] |= (0b110100110100); else if (!isBoundary2D[tet.v3]) added[idx] |= (0b110100110100);
		if (!(added[idx] & (1 << 0))) added[idx] |= (addFulledgeCounter(vertexNumSimpleFulledges, simpleFulledges, v0, v1, counters, iter) << 0);
		if (!(added[idx] & (1 << 1))) added[idx] |= (addFulledgeCounter(vertexNumSimpleFulledges, simpleFulledges, v0, v2, counters, iter) << 1);
		if (!(added[idx] & (1 << 2))) added[idx] |= (addFulledgeCounter(vertexNumSimpleFulledges, simpleFulledges, v0, v3, counters, iter) << 2);
		if (!(added[idx] & (1 << 3))) added[idx] |= (addFulledgeCounter(vertexNumSimpleFulledges, simpleFulledges, v1, v2, counters, iter) << 3);
		if (!(added[idx] & (1 << 4))) added[idx] |= (addFulledgeCounter(vertexNumSimpleFulledges, simpleFulledges, v1, v3, counters, iter) << 4);
		if (!(added[idx] & (1 << 5))) added[idx] |= (addFulledgeCounter(vertexNumSimpleFulledges, simpleFulledges, v2, v3, counters, iter) << 5);

		if (!(added[idx] & (1 << 6))) added[idx] |= (addFulledgeCounter(vertexNumSimpleFulledges, simpleFulledges, v1, v0, counters, iter) << 6);
		if (!(added[idx] & (1 << 7))) added[idx] |= (addFulledgeCounter(vertexNumSimpleFulledges, simpleFulledges, v2, v0, counters, iter) << 7);
		if (!(added[idx] & (1 << 8))) added[idx] |= (addFulledgeCounter(vertexNumSimpleFulledges, simpleFulledges, v3, v0, counters, iter) << 8);
		if (!(added[idx] & (1 << 9))) added[idx] |= (addFulledgeCounter(vertexNumSimpleFulledges, simpleFulledges, v2, v1, counters, iter) << 9);
		if (!(added[idx] & (1 << 10))) added[idx] |= (addFulledgeCounter(vertexNumSimpleFulledges, simpleFulledges, v3, v1, counters, iter) << 10);
		if (!(added[idx] & (1 << 11))) added[idx] |= (addFulledgeCounter(vertexNumSimpleFulledges, simpleFulledges, v3, v2, counters, iter) << 11);
		if (added[idx] == 0xfff) {
			atomicAdd(addedCounter, 1);
		}
	}
}

// difference is which bool array: isFeature
__global__ void k_constructFulledgesFromTetrahedra1CountersOnlyFeature(int nTetrahedra, Tetrahedron* tetrahedra, int* vertexNumSimpleFulledges, int* simpleFulledges, bool* isFeature,
	int* counters, int* added, int* addedCounter, int iter, int vOffset, int vMax) {
	for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < nTetrahedra; idx += blockDim.x * gridDim.x) {
		// there are 12 directed edges, so bitset of 12 bits
		if (added[idx] == 0xfff) {
			continue;
		}
		const Tetrahedron& tet = tetrahedra[idx];
		int v0 = tet.v0 - vOffset;
		int v1 = tet.v1 - vOffset;
		int v2 = tet.v2 - vOffset;
		int v3 = tet.v3 - vOffset;
		if (iter == 10000) {
			printf("idx %i added %i verts %i %i %i %i\n", idx, added[idx], v0,v1,v2,v3);
		}
		// todo move this to a init step, dont do in every iter.
		if (v0 < 0 || v0 >= vMax) added[idx] |= (0b000111000111); else if (!isFeature[tet.v0]) added[idx] |= (0b000111000111); // every edge where v0 appears
		if (v1 < 0 || v1 >= vMax) added[idx] |= (0b011001011001); else if (!isFeature[tet.v1]) added[idx] |= (0b011001011001);
		if (v2 < 0 || v2 >= vMax) added[idx] |= (0b101010101010); else if (!isFeature[tet.v2]) added[idx] |= (0b101010101010);
		if (v3 < 0 || v3 >= vMax) added[idx] |= (0b110100110100); else if (!isFeature[tet.v3]) added[idx] |= (0b110100110100);
		if (!(added[idx] & (1 << 0))) added[idx] |= (addFulledgeCounter(vertexNumSimpleFulledges, simpleFulledges, v0, v1, counters, iter) << 0);
		if (!(added[idx] & (1 << 1))) added[idx] |= (addFulledgeCounter(vertexNumSimpleFulledges, simpleFulledges, v0, v2, counters, iter) << 1);
		if (!(added[idx] & (1 << 2))) added[idx] |= (addFulledgeCounter(vertexNumSimpleFulledges, simpleFulledges, v0, v3, counters, iter) << 2);
		if (!(added[idx] & (1 << 3))) added[idx] |= (addFulledgeCounter(vertexNumSimpleFulledges, simpleFulledges, v1, v2, counters, iter) << 3);
		if (!(added[idx] & (1 << 4))) added[idx] |= (addFulledgeCounter(vertexNumSimpleFulledges, simpleFulledges, v1, v3, counters, iter) << 4);
		if (!(added[idx] & (1 << 5))) added[idx] |= (addFulledgeCounter(vertexNumSimpleFulledges, simpleFulledges, v2, v3, counters, iter) << 5);

		if (!(added[idx] & (1 << 6))) added[idx] |= (addFulledgeCounter(vertexNumSimpleFulledges, simpleFulledges, v1, v0, counters, iter) << 6);
		if (!(added[idx] & (1 << 7))) added[idx] |= (addFulledgeCounter(vertexNumSimpleFulledges, simpleFulledges, v2, v0, counters, iter) << 7);
		if (!(added[idx] & (1 << 8))) added[idx] |= (addFulledgeCounter(vertexNumSimpleFulledges, simpleFulledges, v3, v0, counters, iter) << 8);
		if (!(added[idx] & (1 << 9))) added[idx] |= (addFulledgeCounter(vertexNumSimpleFulledges, simpleFulledges, v2, v1, counters, iter) << 9);
		if (!(added[idx] & (1 << 10))) added[idx] |= (addFulledgeCounter(vertexNumSimpleFulledges, simpleFulledges, v3, v1, counters, iter) << 10);
		if (!(added[idx] & (1 << 11))) added[idx] |= (addFulledgeCounter(vertexNumSimpleFulledges, simpleFulledges, v3, v2, counters, iter) << 11);
		if (added[idx] == 0xfff) {
			atomicAdd(addedCounter, 1);
		}
	}
}

__global__ void k_initBoundary1D(int nVerticesSurf, int* vertexNumHalfedges, Halfedge* halfedges, bool* vertexIsBoundary1D) {
	//const int MAX_HE_PER_VERTEX = 128; // TODO
	for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < nVerticesSurf; idx += blockDim.x * gridDim.x) {
		//if at least one halfedge has no opposite, we have a boundary vertex
		int num = vertexNumHalfedges[idx];
		for (int j = 0; j < num; ++j) {
			Halfedge& halfedge = halfedges[idx * MAX_HE_PER_VERTEX + j];
			if (halfedge.oppositeHE == -1) {
				vertexIsBoundary1D[idx] = true;
				break;
			}
		}
	}
}

__global__ void k_initBoundary1DRowPtr(int nVerticesSurf, int* rowPtr, Halfedge* halfedges, bool* vertexIsBoundary1D) {
	for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < nVerticesSurf; idx += blockDim.x * gridDim.x) {
		//if at least one halfedge has no opposite, we have a boundary vertex
		int num = rowPtr[idx+1] - rowPtr[idx];
		for (int j = 0; j < num; ++j) {
			Halfedge& halfedge = halfedges[rowPtr[idx] + j];
			if (halfedge.oppositeHE == -1) {
				vertexIsBoundary1D[idx] = true;
				break;
			}
		}
	}
}

// adds simple Fulledge to buffer
__global__ void k_constructFulledgesFromTriangles1Pass1(int nTriangles, Triangle* triangles, int* vertexNumHalfedges, Halfedge* halfedges, bool* vertexIsBoundary1D) {
	for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < nTriangles; idx += blockDim.x * gridDim.x) {
		const Triangle& tri = triangles[idx];
		int v0 = tri.v0;
		int v1 = tri.v1;
		int v2 = tri.v2;
		addHalfedge(vertexNumHalfedges, halfedges, v0, v1, idx);
		addHalfedge(vertexNumHalfedges, halfedges, v1, v2, idx);
		addHalfedge(vertexNumHalfedges, halfedges, v2, v0, idx);
	}
}

// adds simple Fulledge to buffer
__global__ void k_constructFulledgesFromTriangles1Pass2(int nTriangles, Triangle* triangles, int* vertexNumHalfedges, Halfedge* halfedges, bool* vertexIsBoundary1D) {
	for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < nTriangles; idx += blockDim.x * gridDim.x) {
		const Triangle& tri = triangles[idx];
		int v0 = tri.v0;
		int v1 = tri.v1;
		int v2 = tri.v2;
		if (vertexIsBoundary1D[v0] && vertexIsBoundary1D[v1]) {
			addHalfedgeCheckExisting(vertexNumHalfedges, halfedges, v1, v0, -1);
		}
		if (vertexIsBoundary1D[v1] && vertexIsBoundary1D[v2]) {
			addHalfedgeCheckExisting(vertexNumHalfedges, halfedges, v2, v1, -1);
		}
		if (vertexIsBoundary1D[v2] && vertexIsBoundary1D[v0]) {
			addHalfedgeCheckExisting(vertexNumHalfedges, halfedges, v0, v2, -1);
		}
	}
}


__global__ void k_constructHalffaces(int nTetrahedra, Tetrahedron* tetrahedra, int* vertexNumTetrahedra, Halfface* vertexTetrahedra) {
	//const int MAX_TETS_PER_VERTEX = 256; // TODO
	for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < nTetrahedra; idx += blockDim.x * gridDim.x) {
		const Tetrahedron& tet = tetrahedra[idx];
		// add tet to oneRing of its vertices
		int v0 = tet.v0;
		int v1 = tet.v1;
		int v2 = tet.v2;
		int v3 = tet.v3; //TODO
		int pos0 = atomicAdd(&vertexNumTetrahedra[v0], 1);
		vertexTetrahedra[v0 * MAX_TETS_PER_VERTEX + pos0] = Halfface{ v1,v2,v3 }; // Halffaces in CCW order according to TetGen ordering
		if (vertexNumTetrahedra[v0] > MAX_TETS_PER_VERTEX) {
			printf("temp buffer too small for tetrahedra\n");
			assert(0);
		}

		int pos1 = atomicAdd(&vertexNumTetrahedra[v1], 1);
		vertexTetrahedra[v1 * MAX_TETS_PER_VERTEX + pos1] = Halfface{ v0,v3,v2 };
		if (vertexNumTetrahedra[v1] > MAX_TETS_PER_VERTEX) {
			printf("temp buffer too small for tetrahedra\n");
			assert(0);
		}

		int pos2 = atomicAdd(&vertexNumTetrahedra[v2], 1);
		vertexTetrahedra[v2 * MAX_TETS_PER_VERTEX + pos2] = Halfface{ v0,v1,v3 };
		if (vertexNumTetrahedra[v2] > MAX_TETS_PER_VERTEX) {
			printf("temp buffer too small for tetrahedra\n");
			assert(0);
		}

		int pos3 = atomicAdd(&vertexNumTetrahedra[v3], 1);
		vertexTetrahedra[v3 * MAX_TETS_PER_VERTEX + pos3] = Halfface{ v0,v2,v1 };
		if (vertexNumTetrahedra[v3] > MAX_TETS_PER_VERTEX) {
			printf("temp buffer too small for tetrahedra\n");
			assert(0);
		}
	}
}

__global__ void k_constructHalffacesNew(int nTetrahedra, Tetrahedron* tetrahedra, Halfface* halffaces, int* keys) {
	for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < nTetrahedra; idx += blockDim.x * gridDim.x) {
		const Tetrahedron& tet = tetrahedra[idx];
		int v0 = tet.v0;
		int v1 = tet.v1;
		int v2 = tet.v2;
		int v3 = tet.v3;
		keys[4 * idx + 0] = v0;
		halffaces[4 * idx + 0] = Halfface{ v1,v2,v3 }; // Halffaces in CCW order according to TetGen ordering

		keys[4 * idx + 1] = v1;
		halffaces[4 * idx + 1] = Halfface{ v0,v3,v2 };

		keys[4 * idx + 2] = v2;
		halffaces[4 * idx + 2] = Halfface{ v0,v1,v3 };

		keys[4 * idx + 3] = v3;
		halffaces[4 * idx + 3] = Halfface{ v0,v2,v1 };
	}
}

__global__ void k_constructHalfhexesNew(int nHexahedra, Hexahedron* hexahedra, Halfhex* halfhexes, int* keys) {
	for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < nHexahedra; idx += blockDim.x * gridDim.x) {
		const Hexahedron& hex = hexahedra[idx];
		int v0 = hex.v0;
		int v1 = hex.v1;
		int v2 = hex.v2;
		int v3 = hex.v3;
		int v4 = hex.v4;
		int v5 = hex.v5;
		int v6 = hex.v6;
		int v7 = hex.v7;
		keys[8 * idx + 0] = v0;
		halfhexes[8 * idx + 0] = Halfhex{ v1,v2,v3,v4,v5,v6,v7 }; // Halffaces in CCW order according to OVM ordering

		keys[8 * idx + 1] = v1;
		halfhexes[8 * idx + 1] = Halfhex{ v5,v6,v2,v0,v4,v7,v3 };

		keys[8 * idx + 2] = v2;
		halfhexes[8 * idx + 2] = Halfhex{ v6,v7,v3,v1,v5,v4,v0 };

		keys[8 * idx + 3] = v3;
		halfhexes[8 * idx + 3] = Halfhex{ v7,v4,v0,v2,v6,v5,v1 };

		keys[8 * idx + 4] = v4;
		halfhexes[8 * idx + 4] = Halfhex{ v0,v3,v7,v5,v1,v2,v6 };

		keys[8 * idx + 5] = v5;
		halfhexes[8 * idx + 5] = Halfhex{ v4,v7,v6,v1,v0,v3,v2 };

		keys[8 * idx + 6] = v6;
		halfhexes[8 * idx + 6] = Halfhex{ v5,v4,v7,v2,v1,v0,v3 };

		keys[8 * idx + 7] = v7;
		halfhexes[8 * idx + 7] = Halfhex{ v6,v5,v4,v3,v2,v1,v0 };
	}
}

__device__ inline Triangle getTriangleSmallestFirst(int a, int b, int c) {
	if (a < b && a < c) {
		return { a, b, c };
	}
	else if (b < a && b < c) {
		return { b, c, a };
	}
	else {
		return { c, a, b };
	}
}

__device__ inline Quad getQuadSmallestFirst(int a, int b, int c, int d) {
	if (a < b && a < c && a < d) {
		return { a, b, c, d };
	}
	else if (b < a && b < c && b < d) {
		return { b, c, d, a };
	}
	else if (c < a && c < b && c < d) {
		return { c, d, a, b };
	}
	else {
		return { d, a, b, c };
	}
}

__global__ void k_constructTrisFromTetrahedra(int nTetrahedra, Tetrahedron* tetrahedra, Triangle* simpleTris) {
	for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < nTetrahedra; idx += blockDim.x * gridDim.x) {
		const Tetrahedron& tet = tetrahedra[idx];
		int v0 = tet.v0;
		int v1 = tet.v1;
		int v2 = tet.v2;
		int v3 = tet.v3;
		simpleTris[4 * idx + 0] = getTriangleSmallestFirst(v2, v1, v0); // all CCW
		simpleTris[4 * idx + 1] = getTriangleSmallestFirst(v0, v1, v3);
		simpleTris[4 * idx + 2] = getTriangleSmallestFirst(v1, v2, v3);
		simpleTris[4 * idx + 3] = getTriangleSmallestFirst(v0, v3, v2);
	}
}

__global__ void k_constructQuadsFromHexahedra(int nHexahedra, Hexahedron* hexahedra, Quad* simpleQuads) {
	for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < nHexahedra; idx += blockDim.x * gridDim.x) {
		const Hexahedron& hex = hexahedra[idx];
		int v0 = hex.v0;
		int v1 = hex.v1;
		int v2 = hex.v2;
		int v3 = hex.v3;
		int v4 = hex.v4;
		int v5 = hex.v5;
		int v6 = hex.v6;
		int v7 = hex.v7;
		simpleQuads[6 * idx + 0] = getQuadSmallestFirst(v0,v1,v2,v3); // all CCW
		simpleQuads[6 * idx + 1] = getQuadSmallestFirst(v1,v5,v6,v2);
		simpleQuads[6 * idx + 2] = getQuadSmallestFirst(v3,v2,v6,v7);
		simpleQuads[6 * idx + 3] = getQuadSmallestFirst(v4,v0,v3,v7);
		simpleQuads[6 * idx + 4] = getQuadSmallestFirst(v1,v0,v4,v5);
		simpleQuads[6 * idx + 5] = getQuadSmallestFirst(v5,v4,v7,v6);
	}
}

__global__ void k_constructSurfaceTrisFromTetrahedra(int nTetrahedra, Tetrahedron* tetrahedra, Triangle* trianglesOut, bool* vertexIsBoundary2D, int* pos_counter) {
	for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < nTetrahedra; idx += blockDim.x * gridDim.x) {
		const Tetrahedron& tet = tetrahedra[idx];
		int v0 = tet.v0;
		int v1 = tet.v1;
		int v2 = tet.v2;
		int v3 = tet.v3;

		unsigned boundaryBitmap = 0; // v3 v2 v1 v0
		boundaryBitmap |= ((unsigned)vertexIsBoundary2D[v0]);
		boundaryBitmap |= ((unsigned)vertexIsBoundary2D[v1] << 1);
		boundaryBitmap |= ((unsigned)vertexIsBoundary2D[v2] << 2);
		boundaryBitmap |= ((unsigned)vertexIsBoundary2D[v3] << 3);
		if ((boundaryBitmap & 0b0111) == 0b0111) {
			Triangle tri{ v2, v1, v0 }; //This is CCW according to TetGen docs (Node ordering)
			int pos = atomicAdd(pos_counter, 1);
			trianglesOut[pos] = tri;
		}
		if ((boundaryBitmap & 0b1011) == 0b1011) {
			Triangle tri{ v0, v1, v3 }; //This is CCW according to TetGen docs (Node ordering)
			int pos = atomicAdd(pos_counter, 1);
			trianglesOut[pos] = tri;
		}
		if ((boundaryBitmap & 0b1110) == 0b1110) {
			Triangle tri{ v1, v2, v3 }; //This is CCW according to TetGen docs (Node ordering)
			int pos = atomicAdd(pos_counter, 1);
			trianglesOut[pos] = tri;
		}
		if ((boundaryBitmap & 0b1101) == 0b1101) {
			Triangle tri{ v0, v3, v2 }; //This is CCW according to TetGen docs (Node ordering)
			int pos = atomicAdd(pos_counter, 1);
			trianglesOut[pos] = tri;
		}
	}
}


// for inner verts
__global__ void k_checkColoring(int nTetrahedra, Tetrahedron* tetrahedra, int startNode, int nNodes, int* coloring) {
	for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < nTetrahedra; idx += blockDim.x * gridDim.x) {
		const Tetrahedron& tet = tetrahedra[idx];
		int v0 = tet.v0 - startNode;
		int v1 = tet.v1 - startNode;
		int v2 = tet.v2 - startNode;
		int v3 = tet.v3 - startNode;
		//printf("nnodes %i colorcheck %i %i %i %i cols %i %i %i %i\n", nNodes, v0, v1, v2, v3, coloring[v0], coloring[v1], coloring[v2], coloring[v3]);
		unsigned testBitmap = 0;
		if (v0 >= 0 && v0 < nNodes) {
			testBitmap |= 0b0001;
		}
		if (v1 >= 0 && v1 < nNodes) {
			testBitmap |= 0b0010;
		}
		if (v2 >= 0 && v2 < nNodes) {
			testBitmap |= 0b0100;
		}
		if (v3 >= 0 && v3 < nNodes) {
			testBitmap |= 0b1000;
		}
		//printf("bitmap %i\n", testBitmap);

		if ((testBitmap & 0b0011) == 0b0011) {
			if (coloring[v0] == coloring[v1]) {
				printf("%i and %i same color %i\n", v0, v1, coloring[v0]);
				assert(0);
			}
		}
		if ((testBitmap & 0b0101) == 0b0101) {
			if (coloring[v0] == coloring[v2]) {
				printf("%i and %i same color %i\n", v0, v2, coloring[v0]);
				assert(0);
			}
		}
		if ((testBitmap & 0b1001) == 0b1001) {
			if (coloring[v0] == coloring[v3]) {
				printf("%i and %i same color %i\n", v0, v3, coloring[v0]);
				assert(0);
			}
		}
		if ((testBitmap & 0b0110) == 0b0110) {
			if (coloring[v1] == coloring[v2]) {
				printf("%i and %i same color %i\n", v1, v2, coloring[v1]);
				assert(0);
			}
		}
		if ((testBitmap & 0b1010) == 0b1010) {
			if (coloring[v1] == coloring[v3]) {
				printf("%i and %i same color %i\n", v1, v3, coloring[v1]);
				assert(0);
			}
		}
		if ((testBitmap & 0b1100) == 0b1100) {
			if (coloring[v2] == coloring[v3]) {
				printf("%i and %i same color %i\n", v2, v3, coloring[v2]);
				assert(0);
			}
		}
	}
}

__global__ void k_checkColoring(int nTriangles, Triangle* triangles, int startNode, int nNodes, int* coloring) {
	for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < nTriangles; idx += blockDim.x * gridDim.x) {
		const Triangle& tri = triangles[idx];
		int v0 = tri.v0 - startNode;
		int v1 = tri.v1 - startNode;
		int v2 = tri.v2 - startNode;
		//printf("nnodes %i colorcheck %i %i %i %i cols %i %i %i %i\n", nNodes, v0, v1, v2, v3, coloring[v0], coloring[v1], coloring[v2], coloring[v3]);
		unsigned testBitmap = 0;
		if (v0 >= 0 && v0 < nNodes) {
			testBitmap |= 0b001;
		}
		if (v1 >= 0 && v1 < nNodes) {
			testBitmap |= 0b010;
		}
		if (v2 >= 0 && v2 < nNodes) {
			testBitmap |= 0b100;
		}
		//printf("bitmap %i\n", testBitmap);

		if ((testBitmap & 0b011) == 0b011) {
			if (coloring[v0] == coloring[v1]) {
				printf("%i and %i same color %i\n", v0, v1, coloring[v0]);
				assert(0);
			}
		}
		if ((testBitmap & 0b101) == 0b101) {
			if (coloring[v0] == coloring[v2]) {
				printf("%i and %i same color %i\n", v0, v2, coloring[v0]);
				assert(0);
			}
		}
		if ((testBitmap & 0b110) == 0b110) {
			if (coloring[v1] == coloring[v2]) {
				printf("%i and %i same color %i\n", v1, v2, coloring[v1]);
				assert(0);
			}
		}
	}
}

__global__ void k_checkColoring(int nTriangles, Quad* quads, int startNode, int nNodes, int* coloring) {
	for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < nTriangles; idx += blockDim.x * gridDim.x) {
		const Quad& tri = quads[idx];
		int v0 = tri.v0 - startNode;
		int v1 = tri.v1 - startNode;
		int v2 = tri.v2 - startNode;
		int v3 = tri.v3 - startNode;
		unsigned testBitmap = 0;
		if (v0 >= 0 && v0 < nNodes) {
			testBitmap |= 0b0001;
		}
		if (v1 >= 0 && v1 < nNodes) {
			testBitmap |= 0b0010;
		}
		if (v2 >= 0 && v2 < nNodes) {
			testBitmap |= 0b0100;
		}
		if (v3 >= 0 && v3 < nNodes) {
			testBitmap |= 0b1000;
		}

		if ((testBitmap & 0b0011) == 0b0011) {
			if (coloring[v0] == coloring[v1]) {
				printf("%i and %i same color %i\n", v0, v1, coloring[v0]);
				assert(0);
			}
		}
		if ((testBitmap & 0b0110) == 0b0110) {
			if (coloring[v1] == coloring[v2]) {
				printf("%i and %i same color %i\n", v1, v2, coloring[v1]);
				assert(0);
			}
		}
		if ((testBitmap & 0b1100) == 0b1100) {
			if (coloring[v2] == coloring[v3]) {
				printf("%i and %i same color %i\n", v2, v3, coloring[v2]);
				assert(0);
			}
		}
		if ((testBitmap & 0b1001) == 0b1001) {
			if (coloring[v3] == coloring[v0]) {
				printf("%i and %i same color %i\n", v0, v3, coloring[v0]);
				assert(0);
			}
		}
	}
}

__global__ void k_checkColoring(int nTetrahedra, Hexahedron* hexahedra, int startNode, int nNodes, int* coloring) {
	for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < nTetrahedra; idx += blockDim.x * gridDim.x) {
		const Hexahedron& tet = hexahedra[idx];
		int v0 = tet.v0 - startNode;
		int v1 = tet.v1 - startNode;
		int v2 = tet.v2 - startNode;
		int v3 = tet.v3 - startNode;
		int v4 = tet.v4 - startNode;
		int v5 = tet.v5 - startNode;
		int v6 = tet.v6 - startNode;
		int v7 = tet.v7 - startNode;
		//printf("nnodes %i colorcheck %i %i %i %i cols %i %i %i %i\n", nNodes, v0, v1, v2, v3, coloring[v0], coloring[v1], coloring[v2], coloring[v3]);
		unsigned testBitmap = 0;
		if (v0 >= 0 && v0 < nNodes) {
			testBitmap |= 0b00000001;
		}
		if (v1 >= 0 && v1 < nNodes) {
			testBitmap |= 0b00000010;
		}
		if (v2 >= 0 && v2 < nNodes) {
			testBitmap |= 0b00000100;
		}
		if (v3 >= 0 && v3 < nNodes) {
			testBitmap |= 0b00001000;
		}
		if (v4 >= 0 && v4 < nNodes) {
			testBitmap |= 0b00010000;
		}
		if (v5 >= 0 && v5 < nNodes) {
			testBitmap |= 0b00100000;
		}
		if (v6 >= 0 && v6 < nNodes) {
			testBitmap |= 0b01000000;
		}
		if (v7 >= 0 && v7 < nNodes) {
			testBitmap |= 0b10000000;
		}
		//printf("bitmap %i\n", testBitmap);

		if ((testBitmap & 0b00000011) == 0b00000011) {
			if (coloring[v0] == coloring[v1]) {
				printf("%i and %i same color %i\n", v0, v1, coloring[v0]);
				assert(0);
			}
		}
		if ((testBitmap & 0b00000110) == 0b00000110) {
			if (coloring[v2] == coloring[v1]) {
				printf("same color\n");
				assert(0);
			}
		}
		if ((testBitmap & 0b00001100) == 0b00001100) {
			if (coloring[v3] == coloring[v2]) {
				printf("same color\n");
				assert(0);
			}
		}
		if ((testBitmap & 0b00001001) == 0b00001001) {
			if (coloring[v0] == coloring[v3]) {
				printf("same color\n");
				assert(0);
			}
		}
		if ((testBitmap & 0b00010001) == 0b00010001) {
			if (coloring[v0] == coloring[v4]) {
				printf("same color\n");
				assert(0);
			}
		}
		if ((testBitmap & 0b00100010) == 0b00100010) {
			if (coloring[v1] == coloring[v5]) {
				printf("same color\n");
				assert(0);
			}
		}
		if ((testBitmap & 0b01000100) == 0b01000100) {
			if (coloring[v2] == coloring[v6]) {
				printf("same color\n");
				assert(0);
			}
		}
		if ((testBitmap & 0b10001000) == 0b10001000) {
			if (coloring[v3] == coloring[v7]) {
				printf("same color\n");
				assert(0);
			}
		}
		if ((testBitmap & 0b00110000) == 0b00110000) {
			if (coloring[v4] == coloring[v5]) {
				printf("same color\n");
				assert(0);
			}
		}
		if ((testBitmap & 0b01100000) == 0b01100000) {
			if (coloring[v5] == coloring[v6]) {
				printf("same color\n");
				assert(0);
			}
		}
		if ((testBitmap & 0b11000000) == 0b11000000) {
			if (coloring[v6] == coloring[v7]) {
				printf("same color\n");
				assert(0);
			}
		}
		if ((testBitmap & 0b10010000) == 0b10010000) {
			if (coloring[v7] == coloring[v4]) {
				printf("same color\n");
				assert(0);
			}
		}
	}
}



#pragma once


#include "MeshTriGPU.h"
#include "MeshGPUCommon.h"
#include "CompressedHalfedges.h"



//#############################################################################
//############################### MeshTriDevice ###############################
//#############################################################################
class MeshBaseDevice {
// all data needed in GPU memory
public:
	int nVerticesFeature;
	int nVerticesSurf;
	int nHalfedges;
	int nColorsFree;
	int nColorsFeature;

	int nVerticesSurfFree;
	bool isFlat;

	Vec3f* vertexPoints; // size: nVerticesSurf
	bool* vertexIsBoundary1D; // size: nVerticesSurf
	Vec3f* vertexNormals; // size: nVerticesSurf
	bool* vertexIsFeature; // size: nVerticesSurf

	Vec3f* faceNormals; // size: nTriangles

	CompressedStorage<Halfedge> halfedges;
	

public:
	//__host__ __device__ MeshTriDevice() {}

	__host__ MeshBaseDevice(MeshBaseGPU& m) :
		nVerticesFeature(m.nVerticesFeature),
		nVerticesSurf(m.nVerticesSurf),
		nHalfedges(m.nHalfedges),
		nColorsFree(m.nColorsFree),
		nColorsFeature(m.nColorsFeature),
		nVerticesSurfFree(nVerticesSurf - nVerticesFeature),
		isFlat(m.isFlat),
		vertexPoints(m.vertexPoints.get()),
		vertexIsBoundary1D(m.vertexIsBoundary1D.get()),
		vertexNormals(m.vertexNormals.get()),
		vertexIsFeature(m.vertexIsFeature.get()),
		faceNormals(m.faceNormals.get()),
		halfedges(m.csrHalfedgeRowPtr.get(), m.csrHalfedgesValues.get())
	{
		
	}

	__device__ CompressedStorage<Halfedge>::Iterator he_ccw_order_begin(int vid) {
		return halfedges.begin(vid);
	}
	__device__ CompressedStorage<Halfedge>::Iterator he_ccw_order_end(int vid) {
		return halfedges.end(vid);
	}

	__device__ CompressedStorage<Halfedge>::RIterator he_ccw_order_rbegin(int vid) {
		return halfedges.rbegin(vid);
	}
	__device__ CompressedStorage<Halfedge>::RIterator he_ccw_order_rend(int vid) {
		return halfedges.rend(vid);
	}

	__device__ bool isFeature(int vid) {
		return nVerticesSurfFree <= vid && vid < nVerticesSurf;
	}
	
};


//#############################################################################
//############################### DMOMeshBase #################################
//#############################################################################
class DMOMeshBase {
public:
	int nVertices;
	int nVerticesFeature;
	int nVerticesSurf;
	//int nTriangles;
	int nHalfedges;
	int nColorsFree;
	int nColorsFeature;
	int nVerticesSurfFree;

	host_vector<int> colorOffsetsFree; //start vertex of col0, ..., start vertex of coln, numFreeVertices // size: nColorsFree + 1
	host_vector<int> colorOffsetsFeature; //start vertex of col0, ..., start vertex of coln, numVerticesSurf // size nColorsFeature + 1
	int maxNumHalfedges;
	bool isFlat_;

	// host mesh data
	device_ptr<Vec3f> vertexPoints; // size: nVerticesSurf
	device_ptr<bool> vertexIsBoundary1D; // size: nVerticesSurf
	device_ptr<Vec3f> vertexNormals; // size: nVerticesSurf
	device_ptr<bool> vertexIsFeature; // size: nVerticesSurf
	//device_ptr<Triangle> triangles; // size: nTriangles
	device_ptr<Vec3f> faceNormals; // size: nTriangles
	//CompressedStorage<Halfedge> halfedges;
	//CompressedHalfedges chalfedges;
	CompressedStorage<Halfedge> halfedges;

	__host__ DMOMeshBase(MeshBaseGPU& m) :
		nVertices(m.nVerticesSurf),
		nVerticesFeature(m.nVerticesFeature),
		nVerticesSurf(m.nVerticesSurf),
		//nTriangles(m.nTriangles),
		nHalfedges(m.nHalfedges),
		nColorsFree((int)m.col_offsets_free.size() - 1),
		nColorsFeature((int)m.col_offsets_feature.size() - 1),
		nVerticesSurfFree(nVerticesSurf - nVerticesFeature),
		colorOffsetsFree(m.col_offsets_free.begin(), m.col_offsets_free.end()),
		colorOffsetsFeature(m.col_offsets_feature.begin(), m.col_offsets_feature.end()),
		maxNumHalfedges(m.maxNumHalfedges),
		isFlat_(m.isFlat),
		vertexPoints(m.vertexPoints.ptr()),
		vertexIsBoundary1D(m.vertexIsBoundary1D.ptr()),
		vertexNormals(m.vertexNormals.ptr()),
		vertexIsFeature(m.vertexIsFeature.ptr()),
		//triangles(m.triangles.ptr()),
		faceNormals(m.faceNormals.ptr()),
		halfedges(m.csrHalfedgeRowPtr.get(), m.csrHalfedgesValues.get())
		//chalfedges(m.csrHalfedgeRowPtr.get(), m.soaHalfedgeTarget.get(), m.soaHalfedgeOpposite.get(), m.soaHalfedgeFace.get())
	{
		
	}

	virtual __host__ ~DMOMeshBase() {
		gpuErrchk(cudaFree(vertexPoints.get()));
		gpuErrchk(cudaFree(vertexIsBoundary1D.get()));
		gpuErrchk(cudaFree(vertexNormals.get()));
		gpuErrchk(cudaFree(vertexIsFeature.get()));
		//gpuErrchk(cudaFree(triangles.get()));
		gpuErrchk(cudaFree(faceNormals.get()));
		gpuErrchk(cudaFree(halfedges.values_));
		//gpuErrchk(cudaFree(halfedges.colInd_));
		gpuErrchk(cudaFree(halfedges.rowPtr_));
	}

	__host__ virtual void updateNormals() = 0;

	virtual __host__ Vec3f* getVertexPoints() {
		return vertexPoints.get();
	}
	virtual __host__ Vec3f* getVertexNormals() {
		return vertexNormals.get();
	}
	virtual __host__ Triangle* getTriangles() = 0;
	virtual __host__ Quad* getQuads() = 0;
	virtual __host__ void* getFaces() = 0;

	virtual __host__ std::array<int, 2> getColorOffsets(int c) {
		if (c < nColorsFree) {
			return std::array<int, 2>{colorOffsetsFree[c], colorOffsetsFree[c + 1]};
		}
		else if (c < nColorsFree + nColorsFeature) {
			c -= nColorsFree;
			return std::array<int, 2>{colorOffsetsFeature[c], colorOffsetsFeature[c + 1]};
		}
		return std::array<int, 2>{0, 0};
	}

	virtual __host__ bool isFlat() {
		return isFlat_;
	}

	virtual __host__ int nFaces() = 0;
};





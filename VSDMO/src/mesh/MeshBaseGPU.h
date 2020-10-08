#pragma once


#include <vector>
#include <array>
#include <thrust/device_vector.h>
#include "MeshTypes.h"
#include "ConfigUsing.h"
#include "CSR.h"
#include "CudaUtil.h"
#include "MeshGPUCommon.h"



struct MeshBaseGPU {
	MeshBaseGPU() :
		findBoundary1D(true),
		nVerticesFeature(0),
		nVerticesSurf(0),
		nHalfedges(0),
		nColorsFree(0),
		nColorsFeature(0),
		maxNumHalfedges(-1),
		isFlat(false)
	{

	}

	bool findBoundary1D;

	int nVerticesFeature;
	int nVerticesSurf;
	//int nTriangles;
	int nHalfedges;

	int nColorsFree;
	int nColorsFeature;

	CudaArray<Vec3f> vertexPoints; // size: nVerticesSurf
	CudaArray<bool> vertexIsBoundary1D; // size: nVerticesSurf
	CudaArray<Vec3f> vertexNormals; // size: nVerticesSurf
	CudaArray<bool> vertexIsFeature; // size: nVerticesSurf
	
	//CudaArray<Triangle> triangles; // size: nTriangles
	CudaArray<Vec3f> faceNormals; // size: nTriangles

	CudaArray<Halfedge> csrHalfedgesValues; // size: nHalfedges
	//CudaArray<int> csrHalfedgeColInd; // size: nHalfedges
	CudaArray<int> csrHalfedgeRowPtr; // size: nVerticesSurf + 1

	CudaArray<int> col_offsets_free; //start vertex of col0, ..., start vertex of coln, numFreeVertices // size: nColorsFree + 1
	CudaArray<int> col_offsets_feature; //start vertex of col0, ..., start vertex of coln, numVerticesSurf // size nColorsFeature + 1

	int maxNumHalfedges;
	bool isFlat;
};




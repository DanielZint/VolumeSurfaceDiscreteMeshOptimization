#pragma once

#include "MeshTriGPU.h"


class MeshTetGPU : public MeshTriGPU {
public:
	__host__ MeshTetGPU();
	__host__ ~MeshTetGPU();
	
	__host__ void setNumVertices(int nV);
	__host__ void setNumTetrahedra(int nT);
	__host__ void setVertexPointsWithBoundary2D(vector<Vec3f>& points, vector<bool>& boundary2d);
	__host__ void setTetrahedra(vector<Tetrahedron>& tets);

	__host__ void init() override;

	__host__ void initBoundary2D();
	__host__ void colorInnerVerticesAndSort();
	__host__ void constructTriangles();
	__host__ void constructHalffaces();
	__host__ void constructHalffacesNew();

public:
	int MAX_TETS_PER_VERTEX = 256;

	bool findBoundary2D;
	int nVertices;
	int nTetrahedra;
	int nColorsInner;
	
	CudaArray<bool> vertexIsBoundary2D;
	
	CudaArray<Tetrahedron> tetrahedra;

	CudaArray<Halfface> csrVertexTetrahedraValues;
	CudaArray<int> csrVertexTetrahedraRowPtr;

	CudaArray<int> col_offsets_inner; //start vertex of col0, ..., start vertex of coln, numFreeVertices // size: nColorsFree + 1

	int maxNumTetrahedra;

private:
	__host__ int removeNonBoundaryTriangles(device_vector<Triangle>& trianglesIn);
	__host__ void makeCSRVertexTetrahedra(device_vector<int>& vertexNumTetrahedra, device_vector<Halfface>& vertexTetrahedra);
	__host__ int sortVerticesByBoundary2D(device_vector<int>& sortMapInverseOut);
	__host__ int sortInnerVerticesRangeByColor(int start, int num, device_vector<int>& colors, CudaArray<int>& col_offsets, device_vector<int>& sortMapInverseOut, bool doReindex);
	__host__ void constructFulledgesFromTetrahedraGPU(device_vector<int>& vertexNumSimpleFulledges, device_vector<int>& simpleFulledges);
	__host__ void constructFulledgesFromTetrahedraGPUOnlyInner(device_vector<int>& vertexNumSimpleFulledges, device_vector<int>& simpleFulledges);
	__host__ void remapElements(device_vector<int>& sortMapInverse) override;
	//__host__ void constructSimpleFulledges(device_vector<int>& vertexNumSimpleFulledges, device_vector<int>& simpleFulledges) override;
	//__host__ void makeSimpleFulledgeVectors(device_vector<int>& vertexNumSimpleFulledges, device_vector<int>& simpleFulledges) override;
	__host__ void checkColorsFree(device_vector<int>& colors) override;
	__host__ void checkColorsFeature(device_vector<int>& colors) override;

	__host__ void constructSimpleFulledgesFree(device_vector<int>& vertexNumSimpleFulledges, device_vector<int>& simpleFulledges) override;
	__host__ void constructSimpleFulledgesFeature(device_vector<int>& vertexNumSimpleFulledges, device_vector<int>& simpleFulledges) override;

	__host__ void constructSimpleFulledgesInnerNew(device_vector<int>& simpleFulledges, device_vector<int>& rowPtr);
	__host__ void constructSimpleFulledgesFreeNew(device_vector<int>& simpleFulledges, device_vector<int>& rowPtr) override;
	__host__ void constructSimpleFulledgesFeatureNew(device_vector<int>& simpleFulledges, device_vector<int>& rowPtr) override;
};







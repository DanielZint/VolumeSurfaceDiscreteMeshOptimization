#pragma once

#include "MeshQuadGPU.h"


class MeshHexGPU : public MeshQuadGPU {
public:
	__host__ MeshHexGPU();
	__host__ ~MeshHexGPU();
	
	__host__ void setNumVertices(int nV);
	__host__ void setNumHexahedra(int nT);
	__host__ void setVertexPointsWithBoundary2D(vector<Vec3f>& points, vector<bool>& boundary2d);
	__host__ void setHexahedra(vector<Hexahedron>& hexes);

	__host__ void init() override;

	__host__ void initBoundary2D();
	__host__ void colorInnerVerticesAndSort();
	__host__ void constructQuads();
	__host__ void constructHalfhexesNew();

public:

	bool findBoundary2D;
	int nVertices;
	int nHexahedra;
	int nColorsInner;
	
	CudaArray<bool> vertexIsBoundary2D;
	
	CudaArray<Hexahedron> hexahedra;

	CudaArray<Halfhex> csrVertexHexahedraValues;
	CudaArray<int> csrVertexHexahedraRowPtr;

	CudaArray<int> col_offsets_inner; //start vertex of col0, ..., start vertex of coln, numFreeVertices // size: nColorsFree + 1

	int maxNumHexahedra;

private:
	__host__ int removeNonBoundaryQuads(device_vector<Quad>& quadsIn);
	__host__ int sortVerticesByBoundary2D(device_vector<int>& sortMapInverseOut);
	__host__ int sortInnerVerticesRangeByColor(int start, int num, device_vector<int>& colors, CudaArray<int>& col_offsets, device_vector<int>& sortMapInverseOut, bool doReindex);
	__host__ void remapElements(device_vector<int>& sortMapInverse) override;
	
	__host__ void checkColorsFree(device_vector<int>& colors) override;
	__host__ void checkColorsFeature(device_vector<int>& colors) override;

	__host__ void constructSimpleFulledgesInnerNew(device_vector<int>& simpleFulledges, device_vector<int>& rowPtr);
	__host__ void constructSimpleFulledgesFreeNew(device_vector<int>& simpleFulledges, device_vector<int>& rowPtr) override;
	__host__ void constructSimpleFulledgesFeatureNew(device_vector<int>& simpleFulledges, device_vector<int>& rowPtr) override;
};







#pragma once


#include <vector>
#include <array>
#include <thrust/device_vector.h>
#include "MeshTypes.h"
#include "ConfigUsing.h"
#include "CSR.h"
#include "CudaUtil.h"
#include "MeshGPUCommon.h"
#include "MeshBaseGPU.h"



class MeshQuadGPU : public MeshBaseGPU {
public:
	__host__ MeshQuadGPU();
	__host__ ~MeshQuadGPU();
	
	__host__ void setNumVerticesSurf(int n);
	__host__ void setNumQuads(int n);
	__host__ void setVertexPoints(vector<Vec3f>& points, bool nonZeroZ = true);
	__host__ void setVertexPointsWithBoundary1D(vector<Vec3f>& points, vector<bool>& boundary1d);
	__host__ void setQuads(vector<Quad>& quads);
	__host__ void fromDeviceData(int nVS, int nQ, CudaArray<Vec3f>& points, CudaArray<Quad>& quads);
	__host__ void fromDeviceData(int nVS, int nQ, device_ptr<Vec3f> points, device_ptr<Quad> quads);

	__host__ virtual void init();
	
	__host__ void updateNormals();
	
public:
	int MAX_HE_PER_VERTEX = 128;

	int nQuads;
	CudaArray<Quad> quads; // size: nTriangles
	//CudaArray<Halfedge> csrHalfedgesValues; // size: nHalfedges

protected:
	__host__ void calcFaceNormals();
	__host__ void initBoundary1DAndFeaturesNew();
	__host__ void colorVerticesAndSort();
	__host__ void constructTriHalfedgesNew();
	__host__ void calcVertexNormals();

	__host__ int sortSurfVerticesByFeature(device_vector<int>& sortMapInverseOut);
	__host__ int sortVerticesRangeByColor(int start, int num, device_vector<int>& colors, CudaArray<int>& col_offsets, device_vector<int>& sortMapInverseOut, bool doReindex);

	__host__ virtual void remapElements(device_vector<int>& sortMapInverse); // For Tetmesh remap tris AND tets
	__host__ virtual void checkColorsFree(device_vector<int>& colors);
	__host__ virtual void checkColorsFeature(device_vector<int>& colors);

	//new
	__host__ void makeSimpleFulledgeFreeVectors(device_vector<int>& vertexNumSimpleFulledges, device_vector<int>& simpleFulledges);
	__host__ void makeSimpleFulledgeFeatureVectors(device_vector<int>& vertexNumSimpleFulledges, device_vector<int>& simpleFulledges);

	__host__ virtual void constructSimpleFulledgesFreeNew(device_vector<int>& simpleFulledges, device_vector<int>& rowPtr);
	__host__ virtual void constructSimpleFulledgesFeatureNew(device_vector<int>& simpleFulledges, device_vector<int>& rowPtr);

};




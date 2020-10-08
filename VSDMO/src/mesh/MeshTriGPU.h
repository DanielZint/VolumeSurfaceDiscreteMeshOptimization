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



class MeshTriGPU : public MeshBaseGPU {
public:
	__host__ MeshTriGPU();
	__host__ ~MeshTriGPU();
	
	__host__ void setNumVerticesSurf(int n);
	__host__ void setNumTriangles(int n);
	__host__ void setVertexPoints(vector<Vec3f>& points, bool nonZeroZ = true);
	__host__ void setVertexPointsWithBoundary1D(vector<Vec3f>& points, vector<bool>& boundary1d);
	__host__ void setTriangles(vector<Triangle>& tris);
	__host__ void fromDeviceData(int nVS, int nT, CudaArray<Vec3f>& points, CudaArray<Triangle>& tris);
	__host__ void fromDeviceData(int nVS, int nT, device_ptr<Vec3f> points, device_ptr<Triangle> tris);

	__host__ virtual void init();
	
	__host__ void updateNormals();
	
public:
	int MAX_HE_PER_VERTEX = 128;

	int nTriangles;
	CudaArray<Triangle> triangles; // size: nTriangles
	//CudaArray<Halfedge> csrHalfedgesValues; // size: nHalfedges
	

protected:
	__host__ void calcFaceNormals();
	__host__ void initBoundary1DAndFeatures();
	__host__ void initBoundary1DAndFeaturesNew();
	__host__ void colorVerticesAndSort();
	__host__ void constructTriHalfedges();
	__host__ void constructTriHalfedgesNew();
	__host__ void calcVertexNormals();
	__host__ void makeHalfedgesSoA();

	//__host__ void initBoundary1D();
	//__host__ void initFeatures();

	//__host__ void constructHalfedgesGPU(device_vector<int>& vertexNumHalfedges, device_vector<Halfedge>& halfedges);

	__host__ void constructHalfedgesWithDummyOpposite(device_vector<int>& vertexNumHalfedges, device_vector<Halfedge>& halfedges);

	//__host__ void constructFulledgesFromTrianglesGPU(device_vector<int>& vertexNumSimpleFulledges, device_vector<int>& simpleFulledges);
	__host__ void constructFulledgesFromTrianglesGPUOnlyFree(device_vector<int>& vertexNumSimpleFulledges, device_vector<int>& simpleFulledges);
	__host__ void constructFulledgesFromTrianglesGPUOnlyFeature(device_vector<int>& vertexNumSimpleFulledges, device_vector<int>& simpleFulledges);

	__host__ void makeCSRHalfedges(device_vector<int>& vertexNumHalfedges, device_vector<Halfedge>& halfedges);

	__host__ int sortSurfVerticesByFeature(device_vector<int>& sortMapInverseOut);
	__host__ int sortVerticesRangeByColor(int start, int num, device_vector<int>& colors, CudaArray<int>& col_offsets, device_vector<int>& sortMapInverseOut, bool doReindex);

	__host__ virtual void remapElements(device_vector<int>& sortMapInverse); // For Tetmesh remap tris AND tets
	//__host__ virtual void constructSimpleFulledges(device_vector<int>& vertexNumSimpleHalfedges, device_vector<int>& simpleHalfedges);
	//__host__ virtual void makeSimpleFulledgeVectors(device_vector<int>& vertexNumSimpleFulledges, device_vector<int>& simpleFulledges); // For Tetmesh nVertices >= nVerticesSurf, so need larger vectors
	__host__ virtual void checkColorsFree(device_vector<int>& colors);
	__host__ virtual void checkColorsFeature(device_vector<int>& colors);

	//new
	__host__ void makeSimpleFulledgeFreeVectors(device_vector<int>& vertexNumSimpleFulledges, device_vector<int>& simpleFulledges);
	__host__ void makeSimpleFulledgeFeatureVectors(device_vector<int>& vertexNumSimpleFulledges, device_vector<int>& simpleFulledges);

	__host__ virtual void constructSimpleFulledgesFree(device_vector<int>& vertexNumSimpleFulledges, device_vector<int>& simpleFulledges);
	__host__ virtual void constructSimpleFulledgesFeature(device_vector<int>& vertexNumSimpleFulledges, device_vector<int>& simpleFulledges);

	__host__ virtual void constructSimpleFulledgesFreeNew(device_vector<int>& simpleFulledges, device_vector<int>& rowPtr);
	__host__ virtual void constructSimpleFulledgesFeatureNew(device_vector<int>& simpleFulledges, device_vector<int>& rowPtr);

};




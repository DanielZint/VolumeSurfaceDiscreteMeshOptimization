#pragma once


#include "MeshTriGPU.h"
#include "MeshGPUCommon.h"
#include "CompressedHalfedges.h"
#include "DMOMeshBase.h"



//#############################################################################
//############################### MeshTriDevice ###############################
//#############################################################################
class MeshTriDevice : public MeshBaseDevice {
// all data needed in GPU memory
public:

	int nTriangles;
	Triangle* triangles; // size: nTriangles
	//CompressedStorage<Halfedge> halfedges;
	
public:
	__host__ MeshTriDevice(MeshTriGPU& m) :
		MeshBaseDevice(m),
		nTriangles(m.nTriangles),
		triangles(m.triangles.get())
		//halfedges(m.csrHalfedgeRowPtr.get(), m.csrHalfedgesValues.get())
	{
		
	}
	
};



class DMOMeshBaseTri : public DMOMeshBase {
public:
	int nTriangles;
	device_ptr<Triangle> triangles; // size: nTriangles
	//CompressedStorage<Halfedge> halfedges;

	__host__ DMOMeshBaseTri(MeshTriGPU& m) : DMOMeshBase(m), nTriangles(m.nTriangles), triangles(m.triangles.ptr()) {
		
	}

	__host__ ~DMOMeshBaseTri() {
		gpuErrchk(cudaFree(triangles.get()));
	}

	virtual __host__ Triangle* getTriangles() override {
		return triangles.get();
	}

	virtual __host__ Quad* getQuads() override {
		return nullptr;
	}

	virtual __host__ void* getFaces() override {
		return (void*)triangles.get();
	}

	__host__ virtual void updateNormals() override {
		const int BLOCK_SIZE = 128;
		k_calcFaceNormals << <getBlockCount(nTriangles, BLOCK_SIZE), BLOCK_SIZE >> > (nTriangles, vertexPoints.get(), faceNormals.get(), triangles.get());
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
		k_calcVertexNormals << <getBlockCount(nVerticesSurf, BLOCK_SIZE), BLOCK_SIZE >> > (nVerticesSurf, vertexNormals.get(), halfedges.values_, halfedges.rowPtr_, faceNormals.get());
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
	}

	virtual __host__ int nFaces() override {
		return nTriangles;
	}
};


//#############################################################################
//############################### DMOMeshTri ##################################
//#############################################################################
class DMOMeshTri : public DMOMeshBaseTri {
public:
	MeshTriDevice* d_mesh;

	__host__ DMOMeshTri(MeshTriGPU& m) : DMOMeshBaseTri(m) {
		MeshTriDevice h_meshDevice(m);
		gpuErrchk(cudaMalloc((void**)&d_mesh, sizeof(MeshTriDevice)));
		gpuErrchk(cudaMemcpy(d_mesh, &h_meshDevice, sizeof(MeshTriDevice), cudaMemcpyHostToDevice));
	}

	__host__ ~DMOMeshTri() {
		gpuErrchk(cudaFree(d_mesh));
	}
};




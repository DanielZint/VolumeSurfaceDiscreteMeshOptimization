#pragma once


#include "MeshQuadGPU.h"
#include "MeshGPUCommon.h"
#include "CompressedHalfedges.h"
#include "DMOMeshBase.h"



//#############################################################################
//############################### MeshQuadDevice ###############################
//#############################################################################
class MeshQuadDevice : public MeshBaseDevice {
	// all data needed in GPU memory
public:

	int nQuads;
	Quad* quads; // size: nQuads

public:
	__host__ MeshQuadDevice(MeshQuadGPU& m) :
		MeshBaseDevice(m),
		nQuads(m.nQuads),
		quads(m.quads.get())
	{

	}

};



class DMOMeshBaseQuad : public DMOMeshBase {
public:
	int nQuads;
	device_ptr<Quad> quads; // size: nQuads

	__host__ DMOMeshBaseQuad(MeshQuadGPU& m) : DMOMeshBase(m), nQuads(m.nQuads), quads(m.quads.ptr()) {

	}

	__host__ ~DMOMeshBaseQuad() {
		gpuErrchk(cudaFree(quads.get()));
	}

	virtual __host__ Triangle* getTriangles() override {
		return nullptr;
	}

	virtual __host__ Quad* getQuads() override {
		return quads.get();
	}

	virtual __host__ void* getFaces() override {
		return (void*)quads.get();
	}

	__host__ virtual void updateNormals() override {
		const int BLOCK_SIZE = 128;
		k_calcFaceNormals << <getBlockCount(nQuads, BLOCK_SIZE), BLOCK_SIZE >> > (nQuads, vertexPoints.get(), faceNormals.get(), quads.get());
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
		k_calcVertexNormals << <getBlockCount(nVerticesSurf, BLOCK_SIZE), BLOCK_SIZE >> > (nVerticesSurf, vertexNormals.get(), halfedges.values_, halfedges.rowPtr_, faceNormals.get());
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
	}

	virtual __host__ int nFaces() override {
		return nQuads;
	}
};


//#############################################################################
//############################### DMOMeshQuad ##################################
//#############################################################################
class DMOMeshQuad : public DMOMeshBaseQuad {
public:
	MeshQuadDevice* d_mesh;

	__host__ DMOMeshQuad(MeshQuadGPU& m) : DMOMeshBaseQuad(m) {
		MeshQuadDevice h_meshDevice(m);
		gpuErrchk(cudaMalloc((void**)&d_mesh, sizeof(MeshQuadDevice)));
		gpuErrchk(cudaMemcpy(d_mesh, &h_meshDevice, sizeof(MeshQuadDevice), cudaMemcpyHostToDevice));
	}

	__host__ ~DMOMeshQuad() {
		gpuErrchk(cudaFree(d_mesh));
	}
};




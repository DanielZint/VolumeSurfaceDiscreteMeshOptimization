#pragma once


#include "MeshHexGPU.h"
#include "DMOMeshQuad.h"



//#############################################################################
//############################### MeshHexDevice ###############################
//#############################################################################
class MeshHexDevice : public MeshQuadDevice {
	// all data needed in GPU memory
public:
	int nVertices;
	int nHexahedra;
	int nColorsInner;

	bool* vertexIsBoundary2D;
	Hexahedron* hexahedra;
	CompressedStorage<Halfhex> halfhexes;

public:
	__host__ MeshHexDevice(MeshHexGPU& m) :
		MeshQuadDevice(m), /*intentional object slice*/
		nVertices(m.nVertices),
		nHexahedra(m.nHexahedra),
		nColorsInner(m.nColorsInner),
		vertexIsBoundary2D(m.vertexIsBoundary2D.get()),
		hexahedra(m.hexahedra.get()),
		halfhexes(m.csrVertexHexahedraRowPtr.get(), m.csrVertexHexahedraValues.get())
	{
		
	}

	__device__ CompressedStorage<Halfhex>::Iterator hf_begin(int vid) {
		return halfhexes.begin(vid);
	}
	__device__ CompressedStorage<Halfhex>::Iterator hf_end(int vid) {
		return halfhexes.end(vid);
	}

};

//#############################################################################
//############################### DMOMeshTet ##################################
//#############################################################################
class DMOMeshHex : public DMOMeshBaseQuad {
	// all CPU data + ptr to GPU data needed for DMO
public:
	int nHexahedra;
	int nColorsInner;
	host_vector<int> colorOffsetsInner;
	int maxNumHexahedra;

	device_ptr<bool> vertexIsBoundary2D;
	device_ptr<Hexahedron> hexahedra;
	CompressedStorage<Halfhex> halfhexes;

	MeshHexDevice* d_mesh;

public:

	__host__ DMOMeshHex(MeshHexGPU& m) :
		DMOMeshBaseQuad(m),
		nHexahedra(m.nHexahedra),
		nColorsInner((int)m.col_offsets_inner.size() - 1),
		colorOffsetsInner(m.col_offsets_inner.begin(), m.col_offsets_inner.end()),
		maxNumHexahedra(m.maxNumHexahedra),
		vertexIsBoundary2D(m.vertexIsBoundary2D.ptr()),
		hexahedra(m.hexahedra.ptr()),
		halfhexes(m.csrVertexHexahedraRowPtr.get(), m.csrVertexHexahedraValues.get())
	{
		nVertices = m.nVertices;

		MeshHexDevice h_meshDevice(m);
		gpuErrchk(cudaMalloc((void**)&d_mesh, sizeof(MeshHexDevice)));
		gpuErrchk(cudaMemcpy(d_mesh, &h_meshDevice, sizeof(MeshHexDevice), cudaMemcpyHostToDevice));
	}

	__host__ ~DMOMeshHex() {
		gpuErrchk(cudaFree(d_mesh));
		gpuErrchk(cudaFree(vertexIsBoundary2D.get()));
		gpuErrchk(cudaFree(hexahedra.get()));
		gpuErrchk(cudaFree(halfhexes.values_));
		gpuErrchk(cudaFree(halfhexes.rowPtr_));
	}

	__host__ Tetrahedron* getTetrahedra() {
		return nullptr;
	}

	__host__ Hexahedron* getHexahedra() {
		return hexahedra.get();
	}

	__host__ void* getCells() {
		return (void*)hexahedra.get();
	}

};






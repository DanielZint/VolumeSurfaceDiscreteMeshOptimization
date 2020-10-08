#pragma once


#include "MeshTetGPU.h"
#include "DMOMeshTri.h"



//#############################################################################
//############################### MeshTetDevice ###############################
//#############################################################################
class MeshTetDevice : public MeshTriDevice {
	// all data needed in GPU memory
public:
	int nVertices;
	int nTetrahedra;
	int nColorsInner;

	bool* vertexIsBoundary2D;
	Tetrahedron* tetrahedra;
	CompressedStorage<Halfface> halffaces;

public:
	__host__ MeshTetDevice(MeshTetGPU& m) :
		MeshTriDevice(m), /*intentional object slice*/
		nVertices(m.nVertices),
		nTetrahedra(m.nTetrahedra),
		nColorsInner(m.nColorsInner),
		vertexIsBoundary2D(m.vertexIsBoundary2D.get()),
		tetrahedra(m.tetrahedra.get()),
		halffaces(m.csrVertexTetrahedraRowPtr.get(), m.csrVertexTetrahedraValues.get())
	{
		
	}

	__device__ CompressedStorage<Halfface>::Iterator hf_begin(int vid) {
		return halffaces.begin(vid);
	}
	__device__ CompressedStorage<Halfface>::Iterator hf_end(int vid) {
		return halffaces.end(vid);
	}

};

//#############################################################################
//############################### DMOMeshTet ##################################
//#############################################################################
class DMOMeshTet : public DMOMeshBaseTri {
	// all CPU data + ptr to GPU data needed for DMO
public:
	int nTetrahedra;
	int nColorsInner;
	host_vector<int> colorOffsetsInner;
	int maxNumTetrahedra;

	device_ptr<bool> vertexIsBoundary2D;
	device_ptr<Tetrahedron> tetrahedra;
	CompressedStorage<Halfface> halffaces;

	MeshTetDevice* d_mesh;

public:

	__host__ DMOMeshTet(MeshTetGPU& m) :
		DMOMeshBaseTri(m),
		nTetrahedra(m.nTetrahedra),
		nColorsInner((int)m.col_offsets_inner.size() - 1),
		colorOffsetsInner(m.col_offsets_inner.begin(), m.col_offsets_inner.end()),
		maxNumTetrahedra(m.maxNumTetrahedra),
		vertexIsBoundary2D(m.vertexIsBoundary2D.ptr()),
		tetrahedra(m.tetrahedra.ptr()),
		halffaces(m.csrVertexTetrahedraRowPtr.get(), m.csrVertexTetrahedraValues.get())
	{
		nVertices = m.nVertices;

		MeshTetDevice h_meshDevice(m);
		gpuErrchk(cudaMalloc((void**)&d_mesh, sizeof(MeshTetDevice)));
		gpuErrchk(cudaMemcpy(d_mesh, &h_meshDevice, sizeof(MeshTetDevice), cudaMemcpyHostToDevice));
	}

	__host__ ~DMOMeshTet() {
		gpuErrchk(cudaFree(d_mesh));
		gpuErrchk(cudaFree(vertexIsBoundary2D.get()));
		gpuErrchk(cudaFree(tetrahedra.get()));
		gpuErrchk(cudaFree(halffaces.values_));
		gpuErrchk(cudaFree(halffaces.rowPtr_));
	}

	__host__ Tetrahedron* getTetrahedra() {
		return tetrahedra.get();
	}

	__host__ Hexahedron* getHexahedra() {
		return nullptr;
	}

	__host__ void* getCells() {
		return (void*)tetrahedra.get();
	}

};






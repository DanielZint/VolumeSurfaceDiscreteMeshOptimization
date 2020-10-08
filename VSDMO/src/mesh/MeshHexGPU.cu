#include "MeshHexGPU.h"

#include "MeshGPUCommon.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <utility>
#include <numeric>
#include <algorithm>
#include <thrust/remove.h>
#include <thrust/sort.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>
#include <thrust/sequence.h>
#include <thrust/unique.h>
#include "VertexColoring.h"
#include "CudaUtil.h"
#include <iostream>
#include <stdio.h>

#include "SortUtil.h"
#include <thrust/extrema.h>


__host__ MeshHexGPU::MeshHexGPU() :
	MeshQuadGPU(),
	findBoundary2D(true),
	nVertices(0),
	nColorsInner(0)
{

}

__host__ MeshHexGPU::~MeshHexGPU() {}

__host__ void MeshHexGPU::init() {
	initBoundary2D(); //done

	colorInnerVerticesAndSort(); //done
	constructQuads(); // done

	MeshQuadGPU::init(); // call last because we need to make and fill triangle buffer first
	constructHalfhexesNew(); // done // call after MeshTri init because it sorts the hexahedra indices! Otherwise we'd have to remap indiced of halffaces too
}

__host__ void MeshHexGPU::setNumVertices(int nV) {
	nVertices = nV;
	vertexPoints.resize(nV);
}

__host__ void MeshHexGPU::setVertexPointsWithBoundary2D(vector<Vec3f>& points, vector<bool>& boundary2d) {
	thrust::copy(points.begin(), points.end(), vertexPoints.begin());
	vertexIsBoundary2D.resize(nVertices);// = device_vector<bool>(nVertices);
	thrust::copy(boundary2d.begin(), boundary2d.end(), vertexIsBoundary2D.begin());
	cout << "bound2d inited" << endl;
	findBoundary2D = false;
}

__host__ void MeshHexGPU::setNumHexahedra(int nT) {
	nHexahedra = nT;
	hexahedra.resize(nT);// = device_vector<Hexahedron>(nT);
}

__host__ void MeshHexGPU::setHexahedra(vector<Hexahedron>& tets) {
	thrust::copy(tets.begin(), tets.end(), hexahedra.begin());
}


__host__ void MeshHexGPU::constructQuads() {
	if (nQuads > 0) {
		// already found and set boundary quads in initBoundary2D
		cout << "already found and set boundary quads in initBoundary2D" << endl;
		return;
	}
	else {
		cout << "MeshTetGPU::constructTriangles" << endl;
		cout << "NOT IMPLEMENTED" << endl;
		while (1);
	}

	//device_vector<Quad> quadsCopy(nHexahedra * 6); //upper bound on num tris
	//device_vector<int> pos_counter(1, 0);

	//const int BLOCK_SIZE = 128;

	//// might contain non-surface tris if all vertices of a tri are boundary. thats why we do removeNonBoundaryTriangles
	//k_constructSurfaceTrisFromTetrahedra << <getBlockCount(nHexahedra, BLOCK_SIZE), BLOCK_SIZE >> > (nHexahedra, raw(hexahedra), raw(trianglesCopy), raw(vertexIsBoundary2D), raw(pos_counter));
	//gpuErrchk(cudaPeekAtLastError());
	//gpuErrchk(cudaDeviceSynchronize());

	//nQuads = removeNonBoundaryQuads(quadsCopy);
	//quads.resize(nQuads);// = device_vector<Quad>(nQuads);
	//thrust::copy(quadsCopy.begin(), quadsCopy.begin() + nQuads, quads.begin());
}


__host__ int MeshHexGPU::removeNonBoundaryQuads(device_vector<Quad>& quadsIn) { // done
	const int BLOCK_SIZE = 128;

	k_reorderVertices << <getBlockCount((int)quadsIn.size(), BLOCK_SIZE), BLOCK_SIZE >> > ((int)quadsIn.size(), raw(quadsIn));
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

	thrust::sort(quadsIn.begin(), quadsIn.end(), quadSort());

	k_markDuplicateQuads << <getBlockCount((int)quadsIn.size(), BLOCK_SIZE), BLOCK_SIZE >> > ((int)quadsIn.size(), raw(quadsIn));
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

	auto endit = thrust::remove_if(quadsIn.begin(), quadsIn.end(), isQuadMarked());
	int newsize = static_cast<int>(endit - quadsIn.begin());
	return newsize;
}



__host__ void MeshHexGPU::constructHalfhexesNew() {
	int nHalfhexes = nHexahedra * 8;
	device_vector<int> keys(nHalfhexes);
	csrVertexHexahedraValues.resize(nHalfhexes);
	csrVertexHexahedraRowPtr.resize(nVertices + 1);
	csrVertexHexahedraRowPtr.set(0, 0);

	const int BLOCK_SIZE = 128;
	k_constructHalfhexesNew << <getBlockCount(nHexahedra, BLOCK_SIZE), BLOCK_SIZE >> > (nHexahedra, raw(hexahedra), raw(csrVertexHexahedraValues), raw(keys));
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

	thrust::sort_by_key(keys.begin(), keys.begin() + nHalfhexes, csrVertexHexahedraValues.begin());
	k_find_first_n << <getBlockCount(nHalfhexes, BLOCK_SIZE), BLOCK_SIZE>> > (nHalfhexes, raw(keys), raw(csrVertexHexahedraRowPtr), nVertices);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

	device_vector<int> halfhexesPerVertex(nVertices);
	thrust::transform(csrVertexHexahedraRowPtr.begin() + 1, csrVertexHexahedraRowPtr.end(), csrVertexHexahedraRowPtr.begin(), halfhexesPerVertex.begin(), thrust::minus<int>());

	auto maxNumTetrahedraIt = thrust::max_element(halfhexesPerVertex.begin(), halfhexesPerVertex.end());
	maxNumHexahedra = *maxNumTetrahedraIt;
	cout << "maxNumTetrahedra " << maxNumHexahedra << endl;
}

__host__ void MeshHexGPU::initBoundary2D() { //done
	const int BLOCK_SIZE = 128;
	if (!findBoundary2D) {
		return;
	}
	// for each tet, add all 4 quads to a list (the 3 vertices in a defined order such as ascending). remove all duplicate quads, the ones left are boundary quads -> boundary vertices
	device_vector<Quad> simpleQuads(nHexahedra * 6);

	k_constructQuadsFromHexahedra << < getBlockCount(nHexahedra, BLOCK_SIZE), BLOCK_SIZE >> > (nHexahedra, raw(hexahedra), raw(simpleQuads));
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	
	nQuads = removeNonBoundaryQuads(simpleQuads);
	quads.resize(nQuads);
	thrust::copy(simpleQuads.begin(), simpleQuads.begin() + nQuads, quads.begin());

	// mark boundary vertices using triangle vector
	cout << "finding bound2d" << endl;
	vertexIsBoundary2D = device_vector<bool>(nVertices, false);
	k_setBoundary2DVertices << <getBlockCount(nQuads, BLOCK_SIZE), BLOCK_SIZE >> > (nQuads, raw(quads), vertexIsBoundary2D);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
}


__host__ int MeshHexGPU::sortVerticesByBoundary2D(device_vector<int>& sortMapInverseOut) { //done
	device_vector<int> sortMap(nVertices);
	thrust::sequence(sortMap.begin(), sortMap.end());

	thrust::sort_by_key(vertexIsBoundary2D.begin(), vertexIsBoundary2D.begin() + nVertices, sortMap.begin(), thrust::greater<bool>());

	auto it = thrust::find(vertexIsBoundary2D.begin(), vertexIsBoundary2D.begin() + nVertices, false);
	int startPos = static_cast<int>(it - vertexIsBoundary2D.begin());

	device_vector<Vec3f> vertexPointsSorted(nVertices);
	thrust::gather(sortMap.begin(), sortMap.end(), vertexPoints.begin(), vertexPointsSorted.begin());
	thrust::copy(vertexPointsSorted.begin(), vertexPointsSorted.end(), vertexPoints.begin());

	// remap hexahedra indices
	sortMapInverseOut = device_vector<int>(sortMap.size());
	const int BLOCK_SIZE = 128;
	k_populateSortMapInverse << <getBlockCount((int)sortMap.size(), BLOCK_SIZE), BLOCK_SIZE >> > (nVertices, sortMap, sortMapInverseOut);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

	return startPos;
}

__host__ int MeshHexGPU::sortInnerVerticesRangeByColor(int start, int num, device_vector<int>& colors, CudaArray<int>& col_offsets, device_vector<int>& sortMapInverseOut, bool doReindex) {
	//need whole sortMap for mapping of elem indices
	device_vector<int> sortMap(nVertices);
	thrust::sequence(sortMap.begin(), sortMap.end());
	thrust::sort_by_key(colors.begin(), colors.end(), sortMap.begin() + start);
	if (doReindex) {
		reindex(colors);
	}

	int numColors;
	thrust::copy(colors.begin() + colors.size() - 1, colors.end(), &numColors);
	numColors++;

	col_offsets.resize(numColors + 1);
	const int BLOCK_SIZE = 128;
	k_find_first_n << <getBlockCount((int)colors.size(), BLOCK_SIZE), BLOCK_SIZE >> > ((int)colors.size(), raw(colors), raw(col_offsets), numColors);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

	sortMapInverseOut = device_vector<int>(sortMap.size());
	k_populateSortMapInverse << <getBlockCount((int)sortMap.size(), BLOCK_SIZE), BLOCK_SIZE >> > (nVertices, sortMap, sortMapInverseOut);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

	thrust::for_each(sortMap.begin() + start, sortMap.begin() + start + num, thrust::placeholders::_1 -= start);

	device_vector<Vec3f> vertexPointsSorted(num);
	thrust::gather(sortMap.begin() + start, sortMap.begin() + start + num, vertexPoints.begin() + start, vertexPointsSorted.begin());
	thrust::copy(vertexPointsSorted.begin(), vertexPointsSorted.end(), vertexPoints.begin() + start);

	return numColors;
}


__host__ void MeshHexGPU::colorInnerVerticesAndSort() {
	// sort feature vertices to beginning and boundary vertices to end
	device_vector<int> sortMapInverse1;
	nVerticesSurf = sortVerticesByBoundary2D(sortMapInverse1);
	if (nQuads > 0) {
		remapElementIndices(nQuads, quads.get(), sortMapInverse1);
	}
	remapElementIndices(nHexahedra, hexahedra.get(), sortMapInverse1);
	int nVerticesInner = nVertices - nVerticesSurf;


	device_vector<int> simpleFulledgesInner;
	device_vector<int> csrFulledgeRowPtrInner;
	constructSimpleFulledgesInnerNew(simpleFulledgesInner, csrFulledgeRowPtrInner);



	device_vector<int> colors(nVerticesInner);
	if (nVerticesInner > 0) {
		int nnz = simpleFulledgesInner.size();
		bool doReindex = true;
		//color_jpl(nVerticesInner, raw(csrFulledgeRowPtrInner), raw(simpleFulledgesInner), colors);
		color_cuSPARSE(nVerticesInner, raw(csrFulledgeRowPtrInner), raw(simpleFulledgesInner), colors, nnz);
		//color_chen_li(nVerticesInner, raw(csrFulledgeRowPtrInner), raw(simpleFulledgesInner), colors, nnz);

		const int BLOCK_SIZE = 128;

		device_vector<int> sortMapInverse2;
		nColorsInner = sortInnerVerticesRangeByColor(nVerticesSurf, nVerticesInner, colors, col_offsets_inner, sortMapInverse2, doReindex);
		if (nQuads > 0) {
			remapElementIndices(nQuads, quads.get(), sortMapInverse2);
		}
		remapElementIndices(nHexahedra, hexahedra.get(), sortMapInverse2);

		k_checkColoring << <getBlockCount(nHexahedra, BLOCK_SIZE), BLOCK_SIZE >> > (nHexahedra, hexahedra.data(), nVerticesSurf, nVerticesInner, raw(colors));
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
	}

	thrust::for_each(col_offsets_inner.begin(), col_offsets_inner.end(), thrust::placeholders::_1 += nVerticesSurf);

} // colorInnerVerticesAndSort

__host__ void MeshHexGPU::remapElements(device_vector<int>& sortMapInverse) {
	remapElementIndices(nQuads, quads.get(), sortMapInverse);
	remapElementIndices(nHexahedra, hexahedra.get(), sortMapInverse);
}


__host__ void MeshHexGPU::checkColorsFree(device_vector<int>& colors) {
	const int BLOCK_SIZE = 128;
	k_checkColoring << <getBlockCount(nHexahedra, BLOCK_SIZE), BLOCK_SIZE >> > (nHexahedra, hexahedra.data(), 0, nVerticesSurf - nVerticesFeature, raw(colors));
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
}

__host__ void MeshHexGPU::checkColorsFeature(device_vector<int>& colors) {
	const int BLOCK_SIZE = 128;
	k_checkColoring << <getBlockCount(nHexahedra, BLOCK_SIZE), BLOCK_SIZE >> > (nHexahedra, hexahedra.data(), nVerticesSurf - nVerticesFeature, nVerticesFeature, raw(colors));
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
}


__host__ void MeshHexGPU::constructSimpleFulledgesInnerNew(device_vector<int>& simpleFulledges, device_vector<int>& rowPtr) {
	int nVerticesInner = nVertices - nVerticesSurf;
	device_vector<int> keys(nHexahedra * 56, -1);
	simpleFulledges.resize(nHexahedra * 56, -1);
	thrust::fill(simpleFulledges.begin(), simpleFulledges.end(), -1);
	rowPtr.resize(nVerticesInner + 1);
	thrust::fill(rowPtr.begin(), rowPtr.end(), -1);
	rowPtr[0] = 0;

	const int BLOCK_SIZE = 128;
	k_constructFulledgesFromHexahedraInnerNewFixed << <getBlockCount(nHexahedra, BLOCK_SIZE), BLOCK_SIZE >> > (nHexahedra, raw(hexahedra), raw(simpleFulledges), raw(keys), raw(vertexIsBoundary2D), nVerticesSurf);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize()); //here

	auto endit = thrust::remove_if(keys.begin(), keys.end(), isNegative());
	keys.erase(endit, keys.end());
	int nFulledges = keys.size();
	//int nFulledges = static_cast<int>(endit - keys.begin());
	auto endit2 = thrust::remove_if(simpleFulledges.begin(), simpleFulledges.end(), isNegative());
	simpleFulledges.erase(endit2, simpleFulledges.end());


	thrust::sort(thrust::make_zip_iterator(thrust::make_tuple(keys.begin(), simpleFulledges.begin())),
		thrust::make_zip_iterator(thrust::make_tuple(keys.begin() + nFulledges, simpleFulledges.begin() + nFulledges)), TupleCompare());

	typedef thrust::device_vector< int >                IntVector;
	typedef IntVector::iterator                         IntIterator;
	typedef thrust::tuple< IntIterator, IntIterator >   IntIteratorTuple;
	typedef thrust::zip_iterator< IntIteratorTuple >    ZipIterator;

	ZipIterator newEnd = thrust::unique(thrust::make_zip_iterator(thrust::make_tuple(keys.begin(), simpleFulledges.begin())),
		thrust::make_zip_iterator(thrust::make_tuple(keys.end(), simpleFulledges.end())));

	IntIteratorTuple endTuple = newEnd.get_iterator_tuple();

	keys.erase(thrust::get<0>(endTuple), keys.end());
	simpleFulledges.erase(thrust::get<1>(endTuple), simpleFulledges.end());

	nFulledges = keys.size();
	if (nFulledges == 0) {
		return;
	}

	k_find_first_n << <getBlockCount(nFulledges, BLOCK_SIZE), BLOCK_SIZE >> > (nFulledges, raw(keys), raw(rowPtr), nVerticesInner);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
}


__host__ void MeshHexGPU::constructSimpleFulledgesFreeNew(device_vector<int>& simpleFulledges, device_vector<int>& rowPtr) {
	int nVerticesFree = nVerticesSurf - nVerticesFeature;
	device_vector<int> keys(nHexahedra * 56, -1);
	simpleFulledges.resize(nHexahedra * 56, -1);
	thrust::fill(simpleFulledges.begin(), simpleFulledges.end(), -1);
	rowPtr.resize(nVerticesFree + 1);
	thrust::fill(rowPtr.begin(), rowPtr.end(), -1);
	rowPtr[0] = 0;

	const int BLOCK_SIZE = 128;
	k_constructFulledgesFromHexahedraFreeNewFixed << <getBlockCount(nHexahedra, BLOCK_SIZE), BLOCK_SIZE >> > (nHexahedra, raw(hexahedra), raw(simpleFulledges), raw(keys), raw(vertexIsBoundary2D), 0, nVerticesFree);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize()); //here

	auto endit = thrust::remove_if(keys.begin(), keys.end(), isNegative());
	keys.erase(endit, keys.end());
	int nFulledges = keys.size();
	//int nFulledges = static_cast<int>(endit - keys.begin());
	auto endit2 = thrust::remove_if(simpleFulledges.begin(), simpleFulledges.end(), isNegative());
	simpleFulledges.erase(endit2, simpleFulledges.end());


	thrust::sort(thrust::make_zip_iterator(thrust::make_tuple(keys.begin(), simpleFulledges.begin())),
		thrust::make_zip_iterator(thrust::make_tuple(keys.begin() + nFulledges, simpleFulledges.begin() + nFulledges)), TupleCompare());

	typedef thrust::device_vector< int >                IntVector;
	typedef IntVector::iterator                         IntIterator;
	typedef thrust::tuple< IntIterator, IntIterator >   IntIteratorTuple;
	typedef thrust::zip_iterator< IntIteratorTuple >    ZipIterator;

	ZipIterator newEnd = thrust::unique(thrust::make_zip_iterator(thrust::make_tuple(keys.begin(), simpleFulledges.begin())),
		thrust::make_zip_iterator(thrust::make_tuple(keys.end(), simpleFulledges.end())));

	IntIteratorTuple endTuple = newEnd.get_iterator_tuple();

	keys.erase(thrust::get<0>(endTuple), keys.end());
	simpleFulledges.erase(thrust::get<1>(endTuple), simpleFulledges.end());

	nFulledges = keys.size();
	if (nFulledges == 0) {
		return;
	}

	k_find_first_n << <getBlockCount(nFulledges, BLOCK_SIZE), BLOCK_SIZE >> > (nFulledges, raw(keys), raw(rowPtr), nVerticesFree);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

}

__host__ void MeshHexGPU::constructSimpleFulledgesFeatureNew(device_vector<int>& simpleFulledges, device_vector<int>& rowPtr) {
	int nVerticesFree = nVerticesSurf - nVerticesFeature;
	device_vector<int> keys(nHexahedra * 56, -1);
	simpleFulledges.resize(nHexahedra * 56, -1);
	thrust::fill(simpleFulledges.begin(), simpleFulledges.end(), -1);
	rowPtr.resize(nVerticesFeature + 1);
	thrust::fill(rowPtr.begin(), rowPtr.end(), -1);
	rowPtr[0] = 0;

	const int BLOCK_SIZE = 128;
	k_constructFulledgesFromHexahedraFeatureNewFixed << <getBlockCount(nHexahedra, BLOCK_SIZE), BLOCK_SIZE >> > (nHexahedra, raw(hexahedra), raw(simpleFulledges), raw(keys), raw(vertexIsBoundary2D), nVerticesFree, nVerticesFeature);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize()); //here

	auto endit = thrust::remove_if(keys.begin(), keys.end(), isNegative());
	keys.erase(endit, keys.end());
	int nFulledges = keys.size();
	//int nFulledges = static_cast<int>(endit - keys.begin());
	auto endit2 = thrust::remove_if(simpleFulledges.begin(), simpleFulledges.end(), isNegative());
	simpleFulledges.erase(endit2, simpleFulledges.end());


	thrust::sort(thrust::make_zip_iterator(thrust::make_tuple(keys.begin(), simpleFulledges.begin())),
		thrust::make_zip_iterator(thrust::make_tuple(keys.begin() + nFulledges, simpleFulledges.begin() + nFulledges)), TupleCompare());

	typedef thrust::device_vector< int >                IntVector;
	typedef IntVector::iterator                         IntIterator;
	typedef thrust::tuple< IntIterator, IntIterator >   IntIteratorTuple;
	typedef thrust::zip_iterator< IntIteratorTuple >    ZipIterator;

	ZipIterator newEnd = thrust::unique(thrust::make_zip_iterator(thrust::make_tuple(keys.begin(), simpleFulledges.begin())),
		thrust::make_zip_iterator(thrust::make_tuple(keys.end(), simpleFulledges.end())));

	IntIteratorTuple endTuple = newEnd.get_iterator_tuple();

	keys.erase(thrust::get<0>(endTuple), keys.end());
	simpleFulledges.erase(thrust::get<1>(endTuple), simpleFulledges.end());

	nFulledges = keys.size();
	if (nFulledges == 0) {
		return;
	}

	k_find_first_n << <getBlockCount(nFulledges, BLOCK_SIZE), BLOCK_SIZE >> > (nFulledges, raw(keys), raw(rowPtr), nVerticesFeature);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
}









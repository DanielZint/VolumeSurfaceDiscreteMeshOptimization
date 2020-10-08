#include "MeshTetGPU.h"

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


__host__ MeshTetGPU::MeshTetGPU() :
	MeshTriGPU(),
	findBoundary2D(true),
	nVertices(0),
	nColorsInner(0)
{

}

__host__ MeshTetGPU::~MeshTetGPU() {}

__host__ void MeshTetGPU::init() {
	initBoundary2D();
	colorInnerVerticesAndSort();
	constructTriangles();

	MeshTriGPU::init(); // call last because we need to make and fill triangle buffer first
	constructHalffacesNew(); // call after MeshTri init because it sorts the tetrahedra indices! Otherwise we'd have to remap indiced of halffaces too
}

__host__ void MeshTetGPU::setNumVertices(int nV) {
	nVertices = nV;
	vertexPoints.resize(nV);
}

__host__ void MeshTetGPU::setVertexPointsWithBoundary2D(vector<Vec3f>& points, vector<bool>& boundary2d) {
	thrust::copy(points.begin(), points.end(), vertexPoints.begin());
	vertexIsBoundary2D.resize(nVertices);// = device_vector<bool>(nVertices);
	thrust::copy(boundary2d.begin(), boundary2d.end(), vertexIsBoundary2D.begin());
	cout << "bound2d inited" << endl;
	findBoundary2D = false;
}

__host__ void MeshTetGPU::setNumTetrahedra(int nT) {
	nTetrahedra = nT;
	tetrahedra.resize(nT);// = device_vector<Tetrahedron>(nT);
}

__host__ void MeshTetGPU::setTetrahedra(vector<Tetrahedron>& tets) {
	thrust::copy(tets.begin(), tets.end(), tetrahedra.begin());
}


__host__ void MeshTetGPU::constructTriangles() {
	if (nTriangles > 0) {
		// already found and set boundary triangles in initBoundary2D
		cout << "already found and set boundary triangles in initBoundary2D" << endl;
		return;
	}
	else {
		cout << "MeshTetGPU::constructTriangles" << endl;
	}

	device_vector<Triangle> trianglesCopy(nTetrahedra * 4); //upper bound on num tris
	device_vector<int> pos_counter(1, 0);

	const int BLOCK_SIZE = 128;

	// might contain non-surface tris if all vertices of a tri are boundary. thats why we do removeNonBoundaryTriangles
	k_constructSurfaceTrisFromTetrahedra << <getBlockCount(nTetrahedra, BLOCK_SIZE), BLOCK_SIZE >> > (nTetrahedra, raw(tetrahedra), raw(trianglesCopy), raw(vertexIsBoundary2D), raw(pos_counter));
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

	nTriangles = removeNonBoundaryTriangles(trianglesCopy);
	triangles.resize(nTriangles);// = device_vector<Triangle>(nTriangles);
	thrust::copy(trianglesCopy.begin(), trianglesCopy.begin() + nTriangles, triangles.begin());
}


__host__ int MeshTetGPU::removeNonBoundaryTriangles(device_vector<Triangle>& trianglesIn) {
	const int BLOCK_SIZE = 128;

	k_reorderVertices << <getBlockCount((int)trianglesIn.size(), BLOCK_SIZE), BLOCK_SIZE >> > ((int)trianglesIn.size(), raw(trianglesIn));
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

	thrust::sort(trianglesIn.begin(), trianglesIn.end(), triangleSort());

	k_markDuplicateTris << <getBlockCount((int)trianglesIn.size(), BLOCK_SIZE), BLOCK_SIZE >> > ((int)trianglesIn.size(), raw(trianglesIn));
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

	auto endit = thrust::remove_if(trianglesIn.begin(), trianglesIn.end(), isTriangleMarked());
	int newsize = static_cast<int>(endit - trianglesIn.begin());
	return newsize;
}



__host__ void MeshTetGPU::constructHalffacesNew() {
	int nHalffaces = nTetrahedra * 4;
	device_vector<int> keys(nHalffaces);
	csrVertexTetrahedraValues.resize(nHalffaces);
	csrVertexTetrahedraRowPtr.resize(nVertices + 1);
	csrVertexTetrahedraRowPtr.set(0, 0);

	const int BLOCK_SIZE = 128;
	k_constructHalffacesNew << <getBlockCount(nTetrahedra, BLOCK_SIZE), BLOCK_SIZE >> > (nTetrahedra, raw(tetrahedra), raw(csrVertexTetrahedraValues), raw(keys));
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

	thrust::sort_by_key(keys.begin(), keys.begin() + nHalffaces, csrVertexTetrahedraValues.begin());
	k_find_first_n << <getBlockCount(nHalffaces, BLOCK_SIZE), BLOCK_SIZE>> > (nHalffaces, raw(keys), raw(csrVertexTetrahedraRowPtr), nVertices);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

	device_vector<int> halffacesPerVertex(nVertices);
	thrust::transform(csrVertexTetrahedraRowPtr.begin() + 1, csrVertexTetrahedraRowPtr.end(), csrVertexTetrahedraRowPtr.begin(), halffacesPerVertex.begin(), thrust::minus<int>());

	//auto zbegin = thrust::make_zip_iterator(thrust::make_tuple(csrVertexTetrahedraRowPtr.begin(), csrVertexTetrahedraRowPtr.begin() + 1));
	//auto zend = thrust::make_zip_iterator(thrust::make_tuple(csrVertexTetrahedraRowPtr.begin() + nVertices, csrVertexTetrahedraRowPtr.begin() + nVertices + 1));
	//thrust::max_element(zbegin, zend, diffLess<decltype(*zbegin)>());

	auto maxNumTetrahedraIt = thrust::max_element(halffacesPerVertex.begin(), halffacesPerVertex.end());
	maxNumTetrahedra = *maxNumTetrahedraIt;
	cout << "maxNumTetrahedra " << maxNumTetrahedra << endl;

	
}

__host__ void MeshTetGPU::constructHalffaces() {
	const int BLOCK_SIZE = 128;
	device_vector<int> vertexNumTetrahedra(nVertices, 0);
	device_vector<Halfface> vertexTetrahedra(nVertices * MAX_TETS_PER_VERTEX, { -1,-1,-1 }); // halffaces
	if (vertexTetrahedra.size() != nVertices * MAX_TETS_PER_VERTEX) cout << "vertexTetrahedra alloc error" << endl;

	k_constructHalffaces << <getBlockCount(nTetrahedra, BLOCK_SIZE), BLOCK_SIZE>> > (nTetrahedra, raw(tetrahedra), raw(vertexNumTetrahedra), raw(vertexTetrahedra));
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	
	auto maxNumTetrahedraIt = thrust::max_element(vertexNumTetrahedra.begin(), vertexNumTetrahedra.end());
	maxNumTetrahedra = *maxNumTetrahedraIt;
	cout << "maxNumTetrahedra " << maxNumTetrahedra << endl;
	
	makeCSRVertexTetrahedra(vertexNumTetrahedra, vertexTetrahedra);
}

__host__ void MeshTetGPU::makeCSRVertexTetrahedra(device_vector<int>& vertexNumTetrahedra, device_vector<Halfface>& vertexTetrahedra) {

	// TODO
	int nVertexTetrahedra = thrust::reduce(vertexNumTetrahedra.begin(), vertexNumTetrahedra.end(), 0);
	csrVertexTetrahedraValues.resize(nVertexTetrahedra);
	csrVertexTetrahedraRowPtr.resize(nVertices + 1);

	//RowPtr
	csrVertexTetrahedraRowPtr.set(0, 0);
	thrust::inclusive_scan(vertexNumTetrahedra.begin(), vertexNumTetrahedra.end(), csrVertexTetrahedraRowPtr.begin() + 1);

	//Values
	auto endit = thrust::remove_if(vertexTetrahedra.begin(), vertexTetrahedra.end(), isEmptyVertexTetrahedraEntry());
	thrust::copy(vertexTetrahedra.begin(), endit, csrVertexTetrahedraValues.begin());
} // makeCSRVertexTetrahedra


__host__ void MeshTetGPU::initBoundary2D() {
	const int BLOCK_SIZE = 128;
	if (!findBoundary2D) {
		return;
	}
	// for each tet, add all 4 triangles to a list (the 3 vertices in a defined order such as ascending). remove all duplicate triangles, the ones left are boundary triangles -> boundary vertices
	device_vector<Triangle> simpleTris(nTetrahedra * 4);

	k_constructTrisFromTetrahedra << < getBlockCount(nTetrahedra, BLOCK_SIZE), BLOCK_SIZE >> > (nTetrahedra, raw(tetrahedra), raw(simpleTris));
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	
	nTriangles = removeNonBoundaryTriangles(simpleTris);
	triangles.resize(nTriangles);// = device_vector<Triangle>(nTriangles);
	thrust::copy(simpleTris.begin(), simpleTris.begin() + nTriangles, triangles.begin());

	// mark boundary vertices using triangle vector
	cout << "finding bound2d" << endl;
	vertexIsBoundary2D = device_vector<bool>(nVertices, false);
	k_setBoundary2DVertices << <getBlockCount(nTriangles, BLOCK_SIZE), BLOCK_SIZE >> > (nTriangles, raw(triangles), vertexIsBoundary2D);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
}


__host__ void MeshTetGPU::constructFulledgesFromTetrahedraGPU(device_vector<int>& vertexNumSimpleFulledges, device_vector<int>& simpleFulledges) {
	//v2 with iterations and atomicAdd or atomicCAS
	const int BLOCK_SIZE = 128;
	device_vector<int> counters(nVertices, 0);
	device_vector<int> added(nTetrahedra, 0);
	device_vector<int> addedCounter(1, 0);
	int addedCounterHost = 0;
	int iter = 0;
	while (addedCounterHost != nTetrahedra) {
		//thrust::fill(counters.begin(), counters.end(), iter); // no fill when using atomicCAS
		k_constructFulledgesFromTetrahedra1Counters << < getBlockCount(nTetrahedra, BLOCK_SIZE), BLOCK_SIZE >> > (nTetrahedra, raw(tetrahedra), raw(vertexNumSimpleFulledges), raw(simpleFulledges),
			raw(counters), raw(added), raw(addedCounter), iter);
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
		addedCounterHost = addedCounter[0];
		++iter;
	}
}

__host__ void MeshTetGPU::constructFulledgesFromTetrahedraGPUOnlyInner(device_vector<int>& vertexNumSimpleFulledges, device_vector<int>& simpleFulledges) {
	//v2 with iterations and atomicAdd or atomicCAS
	const int BLOCK_SIZE = 128;
	device_vector<int> counters(nVertices - nVerticesSurf, 0);
	device_vector<int> added(nTetrahedra, 0);
	device_vector<int> addedCounter(1, 0);
	int addedCounterHost = 0;
	int iter = 0;
	while (addedCounterHost != nTetrahedra) {
		//thrust::fill(counters.begin(), counters.end(), iter); // no fill when using atomicCAS
		k_constructFulledgesFromTetrahedra1CountersOnlyInner << < getBlockCount(nTetrahedra, BLOCK_SIZE), BLOCK_SIZE >> > (nTetrahedra, raw(tetrahedra), raw(vertexNumSimpleFulledges), raw(simpleFulledges),
			raw(vertexIsBoundary2D), raw(counters), raw(added), raw(addedCounter), iter, nVerticesSurf);
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
		addedCounterHost = addedCounter[0];
		++iter;
	}
}

__host__ int MeshTetGPU::sortVerticesByBoundary2D(device_vector<int>& sortMapInverseOut) {
	device_vector<int> sortMap(nVertices);
	thrust::sequence(sortMap.begin(), sortMap.end());

	thrust::sort_by_key(vertexIsBoundary2D.begin(), vertexIsBoundary2D.begin() + nVertices, sortMap.begin(), thrust::greater<bool>());

	auto it = thrust::find(vertexIsBoundary2D.begin(), vertexIsBoundary2D.begin() + nVertices, false);
	int startPos = static_cast<int>(it - vertexIsBoundary2D.begin());

	device_vector<Vec3f> vertexPointsSorted(nVertices);
	thrust::gather(sortMap.begin(), sortMap.end(), vertexPoints.begin(), vertexPointsSorted.begin());
	thrust::copy(vertexPointsSorted.begin(), vertexPointsSorted.end(), vertexPoints.begin());

	// remap tetrahedra indices
	sortMapInverseOut = device_vector<int>(sortMap.size());
	const int BLOCK_SIZE = 128;
	k_populateSortMapInverse << <getBlockCount((int)sortMap.size(), BLOCK_SIZE), BLOCK_SIZE >> > (nVertices, sortMap, sortMapInverseOut);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

	return startPos;
}

__host__ int MeshTetGPU::sortInnerVerticesRangeByColor(int start, int num, device_vector<int>& colors, CudaArray<int>& col_offsets, device_vector<int>& sortMapInverseOut, bool doReindex) {
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


__host__ void MeshTetGPU::colorInnerVerticesAndSort() {
	// sort feature vertices to beginning and boundary vertices to end
	device_vector<int> sortMapInverse1;
	nVerticesSurf = sortVerticesByBoundary2D(sortMapInverse1);
	if (nTriangles > 0) {
		remapElementIndices(nTriangles, triangles.get(), sortMapInverse1);
	}
	remapElementIndices(nTetrahedra, tetrahedra.get(), sortMapInverse1);
	int nVerticesInner = nVertices - nVerticesSurf;

	/*Old*/
	//// make "adjacency" matrix and compress it
	//device_vector<int> vertexNumSimpleFulledgesInner(nVerticesInner, 0);
	//device_vector<int> simpleFulledgesInner((nVerticesInner) * MAX_HE_PER_VERTEX, -1);

	//constructFulledgesFromTetrahedraGPUOnlyInner(vertexNumSimpleFulledgesInner, simpleFulledgesInner); // TODO

	//// Convert SimpleHalfedge structures to compressed format
	//device_vector<int> csrFulledgeRowPtrInner;// (nVerticesInner + 1);

	//compress<int, isEmptySimpleHalfedgeEntry>(simpleFulledgesInner, vertexNumSimpleFulledgesInner, csrFulledgeRowPtrInner, nVerticesInner);
	/*Old End*/

	/*New*/
	device_vector<int> simpleFulledgesInner;
	device_vector<int> csrFulledgeRowPtrInner;
	constructSimpleFulledgesInnerNew(simpleFulledgesInner, csrFulledgeRowPtrInner);
	//device_vector<int> keys(nTetrahedra * 12, -1);
	//device_vector<int> simpleFulledgesInner(nTetrahedra * 12, -1);
	//device_vector<int> csrFulledgeRowPtrInner(nVerticesInner + 1);
	//csrFulledgeRowPtrInner[0] = 0;

	//const int BLOCK_SIZE = 128;
	//k_constructFulledgesFromTetrahedraInnerNew << <getBlockCount(nTetrahedra, BLOCK_SIZE), BLOCK_SIZE >> > (nTetrahedra, raw(tetrahedra), raw(simpleFulledgesInner), raw(keys), raw(vertexIsBoundary2D), nVerticesSurf);
	//gpuErrchk(cudaPeekAtLastError());
	//gpuErrchk(cudaDeviceSynchronize()); //here

	//auto endit = thrust::remove_if(keys.begin(), keys.end(), isNegative());
	//keys.erase(endit, keys.end());
	//int nFulledges = keys.size();
	////int nFulledges = static_cast<int>(endit - keys.begin());
	//auto endit2 = thrust::remove_if(simpleFulledgesInner.begin(), simpleFulledgesInner.end(), isNegative());
	//simpleFulledgesInner.erase(endit2, simpleFulledgesInner.end());


	//thrust::sort(thrust::make_zip_iterator(thrust::make_tuple(keys.begin(), simpleFulledgesInner.begin())),
	//	thrust::make_zip_iterator(thrust::make_tuple(keys.begin() + nFulledges, simpleFulledgesInner.begin() + nFulledges)), TupleCompare());

	//typedef thrust::device_vector< int >                IntVector;
	//typedef IntVector::iterator                         IntIterator;
	//typedef thrust::tuple< IntIterator, IntIterator >   IntIteratorTuple;
	//typedef thrust::zip_iterator< IntIteratorTuple >    ZipIterator;

	//ZipIterator newEnd = thrust::unique(thrust::make_zip_iterator(thrust::make_tuple(keys.begin(), simpleFulledgesInner.begin())),
	//	thrust::make_zip_iterator(thrust::make_tuple(keys.end(), simpleFulledgesInner.end())));

	//IntIteratorTuple endTuple = newEnd.get_iterator_tuple();

	//keys.erase(thrust::get<0>(endTuple), keys.end());
	//simpleFulledgesInner.erase(thrust::get<1>(endTuple), simpleFulledgesInner.end());

	//nFulledges = keys.size();
	//

	//k_find_first_n << <getBlockCount(nFulledges, BLOCK_SIZE), BLOCK_SIZE >> > (nFulledges, raw(keys), raw(csrFulledgeRowPtrInner), nVerticesInner);
	//gpuErrchk(cudaPeekAtLastError());
	//gpuErrchk(cudaDeviceSynchronize());
	/*New End*/


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
		if (nTriangles > 0) {
			remapElementIndices(nTriangles, triangles.get(), sortMapInverse2);
		}
		remapElementIndices(nTetrahedra, tetrahedra.get(), sortMapInverse2);

		k_checkColoring << <getBlockCount(nTetrahedra, BLOCK_SIZE), BLOCK_SIZE >> > (nTetrahedra, tetrahedra.data(), nVerticesSurf, nVerticesInner, raw(colors));
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
	}

	thrust::for_each(col_offsets_inner.begin(), col_offsets_inner.end(), thrust::placeholders::_1 += nVerticesSurf);

} // colorInnerVerticesAndSort

__host__ void MeshTetGPU::remapElements(device_vector<int>& sortMapInverse) {
	remapElementIndices(nTriangles, triangles.get(), sortMapInverse);
	remapElementIndices(nTetrahedra, tetrahedra.get(), sortMapInverse);
}

//__host__ void MeshTetGPU::constructSimpleFulledges(device_vector<int>& vertexNumSimpleFulledges, device_vector<int>& simpleFulledges) {
//	constructFulledgesFromTrianglesGPU(vertexNumSimpleFulledges, simpleFulledges);
//}

//__host__ void MeshTetGPU::makeSimpleFulledgeVectors(device_vector<int>& vertexNumSimpleFulledges, device_vector<int>& simpleFulledges) {
//	vertexNumSimpleFulledges = device_vector<int>(nVertices, 0);
//	simpleFulledges = device_vector<int>(nVertices * MAX_HE_PER_VERTEX, -1);
//}

__host__ void MeshTetGPU::checkColorsFree(device_vector<int>& colors) {
	const int BLOCK_SIZE = 128;
	k_checkColoring << <getBlockCount(nTetrahedra, BLOCK_SIZE), BLOCK_SIZE >> > (nTetrahedra, tetrahedra.data(), 0, nVerticesSurf - nVerticesFeature, raw(colors));
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
}

__host__ void MeshTetGPU::checkColorsFeature(device_vector<int>& colors) {
	const int BLOCK_SIZE = 128;
	k_checkColoring << <getBlockCount(nTetrahedra, BLOCK_SIZE), BLOCK_SIZE >> > (nTetrahedra, tetrahedra.data(), nVerticesSurf - nVerticesFeature, nVerticesFeature, raw(colors));
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
}

__host__ void MeshTetGPU::constructSimpleFulledgesFree(device_vector<int>& vertexNumSimpleFulledges, device_vector<int>& simpleFulledges) {
	const int BLOCK_SIZE = 128;
	device_vector<int> counters(nVerticesSurf - nVerticesFeature, 0);
	device_vector<int> added(nTetrahedra, 0);
	device_vector<int> addedCounter(1, 0);
	int addedCounterHost = 0;
	int iter = 0;
	while (addedCounterHost != nTetrahedra) {
		//thrust::fill(counters.begin(), counters.end(), iter); // no fill when using atomicCAS
		k_constructFulledgesFromTetrahedra1CountersOnlyFree << < getBlockCount(nTetrahedra, BLOCK_SIZE), BLOCK_SIZE >> > (nTetrahedra, raw(tetrahedra), raw(vertexNumSimpleFulledges), raw(simpleFulledges),
			raw(vertexIsBoundary2D), raw(counters), raw(added), raw(addedCounter), iter, 0, nVerticesSurf - nVerticesFeature);
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
		addedCounterHost = addedCounter[0];
		++iter;
	}
}

__host__ void MeshTetGPU::constructSimpleFulledgesFeature(device_vector<int>& vertexNumSimpleFulledges, device_vector<int>& simpleFulledges) {
	const int BLOCK_SIZE = 128;
	device_vector<int> counters(nVerticesFeature, 0);
	device_vector<int> added(nTetrahedra, 0);
	device_vector<int> addedCounter(1, 0);
	int addedCounterHost = 0;
	int iter = 0;
	while (addedCounterHost != nTetrahedra) {
		//thrust::fill(counters.begin(), counters.end(), iter); // no fill when using atomicCAS
		k_constructFulledgesFromTetrahedra1CountersOnlyFeature << < getBlockCount(nTetrahedra, BLOCK_SIZE), BLOCK_SIZE >> > (nTetrahedra, raw(tetrahedra), raw(vertexNumSimpleFulledges), raw(simpleFulledges),
			raw(vertexIsFeature), raw(counters), raw(added), raw(addedCounter), iter, nVerticesSurf - nVerticesFeature, nVerticesFeature);
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
		addedCounterHost = addedCounter[0];
		++iter;
	}
}



__host__ void MeshTetGPU::constructSimpleFulledgesInnerNew(device_vector<int>& simpleFulledges, device_vector<int>& rowPtr) {
	int nVerticesInner = nVertices - nVerticesSurf;
	device_vector<int> keys(nTetrahedra * 12, -1);
	simpleFulledges.resize(nTetrahedra * 12, -1);
	thrust::fill(simpleFulledges.begin(), simpleFulledges.end(), -1);
	rowPtr.resize(nVerticesInner + 1);
	thrust::fill(rowPtr.begin(), rowPtr.end(), -1);
	rowPtr[0] = 0;

	const int BLOCK_SIZE = 128;
	k_constructFulledgesFromTetrahedraInnerNew << <getBlockCount(nTetrahedra, BLOCK_SIZE), BLOCK_SIZE >> > (nTetrahedra, raw(tetrahedra), raw(simpleFulledges), raw(keys), raw(vertexIsBoundary2D), nVerticesSurf);
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


__host__ void MeshTetGPU::constructSimpleFulledgesFreeNew(device_vector<int>& simpleFulledges, device_vector<int>& rowPtr) {
	int nVerticesFree = nVerticesSurf - nVerticesFeature;
	device_vector<int> keys(nTetrahedra * 12, -1);
	simpleFulledges.resize(nTetrahedra * 12, -1);
	thrust::fill(simpleFulledges.begin(), simpleFulledges.end(), -1);
	rowPtr.resize(nVerticesFree + 1);
	thrust::fill(rowPtr.begin(), rowPtr.end(), -1);
	rowPtr[0] = 0;

	const int BLOCK_SIZE = 128;
	k_constructFulledgesFromTetrahedraFreeNew << <getBlockCount(nTetrahedra, BLOCK_SIZE), BLOCK_SIZE >> > (nTetrahedra, raw(tetrahedra), raw(simpleFulledges), raw(keys), raw(vertexIsBoundary2D), 0, nVerticesFree);
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

__host__ void MeshTetGPU::constructSimpleFulledgesFeatureNew(device_vector<int>& simpleFulledges, device_vector<int>& rowPtr) {
	int nVerticesFree = nVerticesSurf - nVerticesFeature;
	device_vector<int> keys(nTetrahedra * 12, -1);
	simpleFulledges.resize(nTetrahedra * 12, -1);
	thrust::fill(simpleFulledges.begin(), simpleFulledges.end(), -1);
	rowPtr.resize(nVerticesFeature + 1);
	thrust::fill(rowPtr.begin(), rowPtr.end(), -1);
	rowPtr[0] = 0;

	const int BLOCK_SIZE = 128;
	k_constructFulledgesFromTetrahedraFeatureNew << <getBlockCount(nTetrahedra, BLOCK_SIZE), BLOCK_SIZE >> > (nTetrahedra, raw(tetrahedra), raw(simpleFulledges), raw(keys), raw(vertexIsBoundary2D), nVerticesFree, nVerticesFeature);
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









#include "MeshTriGPU.h"
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
#include "Timer.h"
#include <iostream>
#include <stdio.h>

#include "SortUtil.h"
#include <thrust/extrema.h>


__host__ MeshTriGPU::MeshTriGPU() :
	MeshBaseGPU(),
	nTriangles(0)
{

}

__host__ MeshTriGPU::~MeshTriGPU() {

}


__host__ void MeshTriGPU::init() {
	calcFaceNormals();
	initBoundary1DAndFeaturesNew();
	colorVerticesAndSort();

	//auto devptr3 = vertexPoints.ptr();
	//host_vector<Vec3f> vertexPointsHost(devptr3, devptr3 + nVerticesSurf);
	//cout << vertexPointsHost[9826] << endl << endl;

	constructTriHalfedgesNew();
	calcVertexNormals();

	makeHalfedgesSoA();
}

__host__ void MeshTriGPU::setNumVerticesSurf(int n) {
	nVerticesSurf = n;
	vertexPoints.resize(n);
	vertexIsBoundary1D.resize(n);
}

__host__ void MeshTriGPU::setNumTriangles(int n) {
	nTriangles = n;
	triangles.resize(n);
}

__host__ void MeshTriGPU::setVertexPoints(vector<Vec3f>& points, bool nonZeroZ) {
	if (!nonZeroZ) {
		//2d flat
		isFlat = true;
	}
	thrust::copy(points.begin(), points.end(), vertexPoints.begin());
	findBoundary1D = true;
}

__host__ void MeshTriGPU::setVertexPointsWithBoundary1D(vector<Vec3f>& points, vector<bool>& boundary1d) {
	thrust::copy(points.begin(), points.end(), vertexPoints.begin());
	thrust::copy(boundary1d.begin(), boundary1d.end(), vertexIsBoundary1D.begin());
	findBoundary1D = false;
}

__host__ void MeshTriGPU::setTriangles(vector<Triangle>& tris) {
	thrust::copy(tris.begin(), tris.end(), triangles.begin());
}

__host__ void MeshTriGPU::fromDeviceData(int nVS, int nT, CudaArray<Vec3f>& points, CudaArray<Triangle>& tris) {
	nVerticesSurf = nVS;
	nTriangles = nT;
	vertexPoints.resize(nVS);
	thrust::copy(points.begin(), points.begin() + nVS, vertexPoints.begin());
	triangles.resize(nT);
	thrust::copy(tris.begin(), tris.begin() + nT, triangles.begin());
	findBoundary1D = true;
}

__host__ void MeshTriGPU::fromDeviceData(int nVS, int nT, device_ptr<Vec3f> points, device_ptr<Triangle> tris) {
	nVerticesSurf = nVS;
	nTriangles = nT;
	vertexPoints.resize(nVS);
	thrust::copy(points, points + nVS, vertexPoints.begin());
	triangles.resize(nT);
	thrust::copy(tris, tris + nT, triangles.begin());
	findBoundary1D = true;
}



//__host__ void MeshTriGPU::constructHalfedgesGPU(device_vector<int>& vertexNumHalfedges, device_vector<Halfedge>& halfedges) {
//	const int BLOCK_SIZE = 128;
//	k_constructHalfedges1 << <getBlockCount(nTriangles, BLOCK_SIZE), BLOCK_SIZE >> > (nTriangles, raw(triangles), raw(vertexNumHalfedges), raw(halfedges));
//	gpuErrchk(cudaPeekAtLastError());
//	gpuErrchk(cudaDeviceSynchronize());
//}



// Note: Also adds the opposite halfedge, even if it does not belong to a face. This is useful for getting all neighbor vertices of a boundary vertex.
__host__ void MeshTriGPU::constructHalfedgesWithDummyOpposite(device_vector<int>& vertexNumHalfedges, device_vector<Halfedge>& halfedges) {
	const int BLOCK_SIZE = 128;
	k_constructFulledgesFromTriangles1Pass1 << <getBlockCount(nTriangles, BLOCK_SIZE), BLOCK_SIZE >> > (nTriangles, raw(triangles), raw(vertexNumHalfedges), raw(halfedges), raw(vertexIsBoundary1D));
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	k_constructFulledgesFromTriangles1Pass2 << <getBlockCount(nTriangles, BLOCK_SIZE), BLOCK_SIZE >> > (nTriangles, raw(triangles), raw(vertexNumHalfedges), raw(halfedges), raw(vertexIsBoundary1D));
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	k_constructHalfedges2 << <getBlockCount(nVerticesSurf * MAX_HE_PER_VERTEX, BLOCK_SIZE), BLOCK_SIZE >> > (nVerticesSurf * MAX_HE_PER_VERTEX, raw(vertexNumHalfedges), raw(halfedges));
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	// TODO handle assert
}

__host__ void MeshTriGPU::constructTriHalfedgesNew() {
	int nHEMax = nTriangles * 6;
	int nHE = nTriangles * 3;
	device_vector<int> keys(nHEMax, -1);
	device_vector<Halfedge> halfedges(nHEMax, {-1, -1, -1});
	csrHalfedgeRowPtr.resize(nVerticesSurf + 1);
	csrHalfedgeRowPtr.set(0, 0);

	const int BLOCK_SIZE = 128;
	k_constructHalfedges1New << <getBlockCount(nTriangles, BLOCK_SIZE), BLOCK_SIZE >> > (nTriangles, raw(triangles), raw(halfedges), raw(keys)); //HERE
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

	{
		thrust::sort_by_key(keys.begin(), keys.begin() + nHE, halfedges.begin());
		k_find_first_n << <getBlockCount(nHE, BLOCK_SIZE), BLOCK_SIZE >> > (nHE, raw(keys), raw(csrHalfedgeRowPtr), nVerticesSurf);
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());

		device_vector<int> indexToCol(nHE);
		k_fillHelperArray << <getBlockCount(nVerticesSurf, BLOCK_SIZE), BLOCK_SIZE >> > (nVerticesSurf, csrHalfedgeRowPtr, indexToCol);
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
		k_setOppositeHalfedgeHelper << <getBlockCount(nHE, BLOCK_SIZE), BLOCK_SIZE >> > (nHE, halfedges, csrHalfedgeRowPtr, indexToCol);
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
	}

	/////////

	k_addBoundaryHalfedges << <getBlockCount(nHE, BLOCK_SIZE), BLOCK_SIZE >> > (nHE, raw(keys), raw(halfedges));
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

	auto endit = thrust::remove_if(keys.begin(), keys.end(), isNegative());
	nHalfedges = static_cast<int>(endit - keys.begin());
	thrust::remove_if(halfedges.begin(), halfedges.end(), isEmptyHalfedgeEntry());

	csrHalfedgesValues.resize(nHalfedges);
	//csrHalfedgeColInd.resize(nHalfedges);

	
	thrust::sort_by_key(keys.begin(), keys.begin() + nHalfedges, halfedges.begin());
	k_find_first_n << <getBlockCount(nHalfedges, BLOCK_SIZE), BLOCK_SIZE >> > (nHalfedges, raw(keys), raw(csrHalfedgeRowPtr), nVerticesSurf);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

	//auto devptr3 = csrHalfedgeRowPtr.ptr();
	//host_vector<int> csrHalfedgeRowPtrHost(devptr3, devptr3 + nVerticesSurf + 1);
	//for (int i = 0; i < nVerticesSurf + 1; ++i) {
	//	cout << csrHalfedgeRowPtrHost[i] << endl;
	//}

	device_vector<int> indexToCol(nHE);
	k_fillHelperArray << <getBlockCount(nVerticesSurf, BLOCK_SIZE), BLOCK_SIZE >> > (nVerticesSurf, csrHalfedgeRowPtr, indexToCol);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	k_setOppositeHalfedgeHelper << <getBlockCount(nHalfedges, BLOCK_SIZE), BLOCK_SIZE >> > (nHalfedges, halfedges, csrHalfedgeRowPtr, indexToCol);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	
	//for (int i = 0; i < 100; ++i) {
	//	Halfedge he = halfedges[i];
	//	cout << he.targetVertex << " " << he.oppositeHE << " " << he.incidentFace << endl;
	//}

	k_sortCCWDummyOpposite << <getBlockCount(nVerticesSurf, BLOCK_SIZE), BLOCK_SIZE >> > (nVerticesSurf, csrHalfedgeRowPtr, halfedges, csrHalfedgesValues,
		raw(vertexIsBoundary1D), raw(vertexIsFeature), raw(triangles));
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

	k_setOppositeHalfedgeHelper << <getBlockCount(nHalfedges, BLOCK_SIZE), BLOCK_SIZE >> > (nHalfedges, csrHalfedgesValues, csrHalfedgeRowPtr, indexToCol);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

	k_setNextHalfedgeHelper << <getBlockCount(nHalfedges, BLOCK_SIZE), BLOCK_SIZE >> > (nHalfedges, csrHalfedgesValues, csrHalfedgeRowPtr, indexToCol, nVerticesSurf);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

	//TODO
	device_vector<int> halfedgesPerVertex(nVerticesSurf);
	thrust::transform(csrHalfedgeRowPtr.begin() + 1, csrHalfedgeRowPtr.end(), csrHalfedgeRowPtr.begin(), halfedgesPerVertex.begin(), thrust::minus<int>());

	auto maxNumHalfedgesIt = thrust::max_element(halfedgesPerVertex.begin(), halfedgesPerVertex.end());
	maxNumHalfedges = *maxNumHalfedgesIt;
	cout << "maxNumHalfedges " << maxNumHalfedges << endl;
}


__host__ void MeshTriGPU::constructTriHalfedges() {
	device_vector<int> vertexNumHalfedges(nVerticesSurf, 0);
	device_vector<Halfedge> halfedges(nVerticesSurf * MAX_HE_PER_VERTEX, { -1, -1, -1 });

	constructHalfedgesWithDummyOpposite(vertexNumHalfedges, halfedges);
	
	auto maxNumHalfedgesIt = thrust::max_element(vertexNumHalfedges.begin(), vertexNumHalfedges.end());
	maxNumHalfedges = vertexNumHalfedges[maxNumHalfedgesIt - vertexNumHalfedges.begin()];
	cout << "maxNumHalfedges " << maxNumHalfedges << endl;

	makeCSRHalfedges(vertexNumHalfedges, halfedges);
}


__host__ void MeshTriGPU::makeCSRHalfedges(device_vector<int>& vertexNumHalfedges, device_vector<Halfedge>& halfedges) {
	nHalfedges = thrust::reduce(vertexNumHalfedges.begin(), vertexNumHalfedges.end(), 0);
	csrHalfedgesValues.resize(nHalfedges);// = device_vector<Halfedge>(nHalfedges);
	//csrHalfedgeColInd.resize(nHalfedges);// = device_vector<int>(nHalfedges);
	csrHalfedgeRowPtr.resize(nVerticesSurf + 1);// = device_vector<int>(nVerticesSurf + 1);

	//RowPtr
	csrHalfedgeRowPtr.set(0, 0);
	thrust::inclusive_scan(vertexNumHalfedges.begin(), vertexNumHalfedges.end(), csrHalfedgeRowPtr.begin() + 1);

	//auto devptr3 = csrHalfedgeRowPtr.ptr();
	//host_vector<int> csrHalfedgeRowPtrHost(devptr3, devptr3 + nVerticesSurf + 1);
	//for (int i = 0; i < nVerticesSurf + 1; ++i) {
	//	cout << csrHalfedgeRowPtrHost[i] << endl;
	//}

	//temp helper array
	device_vector<int> indexToCol(nHalfedges);
	const int BLOCK_SIZE = 128;
	k_fillHelperArray << <getBlockCount(nVerticesSurf, BLOCK_SIZE), BLOCK_SIZE>> > (nVerticesSurf, csrHalfedgeRowPtr, indexToCol);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	
	auto endit = thrust::remove_if(halfedges.begin(), halfedges.end(), isEmptyHalfedgeEntry()); // compress
	int enditPos = static_cast<int>(endit - halfedges.begin());

	// Note: Need the index of opposite halfedge in k_sortCCWDummyOpposite, so set them in advance. Since we sort the HEs, we need to set the oppHEs after that again
	k_setOppositeHalfedgeHelper << <getBlockCount(enditPos, BLOCK_SIZE), BLOCK_SIZE >> > (enditPos, halfedges, csrHalfedgeRowPtr, indexToCol);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

	//for (int i = 0; i < 100; ++i) {
	//	Halfedge he = halfedges[i];
	//	cout << he.targetVertex <<  " " << he.oppositeHE << " " << he.incidentFace << endl;
	//}

	k_sortCCWDummyOpposite << <getBlockCount(nVerticesSurf, BLOCK_SIZE), BLOCK_SIZE >> > (nVerticesSurf, csrHalfedgeRowPtr, halfedges, csrHalfedgesValues,
		raw(vertexIsBoundary1D), raw(vertexIsFeature), raw(triangles));

	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

	k_setOppositeHalfedgeHelper << <getBlockCount(enditPos, BLOCK_SIZE), BLOCK_SIZE >> > (enditPos, csrHalfedgesValues, csrHalfedgeRowPtr, indexToCol);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

	//device_vector<int> sortMap(nHalfedges);
	//thrust::sequence(csrHalfedgeColInd.begin(), csrHalfedgeColInd.end(), 0);
	//thrust::copy(csrHalfedgesValues.begin(), csrHalfedgesValues.end(), halfedges.begin());

} // makeCSRHalfedges



//__host__ void MeshTriGPU::constructSimpleFulledges(device_vector<int>& vertexNumSimpleFulledges, device_vector<int>& simpleFulledges) {
//	constructFulledgesFromTrianglesGPU(vertexNumSimpleFulledges, simpleFulledges);
//}

__host__ void MeshTriGPU::constructSimpleFulledgesFree(device_vector<int>& vertexNumSimpleFulledges, device_vector<int>& simpleFulledges) {
	constructFulledgesFromTrianglesGPUOnlyFree(vertexNumSimpleFulledges, simpleFulledges);
}

__host__ void MeshTriGPU::constructSimpleFulledgesFeature(device_vector<int>& vertexNumSimpleFulledges, device_vector<int>& simpleFulledges) {
	constructFulledgesFromTrianglesGPUOnlyFeature(vertexNumSimpleFulledges, simpleFulledges);
}


__host__ void MeshTriGPU::constructSimpleFulledgesFreeNew(device_vector<int>& simpleFulledges, device_vector<int>& rowPtr) {
	int nVerticesFree = nVerticesSurf - nVerticesFeature;
	device_vector<int> keys(nTriangles * 6, -1);
	simpleFulledges.resize(nTriangles * 6, -1);
	thrust::fill(simpleFulledges.begin(), simpleFulledges.end(), -1);
	rowPtr.resize(nVerticesFree + 1);
	thrust::fill(rowPtr.begin(), rowPtr.end(), -1);
	rowPtr[0] = 0;

	const int BLOCK_SIZE = 128;
	k_constructFulledgesFromTrianglesFreeNew << <getBlockCount(nTriangles, BLOCK_SIZE), BLOCK_SIZE >> > (nTriangles, raw(triangles), raw(simpleFulledges), raw(keys), 0, nVerticesFree);
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

__host__ void MeshTriGPU::constructSimpleFulledgesFeatureNew(device_vector<int>& simpleFulledges, device_vector<int>& rowPtr) {
	int nVerticesFree = nVerticesSurf - nVerticesFeature;
	device_vector<int> keys(nTriangles * 6, -1);
	simpleFulledges.resize(nTriangles * 6, -1);
	thrust::fill(simpleFulledges.begin(), simpleFulledges.end(), -1);
	rowPtr.resize(nVerticesFeature + 1);
	thrust::fill(rowPtr.begin(), rowPtr.end(), -1);
	rowPtr[0] = 0;

	const int BLOCK_SIZE = 128;
	k_constructFulledgesFromTrianglesFeatureNew << <getBlockCount(nTriangles, BLOCK_SIZE), BLOCK_SIZE >> > (nTriangles, raw(triangles), raw(simpleFulledges), raw(keys), nVerticesFree, nVerticesFeature);
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




//__host__ void MeshTriGPU::constructFulledgesFromTrianglesGPU(device_vector<int>& vertexNumSimpleFulledges, device_vector<int>& simpleFulledges) {
//	const int BLOCK_SIZE = 128;
//	k_constructFulledgesFromTriangles1Pass1 << <getBlockCount(nTriangles, BLOCK_SIZE), BLOCK_SIZE >> > (nTriangles, raw(triangles), raw(vertexNumSimpleFulledges), raw(simpleFulledges), raw(vertexIsBoundary1D));
//	gpuErrchk(cudaPeekAtLastError());
//	gpuErrchk(cudaDeviceSynchronize());
//	k_constructFulledgesFromTriangles1Pass2 << <getBlockCount(nTriangles, BLOCK_SIZE), BLOCK_SIZE >> > (nTriangles, raw(triangles), raw(vertexNumSimpleFulledges), raw(simpleFulledges), raw(vertexIsBoundary1D));
//	gpuErrchk(cudaPeekAtLastError());
//	gpuErrchk(cudaDeviceSynchronize());
//	// TODO handle assert
//}

__host__ void MeshTriGPU::constructFulledgesFromTrianglesGPUOnlyFree(device_vector<int>& vertexNumSimpleFulledges, device_vector<int>& simpleFulledges) {
	const int BLOCK_SIZE = 128;
	k_constructFulledgesFromTriangles1OnlyFree << <getBlockCount(nTriangles, BLOCK_SIZE), BLOCK_SIZE >> > (nTriangles, raw(triangles), raw(vertexNumSimpleFulledges), raw(simpleFulledges), raw(vertexIsFeature));
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	// TODO handle assert
}

__host__ void MeshTriGPU::constructFulledgesFromTrianglesGPUOnlyFeature(device_vector<int>& vertexNumSimpleFulledges, device_vector<int>& simpleFulledges) {
	const int BLOCK_SIZE = 128;
	k_constructFulledgesFromTriangles1Pass1OnlyFeature << <getBlockCount(nTriangles, BLOCK_SIZE), BLOCK_SIZE >> > (nTriangles, raw(triangles), raw(vertexNumSimpleFulledges), raw(simpleFulledges), raw(vertexIsFeature), nVerticesSurf - nVerticesFeature);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	k_constructFulledgesFromTriangles1Pass2OnlyFeature << <getBlockCount(nTriangles, BLOCK_SIZE), BLOCK_SIZE >> > (nTriangles, raw(triangles), raw(vertexNumSimpleFulledges), raw(simpleFulledges), raw(vertexIsFeature), nVerticesSurf - nVerticesFeature);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	// TODO handle assert
}






//__host__ void MeshTriGPU::initBoundary1D() {
//	if (findBoundary1D) {
//		cout << "init boundary1d" << endl;
//		device_vector<int> vertexNumHalfedges(nVerticesSurf, 0);
//		device_vector<Halfedge> halfedges(nVerticesSurf * MAX_HE_PER_VERTEX, { -1, -1, -1 });
//		constructHalfedgesGPU(vertexNumHalfedges, halfedges);
//
//		const int BLOCK_SIZE = 128;
//		k_initBoundary1D << <getBlockCount(nVerticesSurf, BLOCK_SIZE), BLOCK_SIZE >> > (nVerticesSurf, raw(vertexNumHalfedges), raw(halfedges), raw(vertexIsBoundary1D));
//		gpuErrchk(cudaPeekAtLastError());
//		gpuErrchk(cudaDeviceSynchronize());
//		findBoundary1D = false;
//	}
//	else {
//		cout << "dont init boundary1d" << endl;
//	}
//}


//__host__ void MeshTriGPU::initFeatures() {
//	device_vector<int> vertexNumHalfedges(nVerticesSurf, 0);
//	device_vector<Halfedge> halfedges(nVerticesSurf * MAX_HE_PER_VERTEX, { -1, -1, -1 });
//	constructHalfedgesGPU(vertexNumHalfedges, halfedges);
//
//	vertexIsFeature = device_vector<bool>(nVerticesSurf, false);
//	const int BLOCK_SIZE = 128;
//	k_initFeatures << <getBlockCount(nVerticesSurf, BLOCK_SIZE), BLOCK_SIZE >> > (nVerticesSurf, vertexIsFeature, vertexNumHalfedges, halfedges, raw(faceNormals), MAX_HE_PER_VERTEX, 20.f);
//	gpuErrchk(cudaPeekAtLastError());
//	gpuErrchk(cudaDeviceSynchronize());
//}

__host__ void MeshTriGPU::initBoundary1DAndFeaturesNew() {
	int nHE = nTriangles * 3;
	device_vector<int> keys(nHE);
	device_vector<Halfedge> halfedges(nHE);
	device_vector<int> rowPtr(nVerticesSurf + 1);
	rowPtr[0] = 0;

	const int BLOCK_SIZE = 128;
	k_constructHalfedges1New << <getBlockCount(nTriangles, BLOCK_SIZE), BLOCK_SIZE >> > (nTriangles, raw(triangles), raw(halfedges), raw(keys)); //HERE
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

	thrust::sort_by_key(keys.begin(), keys.begin() + nHE, halfedges.begin());
	k_find_first_n << <getBlockCount(nHE, BLOCK_SIZE), BLOCK_SIZE >> > (nHE, raw(keys), raw(rowPtr), nVerticesSurf);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

	//device_vector<int> halffacesPerVertex(nVertices);
	//thrust::transform(csrVertexTetrahedraRowPtr.begin() + 1, csrVertexTetrahedraRowPtr.end(), csrVertexTetrahedraRowPtr.begin(), halffacesPerVertex.begin(), thrust::minus<int>());

	device_vector<int> indexToCol(nHE);
	k_fillHelperArray << <getBlockCount(nVerticesSurf, BLOCK_SIZE), BLOCK_SIZE >> > (nVerticesSurf, rowPtr, indexToCol);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	k_setOppositeHalfedgeHelper << <getBlockCount(nHE, BLOCK_SIZE), BLOCK_SIZE >> > (nHE, halfedges, rowPtr, indexToCol);

	if (findBoundary1D) {
		cout << "init boundary1d" << endl;
		vertexIsBoundary1D = device_vector<bool>(nVerticesSurf, false);
		k_initBoundary1DRowPtr << <getBlockCount(nVerticesSurf, BLOCK_SIZE), BLOCK_SIZE >> > (nVerticesSurf, raw(rowPtr), raw(halfedges), raw(vertexIsBoundary1D));
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
		findBoundary1D = false;
	}
	else {
		cout << "dont init boundary1d" << endl;
	}

	vertexIsFeature = device_vector<bool>(nVerticesSurf, false);
	if (isFlat) {
		thrust::copy(vertexIsBoundary1D.begin(), vertexIsBoundary1D.end(), vertexIsFeature.begin());
	}
	else {
		k_initFeaturesRowPtr << <getBlockCount(nVerticesSurf, BLOCK_SIZE), BLOCK_SIZE >> > (nVerticesSurf, vertexIsFeature, rowPtr, halfedges, raw(faceNormals), 20.f);
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
	}
}

__host__ void MeshTriGPU::initBoundary1DAndFeatures() {
	const int BLOCK_SIZE = 128;
	device_vector<Halfedge> halfedges;
	device_vector<int> rowPtr;
	int nValid;
	if (nVerticesSurf > 100000) {
		int nBuckets = 1024*128;
		int sizeBucket = 512;
		device_vector<int> keys(1024*1024*64);
		device_vector<Halfedge> vals(1024*1024*64); // these 2 vectors are 1gb together: 64*1024*1024*16 B

		nValid = 0;
		const int batchSize = 131072;
		int nBatches = nTriangles / batchSize;
		int r = nTriangles % batchSize;
		device_vector<int> keyBuffer(nBuckets * sizeBucket, 0);
		device_vector<Halfedge> valBuffer(nBuckets * sizeBucket, { -1, -1, -1 });
		device_vector<int> countBuffer(nBuckets, 0);
		for (int i = 0; i < nBatches; ++i) {
			thrust::fill(countBuffer.begin(), countBuffer.end(), 0);
			k_constructHalfedges1Batch << <getBlockCount(nTriangles, BLOCK_SIZE), BLOCK_SIZE >> > (batchSize, raw(triangles), raw(countBuffer), raw(keyBuffer), raw(valBuffer),
				nBuckets, sizeBucket, i, batchSize);
			gpuErrchk(cudaPeekAtLastError());
			gpuErrchk(cudaDeviceSynchronize());
			int nnz = compressBatch<int, Halfedge>(keyBuffer, keys, valBuffer, vals, countBuffer, nBuckets, sizeBucket, nValid);
			nValid += nnz;
		}
		{
			thrust::fill(countBuffer.begin(), countBuffer.end(), 0);
			k_constructHalfedges1Batch << <getBlockCount(nTriangles, BLOCK_SIZE), BLOCK_SIZE >> > (r, raw(triangles), raw(countBuffer), raw(keyBuffer), raw(valBuffer),
				nBuckets, sizeBucket, nBatches, batchSize);
			gpuErrchk(cudaPeekAtLastError());
			gpuErrchk(cudaDeviceSynchronize());
			int nnz = compressBatch<int, Halfedge>(keyBuffer, keys, valBuffer, vals, countBuffer, nBuckets, sizeBucket, nValid);
			nValid += nnz;
		}
		keys.resize(nValid);
		vals.resize(nValid);
		thrust::sort_by_key(keys.begin(), keys.begin() + nValid, vals.begin());
		rowPtr.resize(nVerticesSurf + 1);
		k_find_first_n<<<getBlockCount(nValid, BLOCK_SIZE), BLOCK_SIZE>>>(nValid, raw(keys), raw(rowPtr), nVerticesSurf);
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
		halfedges = vals;
	}
	else {
		device_vector<int> vertexNumHalfedges(nVerticesSurf, 0);
		halfedges.resize(nVerticesSurf * MAX_HE_PER_VERTEX, { -1, -1, -1 });
		//constructHalfedgesGPU(vertexNumHalfedges, halfedges);
		k_constructHalfedges1 << <getBlockCount(nTriangles, BLOCK_SIZE), BLOCK_SIZE >> > (nTriangles, raw(triangles), raw(vertexNumHalfedges), raw(halfedges));
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());

		nValid = compress<Halfedge, isEmptyHalfedgeEntry>(halfedges, vertexNumHalfedges, rowPtr, nVerticesSurf);
	}

	device_vector<int> indexToCol(nValid);
	k_fillHelperArray << <getBlockCount(nVerticesSurf, BLOCK_SIZE), BLOCK_SIZE >> > (nVerticesSurf, rowPtr, indexToCol);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	k_setOppositeHalfedgeHelper << <getBlockCount(nValid, BLOCK_SIZE), BLOCK_SIZE >> > (nValid, halfedges, rowPtr, indexToCol);

	if (findBoundary1D) {
		cout << "init boundary1d" << endl;
		vertexIsBoundary1D = device_vector<bool>(nVerticesSurf, false);
		k_initBoundary1DRowPtr << <getBlockCount(nVerticesSurf, BLOCK_SIZE), BLOCK_SIZE >> > (nVerticesSurf, raw(rowPtr), raw(halfedges), raw(vertexIsBoundary1D));
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
		findBoundary1D = false;
}
	else {
		cout << "dont init boundary1d" << endl;
	}

	vertexIsFeature = device_vector<bool>(nVerticesSurf, false);
	if (isFlat) {
		thrust::copy(vertexIsBoundary1D.begin(), vertexIsBoundary1D.end(), vertexIsFeature.begin());
	}
	else {
		k_initFeaturesRowPtr << <getBlockCount(nVerticesSurf, BLOCK_SIZE), BLOCK_SIZE >> > (nVerticesSurf, vertexIsFeature, rowPtr, halfedges, raw(faceNormals), 20.f);
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
	}
}

__host__ int MeshTriGPU::sortVerticesRangeByColor(int start, int num, device_vector<int>& colors, CudaArray<int>& col_offsets, device_vector<int>& sortMapInverseOut, bool doReindex) {
	//need whole sortMap for mapping of elem indices
	device_vector<int> sortMap(nVerticesSurf);
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
	k_populateSortMapInverse << <getBlockCount((int)sortMap.size(), BLOCK_SIZE), BLOCK_SIZE >> > (nVerticesSurf, sortMap, sortMapInverseOut);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

	thrust::for_each(sortMap.begin() + start, sortMap.begin() + start + num, thrust::placeholders::_1 -= start);

	device_vector<Vec3f> vertexPointsSorted(num);
	thrust::gather(sortMap.begin() + start, sortMap.begin() + start + num, vertexPoints.begin() + start, vertexPointsSorted.begin());
	thrust::copy(vertexPointsSorted.begin(), vertexPointsSorted.end(), vertexPoints.begin() + start);

	if (vertexIsBoundary1D.size() > 0) {
		device_vector<bool> vertexIsBoundary1DSorted(num);
		thrust::gather(sortMap.begin() + start, sortMap.begin() + start + num, vertexIsBoundary1D.begin() + start, vertexIsBoundary1DSorted.begin());
		thrust::copy(vertexIsBoundary1DSorted.begin(), vertexIsBoundary1DSorted.end(), vertexIsBoundary1D.begin() + start);
	}
	if (vertexIsFeature.size() > 0) { // TODO we sort by feature before, so no need to sort this?
		device_vector<bool> vertexIsFeatureSorted(num);
		thrust::gather(sortMap.begin() + start, sortMap.begin() + start + num, vertexIsFeature.begin() + start, vertexIsFeatureSorted.begin());
		thrust::copy(vertexIsFeatureSorted.begin(), vertexIsFeatureSorted.end(), vertexIsFeature.begin() + start);
	}
	//if (vertexNormals.size() > 0) {
	//	device_vector<Vec3f> vertexNormalsSorted(num);
	//	thrust::gather(sortMap.begin() + start, sortMap.begin() + start + num, vertexNormals.begin() + start, vertexNormalsSorted.begin());
	//	thrust::copy(vertexNormalsSorted.begin(), vertexNormalsSorted.end(), vertexNormals.begin() + start);
	//}

	return numColors;
}

__host__ int MeshTriGPU::sortSurfVerticesByFeature(device_vector<int>& sortMapInverseOut) {
	device_vector<int> sortMap(nVerticesSurf);
	thrust::sequence(sortMap.begin(), sortMap.end());

	thrust::sort_by_key(vertexIsFeature.begin(), vertexIsFeature.begin() + nVerticesSurf, sortMap.begin());

	auto it = thrust::find(vertexIsFeature.begin(), vertexIsFeature.end(), true);
	int startPos = static_cast<int>(it - vertexIsFeature.begin());

	device_vector<Vec3f> vertexPointsSorted(nVerticesSurf);
	thrust::gather(sortMap.begin(), sortMap.end(), vertexPoints.begin(), vertexPointsSorted.begin());
	thrust::copy(vertexPointsSorted.begin(), vertexPointsSorted.end(), vertexPoints.begin());

	device_vector<bool> vertexIsBoundary1DSorted(nVerticesSurf);
	thrust::gather(sortMap.begin(), sortMap.end(), vertexIsBoundary1D.begin(), vertexIsBoundary1DSorted.begin());
	thrust::copy(vertexIsBoundary1DSorted.begin(), vertexIsBoundary1DSorted.end(), vertexIsBoundary1D.begin());

	// inverse sortmap for remapping triangle indices
	sortMapInverseOut = device_vector<int>(sortMap.size());
	const int BLOCK_SIZE = 128;
	k_populateSortMapInverse << <getBlockCount((int)sortMap.size(), BLOCK_SIZE), BLOCK_SIZE >> > (nVerticesSurf, sortMap, sortMapInverseOut);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

	return startPos;
}


//__host__ void MeshTriGPU::makeSimpleFulledgeVectors(device_vector<int>& vertexNumSimpleFulledges, device_vector<int>& simpleFulledges) {
//	vertexNumSimpleFulledges = device_vector<int>(nVerticesSurf, 0);
//	simpleFulledges = device_vector<int>(nVerticesSurf * MAX_HE_PER_VERTEX, -1);
//}

// TODO handle size 0
__host__ void MeshTriGPU::makeSimpleFulledgeFreeVectors(device_vector<int>& vertexNumSimpleFulledges, device_vector<int>& simpleFulledges) {
	vertexNumSimpleFulledges = device_vector<int>(nVerticesSurf - nVerticesFeature, 0);
	simpleFulledges = device_vector<int>((nVerticesSurf - nVerticesFeature) * MAX_HE_PER_VERTEX, -1);
}

__host__ void MeshTriGPU::makeSimpleFulledgeFeatureVectors(device_vector<int>& vertexNumSimpleFulledges, device_vector<int>& simpleFulledges) {
	vertexNumSimpleFulledges = device_vector<int>(nVerticesFeature, 0);
	simpleFulledges = device_vector<int>(nVerticesFeature * MAX_HE_PER_VERTEX, -1);
}

__host__ void MeshTriGPU::colorVerticesAndSort() {
	// sort feature vertices to beginning and boundary vertices to end
	device_vector<int> sortMapInverse1;
	int nVerticesFree = sortSurfVerticesByFeature(sortMapInverse1);
	nVerticesFeature = nVerticesSurf - nVerticesFree;
	remapElements(sortMapInverse1);

	Timer timer;
	/*Old*/
	// make "adjacency" matrix and compress it
	//device_vector<int> vertexNumSimpleFulledgesFree;
	//device_vector<int> simpleFulledgesFree;
	//device_vector<int> vertexNumSimpleFulledgesFeature;
	//device_vector<int> simpleFulledgesFeature;
	//device_vector<int> csrFulledgeRowPtrFree;// ((nVerticesFree)+1);
	//device_vector<int> csrFulledgeRowPtrFeature;// (nVerticesFeature + 1);

	//makeSimpleFulledgeFreeVectors(vertexNumSimpleFulledgesFree, simpleFulledgesFree);
	//makeSimpleFulledgeFeatureVectors(vertexNumSimpleFulledgesFeature, simpleFulledgesFeature);
	//
	//constructSimpleFulledgesFree(vertexNumSimpleFulledgesFree, simpleFulledgesFree);
	//constructSimpleFulledgesFeature(vertexNumSimpleFulledgesFeature, simpleFulledgesFeature);

	//// Convert SimpleHalfedge structures to compressed format
	//compress<int, isEmptySimpleHalfedgeEntry>(simpleFulledgesFree, vertexNumSimpleFulledgesFree, csrFulledgeRowPtrFree, nVerticesFree);
	//compress<int, isEmptySimpleHalfedgeEntry>(simpleFulledgesFeature, vertexNumSimpleFulledgesFeature, csrFulledgeRowPtrFeature, nVerticesFeature);

	//{
	//	std::ofstream ofs("oldRowPtr.txt");
	//	host_vector<int> rowPtrHost(csrFulledgeRowPtrFree);
	//	for (int i : rowPtrHost) {
	//		ofs << i << endl;
	//	}
	//	std::ofstream ofs2("old3.txt");
	//	host_vector<int> simpleFulledgesHost(simpleFulledgesFree);
	//	for (int i = 0; i < nVerticesFree; ++i) {
	//		for (int j = rowPtrHost[i]; j < rowPtrHost[i + 1]; ++j) {
	//			ofs2 << i << " " << simpleFulledgesHost[j] << endl;
	//		}
	//	}
	//}
	/*Old End*/

	/*New*/
	device_vector<int> simpleFulledgesFree;
	device_vector<int> simpleFulledgesFeature;
	device_vector<int> csrFulledgeRowPtrFree;
	device_vector<int> csrFulledgeRowPtrFeature;

	constructSimpleFulledgesFreeNew(simpleFulledgesFree, csrFulledgeRowPtrFree);
	constructSimpleFulledgesFeatureNew(simpleFulledgesFeature, csrFulledgeRowPtrFeature);
	/*New End*/

	cout << "Fulledges time: " << timer.timeInSeconds() << "s" << endl;

	// Color both halves
	bool doReindex = true;

	device_vector<int> colors1(nVerticesFree);
	if (nVerticesFree > 0) {
		int nnz = simpleFulledgesFree.size();
		//color_jpl(nVerticesFree, raw(csrFulledgeRowPtrFree), raw(simpleFulledgesFree), colors1);
		color_cuSPARSE(nVerticesFree, raw(csrFulledgeRowPtrFree), raw(simpleFulledgesFree), colors1, nnz);
		//color_chen_li(nVerticesFree, raw(csrFulledgeRowPtrFree), raw(simpleFulledgesFree), colors1, nnz);
		cout << "check colors free" << endl;
		checkColorsFree(colors1);

		device_vector<int> sortMapInverse2;
		nColorsFree = sortVerticesRangeByColor(0, nVerticesFree, colors1, col_offsets_free, sortMapInverse2, doReindex);
		remapElements(sortMapInverse2);
	}
	device_vector<int> colors2(nVerticesFeature);
	if (nVerticesFeature > 0) {
		int nnz = simpleFulledgesFeature.size();
		//color_jpl(nVerticesFeature, raw(csrFulledgeRowPtrFeature), raw(simpleFulledgesFeature), colors2);
		color_cuSPARSE(nVerticesFeature, raw(csrFulledgeRowPtrFeature), raw(simpleFulledgesFeature), colors2, nnz);
		//color_chen_li(nVerticesFeature, raw(csrFulledgeRowPtrFeature), raw(simpleFulledgesFeature), colors2, nnz);
		cout << "check colors feature" << endl;
		checkColorsFeature(colors2);

		device_vector<int> sortMapInverse2;
		nColorsFeature = sortVerticesRangeByColor(nVerticesFree, nVerticesFeature, colors2, col_offsets_feature, sortMapInverse2, doReindex);
		remapElements(sortMapInverse2);
	}

	//if (nVerticesFree > 0) {
	//	
	//}
	//if (nVerticesFeature > 0) {
	//	
	//}

	thrust::for_each(col_offsets_feature.begin(), col_offsets_feature.end(), thrust::placeholders::_1 += nVerticesFree);

} // colorVerticesAndSort

__host__ void MeshTriGPU::remapElements(device_vector<int>& sortMapInverse) {
	remapElementIndices(nTriangles, triangles.get(), sortMapInverse);
}

__host__ void MeshTriGPU::checkColorsFree(device_vector<int>& colors) {
	const int BLOCK_SIZE = 128;
	k_checkColoring << <getBlockCount(nTriangles, BLOCK_SIZE), BLOCK_SIZE >> > (nTriangles, triangles.data(), 0, nVerticesSurf - nVerticesFeature, raw(colors));
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
}

__host__ void MeshTriGPU::checkColorsFeature(device_vector<int>& colors) {
	const int BLOCK_SIZE = 128;
	k_checkColoring << <getBlockCount(nTriangles, BLOCK_SIZE), BLOCK_SIZE >> > (nTriangles, triangles.data(), nVerticesSurf - nVerticesFeature, nVerticesFeature, raw(colors));
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
}

__host__ void MeshTriGPU::calcFaceNormals() {
	faceNormals.resize(nTriangles);// = device_vector<Vec3f>(nTriangles);
	const int BLOCK_SIZE = 128;
	k_calcFaceNormals << <getBlockCount(nTriangles, BLOCK_SIZE), BLOCK_SIZE >> > (nTriangles, raw(vertexPoints), raw(faceNormals), raw(triangles));
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
}


__host__ void MeshTriGPU::calcVertexNormals() {
	vertexNormals.resize(nVerticesSurf);// = device_vector<Vec3f>(nVerticesSurf);
	const int BLOCK_SIZE = 128;
	k_calcVertexNormals << <getBlockCount(nVerticesSurf, BLOCK_SIZE), BLOCK_SIZE >> > (nVerticesSurf, raw(vertexNormals), raw(csrHalfedgesValues), raw(csrHalfedgeRowPtr), raw(faceNormals));
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
}

__host__ void MeshTriGPU::updateNormals() {
	const int BLOCK_SIZE = 128;
	k_calcFaceNormals << <getBlockCount(nTriangles, BLOCK_SIZE), BLOCK_SIZE >> > (nTriangles, raw(vertexPoints), raw(faceNormals), raw(triangles));
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	k_calcVertexNormals << <getBlockCount(nVerticesSurf, BLOCK_SIZE), BLOCK_SIZE >> > (nVerticesSurf, raw(vertexNormals), raw(csrHalfedgesValues), raw(csrHalfedgeRowPtr), raw(faceNormals));
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
}

__global__ void k_halfedgesToSoA(int size, Halfedge* halfedges, int* targetVertex_, int* oppositeHE_, int* incidentFace_) {
	for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < size; idx += blockDim.x * gridDim.x) {
		targetVertex_[idx] = halfedges[idx].targetVertex;
		oppositeHE_[idx] = halfedges[idx].oppositeHE;
		incidentFace_[idx] = halfedges[idx].incidentFace;
	}
}

__host__ void MeshTriGPU::makeHalfedgesSoA() {
	//soaHalfedgeTarget.resize(nHalfedges);
	//soaHalfedgeOpposite.resize(nHalfedges);
	//soaHalfedgeFace.resize(nHalfedges);

	//const int BLOCK_SIZE = 128;
	//k_halfedgesToSoA << <getBlockCount(nHalfedges, BLOCK_SIZE), BLOCK_SIZE >> > (nHalfedges, csrHalfedgesValues.data(), soaHalfedgeTarget.data(), soaHalfedgeOpposite.data(), soaHalfedgeFace.data());
	//gpuErrchk(cudaPeekAtLastError());
	//gpuErrchk(cudaDeviceSynchronize());
}











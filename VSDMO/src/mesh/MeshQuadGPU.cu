#include "MeshQuadGPU.h"
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


__host__ MeshQuadGPU::MeshQuadGPU() :
	MeshBaseGPU(),
	nQuads(0)
{

}

__host__ MeshQuadGPU::~MeshQuadGPU() {

}


__host__ void MeshQuadGPU::init() {
	calcFaceNormals(); //done

	initBoundary1DAndFeaturesNew(); //done i think

	colorVerticesAndSort(); //done i think

	constructTriHalfedgesNew();
	calcVertexNormals();

	//{
	//	cout << "vertices" << endl;
	//	auto devptr1 = vertexPoints.ptr();
	//	host_vector<Vec3f> vec(devptr1, devptr1 + nVerticesSurf);
	//	for (auto v : vec) {
	//		cout << v.x << " " << v.y << " " << v.z << endl;
	//	}
	//}

	//{
	//	cout << "bound1d" << endl;
	//	auto devptr1 = vertexIsBoundary1D.ptr();
	//	host_vector<bool> bound1d(devptr1, devptr1 + nVerticesSurf);
	//	for (auto b : bound1d) {
	//		cout << b << endl;
	//	}
	//}

	//{
	//	cout << "feature" << endl;
	//	auto devptr1 = vertexIsFeature.ptr();
	//	host_vector<bool> bound1d(devptr1, devptr1 + nVerticesSurf);
	//	for (auto b : bound1d) {
	//		cout << b << endl;
	//	}
	//}

	//{
	//	cout << "quads" << endl;
	//	auto devptr1 = quads.ptr();
	//	host_vector<Quad> vec(devptr1, devptr1 + nQuads);
	//	for (auto v : vec) {
	//		cout << v.v0 << " " << v.v1 << " " << v.v2 << " " << v.v3 << endl;
	//	}
	//}

	//{
	//	cout << "faceNormals" << endl;
	//	auto devptr1 = faceNormals.ptr();
	//	host_vector<Vec3f> vec(devptr1, devptr1 + nQuads);
	//	for (auto v : vec) {
	//		cout << v.x << " " << v.y << " " << v.z << endl;
	//	}
	//}

	//{
	//	cout << "hes" << endl;
	//	auto devptr1 = csrHalfedgesValues.ptr();
	//	host_vector<Halfedge> vec(devptr1, devptr1 + nHalfedges);
	//	for (int i = 0; i < nHalfedges; ++i) {
	//		auto v = vec[i];
	//		cout << v.targetVertex << " " << v.oppositeHE << " " << v.nextEdge << " " << v.incidentFace << endl;
	//	}
	//}


}

__host__ void MeshQuadGPU::setNumVerticesSurf(int n) {
	nVerticesSurf = n;
	vertexPoints.resize(n);
	vertexIsBoundary1D.resize(n);
}

__host__ void MeshQuadGPU::setNumQuads(int n) {
	nQuads = n;
	quads.resize(n);
}

__host__ void MeshQuadGPU::setVertexPoints(vector<Vec3f>& points, bool nonZeroZ) {
	if (!nonZeroZ) {
		//2d flat
		isFlat = true;
	}
	thrust::copy(points.begin(), points.end(), vertexPoints.begin());
	findBoundary1D = true;
}

__host__ void MeshQuadGPU::setVertexPointsWithBoundary1D(vector<Vec3f>& points, vector<bool>& boundary1d) {
	thrust::copy(points.begin(), points.end(), vertexPoints.begin());
	thrust::copy(boundary1d.begin(), boundary1d.end(), vertexIsBoundary1D.begin());
	findBoundary1D = false;
}

__host__ void MeshQuadGPU::setQuads(vector<Quad>& q) {
	thrust::copy(q.begin(), q.end(), quads.begin());
}

__host__ void MeshQuadGPU::fromDeviceData(int nVS, int nT, CudaArray<Vec3f>& points, CudaArray<Quad>& q) {
	nVerticesSurf = nVS;
	nQuads = nT;
	vertexPoints.resize(nVS);
	thrust::copy(points.begin(), points.begin() + nVS, vertexPoints.begin());
	quads.resize(nT);
	thrust::copy(q.begin(), q.begin() + nT, quads.begin());
	findBoundary1D = true;
}

__host__ void MeshQuadGPU::fromDeviceData(int nVS, int nT, device_ptr<Vec3f> points, device_ptr<Quad> q) {
	nVerticesSurf = nVS;
	nQuads = nT;
	vertexPoints.resize(nVS);
	thrust::copy(points, points + nVS, vertexPoints.begin());
	quads.resize(nT);
	thrust::copy(q, q + nT, quads.begin());
	findBoundary1D = true;
}


__host__ void MeshQuadGPU::constructTriHalfedgesNew() {
	int nHEMax = nQuads * 8;
	int nHE = nQuads * 4;
	device_vector<int> keys(nHEMax, -1);
	device_vector<Halfedge> halfedges(nHEMax, {-1, -1, -1, -1});
	csrHalfedgeRowPtr.resize(nVerticesSurf + 1);
	csrHalfedgeRowPtr.set(0, 0);

	const int BLOCK_SIZE = 128;
	k_constructHalfedges1New << <getBlockCount(nQuads, BLOCK_SIZE), BLOCK_SIZE >> > (nQuads, raw(quads), raw(halfedges), raw(keys)); //HERE
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

	//HERE TODO
	k_sortCCWDummyOpposite << <getBlockCount(nVerticesSurf, BLOCK_SIZE), BLOCK_SIZE >> > (nVerticesSurf, csrHalfedgeRowPtr, halfedges, csrHalfedgesValues,
		raw(vertexIsBoundary1D), raw(vertexIsFeature), raw(quads));
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

	k_setOppositeHalfedgeHelper << <getBlockCount(nHalfedges, BLOCK_SIZE), BLOCK_SIZE >> > (nHalfedges, csrHalfedgesValues, csrHalfedgeRowPtr, indexToCol);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

	k_setNextHalfedgeHelper << <getBlockCount(nHalfedges, BLOCK_SIZE), BLOCK_SIZE >> > (nHalfedges, csrHalfedgesValues, csrHalfedgeRowPtr, indexToCol, nVerticesSurf);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

	device_vector<int> halfedgesPerVertex(nVerticesSurf);
	thrust::transform(csrHalfedgeRowPtr.begin() + 1, csrHalfedgeRowPtr.end(), csrHalfedgeRowPtr.begin(), halfedgesPerVertex.begin(), thrust::minus<int>());

	auto maxNumHalfedgesIt = thrust::max_element(halfedgesPerVertex.begin(), halfedgesPerVertex.end());
	maxNumHalfedges = *maxNumHalfedgesIt;
	cout << "maxNumHalfedges " << maxNumHalfedges << endl;
}



__host__ void MeshQuadGPU::constructSimpleFulledgesFreeNew(device_vector<int>& simpleFulledges, device_vector<int>& rowPtr) {
	int nVerticesFree = nVerticesSurf - nVerticesFeature;
	device_vector<int> keys(nQuads * 12, -1);
	simpleFulledges.resize(nQuads * 12, -1);
	thrust::fill(simpleFulledges.begin(), simpleFulledges.end(), -1);
	rowPtr.resize(nVerticesFree + 1);
	thrust::fill(rowPtr.begin(), rowPtr.end(), -1);
	rowPtr[0] = 0;

	const int BLOCK_SIZE = 128;
	k_constructFulledgesFromQuadsFreeNew << <getBlockCount(nQuads, BLOCK_SIZE), BLOCK_SIZE >> > (nQuads, raw(quads), raw(simpleFulledges), raw(keys), 0, nVerticesFree);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

	auto endit = thrust::remove_if(keys.begin(), keys.end(), isNegative());
	keys.erase(endit, keys.end());
	int nFulledges = keys.size();
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

__host__ void MeshQuadGPU::constructSimpleFulledgesFeatureNew(device_vector<int>& simpleFulledges, device_vector<int>& rowPtr) {
	int nVerticesFree = nVerticesSurf - nVerticesFeature;
	device_vector<int> keys(nQuads * 12, -1);
	simpleFulledges.resize(nQuads * 12, -1);
	thrust::fill(simpleFulledges.begin(), simpleFulledges.end(), -1);
	rowPtr.resize(nVerticesFeature + 1);
	thrust::fill(rowPtr.begin(), rowPtr.end(), -1);
	rowPtr[0] = 0;

	const int BLOCK_SIZE = 128;
	k_constructFulledgesFromQuadsFeatureNew << <getBlockCount(nQuads, BLOCK_SIZE), BLOCK_SIZE >> > (nQuads, raw(quads), raw(simpleFulledges), raw(keys), nVerticesFree, nVerticesFeature);
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



__host__ void MeshQuadGPU::initBoundary1DAndFeaturesNew() {
	int nHE = nQuads * 4;
	device_vector<int> keys(nHE);
	device_vector<Halfedge> halfedges(nHE);
	device_vector<int> rowPtr(nVerticesSurf + 1);
	rowPtr[0] = 0;

	const int BLOCK_SIZE = 128;
	k_constructHalfedges1New << <getBlockCount(nQuads, BLOCK_SIZE), BLOCK_SIZE >> > (nQuads, raw(quads), raw(halfedges), raw(keys));
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

	thrust::sort_by_key(keys.begin(), keys.begin() + nHE, halfedges.begin());
	k_find_first_n << <getBlockCount(nHE, BLOCK_SIZE), BLOCK_SIZE >> > (nHE, raw(keys), raw(rowPtr), nVerticesSurf);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

	device_vector<int> indexToCol(nHE);
	k_fillHelperArray << <getBlockCount(nVerticesSurf, BLOCK_SIZE), BLOCK_SIZE >> > (nVerticesSurf, rowPtr, indexToCol);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

	k_setOppositeHalfedgeHelper << <getBlockCount(nHE, BLOCK_SIZE), BLOCK_SIZE >> > (nHE, halfedges, rowPtr, indexToCol);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

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


__host__ int MeshQuadGPU::sortVerticesRangeByColor(int start, int num, device_vector<int>& colors, CudaArray<int>& col_offsets, device_vector<int>& sortMapInverseOut, bool doReindex) {
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

__host__ int MeshQuadGPU::sortSurfVerticesByFeature(device_vector<int>& sortMapInverseOut) {
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



// TODO handle size 0
__host__ void MeshQuadGPU::makeSimpleFulledgeFreeVectors(device_vector<int>& vertexNumSimpleFulledges, device_vector<int>& simpleFulledges) {
	vertexNumSimpleFulledges = device_vector<int>(nVerticesSurf - nVerticesFeature, 0);
	simpleFulledges = device_vector<int>((nVerticesSurf - nVerticesFeature) * MAX_HE_PER_VERTEX, -1);
}

__host__ void MeshQuadGPU::makeSimpleFulledgeFeatureVectors(device_vector<int>& vertexNumSimpleFulledges, device_vector<int>& simpleFulledges) {
	vertexNumSimpleFulledges = device_vector<int>(nVerticesFeature, 0);
	simpleFulledges = device_vector<int>(nVerticesFeature * MAX_HE_PER_VERTEX, -1);
}

__host__ void MeshQuadGPU::colorVerticesAndSort() {
	// sort feature vertices to beginning and boundary vertices to end
	device_vector<int> sortMapInverse1;
	int nVerticesFree = sortSurfVerticesByFeature(sortMapInverse1);
	nVerticesFeature = nVerticesSurf - nVerticesFree;
	remapElements(sortMapInverse1);

	Timer timer;

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

		host_vector<int> colors2Host(colors2);

		device_vector<int> sortMapInverse2;
		nColorsFeature = sortVerticesRangeByColor(nVerticesFree, nVerticesFeature, colors2, col_offsets_feature, sortMapInverse2, doReindex);
		remapElements(sortMapInverse2);
	}

	thrust::for_each(col_offsets_feature.begin(), col_offsets_feature.end(), thrust::placeholders::_1 += nVerticesFree);

} // colorVerticesAndSort

__host__ void MeshQuadGPU::remapElements(device_vector<int>& sortMapInverse) {
	remapElementIndices(nQuads, quads.get(), sortMapInverse);
}

__host__ void MeshQuadGPU::checkColorsFree(device_vector<int>& colors) {
	const int BLOCK_SIZE = 128;
	k_checkColoring << <getBlockCount(nQuads, BLOCK_SIZE), BLOCK_SIZE >> > (nQuads, quads.data(), 0, nVerticesSurf - nVerticesFeature, raw(colors));
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
}

__host__ void MeshQuadGPU::checkColorsFeature(device_vector<int>& colors) {
	const int BLOCK_SIZE = 128;
	k_checkColoring << <getBlockCount(nQuads, BLOCK_SIZE), BLOCK_SIZE >> > (nQuads, quads.data(), nVerticesSurf - nVerticesFeature, nVerticesFeature, raw(colors));
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
}

__host__ void MeshQuadGPU::calcFaceNormals() {
	faceNormals.resize(nQuads);// = device_vector<Vec3f>(nTriangles);
	const int BLOCK_SIZE = 128;
	k_calcFaceNormals << <getBlockCount(nQuads, BLOCK_SIZE), BLOCK_SIZE >> > (nQuads, raw(vertexPoints), raw(faceNormals), raw(quads));
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
}


__host__ void MeshQuadGPU::calcVertexNormals() {
	vertexNormals.resize(nVerticesSurf);// = device_vector<Vec3f>(nVerticesSurf);
	const int BLOCK_SIZE = 128;
	k_calcVertexNormals << <getBlockCount(nVerticesSurf, BLOCK_SIZE), BLOCK_SIZE >> > (nVerticesSurf, raw(vertexNormals), raw(csrHalfedgesValues), raw(csrHalfedgeRowPtr), raw(faceNormals));
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
}

__host__ void MeshQuadGPU::updateNormals() {
	const int BLOCK_SIZE = 128;
	k_calcFaceNormals << <getBlockCount(nQuads, BLOCK_SIZE), BLOCK_SIZE >> > (nQuads, raw(vertexPoints), raw(faceNormals), raw(quads));
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	k_calcVertexNormals << <getBlockCount(nVerticesSurf, BLOCK_SIZE), BLOCK_SIZE >> > (nVerticesSurf, raw(vertexNormals), raw(csrHalfedgesValues), raw(csrHalfedgeRowPtr), raw(faceNormals));
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
}











#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "MeshTypes.h"
#include "CudaUtil.h"



//template<class T1, class T2, class T3>
//class CompressedSoA {
//public:
//	CompressedStorage() : rowPtr_(nullptr), values1_(nullptr), values2_(nullptr), values3_(nullptr) {}
//	CompressedStorage(int* rowPtr) : rowPtr_(rowPtr), values1_(nullptr), values2_(nullptr), values3_(nullptr) {}
//	~CompressedStorage() {}
//
//	int numItems(int col) {
//		return rowPtr_[col + 1] - rowPtr_[col];
//	}
//
//	int* rowPtr_;
//	T1* values1_;
//	T2* values2_;
//	T3* values3_;
//};



class CompressedHalfedges {
public:
	CompressedHalfedges() : rowPtr_(nullptr), targetVertex_(nullptr), oppositeHE_(nullptr), incidentFace_(nullptr) {}
	CompressedHalfedges(int* rowPtr, int* targetVertex, int* oppositeHE, int* incidentFace) : rowPtr_(rowPtr), targetVertex_(targetVertex), oppositeHE_(oppositeHE), incidentFace_(incidentFace) {
		
	}
	~CompressedHalfedges() {}

	__device__ int numItems(int col) {
		return rowPtr_[col + 1] - rowPtr_[col];
	}

	__device__ int targetVertex(int col, int index) {
		return targetVertex_[rowPtr_[col] + index];
	}

	__device__ int oppositeHE(int col, int index) {
		return oppositeHE_[rowPtr_[col] + index];
	}

	__device__ int incidentFace(int col, int index) {
		return incidentFace_[rowPtr_[col] + index];
	}

	int* rowPtr_;
	int* targetVertex_;
	int* oppositeHE_;
	int* incidentFace_;
};







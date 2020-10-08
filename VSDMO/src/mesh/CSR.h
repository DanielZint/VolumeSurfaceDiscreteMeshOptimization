#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


template<class ValueType>
class CompressedStorage {
public:
	CompressedStorage() : rowPtr_(nullptr), values_(nullptr) {}
	CompressedStorage(int* rowPtr, ValueType* values) : rowPtr_(rowPtr), values_(values) {}
	~CompressedStorage() {}

	int numItems(int col) {
		return rowPtr_[col + 1] - rowPtr_[col];
	}

	ValueType& at(int col, int index) {
		return values_[rowPtr_[col] + index];
	}

	ValueType at(int col, int index) const {
		return values_[rowPtr_[col] + index];
	}
	
	
	class Iterator;
	__host__ __device__ Iterator begin(int v) {
		return Iterator(*this, v);
	}
	__host__ __device__ Iterator end(int v) {
		return Iterator(*this, v + 1);
	}

	class RIterator;
	__host__ __device__ RIterator rbegin(int v) {
		return RIterator(*this, v + 1);
	}
	__host__ __device__ RIterator rend(int v) {
		return RIterator(*this, v);
	}


	class Iterator {
	public:
		__host__ __device__ Iterator(const CompressedStorage& cs) : cs_(cs), v_(0), curr_(nullptr) {}
		__host__ __device__ Iterator(const CompressedStorage& cs, int v) : cs_(cs), v_(v), curr_(cs.values_ + cs.rowPtr_[v]) {}

		__host__ __device__ Iterator& operator++() {
			++curr_;
			return *this;
		}

		__host__ __device__ int operator-(const Iterator& other) {
			return int(curr_ - other.curr_);
		}

		__host__ __device__ bool operator==(const Iterator& other) const {
			return curr_ == other.curr_;
		}

		__host__ __device__ bool operator!=(const Iterator& other) const {
			return curr_ != other.curr_;
		}

		__host__ __device__ const ValueType& operator*() const {
			return *curr_;
		}

		__host__ __device__ const ValueType* operator->() const {
			return curr_;
		}

		__host__ __device__ bool valid() const {
			return curr_ < (cs_.values_ + cs_.rowPtr_[v_ + 1]);
		}

	private:
		const CompressedStorage& cs_;
		const int v_;
		const ValueType* curr_;
	};


	class RIterator {
	public:
		__host__ __device__ RIterator(const CompressedStorage& cs) : cs_(cs), v_(0), curr_(nullptr) {}
		__host__ __device__ RIterator(const CompressedStorage& cs, int v) : cs_(cs), v_(v), curr_(cs.values_ + cs.rowPtr_[v] - 1) {}

		__host__ __device__ RIterator& operator++() {
			--curr_;
			return *this;
		}

		__host__ __device__ bool operator==(const RIterator& other) const {
			return curr_ == other.curr_;
		}

		__host__ __device__ bool operator!=(const RIterator& other) const {
			return curr_ != other.curr_;
		}

		__host__ __device__ const ValueType& operator*() const {
			return *curr_;
		}

		__host__ __device__ const ValueType* operator->() const {
			return curr_;
		}

		__host__ __device__ bool valid() const {
			return curr_ >= (cs_.values_ + cs_.rowPtr_[v_ - 1] - 1);
		}

	private:
		const CompressedStorage& cs_;
		const int v_;
		const ValueType* curr_;
	};


	int* rowPtr_;
	ValueType* values_;
};



template<class ValueType>
class CSR {
public:
	CSR() : rowPtr_(nullptr), colInd_(nullptr), values_(nullptr) {}
	CSR(int* rowPtr, int* colInd, ValueType* values) : rowPtr_(rowPtr), colInd_(colInd), values_(values) {}
	~CSR() {}
	
	//indexed and ordered iterator
	class Iterator;
	__host__ __device__ Iterator ordered_begin(int v) {
		return Iterator(*this, v);
	}
	__host__ __device__ Iterator ordered_end(int v) {
		return Iterator(*this, v + 1);
	}

	class RIterator;
	__host__ __device__ RIterator ordered_rbegin(int v) {
		return RIterator(*this, v + 1);
	}
	__host__ __device__ RIterator ordered_rend(int v) {
		return RIterator(*this, v);
	}

	class IndexedIterator;
	__host__ __device__ IndexedIterator indexed_begin(int v) {
		return IndexedIterator(*this, v);
	}
	__host__ __device__ IndexedIterator indexed_end(int v) {
		return IndexedIterator(*this, v + 1);
	}


	class Iterator {
	public:
		__host__ __device__ Iterator(const CSR& csr) : csr_(csr), v_(0), curr_(nullptr) {}
		__host__ __device__ Iterator(const CSR& csr, int v) : csr_(csr), v_(v), curr_(csr.values_ + csr.rowPtr_[v]) {}

		__host__ __device__ Iterator& operator++() {
			++curr_;
			return *this;
		}

		__host__ __device__ bool operator==(const Iterator& other) const {
			return curr_ == other.curr_;
		}

		__host__ __device__ bool operator!=(const Iterator& other) const {
			return curr_ != other.curr_;
		}

		__host__ __device__ const ValueType& operator*() const {
			return *curr_;
		}

		__host__ __device__ const ValueType* operator->() const {
			return curr_;
		}

		__host__ __device__ bool valid() const {
			return curr_ < (csr_.values_ + csr_.rowPtr_[v_ + 1]);
		}

	private:
		const CSR& csr_;
		const int v_;
		const ValueType* curr_;
	};


	class RIterator {
	public:
		__host__ __device__ RIterator(const CSR& csr) : csr_(csr), v_(0), curr_(nullptr) {}
		__host__ __device__ RIterator(const CSR& csr, int v) : csr_(csr), v_(v), curr_(csr.values_ + csr.rowPtr_[v] - 1) {}

		__host__ __device__ RIterator& operator++() {
			--curr_;
			return *this;
		}

		__host__ __device__ bool operator==(const RIterator& other) const {
			return curr_ == other.curr_;
		}

		__host__ __device__ bool operator!=(const RIterator& other) const {
			return curr_ != other.curr_;
		}

		__host__ __device__ const ValueType& operator*() const {
			return *curr_;
		}

		__host__ __device__ const ValueType* operator->() const {
			return curr_;
		}

		__host__ __device__ bool valid() const {
			return curr_ >= (csr_.values_ + csr_.rowPtr_[v_ - 1] - 1);
		}

	private:
		const CSR& csr_;
		const int v_;
		const ValueType* curr_;
	};


	class IndexedIterator {
	public:
		__host__ __device__ IndexedIterator(const CSR& csr) : csr_(csr), v_(0), curr_(0) {}
		__host__ __device__ IndexedIterator(const CSR& csr, int v) : csr_(csr), v_(v), curr_(csr.rowPtr_[v]) {}

		__host__ __device__ IndexedIterator& operator++() {
			++curr_;
			return *this;
		}

		__host__ __device__ bool operator==(const IndexedIterator& other) const {
			return curr_ == other.curr_;
		}

		__host__ __device__ bool operator!=(const IndexedIterator& other) const {
			return curr_ != other.curr_;
		}

		__host__ __device__ const ValueType& operator*() const {
			return csr_.values_[csr_.colInd_[curr_]];
		}

		__host__ __device__ const ValueType& getOrderedValue() const {
			return csr_.values_[curr_];
		}

		__host__ __device__ bool valid() const {
			return curr_ < csr_.rowPtr_[v_ + 1];
		}

	private:
		const CSR& csr_;
		const int v_;
		int curr_;
	};

	int* rowPtr_;
	int* colInd_;
	ValueType* values_;

};





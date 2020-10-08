#pragma once

#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>
#include "cuda_runtime.h"
#include "ConfigUsing.h"

using thrust::device_ptr;

#define raw(device_vec) raw_pointer_cast(device_vec.data())

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s. File %s Line %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

template<class T>
class CudaArray {
// Stealable data, ArrayView for host side
private:
	device_ptr<T> ptr_;
	size_t size_;
public:
	CudaArray() :
		size_(0)
	{}

	CudaArray(int _size) :
		size_(_size)
	{
		T* data;
		gpuErrchk(cudaMalloc((void**)&data, sizeof(T) * size_));
		ptr_ = thrust::device_pointer_cast(data);
		//ptr = thrust::device_malloc<T>(size);
	}

	CudaArray(const device_vector<T>& dv) :
		CudaArray(dv.size())
	{
		thrust::copy(dv.begin(), dv.end(), ptr_);
	}

	CudaArray(const host_vector<T>& hv) :
		CudaArray(hv.size())
	{
		thrust::copy(hv.begin(), hv.end(), ptr_);
	}

	CudaArray(const std::vector<T>& v) :
		CudaArray(v.size())
	{
		gpuErrchk(cudaMemcpy(ptr_, v.data(), sizeof(T) * size_, cudaMemcpyHostToDevice));
	}

	CudaArray(const CudaArray& other) = delete;
	CudaArray& operator=(const CudaArray& other) = delete;

	CudaArray& operator=(const device_vector<T>& dv) {
		resize(dv.size());
		thrust::copy(dv.begin(), dv.end(), begin());
		return *this;
	}

	/*Does not copy old data over*/
	void resize(size_t newsize) {
		//cout << "here" << newsize << endl;
		if (newsize == size_) return;
		if (size_ > 0) {
			gpuErrchk(cudaFree(ptr_.get()));
		}
		size_ = newsize;
		T* data;
		gpuErrchk(cudaMalloc((void**)&data, sizeof(T) * newsize));
		ptr_ = thrust::device_pointer_cast(data);
	}

	device_ptr<T> begin() {
		return ptr_;
	}

	auto end() -> decltype(ptr_ + size_) {
		return (ptr_ + size_);
	}

	T* get() {
		return ptr_.get();
	}

	T* data() {
		return ptr_.get();
	}

	/*T& operator[](size_t i) {
		return ptr[i];
	}*/

	void set(size_t pos, const T& val) {
		ptr_[pos] = val;
	}

	T operator[] (size_t i) const {
		T ret = ptr_[i];
		return ret;
	}

	void deleteMem() {
		gpuErrchk(cudaFree(ptr_.get()));
	}

	size_t size() const {
		return size_;
	}

	device_ptr<T> ptr() {
		return ptr_;
	}
};




template<class T>
class ArrayView {
public:
	__host__ ArrayView(thrust::device_vector<T>& vec) : data_(thrust::raw_pointer_cast(&vec[0])), size_(vec.size()) {
		
	}

	__host__ ArrayView(CudaArray<T>& a) : data_(a.get()), size_(a.size()) {
	
	}

	__host__ __device__ ~ArrayView() {

	}

	__host__ __device__ T& operator[](size_t i) {
		return data_[i];
	}

	__host__ __device__ size_t size() {
		return size_;
	}

	__host__ __device__ T* data() {
		return data_;
	}
private:
	T *data_;
	size_t size_;
};

template <typename T1, typename T2>
constexpr T1 getBlockCount(T1 problemSize, T2 threadCount) {
	return (problemSize + (threadCount - T2(1))) / (threadCount);
}





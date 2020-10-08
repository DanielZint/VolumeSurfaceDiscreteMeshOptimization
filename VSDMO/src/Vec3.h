#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "ConfigUsing.h"


template<class T = float>
struct Vec3 {
	union {
		struct {
			T x;
			T y;
			T z;
		};
		T data[3];
	};
	
	typedef T value_type;

	__host__ __device__ Vec3() {}
	__host__ __device__ Vec3(T x_, T y_, T z_) : x(x_), y(y_), z(z_) {}
	__host__ __device__ Vec3(float *arr) : x(arr[0]), y(arr[1]), z(arr[2]) {}

	__host__ __device__ Vec3& operator=(const Vec3& v) {
		x = v.x;
		y = v.y;
		z = v.z;
		return *this;
	}

	//__host__ __device__ Vec3& operator=(const Vec3f& v) {
	//	x = v.x;
	//	y = v.y;
	//	z = v.z;
	//	return *this;
	//}

	__host__ __device__ T& operator[](int i) {
		return data[i];
	}

	__host__ __device__ T operator[](int i) const {
		return data[i];
	}

	__host__ __device__ Vec3& operator+=(const Vec3& v) {
		x += v.x;
		y += v.y;
		z += v.z;
		return *this;
	}

	__host__ __device__ Vec3 operator+(const Vec3& v) const {
		Vec3 r = *this;
		r.x += v.x;
		r.y += v.y;
		r.z += v.z;
		return r;
	}

	__host__ __device__ Vec3& operator-=(const Vec3& v) {
		x -= v.x;
		y -= v.y;
		z -= v.z;
		return *this;
	}

	__host__ __device__ Vec3 operator-(const Vec3& v) const {
		Vec3 r = *this;
		r.x -= v.x;
		r.y -= v.y;
		r.z -= v.z;
		return r;
	}

	__host__ __device__ Vec3 operator-() const {
		Vec3 v;
		v.x = -x;
		v.y = -y;
		v.z = -z;
		return v;
	}

	__host__ __device__ T dot(const Vec3& v) const {
		return x * v.x + y * v.y + z * v.z;
	}

	__host__ __device__ Vec3 cross(const Vec3& v) const {
		return Vec3(y*v.z-z*v.y, z*v.x-x*v.z, x*v.y-y*v.x);
	}

	__host__ __device__ T squaredNorm() const {
		return (x * x + y * y + z * z);
	}

	__host__ __device__ T norm() const {
		return sqrtf(x * x + y * y + z * z);
	}

	__host__ __device__ void normalize() {
		T len = norm();
		x /= len;
		y /= len;
		z /= len;
	}

	__host__ __device__ Vec3 normalized() const {
		T len = sqrtf(x * x + y * y + z * z);
		return Vec3(x/len, y/len, z/len);
	}
};

template<class T>
__host__ __device__ Vec3<T> operator*(T s, const Vec3<T>& v) {
	return Vec3<T>(s * v.x, s * v.y, s * v.z);
}

template<class T>
__host__ __device__ Vec3<T> operator*(const Vec3<T>& v, T s) {
	return Vec3<T>(s * v.x, s * v.y, s * v.z);
}



using Vec3f = Vec3<float>;

#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "ConfigUsing.h"
#include "Vec3.h"


template<class T = float>
struct Vec2 {
	union {
		struct {
			T x;
			T y;
		};
		T data[2];
	};
	
	typedef T value_type;

	__host__ __device__ Vec2() {}
	__host__ __device__ Vec2(T x_, T y_) : x(x_), y(y_) {}
	__host__ __device__ Vec2(float *arr) : x(arr[0]), y(arr[1]) {}
	__host__ __device__ Vec2(const Vec3<T>& v) : x(v.x), y(v.y) {}

	__host__ __device__ Vec2& operator=(const Vec2& v) {
		x = v.x;
		y = v.y;
		return *this;
	}

	__host__ __device__ Vec2& operator=(const Vec3<T>& v) {
		x = v.x;
		y = v.y;
		return *this;
	}

	__host__ __device__ T& operator[](int i) {
		return data[i];
	}

	__host__ __device__ T operator[](int i) const {
		return data[i];
	}

	__host__ __device__ Vec2& operator+=(const Vec2& v) {
		x += v.x;
		y += v.y;
		return *this;
	}

	__host__ __device__ Vec2 operator+(const Vec2& v) const {
		Vec2 r = *this;
		r.x += v.x;
		r.y += v.y;
		return r;
	}

	__host__ __device__ Vec2& operator-=(const Vec2& v) {
		x -= v.x;
		y -= v.y;
		return *this;
	}

	__host__ __device__ Vec2 operator-(const Vec2& v) const {
		Vec2 r = *this;
		r.x -= v.x;
		r.y -= v.y;
		return r;
	}

	__host__ __device__ Vec2 operator-() const {
		Vec2 v;
		v.x = -x;
		v.y = -y;
		return v;
	}

	__host__ __device__ T dot(const Vec2& v) const {
		return x * v.x + y * v.y;
	}

	__host__ __device__ Vec2 cross(const Vec2& v) const {
		return x*v.y-y*v.x;
	}

	__host__ __device__ T squaredNorm() const {
		return (x * x + y * y);
	}

	__host__ __device__ T norm() const {
		return sqrtf(x * x + y * y);
	}

	__host__ __device__ void normalize() {
		T len = norm();
		x /= len;
		y /= len;
	}

	__host__ __device__ Vec2 normalized() const {
		T len = sqrtf(x * x + y * y);
		return Vec2(x/len, y/len);
	}
};

template<class T>
__host__ __device__ Vec2<T> operator*(T s, const Vec2<T>& v) {
	return Vec2<T>(s * v.x, s * v.y);
}

template<class T>
__host__ __device__ Vec2<T> operator*(const Vec2<T>& v, T s) {
	return Vec2<T>(s * v.x, s * v.y);
}



using Vec2f = Vec2<float>;

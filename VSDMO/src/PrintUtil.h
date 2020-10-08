#pragma once


#include <iostream>
#include <vector>
#include "ConfigUsing.h"

using std::string;
using std::cout;
using std::endl;

inline void print(Vec3f v) {
	cout << v[0] << " " << v[1] << " " << v[2] << endl;
}

template<class T, int N>
inline void print(std::array<T, N> a) {
	for (int i = 0; i < N; ++i) {
		cout << a[i] << " ";
	}
	cout << endl;
}

template<class T>
inline void print(std::vector<T> v) {
	for (auto x : v) {
		cout << x << " ";
	}
}


#pragma once


#include <string>
#include <sstream>
#include <vector>
#include <algorithm>
#include <iterator>
#include <array>
#include "ConfigUsing.h"

using std::string;

void ignoreComment(string& line);
void ignorePrefixWhitespace(string& line);
Vec3f parseVertexCoord(string line);
int parseInt(string& line);
template<int N> std::array<int, N> parseIntN(string line) {
	std::array<int, N> a;
	std::istringstream iss(line);
	for (int i = 0; i < N; ++i) {
		iss >> a[i];
	}
	return a;
}



inline Vec3f parseVertexCoord(std::istringstream& iss) {
	std::array<float, 3> a;
	iss >> a[0] >> a[1] >> a[2];
	return Vec3f(a.data());
}

inline int parseInt(std::istringstream& iss) {
	int i;
	iss >> i;
	return i;
}

inline float parseFloat(std::istringstream& iss) {
	float f;
	iss >> f;
	return f;
}

template<int N> inline std::array<int, N> parseIntN(std::istringstream& iss) {
	std::array<int, N> a;
	for (int i = 0; i < N; ++i) {
		iss >> a[i];
	}
	return a;
}

#pragma once

#include <fstream>

template<class T>
void writeSurfaces(device_vector<T> dv, const std::string& filename) {
	host_vector<T> hv(dv);
	std::ofstream ofs(filename, std::ios::out | std::ios::binary);
	ofs.write((char*)hv.data(), hv.size() * sizeof(T));
	ofs.close();
}
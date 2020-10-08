#include "ParserHex.h"

#include "ParserUtil.h"
#include <vector>
#include <string>
#include <fstream>
#include <iostream>

using std::cout;
using std::endl;
using std::ifstream;


// Currently assumes well formed input files
//int parseHex2(std::string filename, MeshHexGPU& mesh) {
//	cout << "Reading " << filename << endl;
//	std::ifstream ifs(filename);
//	if (!ifs.is_open()) {
//		cout << "couldn't open file " << filename << endl;
//		return -1;
//	}
//
//	// Find first line
//	std::string line;
//	{
//		std::getline(ifs, line);
//		std::istringstream iss(line);
//		std::string header;
//		int version;
//		iss >> header >> version;
//		cout << header << version << endl;
//		if (header.compare(string("MeshVersionFormatted")) != 0) {
//			throw std::exception("Expected MeshVersionFormatted");
//		}
//	}
//	{
//		std::getline(ifs, line);
//		std::istringstream iss(line);
//		std::string dimensionStr;
//		int dimension;
//		iss >> dimensionStr >> dimension;
//		cout << dimensionStr << dimension << endl;
//		if (dimensionStr.compare(string("Dimension")) != 0) {
//			throw std::exception("Expected Dimension");
//		}
//	}
//	int nVertices;
//	{
//		std::getline(ifs, line);
//		std::istringstream iss(line);
//		std::string verticesStr;
//		iss >> verticesStr >> nVertices;
//		cout << verticesStr << nVertices << endl;
//		if (verticesStr.compare(string("Vertices")) != 0) {
//			throw std::exception("Expected Vertices");
//		}
//	}
//	
//	mesh.setNumVertices(nVertices);
//	std::vector<Vec3f> points(nVertices);
//	
//	for (int j = 0; j < nVertices; ++j) {
//		std::getline(ifs, line);
//		std::istringstream iss(line);
//		Vec3f point = parseVertexCoord(iss);
//		points[j] = point;
//		int ref;
//		iss >> ref;
//	}
//	mesh.setVertexPoints(points);
//
//	int nHexahedra;
//	{
//		std::getline(ifs, line);
//		std::istringstream iss(line);
//		std::string hexahedraStr;
//		iss >> hexahedraStr >> nHexahedra;
//		cout << hexahedraStr << nHexahedra << endl;
//		if (hexahedraStr.compare(string("Hexahedra")) != 0) {
//			throw std::exception("Expected Hexahedra");
//		}
//	}
//
//	
//	std::vector<Hexahedron> hexes(nHexahedra);
//	mesh.setNumHexahedra(nHexahedra);
//	bool reorder = true;
//
//	for (int j = 0; j < nHexahedra; ++j) {
//		std::getline(ifs, line);
//		std::istringstream iss(line);
//		Hexahedron hexIn;
//		hexIn.data2 = parseIntN<8>(iss);
//		for (int& i : hexIn.data2) {
//			--i;
//		}
//		Hexahedron hex;
//		if (reorder) {
//			hex.v0 = hexIn.v4;
//			hex.v1 = hexIn.v5;
//			hex.v2 = hexIn.v6;
//			hex.v3 = hexIn.v7;
//			hex.v4 = hexIn.v0;
//			hex.v5 = hexIn.v1;
//			hex.v6 = hexIn.v2;
//			hex.v7 = hexIn.v3;
//		}
//		else {
//			hex = hexIn;
//		}
//		
//		hexes[j] = hex;
//	}
//	mesh.setHexahedra(hexes);
//	return 0;
//}


// from https://github.com/cnr-isti-vclab/HexaLab/blob/master/src/loader.cpp
int parseHex(std::string filename, MeshHexGPU& mesh) {
	string header;

	vector<Vec3f> vertices;
	vector<Hexahedron> hexes;

	ifstream stream(filename, ifstream::in | ifstream::binary);
	if (!stream.is_open()) {
		cout << "couldn't open file " << filename << endl;
		return -1;
	}

	int precision;
	int dimension;

	while (stream.good()) {
		// Read a line
		stream >> header;

		// Precision
		if (header.compare("MeshVersionFormatted") == 0) {
			stream >> precision;
			// Dimension
		}
		else if (header.compare("Dimension") == 0) {
			stream >> dimension;
			// Vertices
		}
		else if (header.compare("Vertices") == 0) {
			int vertices_count;
			stream >> vertices_count;
			vertices.resize(vertices_count);
			mesh.setNumVertices(vertices_count);
			for (int i = 0; i < vertices_count; ++i) {
				Vec3f v;
				float x;
				stream >> v.x >> v.y >> v.z >> x;
				vertices[i] = v;
			}
			// Quad indices
		}
		else if (header.compare("Quadrilaterals") == 0 || header.compare("Quads") == 0) {
			int quads_count;
			stream >> quads_count;
			for (int i = 0; i < quads_count; ++i) {
				int idx[4];
				int x;
				stream >> idx[0] >> idx[1] >> idx[2] >> idx[3] >> x;
			}
			// Hex indices
		}
		else if (header.compare("Hexahedra") == 0) {
			int hexas_count;
			stream >> hexas_count;
			bool reorder = true;
			hexes.resize(hexas_count);
			mesh.setNumHexahedra(hexas_count);
			for (int h = 0; h < hexas_count; ++h) {

				Hexahedron hexIn;
				int x;
				stream >> hexIn.data[0] >> hexIn.data[1] >> hexIn.data[2] >> hexIn.data[3] >> hexIn.data[4] >> hexIn.data[5] >> hexIn.data[6] >> hexIn.data[7] >> x;
				for (int& i : hexIn.data2) {
					--i;
				}
				Hexahedron hex;
				if (reorder) {
					hex.v0 = hexIn.v4;
					hex.v1 = hexIn.v5;
					hex.v2 = hexIn.v6;
					hex.v3 = hexIn.v7;
					hex.v4 = hexIn.v0;
					hex.v5 = hexIn.v1;
					hex.v6 = hexIn.v2;
					hex.v7 = hexIn.v3;
				}
				else {
					hex = hexIn;
				}

				hexes[h] = hex;
			}
			// End of file
		}
		else if (header.compare("End") == 0) {
			break;
			// Unknown token
		}
		else {
			cout << "Unexpected header tag" << endl;
			return -1;
		}
	}

	vector<int> usedVert(vertices.size(), 0);
	for (int i = 0; i < hexes.size(); ++i) {
		for (int j = 0; j < 8; ++j) {
			int index = hexes[i].data[j];
			if (index < 0 || index >= vertices.size()) {
				cout << "ERROR: hex " << i << " has index out of range with val " << index << endl;
				return -1;
			}
			++usedVert[index];
		}
	}

	// And yell a warning for unreferenced vertices...
	int unrefCnt = 0;
	for (size_t i = 0; i < vertices.size(); ++i) {
		if (usedVert[i] == 0) unrefCnt++;
	}
	if (unrefCnt > 0) {
		cout << "There are unused vertices " << unrefCnt << endl;
		return -1;
	}

	mesh.setVertexPoints(vertices);
	mesh.setHexahedra(hexes);
	return 0;
}


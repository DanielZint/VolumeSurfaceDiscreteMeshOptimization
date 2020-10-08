#include "ParserOFF.h"

#include "ParserUtil.h"
//#include "PrintUtil.h"
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include "ConfigUsing.h"

using std::cout;
using std::endl;

// Currently assumes well formed input files
// filename: file.off
int parseOFF(std::string filename, MeshTriGPU& mesh) {
	cout << "Reading " << filename << endl;
	std::ifstream ifs(filename);
	if (!ifs.is_open()) {
		cout << "couldn't open file " << filename << endl;
		return -1;
	}

	// Find first line
	std::string line;

	std::getline(ifs, line);
	if (line.compare(string("OFF")) != 0) {
		throw std::exception("First line is not OFF");
	}

	do {
		std::getline(ifs, line);
		ignoreComment(line);
		ignorePrefixWhitespace(line);
	} while (line.size() == 0);

	std::istringstream iss(line);
	std::array<int, 3> arr = parseIntN<3>(iss);

	int nPoints = arr[0];
	int nFaces = arr[1];
	int nEdges = arr[2];
	//cout << nPoints << endl;

	// Alloc Vertex Arrays
	mesh.setNumVerticesSurf(nPoints);

	// Read all vertices
	std::vector<Vec3f> points(nPoints);
	//std::vector<bool> boundary(nPoints);

	bool nonZeroZ = false;

	for (int j = 0; j < nPoints; ++j) {
		std::getline(ifs, line);
		std::istringstream iss(line);

		Vec3f point = parseVertexCoord(iss);
		if (point.z != 0.f) {
			nonZeroZ = true;
		}

		points[j] = point;
		//print(point);
	}
	mesh.setVertexPoints(points, nonZeroZ); //HERE

	//--
	std::vector<Triangle> tris(nFaces);
	mesh.setNumTriangles(nFaces);

	// Read all vertices
	for (int j = 0; j < nFaces; ++j) {
		std::getline(ifs, line);
		std::istringstream iss2(line);

		int nVertsFace = parseInt(iss2);
		
		Triangle tri;
		tri.data2 = parseIntN<3>(iss2);
		tris[j] = tri;
	}
	mesh.setTriangles(tris);
	return 0;
}



int parseOFF(std::string filename, MeshQuadGPU& mesh) {
	cout << "Reading " << filename << endl;
	std::ifstream ifs(filename);
	if (!ifs.is_open()) {
		cout << "couldn't open file " << filename << endl;
		return -1;
	}

	// Find first line
	std::string line;

	std::getline(ifs, line);
	if (line.compare(string("OFF")) != 0) {
		throw std::exception("First line is not OFF");
	}

	do {
		std::getline(ifs, line);
		ignoreComment(line);
		ignorePrefixWhitespace(line);
	} while (line.size() == 0);

	std::istringstream iss(line);
	std::array<int, 3> arr = parseIntN<3>(iss);

	int nPoints = arr[0];
	int nFaces = arr[1];
	int nEdges = arr[2];
	//cout << nPoints << endl;

	// Alloc Vertex Arrays
	mesh.setNumVerticesSurf(nPoints);

	// Read all vertices
	std::vector<Vec3f> points(nPoints);
	//std::vector<bool> boundary(nPoints);

	bool nonZeroZ = false;

	for (int j = 0; j < nPoints; ++j) {
		std::getline(ifs, line);
		std::istringstream iss(line);

		Vec3f point = parseVertexCoord(iss);
		if (point.z != 0.f) {
			nonZeroZ = true;
		}

		points[j] = point;
		//print(point);
	}
	mesh.setVertexPoints(points, nonZeroZ); //HERE

	//--
	std::vector<Quad> quads(nFaces);
	mesh.setNumQuads(nFaces);

	// Read all vertices
	for (int j = 0; j < nFaces; ++j) {
		std::getline(ifs, line);
		std::istringstream iss2(line);

		int nVertsFace = parseInt(iss2);

		Quad quad;
		quad.data2 = parseIntN<4>(iss2);
		quads[j] = quad;
	}
	mesh.setQuads(quads);
	return 0;
}


int getFaceType(std::string filename) {
	std::ifstream ifs(filename);
	if (!ifs.is_open()) {
		cout << "couldn't open file " << filename << endl;
		return -1;
	}

	// Find first line
	std::string line;

	std::getline(ifs, line);
	if (line.compare(string("OFF")) != 0) {
		throw std::exception("First line is not OFF");
	}

	do {
		std::getline(ifs, line);
		ignoreComment(line);
		ignorePrefixWhitespace(line);
	} while (line.size() == 0);

	std::istringstream iss(line);
	std::array<int, 3> arr = parseIntN<3>(iss);

	int nPoints = arr[0];
	int nFaces = arr[1];
	int nEdges = arr[2];

	bool nonZeroZ = false;

	for (int j = 0; j < nPoints; ++j) {
		std::getline(ifs, line);
	}

	
	std::getline(ifs, line);
	std::istringstream iss2(line);

	int nVertsFace = parseInt(iss2);

	return nVertsFace;
}


#include "ParserTetGen.h"

#include "ParserUtil.h"
#include <vector>
#include <string>
#include <fstream>
#include <iostream>

using std::cout;
using std::endl;


// Currently assumes well formed input files
int parseTetGen(std::string filename, MeshTetGPU& mesh) {
	cout << "Reading " << filename << ".node" << endl;
	cout << "Reading " << filename << ".ele" << endl;
	std::string filenameNode = filename + ".node";
	std::string filenameEle = filename + ".ele";

	std::ifstream ifsNode(filenameNode);
	if (!ifsNode.is_open()) {
		cout << "couldn't open file " << filenameNode << endl;
		return -1;
	}
	std::ifstream ifsEle(filenameEle);
	if (!ifsEle.is_open()) {
		cout << "couldn't open file " << filenameEle << endl;
		return -1;
	}

	bool startAtOne = false;
	{
		// Find first line
		std::string line;
		do {
			std::getline(ifsNode, line);
			ignoreComment(line);
			ignorePrefixWhitespace(line);
		} while (line.size() == 0);

		std::istringstream iss(line);
		std::array<int, 4> arr = parseIntN<4>(iss);

		int nPoints = arr[0];
		int nAttrNode = arr[2];
		bool boundaryMarker = arr[3];

		
		mesh.setNumVertices(nPoints);

		// Buffers
		std::vector<Vec3f> points(nPoints);
		std::vector<bool> boundary(nPoints);

		for (int j = 0; j < nPoints; ++j) {
			std::getline(ifsNode, line);
			std::istringstream iss(line);
			int vertNum = parseInt(iss);
			if (j == 0 && vertNum == 1) {
				startAtOne = true;
			}
			Vec3f point = parseVertexCoord(iss);

			for (int i = 0; i < nAttrNode; ++i) {
				parseFloat(iss);
			}
			int marker = 0;
			if (boundaryMarker) {
				marker = parseInt(iss);
			}

			points[j] = point;
			boundary[j] = marker;
		}
		if (boundaryMarker) {
			mesh.setVertexPointsWithBoundary2D(points, boundary);
		}
		else {
			mesh.setVertexPoints(points);
		}
		
	}

	{
		std::string line;
		do {
			std::getline(ifsEle, line);
			ignoreComment(line);
			ignorePrefixWhitespace(line);
		} while (line.size() == 0);

		std::istringstream iss2(line);
		std::array<int, 3> arr2 = parseIntN<3>(iss2);

		int nTets = arr2[0];
		int nAttrTet = arr2[2];

		mesh.setNumTetrahedra(nTets);

		std::vector<Tetrahedron> tets(nTets);

		for (int j = 0; j < nTets; ++j) {
			std::getline(ifsEle, line);
			std::istringstream iss2(line);
			int tetNum = parseInt(iss2);
			Tetrahedron tet;
			tet.data2 = parseIntN<4>(iss2);
			if (startAtOne) {
				tet.v0--;
				tet.v1--;
				tet.v2--;
				tet.v3--;
			}
			tets[j] = tet;
		}
		mesh.setTetrahedra(tets);
	}
	return 0;
}


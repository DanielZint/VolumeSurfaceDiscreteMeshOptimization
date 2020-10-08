#include "FileWriter.h"
#include "ConfigUsing.h"

#include <vector>
#include <string>
#include <fstream>
#include <iostream>

using std::cout;
using std::endl;

void writeOFF(std::string filename, DMOMeshTri& mesh) {
	auto slashpos = filename.find_last_of("/\\");
	std::string filepath = filename.substr(0, slashpos+1);
	std::string file = filename.substr(slashpos+1);
	std::string outfilepath = filepath + std::string("out/") + file;
	cout << "Writing " << outfilepath << ".off" << endl;
	std::ofstream ofs(outfilepath + ".off");
	if (!ofs.is_open()) {
		cout << "couldn't open file " << (outfilepath + ".off") << endl;
	}
	thrust::host_vector<Vec3f> verticesHost(mesh.vertexPoints, mesh.vertexPoints + mesh.nVerticesSurf);
	thrust::host_vector<Triangle> trianglesHost(mesh.triangles, mesh.triangles + mesh.nTriangles);

	ofs << "OFF" << '\n';
	ofs << mesh.nVerticesSurf << " " << mesh.nTriangles << " 0" << '\n';
	for (int i = 0; i < mesh.nVerticesSurf; ++i) {
		Vec3f v = verticesHost[i];
		ofs << v.x << " " << v.y << " " << v.z << '\n';
	}
	for (auto it = trianglesHost.begin(); it != trianglesHost.end(); ++it) {
		Triangle t = *it;
		ofs << "3 " << t.v0 << " " << t.v1 << " " << t.v2 << '\n';
	}
	ofs << std::flush;
}

void writeOFF(std::string filename, DMOMeshQuad& mesh) {
	auto slashpos = filename.find_last_of("/\\");
	std::string filepath = filename.substr(0, slashpos + 1);
	std::string file = filename.substr(slashpos + 1);
	std::string outfilepath = filepath + std::string("out/") + file;
	cout << "Writing " << outfilepath << ".off" << endl;
	std::ofstream ofs(outfilepath + ".off");
	if (!ofs.is_open()) {
		cout << "couldn't open file " << (outfilepath + ".off") << endl;
	}
	thrust::host_vector<Vec3f> verticesHost(mesh.vertexPoints, mesh.vertexPoints + mesh.nVerticesSurf);
	thrust::host_vector<Quad> quadsHost(mesh.quads, mesh.quads + mesh.nQuads);

	ofs << "OFF" << '\n';
	ofs << mesh.nVerticesSurf << " " << mesh.nQuads << " 0" << '\n';
	for (int i = 0; i < mesh.nVerticesSurf; ++i) {
		Vec3f v = verticesHost[i];
		ofs << v.x << " " << v.y << " " << v.z << '\n';
	}
	for (auto it = quadsHost.begin(); it != quadsHost.end(); ++it) {
		Quad t = *it;
		ofs << "4 " << t.v0 << " " << t.v1 << " " << t.v2 << " " << t.v3 << '\n';
	}
	ofs << std::flush;
}

void writeTetgen(std::string filename, DMOMeshTet& mesh) {
	auto slashpos = filename.find_last_of("/\\");
	std::string filepath = filename.substr(0, slashpos + 1);
	std::string file = filename.substr(slashpos + 1);
	std::string outfilepath = filepath + std::string("out/") + file;
	cout << "Writing " << outfilepath << ".node" << endl;
	cout << "Writing " << outfilepath << ".ele" << endl;
	std::ofstream ofsNode(outfilepath + ".node");
	if (!ofsNode.is_open()) {
		cout << "couldn't open file " << (outfilepath + ".node") << endl;
	}
	std::ofstream ofsEle(outfilepath + ".ele");
	if (!ofsEle.is_open()) {
		cout << "couldn't open file " << (outfilepath + ".ele") << endl;
	}
	thrust::host_vector<Vec3f> verticesHost(mesh.vertexPoints, mesh.vertexPoints + mesh.nVertices);
	thrust::host_vector<Tetrahedron> tetrahedraHost(mesh.tetrahedra, mesh.tetrahedra + mesh.nTetrahedra);

	ofsNode << "#some comment" << '\n';
	//First line : <# of points> <dimension(must be 3)> <# of attributes> <# of boundary markers(0 or 1)>
	ofsNode << mesh.nVertices << " 3 0 0" << '\n';
	for (int i = 0; i < mesh.nVertices; ++i) {
		Vec3f v = verticesHost[i];
		//<point #> <x> <y> <z> [attributes] [boundary marker]
		ofsNode << i << " " << v.x << " " << v.y << " " << v.z << '\n';
	}
	ofsNode << std::flush;

	ofsEle << "#some comment" << '\n';
	//First line: <# of tetrahedra> <nodes per tetrahedron> <# of attributes> 
	ofsEle << mesh.nTetrahedra << " 4 0" << '\n';
	for (int i = 0; i < mesh.nTetrahedra; ++i) {
		Tetrahedron t = tetrahedraHost[i];
		//<tetrahedron #> <node> <node> <node> <node> ... [attributes]
		ofsEle << i << " " << t.v0 << " " << t.v1 << " " << t.v2 << " " << t.v3 << '\n';
	}
	ofsEle << std::flush;
}


void writeHex(std::string filename, DMOMeshHex& mesh) {
	auto slashpos = filename.find_last_of("/\\");
	std::string filepath = filename.substr(0, slashpos + 1);
	std::string file = filename.substr(slashpos + 1);
	std::string outfilepath = filepath + std::string("out/") + file;
	cout << "Writing " << outfilepath << ".mesh" << endl;
	std::ofstream ofs(outfilepath + ".mesh");
	if (!ofs.is_open()) {
		cout << "couldn't open file " << (outfilepath + ".mesh") << endl;
	}
	thrust::host_vector<Vec3f> verticesHost(mesh.vertexPoints, mesh.vertexPoints + mesh.nVertices);
	thrust::host_vector<Hexahedron> hexahedraHost(mesh.hexahedra, mesh.hexahedra + mesh.nHexahedra);

	ofs << "MeshVersionFormatted 1" << '\n';
	ofs << "Dimension 3" << '\n';
	ofs << "Vertices " << mesh.nVertices << '\n';
	for (int i = 0; i < mesh.nVertices; ++i) {
		Vec3f v = verticesHost[i];
		ofs << v.x << " " << v.y << " " << v.z << " 0" << '\n';
	}
	bool reorder = true;
	ofs << "Hexahedra " << mesh.nHexahedra << '\n';
	for (auto it = hexahedraHost.begin(); it != hexahedraHost.end(); ++it) {
		Hexahedron hex = *it;
		for (int& i : hex.data2) {
			i++;
		}
		Hexahedron hexOut = hex;
		if (reorder) {
			hexOut.v0 = hex.v4;
			hexOut.v1 = hex.v5;
			hexOut.v2 = hex.v6;
			hexOut.v3 = hex.v7;
			hexOut.v4 = hex.v0;
			hexOut.v5 = hex.v1;
			hexOut.v6 = hex.v2;
			hexOut.v7 = hex.v3;
		}
		ofs << hexOut.v0 << " " << hexOut.v1 << " " << hexOut.v2 << " " << hexOut.v3 << " " << hexOut.v4 << " " << hexOut.v5 << " " << hexOut.v6 << " " << hexOut.v7 << " 0" << '\n';
	}
	ofs << "End" << '\n';
	ofs << std::flush;
}


#pragma once


#include <array>

struct Halfedge {
	int targetVertex;
	int oppositeHE; //index into halfedges
	int incidentFace;
	int nextEdge;
};

//struct CSRHalfedge {
//	int oppositeHE; //index into csrHalfedgesValues
//	int incidentFace;
//};

struct Edge {
	int v0, v1;
	int t0, t1;
};

struct Triangle {
	static constexpr int numVertices = 3;
	union {
		int data[3];
		std::array<int, 3> data2;
		struct {
			int v0, v1, v2;
		};
	};
};

using Halfface = Triangle;

struct Tetrahedron {
	static constexpr int numVertices = 4;
	union {
		int data[4];
		std::array<int, 4> data2;
		struct {
			int v0, v1, v2, v3;
		};
	};
};

struct Hexahedron {
	static constexpr int numVertices = 8;
	union {
		int data[8];
		std::array<int, 8> data2;
		struct {
			int v0, v1, v2, v3, v4, v5, v6, v7;
		};
	};
};

struct Halfhex {
	static constexpr int numVertices = 7;
	union {
		int data[7];
		std::array<int, 7> data2;
		struct {
			int v0, v1, v2, v3, v4, v5, v6;
		};
	};
};


struct HalfedgeQ {
	int targetVertex;
	int oppositeHE; //index into halfedges
	int incidentFace;
	int nextEdge;
};

struct Quad {
	static constexpr int numVertices = 4;
	union {
		int data[4];
		std::array<int, 4> data2;
		struct {
			int v0, v1, v2, v3;
		};
	};
};



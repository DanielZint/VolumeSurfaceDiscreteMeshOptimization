#pragma once

#include "ConfigUsing.h"
#include "MeshTriGPU.h"
#include "MeshTetGPU.h"
#include "MeshQuadGPU.h"
#include "DMOMeshTri.h"
#include "DMOMeshTet.h"
#include "DMOMeshQuad.h"
#include "DMOMeshHex.h"
#include "io/ParserOFF.h"
#include "io/ParserTetGen.h"
#include "io/ParserHex.h"



class DMOMeshTriFactory {
public:
	static std::shared_ptr<DMOMeshTri> create(std::string filename);
	static std::shared_ptr<DMOMeshTri> create(int nVS, int nT, CudaArray<Vec3f>& points, CudaArray<Triangle>& tris);
	static std::shared_ptr<DMOMeshTri> create(int nVS, int nT, device_ptr<Vec3f>& points, device_ptr<Triangle>& tris);
	static std::shared_ptr<DMOMeshTri> create(DMOMeshTet& tetmesh);
};

class DMOMeshTetFactory {
public:
	static std::shared_ptr<DMOMeshTet> create(std::string filename);
};

class DMOMeshQuadFactory {
public:
	static std::shared_ptr<DMOMeshQuad> create(std::string filename);
	static std::shared_ptr<DMOMeshQuad> create(int nVS, int nT, CudaArray<Vec3f>& points, CudaArray<Quad>& tris);
	static std::shared_ptr<DMOMeshQuad> create(int nVS, int nT, device_ptr<Vec3f>& points, device_ptr<Quad>& tris);
	static std::shared_ptr<DMOMeshQuad> create(DMOMeshHex& hexmesh);
};

class DMOMeshHexFactory {
public:
	static std::shared_ptr<DMOMeshHex> create(std::string filename);
};


//class DMOMeshFactory {
//public:
//	static std::shared_ptr<DMOMeshBase> create(std::string filename);
//};


#include "MeshFactory.h"

std::shared_ptr<DMOMeshTri> DMOMeshTriFactory::create(std::string filename) {
	MeshTriGPU m;
	if (parseOFF(filename, m)) {
		return nullptr;
	}
	cout << "parsing done" << endl;
	m.init();

	auto ret = std::make_shared<DMOMeshTri>(m);
	return ret;
}

std::shared_ptr<DMOMeshTri> DMOMeshTriFactory::create(int nVS, int nT, CudaArray<Vec3f>& points, CudaArray<Triangle>& tris) {
	MeshTriGPU m;
	m.fromDeviceData(nVS, nT, points, tris);
	m.init();

	auto ret = std::make_shared<DMOMeshTri>(m);
	return ret;
}

std::shared_ptr<DMOMeshTri> DMOMeshTriFactory::create(int nVS, int nT, device_ptr<Vec3f>& points, device_ptr<Triangle>& tris) {
	MeshTriGPU m;
	m.fromDeviceData(nVS, nT, points, tris);
	m.init();

	auto ret = std::make_shared<DMOMeshTri>(m);
	return ret;
}

std::shared_ptr<DMOMeshTri> DMOMeshTriFactory::create(DMOMeshTet& tetmesh) {
	MeshTriGPU m;
	m.fromDeviceData(tetmesh.nVerticesSurf, tetmesh.nTriangles, tetmesh.vertexPoints, tetmesh.triangles);
	m.init();

	auto ret = std::make_shared<DMOMeshTri>(m);
	return ret;
}




std::shared_ptr<DMOMeshTet> DMOMeshTetFactory::create(std::string filename) {
	MeshTetGPU m;
	if (parseTetGen(filename, m)) {
		return nullptr;
	}
	cout << "parsing done" << endl;
	m.init();

	auto ret = std::make_shared<DMOMeshTet>(m);
	return ret;
}




std::shared_ptr<DMOMeshQuad> DMOMeshQuadFactory::create(std::string filename) {
	MeshQuadGPU m;
	if (parseOFF(filename, m)) {
		return nullptr;
	}
	cout << "parsing done" << endl;
	m.init();

	auto ret = std::make_shared<DMOMeshQuad>(m);
	return ret;
}

std::shared_ptr<DMOMeshQuad> DMOMeshQuadFactory::create(int nVS, int nT, CudaArray<Vec3f>& points, CudaArray<Quad>& quads) {
	MeshQuadGPU m;
	m.fromDeviceData(nVS, nT, points, quads);
	m.init();

	auto ret = std::make_shared<DMOMeshQuad>(m);
	return ret;
}

std::shared_ptr<DMOMeshQuad> DMOMeshQuadFactory::create(int nVS, int nT, device_ptr<Vec3f>& points, device_ptr<Quad>& quads) {
	MeshQuadGPU m;
	m.fromDeviceData(nVS, nT, points, quads);
	m.init();

	auto ret = std::make_shared<DMOMeshQuad>(m);
	return ret;
}

std::shared_ptr<DMOMeshQuad> DMOMeshQuadFactory::create(DMOMeshHex& hexmesh) {
	MeshQuadGPU m;
	m.fromDeviceData(hexmesh.nVerticesSurf, hexmesh.nQuads, hexmesh.vertexPoints, hexmesh.quads);
	m.init();

	auto ret = std::make_shared<DMOMeshQuad>(m);
	return ret;
}



std::shared_ptr<DMOMeshHex> DMOMeshHexFactory::create(std::string filename) {
	MeshHexGPU m;
	if (parseHex(filename, m)) {
		return nullptr;
	}
	cout << "parsing done" << endl;
	m.init();

	auto ret = std::make_shared<DMOMeshHex>(m);
	return ret;
}



//std::shared_ptr<DMOMeshBase> DMOMeshFactory::create(std::string filename) {
//	MeshQuadGPU m;
//	if (parseOFF(filename, m)) {
//		return nullptr;
//	}
//	cout << "parsing done" << endl;
//	m.init();
//
//	auto ret = std::make_shared<DMOMeshQuad>(m);
//	return ret;
//}



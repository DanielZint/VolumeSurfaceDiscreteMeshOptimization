#include "io/ParserUtil.h"
#include "io/ParserTetGen.h"
#include "io/ParserOFF.h"
#include "io/ParserHex.h"
#include "mesh/MeshTriGPU.h"
#include "mesh/MeshFactory.h"
#include "mesh/DMOMeshTri.h"
#include "mesh/DMOMeshTet.h"
#include "mesh/DMOMeshQuad.h"
#include "mesh/DMOMeshHex.h"
#include "dmo/DMOTri.h"
#include "dmo/DMOTet.h"
#include "dmo/DMOQuad.h"
#include "dmo/DMOHex.h"
#include "dmo/Surface1D.h"
#include "io/FileWriter.h"
#include "ConfigUsing.h"
#include "ConfigOpenMesh.h"
#include "Timer.h"

#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>
#include <thrust/sequence.h>

#include "graphics/Viewer.h"
#include "graphics/shaderManager.h"

//#include <filesystem>

//using namespace std::filesystem; //error : namespace "std" has no member "filesystem"

void initCuda() {
	cudaSetDevice(0);
	cudaFree(0);
}

void launchGui() {
	//Graphics::Viewer viewer;
	//viewer.renderLoop();
	//viewer.shutdown();
	std::unique_ptr<Graphics::Viewer> viewer = std::make_unique<Graphics::Viewer>();
	viewer->renderLoop();
	viewer->shutdown();

}



void dmoVolumeTet(const std::string& filename) {
	cout << "Processing Tetrahedral mesh " << filename << endl;
	Timer timer;
	// optimize surface of volume mesh and then generate surface mesh and save
	auto dmo_mesh = DMOMeshTetFactory::create(filename);
	if (dmo_mesh == nullptr) {
		return;
	}
	cout << "Mesh Data Structure Time: " << timer.timeInSeconds() << "s" << endl;

	auto dmo_mesh3 = DMOMeshTriFactory::create(*dmo_mesh);

	writeOFF(filename + "_in_surf", *dmo_mesh3);

	DMO::DMOTetClass dmo(*dmo_mesh);
	dmo.displayQualityGPU();
	dmo.optimize();
	dmo.displayQualityGPU();
	writeTetgen(std::string(filename + "_out"), *dmo_mesh);

	auto dmo_mesh2 = DMOMeshTriFactory::create(*dmo_mesh);
	writeOFF(filename + "_out_surf", *dmo_mesh2);
}

void dmoVolumeHex(const std::string& inputFile) {
	cout << "Processing Hexahedral mesh " << inputFile << endl;
	Timer timer;
	std::size_t pointPos = inputFile.find_last_of(".");

	std::string fileName = inputFile.substr(0, pointPos);
	std::string fileExtension = inputFile.substr(pointPos);
	std::string outputFile = fileName + std::string("_out");

	// optimize surface of volume mesh and then generate surface mesh and save
	auto dmo_mesh = DMOMeshHexFactory::create(inputFile);
	if (dmo_mesh == nullptr) {
		return;
	}
	cout << "Mesh Data Structure Time: " << timer.timeInSeconds() << "s" << endl;

	auto dmo_mesh3 = DMOMeshQuadFactory::create(*dmo_mesh);

	writeOFF(fileName + "_in_surf", *dmo_mesh3);

	DMO::DMOHexClass dmo(*dmo_mesh);
	dmo.displayQualityGPU();
	dmo.optimize();
	dmo.displayQualityGPU();
	writeHex(std::string(outputFile), *dmo_mesh);

	auto dmo_mesh2 = DMOMeshQuadFactory::create(*dmo_mesh);
	writeOFF(fileName + "_out_surf", *dmo_mesh2);
}

void dmoSurfaceTri(const std::string& inputFile) {
	cout << "Processing Triangle mesh " << inputFile << endl;
	Timer timer;
	std::size_t pointPos = inputFile.find_last_of(".");

	std::string fileName = inputFile.substr(0, pointPos);
	std::string fileExtension = inputFile.substr(pointPos);
	std::string outputFile = fileName + std::string("_out");

	auto dmo_mesh = DMOMeshTriFactory::create(inputFile);
	if (dmo_mesh == nullptr) {
		return;
	}
	cout << "Mesh Data Structure Time: " << timer.timeInSeconds() << "s" << endl;

	DMO::DMOTriClass dmo2(*dmo_mesh);
	dmo2.displayQualityGPU();
	dmo2.optimize();
	dmo2.displayQualityGPU();
	writeOFF(outputFile, *dmo_mesh);
}

void dmoSurfaceQuad(const std::string& inputFile) {
	cout << "Processing Quad mesh " << inputFile << endl;
	Timer timer;
	std::size_t pointPos = inputFile.find_last_of(".");

	std::string fileName = inputFile.substr(0, pointPos);
	std::string fileExtension = inputFile.substr(pointPos);
	std::string outputFile = fileName + std::string("_out");

	auto dmo_mesh = DMOMeshQuadFactory::create(inputFile);
	if (dmo_mesh == nullptr) {
		return;
	}
	cout << "Mesh Data Structure Time: " << timer.timeInSeconds() << "s" << endl;

	DMO::DMOQuadClass dmo2(*dmo_mesh);
	dmo2.displayQualityGPU();
	dmo2.optimize();
	dmo2.displayQualityGPU();
	writeOFF(outputFile, *dmo_mesh);
}

void processFiles(const std::vector<std::string>& inputFiles) {
	for (const std::string& inputFile : inputFiles) {
		//std::string outputFile;
		//std::string fileExtension;
		std::size_t pointPos = inputFile.find_last_of(".");
		std::string fileExtension;
		if (pointPos != std::string::npos) {
			fileExtension = inputFile.substr(pointPos + 1);
			cout << fileExtension << endl;
		}

		if (fileExtension.compare("node") == 0 || fileExtension.compare("ele") == 0) {
			//outputFile = inputFile + std::string("_out");
			dmoVolumeTet(inputFile.substr(0, pointPos));
		}
		else if (fileExtension.compare("mesh") == 0) {
			dmoVolumeHex(inputFile);
		}
		else {
			int nNodesFace = getFaceType(inputFile);
			if (nNodesFace == 3) {
				std::string fileName = inputFile.substr(0, pointPos);
				//fileExtension = inputFile.substr(pointPos);
				//outputFile = fileName + std::string("_out") + fileExtension;
				dmoSurfaceTri(inputFile);
			}
			else if (nNodesFace == 4) {
				std::string fileName = inputFile.substr(0, pointPos);
				//fileExtension = inputFile.substr(pointPos);
				//outputFile = fileName + std::string("_out") + fileExtension;
				dmoSurfaceQuad(inputFile);
			}
			
		}
	}
}


//void testVolumeDeterministic(const std::string& filename, int n) {
//	for (int i = 0; i < n; ++i) {
//		auto dmo_mesh = DMOMeshTetFactory::create(filename);
//		DMO::DMOTetClass dmo(*dmo_mesh);
//		//dmo.displayQualityGPU();
//		dmo.optimize();
//		//dmo.displayQualityGPU();
//		std::string outname = std::string("res/test") + std::to_string(i);
//		writeTetgen(std::string(outname), *dmo_mesh);
//	}
//}
//
//void testSurfaceDeterministic(const std::string& filename, int n) {
//	for (int i = 0; i < n; ++i) {
//		auto dmo_mesh = DMOMeshTriFactory::create(filename);
//		DMO::DMOTriClass dmo(*dmo_mesh);
//		//dmo.displayQualityGPU();
//		dmo.optimize();
//		//dmo.displayQualityGPU();
//		std::string outname = std::string("res/test") + std::to_string(i);
//		writeOFF(std::string(outname), *dmo_mesh);
//	}
//}

string findResPath(string exePath) {
	vector<string> pathParts;
	size_t found = exePath.find("\\");
	pathParts.push_back(exePath.substr(0, found));
	size_t nfound = 0;
	while ((nfound = exePath.find("\\", found + 1)) != string::npos) {
		pathParts.push_back(exePath.substr(found + 1, nfound - found - 1));
		found = nfound;
	}
	int j = 0;
	for (int i = pathParts.size() - 1; i >= 0; --i) {
		if (pathParts[i].compare("bin") == 0) {
			j = i;
			break;
		}
	}

	string resPath;
	for (int i = 0; i < j; ++i) {
		resPath += pathParts[i];
		resPath += string("\\");
	}
	resPath += string("res");
	return resPath;
}

int main(int argc, char* argv[]) {

	//OMTriMesh me;
	//OpenMesh::IO::read_mesh(me, string("res/m95.off"));


	Timer timer;
	initCuda();


	//string exePath(argv[0]);
	//cout << exePath << endl;
	resPath = findResPath(argv[0]);
	cout << resPath << endl;

	

	bool withGui = true;
	std::vector<std::string> inputFiles;

	for (int i = 1; i < argc; ++i) {
		if (std::string("-p").compare(argv[i]) == 0) {
			withGui = false;
		}
		else {
			inputFiles.push_back(argv[i]);
		}
	}




	if (withGui) {
		launchGui();
	}
	else {
		if (inputFiles.empty()) {
			cout << "No input files" << endl;
			return 0;
		}
		processFiles(inputFiles);
	}
	


	cout << "Time: " << timer.timeInSeconds() << "s" << endl;
	//timer.printTimeInSeconds();

	cudaDeviceReset();

	return 0;
}





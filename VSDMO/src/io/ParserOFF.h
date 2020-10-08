#pragma once

#include "mesh/MeshTriGPU.h"
#include "mesh/MeshQuadGPU.h"
#include <string>


int parseOFF(std::string filename, MeshTriGPU& mesh);
int parseOFF(std::string filename, MeshQuadGPU& mesh);

int getFaceType(std::string filename);

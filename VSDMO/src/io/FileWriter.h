#pragma once

#include "mesh/MeshTriGPU.h"
#include "mesh/MeshTetGPU.h"
#include "mesh/DMOMeshTri.h"
#include "mesh/DMOMeshTet.h"
#include "mesh/MeshQuadGPU.h"
#include "mesh/DMOMeshQuad.h"
#include "mesh/DMOMeshHex.h"
#include <string>

void writeOFF(std::string filename, DMOMeshTri& mesh);

void writeOFF(std::string filename, DMOMeshQuad& mesh);

void writeTetgen(std::string filename, DMOMeshTet& mesh);

void writeHex(std::string filename, DMOMeshHex& mesh);

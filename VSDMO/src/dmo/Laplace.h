#pragma once

#include "cuda_runtime.h"
#include "mesh/MeshTriGPU.h"
#include "mesh/DMOMeshTri.h"
#include "ConfigUsing.h"


void laplaceSmoothing(DMOMeshTri& dmo_mesh, int n_iter);





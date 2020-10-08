#pragma once

#include "Surface1D.h"
using namespace Surf1D;


constexpr bool USE_SURF_OF_NN = false;

#define SURFLS 1

#ifdef SURFLS
#include "surfaceP/Surface.h"
//#include "surfaceP/SurfaceEstimation.h"
#include "surfaceP/SurfaceFeature.h"
using namespace SurfLS;
#else
#include "surfacePN/SurfaceV2.h"
//#include "surfacePN/SurfaceEstimationV2.h"
#include "surfacePN/SurfaceFeatureV2.h"
using namespace SurfV2;
#endif


#pragma once

#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/gather.h>

//#include "mesh/MeshTypes.h"
#include "ConfigUsing.h"


void color_jpl(int nVertices, int* rowPtr, int* colInd, device_vector<int>& colors);

// https://devblogs.nvidia.com/graph-coloring-more-parallelism-for-incomplete-lu-factorization/
//start: offset
//n: number of vertices
//Av: values
//Ao: row/col_ptr
//Ac: indices
//colors: output color per vertex
void color_jpl(int start, int n, int *rowPtr, int *colInd, device_vector<int>& colors);




void color_cuSPARSE(int nVertices, int* rowPtr, int* colInd, device_vector<int>& colors, int nnz);


#pragma once

#include <OpenVolumeMesh/Core/TopologyKernel.hh>
#include <OpenVolumeMesh/Mesh/TetrahedralMesh.hh>
#include <OpenVolumeMesh/FileManager/FileManager.hh>
typedef OpenVolumeMesh::Geometry::Vec3f	OVMVec3f;
typedef OpenVolumeMesh::GeometryKernel<OVMVec3f, OpenVolumeMesh::TetrahedralMeshTopologyKernel> OVMTetMesh;
//typedef OpenVolumeMesh::GeometricTetrahedralMeshV3f	OVMTetMesh;

#include <OpenMesh\Core\IO\MeshIO.hh>
#include <OpenMesh\Core\Mesh\TriMesh_ArrayKernelT.hh>
#include <OpenMesh\Core\Mesh\PolyMesh_ArrayKernelT.hh>
typedef OpenMesh::TriMesh_ArrayKernelT<> OMTriMesh;

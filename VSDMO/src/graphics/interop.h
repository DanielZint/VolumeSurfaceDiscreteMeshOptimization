// interop.h
#pragma once

#include "cuda_runtime.h"
#include <cuda_gl_interop.h>
#include "CudaUtil.h"



namespace Graphics {

	class Interop {
	public:
		Interop() {}
		~Interop() {
			//cout << "Interop Destructor" << endl;
			unregisterBuffer();
		}
		void registerBuffer(GLuint glbuffer) {
			//std::cout << "reg" << std::endl;
			gpuErrchk(cudaGraphicsGLRegisterBuffer(&graphicResource, glbuffer, cudaGraphicsRegisterFlagsNone));
		}
		void unregisterBuffer() {
			//std::cout << "unreg" << std::endl;
			if (graphicResource) gpuErrchk(cudaGraphicsUnregisterResource(graphicResource));
		}
		void map() {
			//std::cout << "map" << std::endl;
			gpuErrchk(cudaGraphicsMapResources(1, &graphicResource, 0));
		}
		void unmap() {
			//std::cout << "unmap" << std::endl;
			gpuErrchk(cudaGraphicsUnmapResources(1, &graphicResource, 0));
		}
		void* ptr() {
			gpuErrchk(cudaGraphicsResourceGetMappedPointer(&device_ptr, &size, graphicResource));
			return device_ptr;
		}

	protected:
		cudaGraphicsResource* graphicResource = nullptr;
		void* device_ptr = nullptr;
		size_t size = 0;
	};

	
}


// drawable.h
#pragma once

#include <vector>
#include <memory>

#include "GlmConfig.h"
#include "camera.h"
#include "GraphicsConfig.h"
#include "buffer.h"

#include "interop.h"
#include "drawableTriMesh.h"



namespace Graphics {

	// ######################################################################## //
	// ### DrawableInteropTetMesh ############################################# //
	// ######################################################################## //
	class DrawableInteropTetMesh : public DrawableInteropTriMesh {
	public:

		DrawableInteropTetMesh(const glm::vec4& color);

		DrawableInteropTetMesh(const DrawableInteropTetMesh& copy) = delete;
		DrawableInteropTetMesh& operator=(const DrawableInteropTetMesh& assign) = delete;

		virtual ~DrawableInteropTetMesh();

		void renderElementQuality(const Graphics::Camera& camera, const glm::vec4& color) const override;
		void renderTetrahedra(const Graphics::Camera& camera) const;
		void init(int nVertices, int nTriangles, int nTetrahedra);

		void setIboCellsDev(void* data, size_t size);
		void setSlicingVal(float z) { slicingZ = z; }
		void setSizeFactor(float s) { sizeFactor = s; }

	protected:
		std::unique_ptr<IndexBuffer> m_iboCells;
		std::unique_ptr<Interop> interopIboCells;
		float slicingZ = 0.f;
		float sizeFactor = 0.f;
	};
	
}


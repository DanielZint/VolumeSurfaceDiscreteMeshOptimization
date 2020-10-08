// drawable.h
#pragma once

#include <vector>
#include <memory>

#include "GlmConfig.h"
#include "camera.h"
#include "GraphicsConfig.h"
#include "buffer.h"

#include "interop.h"
#include "drawableQuadMesh.h"



namespace Graphics {

	// ######################################################################## //
	// ### DrawableInteropHexMesh ############################################# //
	// ######################################################################## //
	class DrawableInteropHexMesh : public DrawableInteropQuadMesh {
	public:

		DrawableInteropHexMesh(const glm::vec4& color);

		DrawableInteropHexMesh(const DrawableInteropHexMesh& copy) = delete;
		DrawableInteropHexMesh& operator=(const DrawableInteropHexMesh& assign) = delete;

		virtual ~DrawableInteropHexMesh();

		void renderElementQuality(const Graphics::Camera& camera, const glm::vec4& color) const override;
		void renderHexahedra(const Graphics::Camera& camera) const;
		void init(int nVertices, int nFaces, int nCells);

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


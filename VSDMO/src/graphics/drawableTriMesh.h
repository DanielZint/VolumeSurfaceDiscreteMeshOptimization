// drawable.h
#pragma once

#include <vector>
#include <memory>

#include "GlmConfig.h"
#include "camera.h"
#include "GraphicsConfig.h"
#include "buffer.h"

#include "interop.h"
#include "drawableMesh.h"



namespace Graphics {


	// ######################################################################## //
	// ### DrawableTriMesh ####################################################### //
	// ######################################################################## //
	class DrawableTriMesh : public DrawableMesh {
	public:

		DrawableTriMesh(const glm::vec4& color);

		DrawableTriMesh(const DrawableTriMesh& copy) = delete;
		DrawableTriMesh& operator=(const DrawableTriMesh& assign) = delete;

		virtual ~DrawableTriMesh();

		virtual void render(const Graphics::Camera& camera) const override;
		virtual void renderFaces(const Graphics::Camera& camera, const glm::vec4& color) const override;
		virtual void renderEdges(const Graphics::Camera& camera, const glm::vec4& color) const override;
		virtual void renderFacesAndEdges(const Graphics::Camera& camera, const glm::vec4& color1, const glm::vec4& color2) const override;
		virtual void renderVertex(const Graphics::Camera& camera, const glm::vec4& color) const override;
		virtual void renderVertices(const Graphics::Camera& camera, int startOffset, int endOffset, const glm::vec4& color) const override;
		virtual void renderElementQuality(const Graphics::Camera& camera, const glm::vec4& color) const override;
		virtual void setVertexHighlighted(int vid) override;
		virtual void init(int nVertices, int nTriangles, int nElements) override;
		virtual void initTbo(int nElements) override;

	protected:
		std::unique_ptr<VertexBuffer> m_vboPos;
		std::unique_ptr<VertexBuffer> m_vboNormal;
		std::unique_ptr<IndexBuffer> m_iboFaces;
		std::unique_ptr<IndexBuffer> m_iboHighlightVertex;
		std::unique_ptr<IndexBuffer> m_iboVertices;
		std::unique_ptr<TextureBuffer> m_tbo;
		GLuint texTbo;

	};



	// ######################################################################## //
	// ### DrawableInteropMesh ################################################ //
	// ######################################################################## //
	class DrawableInteropTriMesh : public DrawableTriMesh {
	public:

		DrawableInteropTriMesh(const glm::vec4& color);

		DrawableInteropTriMesh(const DrawableInteropTriMesh& copy) = delete;
		DrawableInteropTriMesh& operator=(const DrawableInteropTriMesh& assign) = delete;

		virtual ~DrawableInteropTriMesh();

		virtual void init(int nVertices, int nTriangles, int nTetrahedra) override;
		
		void updatePositionsDev(void* points, size_t size) override;
		void updateNormalsDev(void* normals, size_t size) override;
		void setIboDev(void* data, size_t size) override;
		Interop& getTboInterop() override { return *interopTbo; };
		void updateQualitiesDev(void* data);

	protected:
		std::unique_ptr<Interop> interopPos;
		std::unique_ptr<Interop> interopNormal;
		std::unique_ptr<Interop> interopIbo;
		std::unique_ptr<Interop> interopTbo;
	};



	
}


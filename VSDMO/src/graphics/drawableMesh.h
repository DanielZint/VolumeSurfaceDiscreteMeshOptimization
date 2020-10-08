// drawable.h
#pragma once

#include <vector>
#include <memory>

#include "GlmConfig.h"
#include "camera.h"
#include "GraphicsConfig.h"
#include "buffer.h"

#include "interop.h"
#include "drawable.h"



namespace Graphics {

	


	// ######################################################################## //
	// ### DrawableMesh ####################################################### //
	// ######################################################################## //
	class DrawableMesh : public Drawable {
	public:
		enum class RenderingMethod {
			PhongWithNormals,
			PhongWithoutNormals
		};

	public:

		DrawableMesh(const glm::vec4& color) : Drawable(color), m_renderingMethod(RenderingMethod::PhongWithNormals), m_center(0, 0, 0), m_meshScale(0) {}

		DrawableMesh(const DrawableMesh& copy) = delete;
		DrawableMesh& operator=(const DrawableMesh& assign) = delete;

		virtual ~DrawableMesh() {}

		virtual void render(const Graphics::Camera& camera) const override {};
		virtual void renderFaces(const Graphics::Camera& camera, const glm::vec4& color) const = 0;
		virtual void renderEdges(const Graphics::Camera& camera, const glm::vec4& color) const = 0;
		virtual void renderFacesAndEdges(const Graphics::Camera& camera, const glm::vec4& color1, const glm::vec4& color2) const = 0;
		virtual void renderVertex(const Graphics::Camera& camera, const glm::vec4& color) const = 0;
		virtual void renderVertices(const Graphics::Camera& camera, int startOffset, int endOffset, const glm::vec4& color) const = 0;
		virtual void renderElementQuality(const Graphics::Camera& camera, const glm::vec4& color) const = 0;
		virtual void setVertexHighlighted(int vid) = 0;
		virtual void init(int nVertices, int nTriangles, int nElements) = 0;
		virtual void initTbo(int nElements) = 0;

		virtual void updatePositionsDev(void* points, size_t size) = 0;
		virtual void updateNormalsDev(void* normals, size_t size) = 0;
		virtual void setIboDev(void* data, size_t size) = 0;
		virtual Interop& getTboInterop() = 0;

		inline void setRenderingMethod(const RenderingMethod& method) { m_renderingMethod = method; }
		inline void setCenter(const glm::vec3& c) { m_center = c; }
		inline const glm::vec3& getCenter() const { return m_center; }
		inline float getScale() const { return m_meshScale; }
		inline const glm::vec3 getCenterWorldSpace() const { return glm::vec3(m_modelMatrix * glm::vec4(m_center, 1.f)); }


	protected:
		RenderingMethod m_renderingMethod;

		glm::vec3 m_center;
		float m_meshScale;
	};



	
}


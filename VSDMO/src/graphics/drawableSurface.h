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
	// ### DrawableSurfaceEstimation ########################################## //
	// ######################################################################## //

	class DrawableSurfaceEstimation : public Drawable {
	public:

		DrawableSurfaceEstimation(const glm::vec4& color, int nu_, int nv_);

		DrawableSurfaceEstimation(const DrawableSurfaceEstimation& copy) = delete;
		DrawableSurfaceEstimation& operator=(const DrawableSurfaceEstimation& assign) = delete;

		virtual ~DrawableSurfaceEstimation();

		void render(const Graphics::Camera& camera) const override;
		void renderFaces(const Graphics::Camera& camera) const;
		void renderLines(const Graphics::Camera& camera) const;

		virtual void init();

		void updatePositionsDev(void* points);
		//void updateNormalsDev(void* normals, size_t size);
		void calculateIbo();

		Interop& getPosInterop() { return *interopPos; }

		int nu() const { return m_nu; }
		int nv() const { return m_nv; }

	protected:
		std::unique_ptr<VertexBuffer> m_vboPos;
		std::unique_ptr<Interop> interopPos;

		//std::unique_ptr<VertexBuffer> m_vboNormal;

		std::unique_ptr<IndexBuffer> m_iboFaces;
		std::unique_ptr<IndexBuffer> m_iboLines;

		int m_nu;
		int m_nv;

	};

	
}


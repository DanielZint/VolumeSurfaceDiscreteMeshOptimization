// drawable.h
#pragma once

#include <vector>
#include <memory>

#include "GlmConfig.h"
#include "camera.h"
#include "GraphicsConfig.h"
#include "buffer.h"

#include "interop.h"



namespace Graphics {

	// ##################################################################### //
	// ### Drawable ######################################################## //
	// ##################################################################### //
	// as VAOs will not be shared between threads, Drawables must be created
	// by the render thread

	class Drawable {
	public:
		Drawable();
		Drawable(const glm::vec4& color);
		virtual ~Drawable();

		virtual void render(const Graphics::Camera& camera) const = 0;

		void setColor(const glm::vec4& color) { m_color = color; }

		void setModelMatrix(const glm::mat4& matrix) { m_modelMatrix = matrix; }
		inline const glm::mat4& getModelMatrix() { return m_modelMatrix; }
		virtual void recalculateModel() { m_modelMatrix = glm::mat4(1.f); }

	protected:
		void bindVao() const { m_vao.bind(); }
		void unbindVao() const { m_vao.unbind(); }

		// should be implemented to call glVertexAttribArray()
		// the default implementation does nothing
		virtual void updateBufferPointers() const {}

		// should be implemented to setup shader uniforms.
		// the default implementation provides uniforms for a phong shader.
		virtual void setupGlUniforms(GLuint shaderProg, const Graphics::Camera& camera) const;

	protected:
		glm::mat4 m_modelMatrix;
		glm::vec4 m_color;
	private:
		VertexArray m_vao;
	};




	
}


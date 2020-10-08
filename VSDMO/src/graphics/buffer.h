// buffer.h
#pragma once

#include <mutex>
#include <memory>
#include <array>

#include "GraphicsConfig.h"

namespace Graphics {

	// ##################################################################### //
	// ### Buffer ########################################################## //
	// ##################################################################### //

	class Buffer {
	public:
		virtual GLuint getID() const = 0;
		virtual GLenum getTarget() const = 0;
		virtual size_t getSize() const = 0;
		virtual size_t getNumElements() const = 0;

		virtual void setSize(const size_t& size) = 0;
		virtual void setNumElements(const size_t& numElements) = 0;

		// will be called by the render thread
		virtual inline void bind() const { glCall(glBindBuffer(getTarget(), getID())); }
		virtual inline void unbind() const { glCall(glBindBuffer(getTarget(), 0)); }
	};


	// ##################################################################### //
	// ### VertexArray ##################################################### //
	// ##################################################################### //

	class VertexArray {
	public:
		VertexArray();
		~VertexArray();

		inline virtual GLuint getID() const { return m_bufferID; }

		inline virtual void bind() const { glCall(glBindVertexArray(m_bufferID)); }
		inline virtual void unbind() const { glCall(glBindVertexArray(0)); }
	protected:
		GLuint m_bufferID;
	};


	// ##################################################################### //
	// ### VertexBuffer #################################################### //
	// ##################################################################### //

	class VertexBuffer : public Buffer {
	public:
		VertexBuffer();
		~VertexBuffer();

		inline virtual GLuint getID() const override { return m_bufferID; }
		inline virtual GLenum getTarget() const override { return GL_ARRAY_BUFFER; }
		inline virtual size_t getSize() const override { return m_size; }
		inline virtual size_t getNumElements() const override { return m_numElements; }

		inline virtual void setSize(const size_t& size) override { m_size = size; }
		inline virtual void setNumElements(const size_t& numElements) override { m_numElements = numElements; }
	protected:
		GLuint m_bufferID;
		size_t m_size;
		size_t m_numElements;
	};


	// ##################################################################### //
	// ### IndexBuffer ##################################################### //
	// ##################################################################### //

	class IndexBuffer : public Buffer {
	public:
		IndexBuffer();
		~IndexBuffer();

		inline virtual GLuint getID() const override { return m_bufferID; }
		inline virtual GLenum getTarget() const override { return GL_ELEMENT_ARRAY_BUFFER; }
		inline virtual size_t getSize() const override { return m_size; }
		inline virtual size_t getNumElements() const override { return m_numElements; }

		inline virtual void setSize(const size_t& size) override { m_size = size; }
		inline virtual void setNumElements(const size_t& numElements) override { m_numElements = numElements; }

	protected:
		GLuint m_bufferID;
		size_t m_size;
		size_t m_numElements;
	};

	// ##################################################################### //
	// ### TextureBuffer ################################################### //
	// ##################################################################### //

	class TextureBuffer : public Buffer {
	public:
		TextureBuffer();
		~TextureBuffer();

		inline virtual GLuint getID() const override { return m_bufferID; }
		inline virtual GLenum getTarget() const override { return GL_TEXTURE_BUFFER; }
		inline virtual size_t getSize() const override { return m_size; }
		inline virtual size_t getNumElements() const override { return m_numElements; }

		inline virtual void setSize(const size_t& size) override { m_size = size; }
		inline virtual void setNumElements(const size_t& numElements) override { m_numElements = numElements; }
	protected:
		GLuint m_bufferID;
		size_t m_size;
		size_t m_numElements;
	};


	

}


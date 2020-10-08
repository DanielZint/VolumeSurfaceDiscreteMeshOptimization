// buffer.cpp
#include "buffer.h"

namespace Graphics {

	// ##################################################################### //
	// ### VertexArray ##################################################### //
	// ##################################################################### //

	VertexArray::VertexArray()
		: m_bufferID(0)
	{
		glCall(glGenVertexArrays(1, &m_bufferID));
	}

	VertexArray::~VertexArray() {
		glCall(glDeleteVertexArrays(1, &m_bufferID));
	}


	// ##################################################################### //
	// ### VertexBuffer #################################################### //
	// ##################################################################### //

	VertexBuffer::VertexBuffer()
		: Buffer()
		, m_bufferID(0)
		, m_size(0)
	{
		glCall(glGenBuffers(1, &m_bufferID));
		glCall(glBindBuffer(getTarget(), m_bufferID));
		glCall(glBindBuffer(getTarget(), 0));
	}

	VertexBuffer::~VertexBuffer() {
		glCall(glDeleteBuffers(1, &m_bufferID));
	}


	// ##################################################################### //
	// ### IndexBuffer ##################################################### //
	// ##################################################################### //

	IndexBuffer::IndexBuffer()
		: Buffer()
		, m_bufferID(0)
		, m_size(0)
	{
		glCall(glGenBuffers(1, &m_bufferID));
		glCall(glBindBuffer(getTarget(), m_bufferID));
		glCall(glBindBuffer(getTarget(), 0));
	}

	IndexBuffer::~IndexBuffer() {
		glCall(glDeleteBuffers(1, &m_bufferID));
	}


	// ##################################################################### //
	// ### TextureBuffer ################################################### //
	// ##################################################################### //

	TextureBuffer::TextureBuffer()
		: Buffer()
		, m_bufferID(0)
		, m_size(0)
	{
		glCall(glGenBuffers(1, &m_bufferID));
		glCall(glBindBuffer(getTarget(), m_bufferID));
		glCall(glBindBuffer(getTarget(), 0));
	}

	TextureBuffer::~TextureBuffer() {
		glCall(glDeleteBuffers(1, &m_bufferID));
	}

}
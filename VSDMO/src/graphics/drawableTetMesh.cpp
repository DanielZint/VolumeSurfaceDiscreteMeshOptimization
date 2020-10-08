// drawable.cpp
#include "drawableTetMesh.h"

#include "shaderManager.h"
#include "cameraManager.h"



namespace Graphics {


	DrawableInteropTetMesh::DrawableInteropTetMesh(const glm::vec4& color)
		: DrawableInteropTriMesh(color)
		, m_iboCells(nullptr)
		, interopIboCells(nullptr)
	{
		interopIboCells = std::make_unique<Interop>();
	}

	DrawableInteropTetMesh::~DrawableInteropTetMesh() {
		//cout << "DrawableInteropTetMesh Destructor" << endl;
	}

	void DrawableInteropTetMesh::init(int nVertices, int nTriangles, int nTetrahedra) {
		DrawableInteropTriMesh::init(nVertices, nTriangles, nTetrahedra);
		

		m_iboCells = std::make_unique<IndexBuffer>();
		m_iboCells->setNumElements((size_t)nTetrahedra * 4);
		m_iboCells->setSize((size_t)nTetrahedra * 4 * sizeof(int));


		m_iboCells->bind();
		glBufferData(m_iboCells->getTarget(), m_iboCells->getSize(), NULL, GL_DYNAMIC_DRAW);
		m_iboCells->unbind();


		interopIboCells->registerBuffer(m_iboCells->getID());
	}

	void DrawableInteropTetMesh::renderTetrahedra(const Graphics::Camera& camera) const {
		bindVao();
		GLuint shaderProg = Graphics::ShaderManager::getInstance().getProgramGL("tetrahedraToTrisFlat");

		glCall(glUseProgram(shaderProg));

		setupGlUniforms(shaderProg, camera);

		int location = glGetUniformLocation(shaderProg, "slicingZ");
		glCall(glUniform1f(location, slicingZ));
		location = glGetUniformLocation(shaderProg, "sizeFactor");
		glCall(glUniform1f(location, sizeFactor));

		glCall(glEnable(GL_DEPTH_TEST));
		glCall(glDepthMask(GL_TRUE));
		glCall(glEnable(GL_BLEND));
		glCall(glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA));

		glCall(glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)); //GL_LINE/FILL

		m_iboCells->bind();
		glCall(glDrawElements(GL_LINES_ADJACENCY, (int)m_iboCells->getNumElements(), GL_UNSIGNED_INT, 0));
		m_iboCells->unbind();
		m_iboFaces->bind();
		unbindVao();
	}

	void DrawableInteropTetMesh::renderElementQuality(const Graphics::Camera& camera, const glm::vec4& color) const {
		bindVao();
		GLuint shaderProg = Graphics::ShaderManager::getInstance().getProgramGL("tetrahedraToTrisQuality");

		glCall(glUseProgram(shaderProg));

		setupGlUniforms(shaderProg, camera);

		int location = glGetUniformLocation(shaderProg, "color");
		glCall(glUniform4fv(location, 1, &color[0]));
		location = glGetUniformLocation(shaderProg, "slicingZ");
		glCall(glUniform1f(location, slicingZ));
		location = glGetUniformLocation(shaderProg, "sizeFactor");
		glCall(glUniform1f(location, sizeFactor));

		int u_tbo_tex = glGetUniformLocation(shaderProg, "qualities");
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_BUFFER, texTbo);
		glTexBuffer(GL_TEXTURE_BUFFER, GL_R32F, m_tbo->getID());
		glUniform1i(u_tbo_tex, 0);

		glCall(glEnable(GL_DEPTH_TEST));
		glCall(glDepthMask(GL_TRUE));
		glCall(glEnable(GL_BLEND));
		glCall(glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA));

		glCall(glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)); //GL_LINE/FILL

		m_iboCells->bind();
		glCall(glDrawElements(GL_LINES_ADJACENCY, (int)m_iboCells->getNumElements(), GL_UNSIGNED_INT, 0));
		m_iboCells->unbind();
		m_iboFaces->bind();
		unbindVao();
	}

	void DrawableInteropTetMesh::setIboCellsDev(void* data, size_t size) {
		// copies from data to GL buffer
		interopIboCells->map();
		auto ptr = interopIboCells->ptr();
		gpuErrchk(cudaMemcpy(ptr, data, size, cudaMemcpyDeviceToDevice));
		interopIboCells->unmap();
	}

}
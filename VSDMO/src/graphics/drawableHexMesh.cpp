// drawable.cpp
#include "drawableHexMesh.h"

#include "shaderManager.h"
#include "cameraManager.h"
#include "mesh/CopyUtil.h"



namespace Graphics {


	DrawableInteropHexMesh::DrawableInteropHexMesh(const glm::vec4& color)
		: DrawableInteropQuadMesh(color)
		, m_iboCells(nullptr)
		, interopIboCells(nullptr)
	{
		interopIboCells = std::make_unique<Interop>();
	}

	DrawableInteropHexMesh::~DrawableInteropHexMesh() {
		//cout << "DrawableInteropTetMesh Destructor" << endl;
	}

	void DrawableInteropHexMesh::init(int nVertices, int nFaces, int nCells) {
		DrawableInteropQuadMesh::init(nVertices, nFaces, nCells);
		

		m_iboCells = std::make_unique<IndexBuffer>();
		m_iboCells->setNumElements((size_t)nCells * 4 * 6); // store all quads explicitly for now
		m_iboCells->setSize((size_t)nCells * 4 * 6 * sizeof(int));


		m_iboCells->bind();
		glBufferData(m_iboCells->getTarget(), m_iboCells->getSize(), NULL, GL_DYNAMIC_DRAW);
		m_iboCells->unbind();


		interopIboCells->registerBuffer(m_iboCells->getID());
	}

	void DrawableInteropHexMesh::renderHexahedra(const Graphics::Camera& camera) const {
		bindVao();
		GLuint shaderProg = Graphics::ShaderManager::getInstance().getProgramGL("quadsPhongLightingNormalsSlice");

		glCall(glUseProgram(shaderProg));

		setupGlUniforms(shaderProg, camera);

		// TODO
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

	void DrawableInteropHexMesh::renderElementQuality(const Graphics::Camera& camera, const glm::vec4& color) const {
		bindVao();
		GLuint shaderProg = Graphics::ShaderManager::getInstance().getProgramGL("hexquadsQuality");

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

	void DrawableInteropHexMesh::setIboCellsDev(void* srcData, size_t sizeBytes) {
		// copies from data to GL buffer
		interopIboCells->map();
		auto ptr = interopIboCells->ptr();
		copyHexahedraToQuads(srcData, ptr, sizeBytes);
		//gpuErrchk(cudaMemcpy(ptr, srcData, sizeBytes, cudaMemcpyDeviceToDevice));
		interopIboCells->unmap();
	}

}
// drawable.cpp
#include "drawableSurface.h"

#include "shaderManager.h"
#include "cameraManager.h"



namespace Graphics {


	// ######################################################################## //
	// ### DrawableSurfaceEstimation ########################################## //
	// ######################################################################## //

	DrawableSurfaceEstimation::DrawableSurfaceEstimation(const glm::vec4& color, int nu_, int nv_)
		: Drawable(color)
		, m_vboPos(nullptr)
		, m_iboFaces(nullptr)
		, m_iboLines(nullptr)
		, m_nu(nu_)
		, m_nv(nv_)
	{
		interopPos = std::make_unique<Interop>();
	}

	DrawableSurfaceEstimation::~DrawableSurfaceEstimation() {
		//cout << "DrawableSurfaceEstimation Destructor" << endl;
	}

	void DrawableSurfaceEstimation::init() {
		std::vector<int> indicesFaces;
		for (int i = 0; i < m_nu - 1; ++i) {
			for (int j = 0; j < m_nv - 1; ++j) {
				indicesFaces.push_back(m_nu * j + i);
				indicesFaces.push_back(m_nu * (j + 1) + i);
				indicesFaces.push_back(m_nu * j + i + 1);

				indicesFaces.push_back(m_nu * j + i + 1);
				indicesFaces.push_back(m_nu * (j + 1) + i);
				indicesFaces.push_back(m_nu * (j + 1) + i + 1);
			}
		}

		std::vector<int> indicesLines;
		for (int i = 0; i < m_nu - 1; ++i) {
			for (int j = 0; j < m_nv - 1; ++j) {
				indicesLines.push_back(m_nu * j + i);
				indicesLines.push_back(m_nu * (j + 1) + i);

				indicesLines.push_back(m_nu * j + i);
				indicesLines.push_back(m_nu * j + i + 1);
			}
		}
		for (int i = 0; i < m_nu - 1; ++i) {
			int j = m_nv - 1;
			indicesLines.push_back(m_nu * j + i);
			indicesLines.push_back(m_nu * j + i + 1);
		}
		for (int j = 0; j < m_nv - 1; ++j) {
			int i = m_nu - 1;
			indicesLines.push_back(m_nu * j + i);
			indicesLines.push_back(m_nu * (j + 1) + i);
		}

		m_vboPos = std::make_unique<VertexBuffer>();
		m_vboPos->setNumElements((size_t)m_nu * m_nv * 3);
		m_vboPos->setSize((size_t)m_nu * m_nv * 3 * sizeof(float));

		m_iboFaces = std::make_unique<IndexBuffer>();
		m_iboFaces->setNumElements(indicesFaces.size()); //2 tris per "cell", 3 ints per tri // ((m_nu - 1) * (m_nv - 1) * 2 * 3)
		m_iboFaces->setSize(indicesFaces.size() * sizeof(int));

		m_iboLines = std::make_unique<IndexBuffer>();
		m_iboLines->setNumElements(indicesLines.size()); //2 lines per "cell", 2 ints per line // ((m_nu - 1) * (m_nv - 1) * 2 * 2)
		m_iboLines->setSize(indicesLines.size() * sizeof(int));

		bindVao();

		m_vboPos->bind();
		glBufferData(m_vboPos->getTarget(), m_vboPos->getSize(), NULL, GL_DYNAMIC_DRAW);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
		glEnableVertexAttribArray(0);

		//m_vboNormal->bind();
		//glBufferData(m_vboNormal->getTarget(), m_vboNormal->getSize(), NULL, GL_DYNAMIC_DRAW);
		//glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
		//glEnableVertexAttribArray(1);

		m_iboLines->bind();
		glBufferData(m_iboLines->getTarget(), m_iboLines->getSize(), indicesLines.data(), GL_STATIC_DRAW);
		
		m_iboFaces->bind();
		glBufferData(m_iboFaces->getTarget(), m_iboFaces->getSize(), indicesFaces.data(), GL_STATIC_DRAW);
		//glBufferData(m_iboFaces->getTarget(), sizeof(int) * 3 * (m_nu - 1) * (m_nv - 1) * 2, NULL, GL_DYNAMIC_DRAW);
		

		unbindVao();
		m_vboPos->unbind();
		m_iboFaces->unbind();

		interopPos->registerBuffer(m_vboPos->getID());

		//calculateIbo();
	}

	void DrawableSurfaceEstimation::render(const Graphics::Camera& camera) const {
		renderLines(camera);
	}

	void DrawableSurfaceEstimation::renderFaces(const Graphics::Camera& camera) const {
		bindVao();
		GLuint shaderProg = Graphics::ShaderManager::getInstance().getProgramGL("flatShadingShowNormals");

		glCall(glUseProgram(shaderProg));

		setupGlUniforms(shaderProg, camera);

		glCall(glEnable(GL_DEPTH_TEST));
		glCall(glDepthMask(GL_TRUE));
		glCall(glEnable(GL_BLEND));
		glCall(glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA));

		glCall(glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)); //GL_LINE/FILL
		m_iboFaces->bind();
		glCall(glDrawElements(GL_TRIANGLES, (int)m_iboFaces->getNumElements(), GL_UNSIGNED_INT, 0));
		m_iboFaces->unbind();
		unbindVao();
	}

	void DrawableSurfaceEstimation::renderLines(const Graphics::Camera& camera) const {
		bindVao();
		GLuint shaderProg = Graphics::ShaderManager::getInstance().getProgramGL("phongLightingUniform");

		glCall(glUseProgram(shaderProg));

		setupGlUniforms(shaderProg, camera);

		glCall(glEnable(GL_DEPTH_TEST));
		glCall(glDepthMask(GL_TRUE));
		glCall(glEnable(GL_BLEND));
		glCall(glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA));

		glCall(glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)); //GL_LINE/FILL

		m_iboLines->bind();
		glCall(glDrawElements(GL_LINES, (int)m_iboLines->getNumElements(), GL_UNSIGNED_INT, 0));
		m_iboLines->unbind();
		unbindVao();
	}

	void DrawableSurfaceEstimation::updatePositionsDev(void* points) {
		interopPos->map();
		auto posptr = interopPos->ptr();
		gpuErrchk(cudaMemcpy(posptr, points, m_vboPos->getSize(), cudaMemcpyDeviceToDevice));
		interopPos->unmap();
	}

	void DrawableSurfaceEstimation::calculateIbo() {
		// TODO BROKEN ATM
		std::vector<int> indices;
		//std::vector<int> indices((m_nu - 1) * (m_nv - 1) * 2 * 3);
		for (int i = 0; i < m_nu - 1; ++i) {
			for (int j = 0; j < m_nv - 1; ++j) {
				indices.push_back(m_nu * j + i);
				indices.push_back(m_nu * (j + 1) + i);
				indices.push_back(m_nu * j + i + 1);

				indices.push_back(m_nu * j + i + 1);
				indices.push_back(m_nu * (j + 1) + i);
				indices.push_back(m_nu * (j + 1) + i + 1);
			}
		}
		bindVao();
		m_iboFaces->bind();
		glBufferSubData(GL_ELEMENT_ARRAY_BUFFER, 0, indices.size() * sizeof(int), (void *)indices.data());
		unbindVao();
		m_iboFaces->unbind();
	}




}
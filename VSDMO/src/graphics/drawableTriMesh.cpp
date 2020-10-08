// drawable.cpp
#include "drawableTriMesh.h"

#include "shaderManager.h"
#include "cameraManager.h"



namespace Graphics {



	// ######################################################################## //
	// ### DrawableMesh ####################################################### //
	// ######################################################################## //

	DrawableTriMesh::DrawableTriMesh(const glm::vec4& color)
		: DrawableMesh(color)
		, m_vboPos(nullptr)
		, m_vboNormal(nullptr)
		, m_iboFaces(nullptr)
		, m_iboHighlightVertex(nullptr)
		, m_iboVertices(nullptr)
		, m_tbo(nullptr)
		, texTbo(0)
	{
		
	}

	DrawableTriMesh::~DrawableTriMesh() {
		//cout << "DrawableMesh Destructor" << endl;
	}

	void DrawableTriMesh::init(int nVertices, int nTriangles, int nElements) {
		int highlistVertexIndices[1] = {0};

		vector<int> pointIndices(nVertices);
		for (int i = 0; i < nVertices; ++i) {
			pointIndices[i] = i;
		}

		m_vboPos = std::make_unique<VertexBuffer>();
		m_vboPos->setNumElements((size_t)nVertices * 3);
		m_vboPos->setSize((size_t)nVertices * 3 * sizeof(float));

		m_vboNormal = std::make_unique<VertexBuffer>();
		m_vboNormal->setNumElements((size_t)nVertices * 3);
		m_vboNormal->setSize((size_t)nVertices * 3 * sizeof(float));

		m_iboFaces = std::make_unique<IndexBuffer>();
		m_iboFaces->setNumElements((size_t)nTriangles * 3);
		m_iboFaces->setSize((size_t)nTriangles * 3 * sizeof(int));

		m_iboHighlightVertex = std::make_unique<IndexBuffer>();
		m_iboHighlightVertex->setNumElements(1);
		m_iboHighlightVertex->setSize(1 * sizeof(int));

		m_iboVertices = std::make_unique<IndexBuffer>();
		m_iboVertices->setNumElements(nVertices);
		m_iboVertices->setSize(nVertices * sizeof(int));

		bindVao();

		m_vboPos->bind();
		glBufferData(m_vboPos->getTarget(), m_vboPos->getSize(), NULL, GL_DYNAMIC_DRAW);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
		glEnableVertexAttribArray(0);

		m_vboNormal->bind();
		glBufferData(m_vboNormal->getTarget(), m_vboNormal->getSize(), NULL, GL_DYNAMIC_DRAW);
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
		glEnableVertexAttribArray(1);

		m_iboHighlightVertex->bind();
		glBufferData(m_iboHighlightVertex->getTarget(), m_iboHighlightVertex->getSize(), highlistVertexIndices, GL_DYNAMIC_DRAW);

		m_iboVertices->bind();
		glBufferData(m_iboVertices->getTarget(), m_iboVertices->getSize(), pointIndices.data(), GL_STATIC_DRAW);

		m_iboFaces->bind();
		glBufferData(m_iboFaces->getTarget(), m_iboFaces->getSize(), NULL, GL_DYNAMIC_DRAW);

		unbindVao();
		m_vboNormal->unbind();
		m_iboFaces->unbind();

		initTbo(nElements);
	}

	void DrawableTriMesh::initTbo(int nElements) {
		m_tbo = std::make_unique<TextureBuffer>();
		m_tbo->setNumElements(nElements);
		m_tbo->setSize(nElements * sizeof(float));
		m_tbo->bind();
		glBufferData(GL_TEXTURE_BUFFER, m_tbo->getSize(), NULL, GL_DYNAMIC_DRAW);
		glGenTextures(1, &texTbo);
		m_tbo->unbind();
	}

	void DrawableTriMesh::render(const Graphics::Camera& camera) const {
		bindVao();
		GLuint shaderProg = Graphics::ShaderManager::getInstance().getProgramGL("phongLightingNormals");

		glCall(glUseProgram(shaderProg));

		setupGlUniforms(shaderProg, camera);

		glCall(glEnable(GL_DEPTH_TEST));
		glCall(glDepthMask(GL_TRUE));
		glCall(glEnable(GL_BLEND));
		glCall(glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA));

		glCall(glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)); //GL_LINE/FILL

		glCall(glDrawElements(GL_TRIANGLES, (int)m_iboFaces->getNumElements(), GL_UNSIGNED_INT, 0));
		unbindVao();
	}

	void DrawableTriMesh::renderFaces(const Graphics::Camera& camera, const glm::vec4& color) const {
		bindVao();
		GLuint shaderProg = Graphics::ShaderManager::getInstance().getProgramGL("phongLightingNormals");
		//GLuint shaderProg = Graphics::ShaderManager::getInstance().getProgramGL("flatShading");

		glCall(glUseProgram(shaderProg));

		setupGlUniforms(shaderProg, camera);

		int location = glGetUniformLocation(shaderProg, "color");
		glCall(glUniform4fv(location, 1, &color[0]));

		glCall(glEnable(GL_DEPTH_TEST));
		glCall(glDepthMask(GL_TRUE));
		glCall(glEnable(GL_BLEND));
		glCall(glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA));

		glCall(glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)); //GL_LINE/FILL

		glCall(glDrawElements(GL_TRIANGLES, (int)m_iboFaces->getNumElements(), GL_UNSIGNED_INT, 0));
		unbindVao();
	}

	void DrawableTriMesh::renderEdges(const Graphics::Camera& camera, const glm::vec4& color) const {
		bindVao();
		GLuint shaderProg = Graphics::ShaderManager::getInstance().getProgramGL("trisToEdges");

		glCall(glUseProgram(shaderProg));

		setupGlUniforms(shaderProg, camera);

		int location = glGetUniformLocation(shaderProg, "color");
		glCall(glUniform4fv(location, 1, &color[0]));

		glCall(glEnable(GL_DEPTH_TEST));
		glCall(glDepthMask(GL_TRUE));
		glCall(glEnable(GL_BLEND));
		glCall(glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA));

		glCall(glDrawElements(GL_TRIANGLES, (int)m_iboFaces->getNumElements(), GL_UNSIGNED_INT, 0));

		unbindVao();
	}

	void DrawableTriMesh::renderFacesAndEdges(const Graphics::Camera& camera, const glm::vec4& color1, const glm::vec4& color2) const {
		glCall(glEnable(GL_POLYGON_OFFSET_FILL));
		glCall(glLineWidth(1.f));
		glCall(glPolygonOffset(1.f, 1.f));
		glCall(glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)); //GL_LINE/FILL
		renderFaces(camera, color1);
		renderEdges(camera, color2);
		glCall(glDisable(GL_POLYGON_OFFSET_FILL));
	}

	void DrawableTriMesh::renderVertex(const Graphics::Camera& camera, const glm::vec4& color) const {
		bindVao();
		GLuint shaderProg = Graphics::ShaderManager::getInstance().getProgramGL("phongLightingUniform");

		glCall(glUseProgram(shaderProg));

		setupGlUniforms(shaderProg, camera);

		int location = glGetUniformLocation(shaderProg, "color");
		glCall(glUniform4fv(location, 1, &color[0]));

		glCall(glDisable(GL_DEPTH_TEST));
		glCall(glDepthMask(GL_TRUE));
		glCall(glEnable(GL_BLEND));
		glCall(glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA));

		glCall(glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)); //GL_LINE/FILL
		glCall(glPointSize(4.f));

		m_iboHighlightVertex->bind();
		glCall(glDrawElements(GL_POINTS, (int)m_iboHighlightVertex->getNumElements(), GL_UNSIGNED_INT, 0));
		m_iboHighlightVertex->unbind();
		m_iboFaces->bind();
		unbindVao();
		glCall(glEnable(GL_DEPTH_TEST));
	}

	void DrawableTriMesh::renderVertices(const Graphics::Camera& camera, int startOffset, int endOffset, const glm::vec4& color) const {
		glCall(glEnable(GL_POLYGON_OFFSET_FILL));
		bindVao();
		// draws an index range of vertices with different color
		GLuint shaderProg = Graphics::ShaderManager::getInstance().getProgramGL("pointRange");

		glCall(glUseProgram(shaderProg));

		setupGlUniforms(shaderProg, camera);

		int location = glGetUniformLocation(shaderProg, "color");
		glCall(glUniform4fv(location, 1, &color[0]));
		location = glGetUniformLocation(shaderProg, "startOffset");
		glCall(glUniform1i(location, startOffset));
		location = glGetUniformLocation(shaderProg, "endOffset");
		glCall(glUniform1i(location, endOffset));
		location = glGetUniformLocation(shaderProg, "depthOffset");
		glCall(glUniform1f(location, 0.01f));

		//glCall(glDisable(GL_DEPTH_TEST));
		glCall(glDepthMask(GL_TRUE));
		glCall(glEnable(GL_BLEND));
		glCall(glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA));

		glCall(glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)); //GL_LINE/FILL
		glCall(glPointSize(3.f));

		m_iboVertices->bind();
		glCall(glDrawElements(GL_POINTS, (int)m_iboVertices->getNumElements(), GL_UNSIGNED_INT, 0));
		m_iboVertices->unbind();
		m_iboFaces->bind();

		unbindVao();
		//glCall(glEnable(GL_DEPTH_TEST));
		glCall(glDisable(GL_POLYGON_OFFSET_FILL));
	}

	void DrawableTriMesh::renderElementQuality(const Graphics::Camera& camera, const glm::vec4& color) const {
		bindVao();
		
		GLuint shaderProg = Graphics::ShaderManager::getInstance().getProgramGL("triQuality");
		//GLuint shaderProg = Graphics::ShaderManager::getInstance().getProgramGL("flatShading");

		glCall(glUseProgram(shaderProg));

		setupGlUniforms(shaderProg, camera);

		int location = glGetUniformLocation(shaderProg, "color");
		glCall(glUniform4fv(location, 1, &color[0]));

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

		glCall(glDrawElements(GL_TRIANGLES, (int)m_iboFaces->getNumElements(), GL_UNSIGNED_INT, 0));
		unbindVao();
	}

	void DrawableTriMesh::setVertexHighlighted(int vid) {
		m_iboHighlightVertex->bind();
		//glBufferData(m_iboHighlightVertex->getTarget(), m_iboHighlightVertex->getSize(), highlistVertexIndices, GL_DYNAMIC_DRAW);
		glBufferSubData(GL_ELEMENT_ARRAY_BUFFER, 0, sizeof(int), &vid);
		m_iboHighlightVertex->unbind();
	}


	// ######################################################################## //
	// ### DrawableInteropMesh ################################################ //
	// ######################################################################## //

	DrawableInteropTriMesh::DrawableInteropTriMesh(const glm::vec4& color)
		: DrawableTriMesh(color)
	{
		interopPos = std::make_unique<Interop>();
		interopNormal = std::make_unique<Interop>();
		interopIbo = std::make_unique<Interop>();
		interopTbo = std::make_unique<Interop>();
	}

	DrawableInteropTriMesh::~DrawableInteropTriMesh() {
		//cout << "DrawableInteropMesh Destructor" << endl;
		//interopPos->unregisterBuffer();
		//interopNormal->unregisterBuffer();
		//interopIbo->unregisterBuffer();
	}

	
	void DrawableInteropTriMesh::init(int nVertices, int nTriangles, int nElements) {
		DrawableTriMesh::init(nVertices, nTriangles, nElements);
		interopPos->registerBuffer(m_vboPos->getID());
		interopNormal->registerBuffer(m_vboNormal->getID());
		interopIbo->registerBuffer(m_iboFaces->getID());
		interopTbo->registerBuffer(m_tbo->getID());
	}


	void DrawableInteropTriMesh::updatePositionsDev(void* points, size_t size) {
		interopPos->map();
		auto posptr = interopPos->ptr();
		gpuErrchk(cudaMemcpy(posptr, points, size, cudaMemcpyDeviceToDevice));
		interopPos->unmap();
	}

	void DrawableInteropTriMesh::updateNormalsDev(void* normals, size_t size) {
		interopNormal->map();
		auto norptr = interopNormal->ptr();
		cudaMemcpy(norptr, normals, size, cudaMemcpyDeviceToDevice);
		interopNormal->unmap();
	}

	void DrawableInteropTriMesh::setIboDev(void* data, size_t size) {
		interopIbo->map();
		auto ptr = interopIbo->ptr();
		gpuErrchk(cudaMemcpy(ptr, data, size, cudaMemcpyDeviceToDevice));
		interopIbo->unmap();
	}

	void DrawableInteropTriMesh::updateQualitiesDev(void* data) {
		interopTbo->map();
		auto posptr = interopTbo->ptr();
		gpuErrchk(cudaMemcpy(posptr, data, m_tbo->getSize(), cudaMemcpyDeviceToDevice));
		interopTbo->unmap();
	}

}
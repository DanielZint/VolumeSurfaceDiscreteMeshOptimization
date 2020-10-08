// drawable.cpp
#include "drawable.h"

#include "shaderManager.h"
#include "cameraManager.h"



namespace Graphics {

	// ######################################################################## //
	// ### Drawable ########################################################### //
	// ######################################################################## //

	Drawable::Drawable()
		: Drawable(glm::vec4(0, 1, 0, 1))
	{}

	Drawable::Drawable(const glm::vec4& color)
		: m_vao()
		, m_color(color)
		, m_modelMatrix(glm::mat4(1))
	{
	}

	Drawable::~Drawable() {
		//cout << "Drawable Destructor" << endl;
	}

	void Drawable::setupGlUniforms(GLuint shaderProg, const Graphics::Camera& camera) const {
		int location;

		location = glGetUniformLocation(shaderProg, "proj");
		const glm::mat4& proj = camera.getProjectionMatrix();
		glCall(glUniformMatrix4fv(location, 1, GL_FALSE, &proj[0][0]));

		location = glGetUniformLocation(shaderProg, "view");
		const glm::mat4& view = camera.getViewMatrix();
		glCall(glUniformMatrix4fv(location, 1, GL_FALSE, &view[0][0]));

		location = glGetUniformLocation(shaderProg, "model");
		glCall(glUniformMatrix4fv(location, 1, GL_FALSE, &m_modelMatrix[0][0]));

		location = glGetUniformLocation(shaderProg, "color");
		glCall(glUniform4fv(location, 1, &m_color[0]));

		location = glGetUniformLocation(shaderProg, "depthOffset");
		glCall(glUniform1f(location, 0.f));

		location = glGetUniformLocation(shaderProg, "ambient");
		glCall(glUniform1f(location, .4f));

		location = glGetUniformLocation(shaderProg, "diffuse");
		glCall(glUniform1f(location, .4f));

		location = glGetUniformLocation(shaderProg, "specular");
		glCall(glUniform1f(location, .01f));

		location = glGetUniformLocation(shaderProg, "shiny");
		glCall(glUniform1f(location, 30.f));

		location = glGetUniformLocation(shaderProg, "lightDirection");
		glCall(glUniform3f(location, -.5f, .5f, -1.f));

		location = glGetUniformLocation(shaderProg, "lightColor");
		glCall(glUniform4f(location, .95f, .95f, .95f, 1.f));

		location = glGetUniformLocation(shaderProg, "cameraPosition");
		glCall(glUniform3fv(location, 1, &camera.getPosition()[0]));

		location = glGetUniformLocation(shaderProg, "inverseColor");
		glCall(glUniform1i(location, 0));

		location = glGetUniformLocation(shaderProg, "lineLength");
		glCall(glUniform1f(location, 1.f));
	}


	


}
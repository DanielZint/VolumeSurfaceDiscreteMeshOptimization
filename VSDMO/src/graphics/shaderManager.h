// shaderManager.h
#pragma once

#include <GL/glew.h>
#include <map>
#include <string>
#include <vector>
#include <string>

#define SHADER_VERSION "#version 430 core\n"

using std::string;

extern string resPath;

namespace Graphics {

	enum ShaderType : int {
		NONE = -1,
		VERTEX = 0,
		TESSCONTROL = 1,
		TESSEVALUATION = 2,
		GEOMETRY = 3,
		FRAGMENT = 4,
		COMPUTE = 5
	};

	class Shader {
		GLuint m_program;
		std::string m_filePath;
		std::vector<ShaderType> m_shaderFlags;

	public:
		Shader(const std::string& shaderDir, const std::string& filePath, const std::string& fileExtension);
		~Shader();

		GLuint getShader() const { return m_program; }
		std::string getFilePath() const { return m_filePath; }

		void update(const std::string& shaderDir);
	private:
		void parseShader(const std::string& shaderDir, const std::string& filePath);
		void handleInclude(const std::string& shaderDir, const std::string& includePath, std::stringstream& target);
		void compileAndLinkShaders(std::vector<std::string>& sourceCode);

	};

	// Singleton class
	class ShaderManager {
	public:

		static ShaderManager& getInstance();

		~ShaderManager();

		void setShaderDir(const std::string& shaderDir) { m_shaderDir = shaderDir; }
		void setFileExtension(const std::string& fileExtension) { m_fileExtension = fileExtension; }

		void registerProgram(const std::string& filePath);
		GLuint getProgramGL(const std::string& id) const;

		void update();
	private:
		ShaderManager(const std::string& shaderDir, const std::string& fileExtension);

		std::string m_shaderDir;
		std::string m_fileExtension;
		std::map<std::string, Shader *> m_programs;
	};

}
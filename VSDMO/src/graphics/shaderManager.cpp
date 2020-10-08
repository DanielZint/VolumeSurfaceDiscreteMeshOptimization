// shaderManager.cpp
#include "shaderManager.h"
#include "GraphicsConfig.h"

#include <fstream>
#include <sstream>
#include <algorithm>
#include <iostream>

string resPath;

namespace Graphics {

	// ######################################################################### //
	// ### ShaderManager######################################################## //
	// ######################################################################### //

	ShaderManager& ShaderManager::getInstance() {
		string dirPath = resPath + string("/shaders/");
		//static ShaderManager manager("res/shaders/", ".glsl");
		static ShaderManager manager(dirPath, ".glsl");
		return manager;
	}

	ShaderManager::ShaderManager(const std::string& shaderDir, const std::string& fileExtension)
		: m_programs()
		, m_shaderDir(shaderDir)
		, m_fileExtension(fileExtension)
	{
	}

	ShaderManager::~ShaderManager() {
	}

	void ShaderManager::registerProgram(const std::string& filePath) {
		Shader *shader = new Shader(m_shaderDir, filePath, m_fileExtension);
		m_programs[filePath] = shader;
	}

	GLuint ShaderManager::getProgramGL(const std::string& id) const {
		return m_programs.at(id)->getShader();
	}

	void ShaderManager::update() {
		for (auto s : m_programs) {
			s.second->update(m_shaderDir);
		}
	}


	// ######################################################################### //
	// ### Shader ############################################################## //
	// ######################################################################### //

	Shader::Shader(const std::string& shaderDir, const std::string& filePath, const std::string& fileExtension)
		: m_filePath()
		, m_program(0)
		, m_shaderFlags()
	{
		parseShader(shaderDir, filePath + fileExtension);
	}

	Shader::~Shader() {
	}

	void Shader::update(const std::string& shaderDir) {
		m_shaderFlags.clear();
		parseShader(shaderDir, m_filePath);
	}

	void Shader::parseShader(const std::string& shaderDir, const std::string& filePath) {
		m_filePath = filePath;

		std::ifstream stream(shaderDir + m_filePath);
		if (!stream.is_open()) {
			std::cout << "couldn't open file " << (shaderDir + m_filePath) << std::endl;
		}
		
		std::string line;
		std::stringstream ss[6];
		ShaderType type = ShaderType::NONE;

		// parse shaders
		while (getline(stream, line)) {
			if (line.find("---") != std::string::npos) {
				// set shader type
				if (line.find("vertex") != std::string::npos) {
					type = ShaderType::VERTEX;
				}
				else if (line.find("tesscontrol") != std::string::npos) {
					type = ShaderType::TESSCONTROL;
				}
				else if (line.find("tessevaluation") != std::string::npos) {
					type = ShaderType::TESSEVALUATION;
				}
				else if (line.find("geometry") != std::string::npos) {
					type = ShaderType::GEOMETRY;
				}
				else if (line.find("fragment") != std::string::npos) {
					type = ShaderType::FRAGMENT;
				}
				else if (line.find("compute") != std::string::npos) {
					type = ShaderType::COMPUTE;
				}
				m_shaderFlags.push_back(type);
				ss[type] << SHADER_VERSION;
			}
			// handle includes
			else if (line.find("#include") != std::string::npos) {
				// get the filename of the include
				handleInclude(shaderDir, line, ss[type]);
			}
			else {
				if (type != ShaderType::NONE) {
					ss[type] << line << "\n";
				}
			}
		}

		std::vector<std::string> sourceCode;
		for (int i = 0; i < 6; ++i) {
			std::string code = ss[i].str();
			if (code.size() > 0) {
				sourceCode.push_back(code);
			}
		}

		compileAndLinkShaders(sourceCode);
	}

	void Shader::handleInclude(const std::string& shaderDir, const std::string& includeLine, std::stringstream& target) {
		std::string line;

		line = includeLine.substr(9);
		line.erase(std::remove(line.begin(), line.end(), '\"'), line.end());
		std::ifstream include(shaderDir + line);
		if (!include.is_open()) {
			std::cout << "couldn't open file " << (shaderDir + line) << std::endl;
		}

		while (getline(include, line)) {
			if (line.find("#include") != std::string::npos) {
				handleInclude(shaderDir, line, target);
			}
			else {
				target << line << "\n";
			}
		}
	}

	void Shader::compileAndLinkShaders(std::vector<std::string>& sourceCode) {
		std::string source;

		std::vector<GLuint> shaders;

		// compile shaders
		int compiled = 0;
		for (int type = 0; type < 6 && compiled < m_shaderFlags.size(); ++type) {
			GLuint shader;
			switch (m_shaderFlags[compiled]) {
			case ShaderType::VERTEX:
				shader = glCreateShader(GL_VERTEX_SHADER);
				break;
			case ShaderType::TESSCONTROL:
				shader = glCreateShader(GL_TESS_CONTROL_SHADER);
				break;
			case ShaderType::TESSEVALUATION:
				shader = glCreateShader(GL_TESS_EVALUATION_SHADER);
				break;
			case ShaderType::GEOMETRY:
				shader = glCreateShader(GL_GEOMETRY_SHADER);
				break;
			case ShaderType::FRAGMENT:
				shader = glCreateShader(GL_FRAGMENT_SHADER);
				break;
			case ShaderType::COMPUTE:
				shader = glCreateShader(GL_COMPUTE_SHADER);
				break;
			default:
				continue;
			}
			source = sourceCode[compiled];
			const char *src = source.c_str();
			glCall(glShaderSource(shader, 1, &src, nullptr));
			glCall(glCompileShader(shader));

			// error handling:
			int result;
			glCall(glGetShaderiv(shader, GL_COMPILE_STATUS, &result));
			if (result == GL_FALSE) {
				int length;
				glCall(glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &length));
				char *message = (char *)alloca(length * sizeof(char));
				glCall(glGetShaderInfoLog(shader, length, &length, message));
				std::cerr << "Failed to compile ";
				switch (m_shaderFlags[compiled]) {
				case ShaderType::VERTEX:
					std::cerr << "VERTEX_SHADER";
					break;
				case ShaderType::TESSCONTROL:
					std::cerr << "TESS_CONTROL_SHADER";
					break;
				case ShaderType::TESSEVALUATION:
					std::cerr << "TESS_EVALUATION_SHADER";
					break;
				case ShaderType::GEOMETRY:
					std::cerr << "GEOMETRY_SHADER";
					break;
				case ShaderType::FRAGMENT:
					std::cerr << "FRAGMENT_SHADER";
					break;
				case ShaderType::COMPUTE:
					std::cerr << "COMPUTE_SHADER";
					break;
				}
				std::cerr << " [" << m_filePath << "]: " << message << std::endl;
				glCall(glDeleteShader(shader));
				return;
			}
			++compiled;

			shaders.push_back(shader);
		}

		// link shaders
		m_program = glCreateProgram();
		for (GLuint shader : shaders) {
			glCall(glAttachShader(m_program, shader));
		}

		glCall(glLinkProgram(m_program));
		glCall(glValidateProgram(m_program));

		// clean up intermediates
		for (GLuint shader : shaders) {
			glCall(glDeleteShader(shader));
		}

		std::cout << "[ " << m_filePath << " ] compiled successfully." << std::endl;
	}

}
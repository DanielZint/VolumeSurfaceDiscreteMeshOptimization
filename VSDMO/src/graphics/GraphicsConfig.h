// GraphicsConfig.h
#pragma once

#include <GL/glew.h>
#include <iostream>

#ifndef NDEBUG
	#ifdef _WINDOWS
	#define ASSERT(x) if(!(x)) __debugbreak();
	#else
	#define ASSERT(x) if(!(x)) throw -1;
	#endif
	#define glCall(x) \
		{while(glGetError() != GL_NO_ERROR){}\
		x;\
		bool noError = true;\
		while (GLenum error = glGetError()) {\
			std::cout << "[ OpenGL Error ] [ " << error << " ]: " << gluErrorString(error) << std::endl; \
			noError = false; \
		}\
		ASSERT(noError);}
#else
	#define ASSERT(x) x
	#define glCall(x) x
#endif

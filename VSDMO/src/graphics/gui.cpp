// gui.cpp
#include "gui.h"

#include <iostream>
#include <chrono>

#include "imgui/imgui_impl_opengl3.h"
#include "imgui/imgui_impl_sdl.h"
#include "GraphicsConfig.h"

// --- shared contexts --- //
#ifdef WINDOWS
//#include <Windows.h>
#endif
// --- shared contexts --- //


namespace Graphics {

	// ######################################################################### //
	// ### ImGuiRenderer ####################################################### //
	// ######################################################################### //

	ImGuiRenderer::ImGuiRenderer()
		: m_window(nullptr)
	{

	}

	ImGuiRenderer::~ImGuiRenderer() {
		ImGui_ImplOpenGL3_Shutdown();
		ImGui_ImplSDL2_Shutdown();
		ImGui::DestroyContext();
	}

	void ImGuiRenderer::init(SDL_Window *window, SDL_GLContext context, const char *glsl_version) {
		IMGUI_CHECKVERSION();
		ImGui::CreateContext();

		ImGui_ImplSDL2_InitForOpenGL(window, context);
		ImGui_ImplOpenGL3_Init(glsl_version);

		ImGui::StyleColorsDark();

		m_window = window;

		std::cout << "ImGui initialized." << std::endl;
	}

	void ImGuiRenderer::handleSDLEvent(SDL_Event event) {
		ImGui_ImplSDL2_ProcessEvent(&event);
	}

	void ImGuiRenderer::startFrame() {
		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplSDL2_NewFrame(m_window);
		ImGui::NewFrame();
	}

	void ImGuiRenderer::endFrame() {
		ImGui::Render();
		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
	}


	// ######################################################################### //
	// ### Gui ################################################################# //
	// ######################################################################### //

	Gui::Gui()
		: m_window(nullptr)
		, m_activeCursor(nullptr)
		, m_glContext()
		, m_exit(false)
		, m_updateMouse(false)
	{
		// init Gui

		// setup SDL
		if (SDL_Init(SDL_INIT_VIDEO) != 0) {
			std::cerr << "SDL Init Error: " << SDL_GetError() << std::endl;
			return;
		}

		const char *glsl_version = "#version 430";
		SDL_GL_SetAttribute(SDL_GL_CONTEXT_FLAGS, 0);
		SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
		SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 4);
		SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 5);

		SDL_GL_SetAttribute(SDL_GL_RED_SIZE, 8);
		SDL_GL_SetAttribute(SDL_GL_GREEN_SIZE, 8);
		SDL_GL_SetAttribute(SDL_GL_BLUE_SIZE, 8);
		SDL_GL_SetAttribute(SDL_GL_ALPHA_SIZE, 8);
		SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
		SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);
		SDL_GL_SetAttribute(SDL_GL_STENCIL_SIZE, 8);

		SDL_DisplayMode current;
		SDL_GetCurrentDisplayMode(0, &current);
		m_window = SDL_CreateWindow("VSDMO", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, 1280, 720, SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE);
		m_glContext = SDL_GL_CreateContext(m_window);

		// --- shared context --- //
#ifdef WINDOWS
		//m_glLoaderContext = SDL_GL_CreateContext(m_window);
		//BOOL wglerror = wglShareLists((HGLRC)m_glContext, (HGLRC)m_glLoaderContext);
		//if (wglerror == FALSE) {
		//	throw std::runtime_error("wglShareLists failed");
		//}
		//SDL_GL_MakeCurrent(m_window, m_glContext);

		////GraphicsUpdater::initializeWindows(m_window, m_glLoaderContext);
		//std::cout << "Shared Contexts created and loader thread started." << std::endl;
#else
		//GraphicsUpdater::initializeLinux();
#endif
		// --- shared context --- //

		SDL_GL_SetSwapInterval(1); // Enable vsync

		m_activeCursor = SDL_CreateSystemCursor(SDL_SYSTEM_CURSOR_ARROW);

		std::cout << "SDL initialized: " << SDL_GetError() << std::endl;

		// setup GLEW
		GLenum error = glewInit();
		if (error != GLEW_OK) {

			std::cerr << "Glew Initialization failed! [" << error << "]: " << glewGetErrorString(error) << std::endl;
			return;
		}

		std::cout << "glew initialized." << std::endl;

		std::cout << "OpenGL version: " << glGetString(GL_VERSION) << std::endl;
		std::cout << "GLSL version: " << glGetString(GL_SHADING_LANGUAGE_VERSION) << std::endl;
		std::cout << "Renderer: " << glGetString(GL_RENDERER) << std::endl;

		// setup Dear ImGui
		m_imgui.init(m_window, m_glContext, glsl_version);
	}

	Gui::~Gui() {
		ImGui_ImplSDL2_Shutdown();
	}

	void Gui::shutdown() {
		//GraphicsUpdater::get().shutdown();
	}

	void Gui::renderLoop() {
		static bool updateNow = false;

		int startTime = SDL_GetTicks();
		int previousTime = startTime;

		int recalculateFps = 0;
		float fps = 0.f;

		float screenresX = 0.f;
		float screenresY = 0.f;

		while (!m_exit) {
			int currentTime = SDL_GetTicks() - startTime;

			if (m_updateMouse) {
				updateMouse();
				m_updateMouse = false;
			}
			SDL_SetCursor(m_activeCursor);

			SDL_Event event;
			
			while (SDL_PollEvent(&event) > 0) {
				// handle event
				handleSDLEvent(event);
				m_imgui.handleSDLEvent(event);

				// exit program if requested
				if (event.type == SDL_QUIT)
					m_exit = true;
				if (event.type == SDL_WINDOWEVENT && event.window.event == SDL_WINDOWEVENT_CLOSE && event.window.windowID == SDL_GetWindowID(m_window))
					m_exit = true;
				if (event.type == SDL_KEYDOWN && event.key.keysym.sym == SDLK_ESCAPE)
					m_exit = true;

			}

			if (m_exit) break;

#ifndef WINDOWS
			//GraphicsUpdater::get().handleRequestsExternal(30);
#endif // !WINDOWS

			//TODO: only update if buttons are pressed or events have happened
			updateNow = true;

			if (updateNow) {
				// rendering
				glCall(glClearColor(.9f, .9f, .9f, 0));
				glCall(glClearDepth(1.f));
				glCall(glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT));


				update(float(currentTime - previousTime) / 1000.f);
				if (recalculateFps-- < 1) {
					fps = 1000.f / (currentTime - previousTime);
					recalculateFps = 10;
				}

				previousTime = currentTime;
				render();

				m_imgui.startFrame();

				renderGui(fps);

				m_imgui.endFrame();

				SDL_GL_MakeCurrent(m_window, m_glContext);
				SDL_GL_SwapWindow(m_window);

				// allow resizing of the window
				ImGuiIO& io = ImGui::GetIO();
				if (screenresX != io.DisplaySize.x || screenresY != io.DisplaySize.y) {
					screenresX = io.DisplaySize.x;
					screenresY = io.DisplaySize.y;
					resize();
				}
				glViewport(0, 0, (int)io.DisplaySize.x, (int)io.DisplaySize.y);

			}

			updateNow = false;
		}
	}
}
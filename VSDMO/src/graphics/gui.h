// gui.h
#pragma once

#include <SDL2/SDL.h>
#include <GL/glew.h>
#include "imgui/imgui.h"

namespace Graphics {

	class ImGuiRenderer {
	public:
		ImGuiRenderer();
		~ImGuiRenderer();

		void init(SDL_Window *window, SDL_GLContext context, const char *glsl_version);

		void handleSDLEvent(SDL_Event event);

		void startFrame();
		void endFrame();
	private:
		SDL_Window *m_window;

	};

	class Gui {
	public:
		Gui();
		virtual ~Gui();

		void renderLoop();

		virtual void shutdown();
	protected:
		virtual void renderGui(float fps) {
			// an example ImGui Window
			ImGui::Begin("Menu");
			ImGui::Text("FPS: %.1f", fps);
			ImGui::Button("Button");
			ImGui::End();
		}
		virtual void update(float timePassed) {	/*do nothing by default*/ }
		virtual void render() {	/*do nothing by default*/ }
		virtual void handleSDLEvent(SDL_Event event) { /*do nothing by default*/ }

		virtual void resize() = 0;

		// sets m_activeCursor
		virtual void updateMouse() { /*do nothing by default*/ }


	protected:
		SDL_Cursor *m_activeCursor;
		SDL_Window *m_window;
		SDL_GLContext m_glContext;
		SDL_GLContext m_glLoaderContext;
		ImGuiRenderer m_imgui;

		bool m_exit;
		bool m_updateMouse;
	};

}
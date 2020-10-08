// imguiWindows.h
#pragma once

#include "Viewer.h"

namespace Graphics {

	// Interface and Superclass of all ImGuiWindows
	class ImGuiWindow {
	public:
		ImGuiWindow(Viewer& app) : m_app(app) {}
		virtual ~ImGuiWindow() {}

		virtual void render(float fps) = 0;
	protected:
		static const ImVec2 s_buttonSize;
		static const int s_itemWidth;

		Viewer& m_app;
	};


	// Renders the Main Menu, that lists all Tools, etc
	class MainWindow : public ImGuiWindow {
	public:
		MainWindow(Viewer& app, bool& showMe, std::shared_ptr<Viewer::Options>& options_);
		~MainWindow() {}

		void render(float fps) override;

	private:
		bool& m_showMe;
		int m_level;
		std::shared_ptr<Viewer::Options> options;
	};

	// Renders and Manages all ImGui windows
	class MainMenu : public ImGuiWindow {
	public:
		MainMenu(Viewer& app, std::shared_ptr<Viewer::Options>& options);
		~MainMenu() {}

		void render(float fps) override;
		
		void toggleMainWindow();

		//MainWindow::Options getOptions() { return m_mainWindow->getOptions(); }

	private:
		//inline void renderFileMenu() const;
		//inline void renderWindowMenu();

		// returns the string to the import destination if successful, 
		// an empty string otherwise
		//std::string renderImport(const std::string& file) const;

		// returns the string to the export destination if successful, 
		// an empty string otherwise
		//std::string renderExport(const std::string& file) const;

	private:

		std::unique_ptr<MainWindow> m_mainWindow;
		bool m_renderMain;
	};

}
// imguiWindows.cpp
#include "imguiWindows.h"

#include "cameraManager.h"


namespace Graphics {

	// ######################################################################## //
	// ### ImGuiWindow ######################################################## //
	// ######################################################################## //

	const ImVec2 ImGuiWindow::s_buttonSize = ImVec2(200, 30);
	const int ImGuiWindow::s_itemWidth = 200;


	// ######################################################################## //
	// ### MainWindow ######################################################### //
	// ######################################################################## //

	MainWindow::MainWindow(Viewer& app, bool& showMe, std::shared_ptr<Viewer::Options>& options_)
		: ImGuiWindow(app)
		, m_showMe(showMe)
		, options(options_)
	{
		//cout << "MainWindow Destructor" << endl;
	}

	void MainWindow::render(float fps) {
		ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiCond_FirstUseEver);
		ImGui::SetNextWindowSize(ImVec2(300, 600), ImGuiCond_FirstUseEver);

		
		
		if (ImGui::Begin("Menu")) {
			ImGui::Text("FPS: %.2f", fps);

			ImGui::Dummy(ImVec2(0.f, 10.f));
			ImGui::Separator();

			// ---------------------- //
			// --- Camera Options --- //
			ImGui::Dummy(ImVec2(0.f, 10.f));
			ImGui::Text("General Options");

			bool camIsLocked = CameraManager::getInstance().getActiveCamera()->isLocked();
			if (ImGui::Checkbox("Orbital Camera", &camIsLocked)) {
				CameraManager::getInstance().getActiveCamera()->toggleLock();
			}
			ImGui::Checkbox("Show Data Window", &options->showDataWindow);

			ImGui::Separator();

			// ---------------------------- //
			// --- Mesh Display Options --- //
			ImGui::Dummy(ImVec2(0.f, 10.f));
			//ImGui::Text("Mesh Display Options");
			if (ImGui::CollapsingHeader("Mesh Display Options")) {
				ImGui::Checkbox("Backface Culling", &options->culling);
				ImGui::Checkbox("Show Mesh", &options->showMesh);
				ImGui::Checkbox("Wireframe", &options->wireframe);
				ImGui::PushItemWidth(s_itemWidth);
				ImGui::InputInt("Highlighted Color", &options->highlightedColor);
				ImGui::PopItemWidth();
				ImGui::Checkbox("Show Coloring", &options->showColoring);
				ImGui::Checkbox("Show Feature Vertices", &options->showFeatureVertices);

				ImGui::Checkbox("Show Element Quality", &options->showElementQuality);

				ImGui::Checkbox("Render Volume", &options->renderVolume);
				ImGui::PushItemWidth(s_itemWidth);
				ImGui::SliderFloat("Slicing Z", &options->slicingZ, options->slicingZMin, options->slicingZMax);
				ImGui::SliderFloat("Size Factor", &options->sizeFactor, -1.f, 1.f);
				ImGui::PopItemWidth();
				//ImGui::Checkbox("Flat Shading", &options->flatShadingMesh);
			}


			ImGui::Separator();

			// ---------------------------------- //
			// --- Surface Estimation (Whole) --- //
			options->surfaceOptionUpdated = false;
			ImGui::Dummy(ImVec2(0.f, 10.f));

			if (ImGui::CollapsingHeader("Surface Estimation Options")) {
				//ImGui::Text("Surface Estimation Options");
				ImGui::PushItemWidth(s_itemWidth);
				options->surfaceOptionUpdated |= ImGui::InputInt("Vertex ID", &options->selectedVertex);
				ImGui::PopItemWidth();
				ImGui::Checkbox("Show Surface", &options->showSurfaceEstimation);
				ImGui::Checkbox("Render Lines", &options->surfaceEstimationRenderLines);

				options->surfaceOptionUpdated |= ImGui::Checkbox("Show Interpolated Surface", &options->showInterpolatedSurface);
				ImGui::PushItemWidth(s_itemWidth);
				options->surfaceOptionUpdated |= ImGui::InputInt("Feature Surface ID", &options->featureSurfaceID);
				ImGui::PopItemWidth();
			}



			//ImGui::Separator();
			//ImGui::Dummy(ImVec2(0.f, 10.f));
			//ImGui::Text("Display Options");


			
			ImGui::Separator();

			// buttons to continue exectuion
			//ImGui::Dummy(ImVec2(0.f, 10.f));
			ImGui::Text("Execution Options");

			const char* items[] = { "Mean Ratio", "Area", "Right Angle", "Jacobian", "Min Angle", "Radius Ratio", "Max Angle" };
			static const char* current_item = items[0];

			if (ImGui::BeginCombo("##combo", current_item)) // The second parameter is the label previewed before opening the combo.
			{
				for (int n = 0; n < IM_ARRAYSIZE(items); n++)
				{
					bool is_selected = (current_item == items[n]); // You can store your selection however you want, outside or inside your objects
					if (ImGui::Selectable(items[n], is_selected)) {
						//cout << "Selectable" << endl;
						current_item = items[n];
						options->selectedMetric = static_cast<QualityCriterium>(n);
					}
					if (is_selected) {
						//cout << "is_selected" << endl;
						ImGui::SetItemDefaultFocus();   // You may set the initial focus when opening the combo (scrolling + for keyboard navigation support)
					}
				}
				ImGui::EndCombo();
			}
			
			static char inputTextInFile[64] = "res/homer.off";
			ImGui::PushItemWidth(s_itemWidth);
			ImGui::InputText("Input File", inputTextInFile, 64);

			static char inputTextOutFile[64] = "res/outfile";
			ImGui::InputText("Output File", inputTextOutFile, 64);
			ImGui::PopItemWidth();
			
			if (ImGui::Button("Load Mesh", s_buttonSize)) {
				m_app.openMesh(inputTextInFile);
			}
			if (ImGui::Button("Save Mesh", s_buttonSize)) {
				m_app.saveMesh(inputTextOutFile);
			}
			if (ImGui::Button("Save Surface Mesh", s_buttonSize)) {
				m_app.saveSurfaceMesh(inputTextOutFile);
			}

			/*ImGui::InputInt("Quality Metric", (int*)&options->selectedMetric);*/


			if (ImGui::Button("Start/Continue DMO", s_buttonSize)) {
				m_app.continueDMO();
			}
			//if (ImGui::Button("Start DMO", s_buttonSize)) {
			//	m_app.startDMO();
			//}
			if (ImGui::Button("Pause DMO", s_buttonSize)) {
				m_app.pauseDMO();
			}
			
			//if (ImGui::Button("Stop DMO", s_buttonSize)) {
			//	m_app.stopDMO();
			//}
			if (ImGui::Button("Step DMO", s_buttonSize)) {
				m_app.stepDMO();
			}
			if (ImGui::Button("Print Quality", s_buttonSize)) {
				m_app.printQuality();
			}

			

			bool disableButton = true;
			if (disableButton) {
				ImGui::PushStyleVar(ImGuiStyleVar_Alpha, ImGui::GetStyle().Alpha * 0.5f);
			}
			if (ImGui::Button("Continue Execution", s_buttonSize)) {
				//if (!disableButton) m_app.resume();
			}
			if (disableButton) {
				ImGui::PopStyleVar();
			}



			ImGui::Separator();

		}
		ImGui::End();
		

		if (!options->showDataWindow) return;

		ImGui::SetNextWindowPos(ImVec2(800, 0), ImGuiCond_FirstUseEver);
		ImGui::SetNextWindowSize(ImVec2(400, 200), ImGuiCond_FirstUseEver);
		if (ImGui::Begin("Quality Info")) {
			ImGui::Text("min quality %f", options->curr_quality);
			ImVec2 plotextent(ImGui::GetContentRegionAvailWidth(), 100);
			const vector<float>& v = options->qualities;
			float minval = 0.f;
			float maxval = 1.f;
			if (v.size() > 0) {
				//minval = v[0];
				maxval = v[v.size() - 1];
			}
			ImGui::PlotLines("", v.data(), (int)v.size(), 0, nullptr, minval, maxval, plotextent);

			const vector<int>& hist = options->histogram;
			vector<float> histf(hist.size());
			for (int i = 0; i < hist.size(); ++i) {
				histf[i] = (float)hist[i];
			}
			ImGui::PlotHistogram("", (float*)histf.data(), (int)histf.size(), 0, NULL, FLT_MAX, FLT_MAX, ImVec2(200,40));
		}
		ImGui::End();
		//ImGui::GetWindowDrawList()->AddText(ImGui::GetWindowFont(), ImGui::GetWindowFontSize(), ImVec2(100.f, 100.f), ImColor(255, 255, 0, 255), "Hello World", 0, 0.0f, 0);
	}


	// ######################################################################## //
	// ### MainMenu ########################################################### //
	// ######################################################################## //

	MainMenu::MainMenu(Viewer& app, std::shared_ptr<Viewer::Options>& options)
		: ImGuiWindow(app)
		, m_mainWindow(nullptr)
		, m_renderMain(true)

	{
		m_mainWindow = std::make_unique<MainWindow>(app, m_renderMain, options);
	}

	void MainMenu::render(float fps) {
		// render Main window
		if (m_renderMain) {
			m_mainWindow->render(fps);
		}
	}

	void MainMenu::toggleMainWindow() {
		m_renderMain = !m_renderMain;
	}
}
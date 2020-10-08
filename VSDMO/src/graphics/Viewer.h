// Viewer.h
#pragma once

#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <vector>
#include <thread>

#include "gui.h"
#include "drawableTetMesh.h"
#include "drawableTriMesh.h"
#include "drawableQuadMesh.h"
#include "drawableHexMesh.h"
#include "drawableSurface.h"

#include "mesh/MeshTetGPU.h"
#include "dmo/DMOTet.h"


namespace Graphics {

	// --- ImGui Window classes have to be forward declared here --- //
	// --- and included in the cpp-File ---------------------------- //
	class MainMenu;
	// ------------------------------------------------------------- //

	/**
	 *	class Viewer
	 *	@brief The Rendering Application responsible for the render loop
	 *	and halting or resuming the working thread.
	 */
	class Viewer : public Gui {
	public:
		Viewer();
		~Viewer();

		Viewer(const Viewer& copy) = delete;
		Viewer& operator=(const Viewer& assign) = delete;

		virtual void shutdown() override;

		virtual void renderGui(float fps) override;
		virtual void update(float timePassed) override;
		virtual void render() override;
		virtual void handleSDLEvent(SDL_Event event) override;
		virtual void resize() override;

		void openMesh(const std::string& file);
		void saveMesh(const std::string& file);
		void saveSurfaceMesh(const std::string& file);

		void startDMO();
		void pauseDMO();
		void continueDMO();
		void stepDMO();
		void printQuality();

		void makeSurfaceEstimation();

	protected:
		void makeDMO();

		void renderMesh(Camera& cam);
		void renderSurface(Camera& cam);
		//void updateSinglethreaded(float timePassed);
		void updateMultithreaded(float timePassed);

		void updateMeshPositionsDev(void* points, size_t size);
		void updateMeshNormalsDev(void* normals, size_t size);
		void updateMeshIndicesDev(void* data, size_t size);
		void updateMeshCellIndicesDev(void* data, size_t size);

		void updateMesh();
		void updateSurfaceEstimation();
		void updateQualities();

		void dmoWorkerFunc();

	protected:
		enum class MeshType {
			TRI,
			TET,
			QUAD,
			HEX,
		} m_meshType = MeshType::TRI;

		enum class DMOExecutionState {
			NONE,
			RUNNING,
			PAUSED,
			DONE,
			STEP,
		} m_dmoState = DMOExecutionState::NONE;

		// Gui Overlay
		std::unique_ptr<MainMenu> m_menu;

		std::unique_ptr<DrawableMesh> m_drawableMesh; // drawable mesh
		std::shared_ptr<DMOMeshBase> m_viewedDMOMesh; // actual mesh used by dmo/ currently loaded
		std::unique_ptr<DMO::DMOBaseClass> m_dmo; // instance of dmo algo
		std::unique_ptr<DrawableSurfaceEstimation> m_drawableSurfaceEstimation; // surface estimation of a selected vertex

	public:
		struct Options {
			// DMO
			int niter = 100;
			// Render
			bool culling = false;
			// Mesh Render
			bool showMesh = true;
			bool wireframe = false;
			bool flatShadingMesh = false;
			int highlightedColor = 0;
			bool showColoring = false;
			bool showFeatureVertices = false;
			bool showElementQuality = false;
			// Surface Render
			int selectedVertex = 0;
			bool showSurfaceEstimation = true;
			bool surfaceEstimationRenderLines = true;
			bool showInterpolatedSurface = false;
			int featureSurfaceID = 0;
			bool surfaceOptionUpdated = false;
			// Volume Render
			bool renderVolume = false;
			float slicingZMin = -1.f;
			float slicingZMax = 1.f;
			float slicingZ = 0.f;
			float sizeFactor = 0.f;

			// Data
			bool showDataWindow = true;
			float curr_quality = 0.f;
			vector<float> qualities;
			vector<int> histogram;

			QualityCriterium selectedMetric = QualityCriterium::MEAN_RATIO;
		};

	protected:
		std::shared_ptr<Options> options;

		std::unique_ptr<std::thread> m_worker;
		volatile bool m_dataAvailable;
		volatile bool m_activeWorker;
		volatile bool m_inIteration;
		volatile bool m_copyDone;
	};

}

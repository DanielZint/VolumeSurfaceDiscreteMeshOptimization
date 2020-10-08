// Viewer.cpp
#include "Viewer.h"

#include <memory>

#include "GraphicsConfig.h"

#include "camera.h"
#include "shaderManager.h"
#include "cameraManager.h"
#include "imguiWindows.h"

#include "mesh/MeshFactory.h"
#include "io/FileWriter.h"
#include "dmo/DMOTet.h"
#include "dmo/DMOTri.h"
#include "dmo/DMOTriFlat.h"
#include "dmo/DMOQuad.h"
#include "dmo/DMOHex.h"
#include "mesh/MeshFunctions.h"

//#include "CudaUtil.h"
#include "Timer.h"


namespace Graphics {

	glm::vec3 mouseToWorldspace() {
		int mouseX = 0;
		int mouseY = 0;

		glm::vec3 wpos(NAN);

		Uint32 buttons = SDL_GetMouseState(&mouseX, &mouseY);
		// this is a right click

		ImGuiIO& io = ImGui::GetIO();
		float screenResX = io.DisplaySize.x;
		float screenResY = io.DisplaySize.y;

		float screenX = (((float)mouseX) / screenResX) * 2.f - 1.f;
		float screenY = ((screenResY - (float)mouseY) / screenResY) * 2.f - 1.f;

		float depth = 1.f;

		// read depth of click
		glCall(glReadPixels((GLint)mouseX, (GLint)(screenResY - mouseY), 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT, &depth));

		// convert from screen space to world space
		const Camera& cam = *(CameraManager::getInstance().getActiveCamera());
		glm::vec4 tmp = glm::vec4(screenX, screenY, 2.f * depth - 1.f, 1.f);
		glm::mat4 inverseProjView = glm::inverse(cam.getProjectionMatrix() * cam.getViewMatrix());
		glm::vec4 result = inverseProjView * tmp;
		result /= result.w;

		wpos = glm::vec3(result);
		return wpos;
	}

	Viewer::Viewer()
		: Gui()
		, m_menu(nullptr)
		, m_drawableMesh(nullptr)
		, m_viewedDMOMesh(nullptr)
		, m_drawableSurfaceEstimation(nullptr)
		, m_dataAvailable(false)
		, m_activeWorker(true)
		, m_inIteration(false)
		, m_copyDone(false)
	{
		// create 2D and 3D cameras
		CameraManager& cameraManager = CameraManager::getInstance();
		std::shared_ptr<Camera> cam3D = std::make_shared<Camera3D>();
		cam3D->forcePosition(glm::vec3(0.27f, 0.4f, 1.13f));
		cameraManager.registerCamera(cam3D, "3D");

		cameraManager.setCameraActive("3D");

		// create shaders
		ShaderManager& shaderManager = ShaderManager::getInstance();
		shaderManager.registerProgram("phongLighting");
		shaderManager.registerProgram("phongLightingUniform");
		shaderManager.registerProgram("phongLightingNormals");
		shaderManager.registerProgram("simpleColor");
		shaderManager.registerProgram("normalDisplay");
		shaderManager.registerProgram("trisToEdges");
		shaderManager.registerProgram("helloTri");
		shaderManager.registerProgram("flatShading");
		shaderManager.registerProgram("flatShadingShowNormals");
		shaderManager.registerProgram("pointRange");
		shaderManager.registerProgram("tetrahedraToTris");
		shaderManager.registerProgram("tetrahedraToTrisFlat");
		shaderManager.registerProgram("tetrahedraToTrisQuality");
		shaderManager.registerProgram("triQuality");
		shaderManager.registerProgram("quadsPhongLightingNormals");
		shaderManager.registerProgram("quadsToEdges");
		shaderManager.registerProgram("quadQuality");
		shaderManager.registerProgram("quadsPhongLightingNormalsSlice");
		shaderManager.registerProgram("hexquadsQuality");
		

		options = std::make_shared<Options>();
		options->histogram.resize(100);

		// create ImGuiWindow
		m_menu = std::make_unique<MainMenu>(*this, options);

	}

	Viewer::~Viewer() {
		// do nothing
		//cout << "Viewer Destructor" << endl;
	}

	void Viewer::shutdown() {
		//cout << "shutdown" << endl;
		m_activeWorker = false;
		m_dmoState = DMOExecutionState::NONE;
		m_copyDone = true;
		if (m_worker != nullptr) {
			//cout << "joining" << endl;
			m_worker->join();
			//cout << "joined" << endl;
		}
		m_worker.reset();
		Gui::shutdown();
	}

	void Viewer::renderGui(float fps) {
		m_menu->render(fps);
	}


	void Viewer::update(float timePassed) {
		updateMultithreaded(timePassed);
	}

	//void Viewer::updateSinglethreaded(float timePassed) {
	//	if (!ImGui::GetIO().WantCaptureMouse) {
	//		CameraManager& cameraManager = CameraManager::getInstance();
	//		cameraManager.getActiveCamera()->update(timePassed);
	//	}

	//	/*if (dataAvailable) {
	//		dataAvailable = false;
	//		updateMesh();
	//		m_dmoContinue.notify_one();
	//	}*/

	//	if (m_dmoState != NONE) {
	//		if (m_dmo->isDone()) {
	//			m_dmoState = DONE;
	//		}
	//		if (m_dmoState == RUNNING) {
	//			m_dmo->doIteration();
	//			updateMesh();
	//		}
	//	}

	//	if (m_drawableSurfaceEstimation != nullptr && options->showSurfaceEstimation) {
	//		int vid = options->selectedVertex;
	//		if (vid >= 0 && vid < m_viewedDMOMesh->nVerticesSurf) {
	//			Interop& interopPos = m_drawableSurfaceEstimation->getPosInterop();
	//			interopPos.map();
	//			void* dstSurfaceVertices = interopPos.ptr();
	//			m_dmo->getLocalSurfacePoints(options->selectedVertex, m_drawableSurfaceEstimation->nu(), m_drawableSurfaceEstimation->nv(), dstSurfaceVertices);
	//			interopPos.unmap();
	//		}
	//	}

	//	if (m_drawableMesh != nullptr) {
	//		m_drawableMesh->setVertexHighlighted(options->selectedVertex);
	//	}
	//}


	void Viewer::updateMultithreaded(float timePassed) {
		if (!ImGui::GetIO().WantCaptureMouse) {
			CameraManager& cameraManager = CameraManager::getInstance();
			cameraManager.getActiveCamera()->update(timePassed);
		}

		if (m_dmoState != DMOExecutionState::NONE) {
			if (m_dmo->isDone()) {
				m_dmoState = DMOExecutionState::DONE;
			}
		}

		bool doUpdateSurfaceEstimation = false;

		if (m_dataAvailable) {
			m_dataAvailable = false;
			if (m_dmoState == DMOExecutionState::STEP) {
				m_dmoState = DMOExecutionState::PAUSED;
			}

			updateMesh();
			updateQualities();
			doUpdateSurfaceEstimation = true;
			m_copyDone = true;
		}
		
		//updateSurfaceEstimation(); // slows down algo because of interop map
		if (doUpdateSurfaceEstimation || options->surfaceOptionUpdated) {
			updateSurfaceEstimation();
		}


		if (m_drawableMesh != nullptr) {
			m_drawableMesh->setVertexHighlighted(options->selectedVertex);
		}
	}

	void Viewer::renderMesh(Camera& cam) {
		if (!options->showMesh || m_drawableMesh == nullptr) {
			return;
		}
		if (m_meshType == MeshType::TET) {
			DrawableInteropTetMesh& drawableTetMesh = dynamic_cast<DrawableInteropTetMesh&>(*m_drawableMesh);
			drawableTetMesh.setSlicingVal(options->slicingZ);
			drawableTetMesh.setSizeFactor(options->sizeFactor);
		}
		if (m_meshType == MeshType::HEX) {
			DrawableInteropHexMesh& drawableHexMesh = dynamic_cast<DrawableInteropHexMesh&>(*m_drawableMesh);
			drawableHexMesh.setSlicingVal(options->slicingZ);
			drawableHexMesh.setSizeFactor(options->sizeFactor);
		}

		if (options->showElementQuality) {
			m_drawableMesh->renderElementQuality(cam, glm::vec4(1, 0, 0, 1));
			return;
		}

		/* Mesh Render */
		if (options->renderVolume && m_meshType == MeshType::TET) {
			DrawableInteropTetMesh& drawableTetMesh = dynamic_cast<DrawableInteropTetMesh&>(*m_drawableMesh);
			drawableTetMesh.renderTetrahedra(cam);
		} else if (options->renderVolume && m_meshType == MeshType::HEX) {
			DrawableInteropHexMesh& drawableHexMesh = dynamic_cast<DrawableInteropHexMesh&>(*m_drawableMesh);
			drawableHexMesh.renderHexahedra(cam);
		} else if (options->wireframe) {
			m_drawableMesh->renderFacesAndEdges(cam, glm::vec4(0.6, 0.6, 0.6, 1), glm::vec4(0.2, 0.2, 0.2, 1));
		}
		else {
			m_drawableMesh->renderFaces(cam, glm::vec4(0.5, 0.5, 0.5, 1));
		}

		/* Vertex Highlighting */
		if (options->showColoring) {
			auto offsets = m_viewedDMOMesh->getColorOffsets(options->highlightedColor);
			m_drawableMesh->renderVertices(cam, offsets[0], offsets[1], glm::vec4(0.5, 1, 0.5, 1));
		}
		if (options->showFeatureVertices) {
			m_drawableMesh->renderVertices(cam, m_viewedDMOMesh->nVerticesSurfFree, m_viewedDMOMesh->nVerticesSurf, glm::vec4(1, 0.5, 0, 1));
		}
		m_drawableMesh->renderVertex(cam, glm::vec4(1, 0, 0, 1));
		
	}

	void Viewer::renderSurface(Camera& cam) {
		if (!options->showSurfaceEstimation || m_drawableSurfaceEstimation == nullptr) {
			return;
		}

		if (options->surfaceEstimationRenderLines) {
			m_drawableSurfaceEstimation->renderLines(cam);
		}
		else {
			m_drawableSurfaceEstimation->renderFaces(cam);
		}
	}

	void Viewer::render() {
		Camera& cam = *(CameraManager::getInstance().getActiveCamera());
		//cout << "camera: " << cam.getPosition() << endl;

		glCullFace(GL_BACK);
		if (options->culling) {
			glEnable(GL_CULL_FACE);
		}
		else {
			glDisable(GL_CULL_FACE);
		}
		
		renderMesh(cam);
		renderSurface(cam);
	}

	void Viewer::handleSDLEvent(SDL_Event event) {
		if (event.type == SDL_KEYDOWN) {
			CameraManager& cameraManager = CameraManager::getInstance();
			if (event.key.keysym.sym == SDLK_F10 || event.key.keysym.sym == SDLK_F11) {
				//resume();
			}
			else if (event.key.keysym.sym == SDLK_TAB) {
				m_menu->toggleMainWindow();
			}
			else if (event.key.keysym.sym == SDLK_F6) {
				ShaderManager& shaderManager = ShaderManager::getInstance();
				shaderManager.update();
			}
		}

		if (event.type == SDL_MOUSEBUTTONDOWN) {
			if (event.button.button == SDL_BUTTON_RIGHT) {
				// highlight the closest vertex, edge or face
				// convert mouse position to world space position
				glm::vec3 mousePos = mouseToWorldspace();
				cout << "clicked position: " << mousePos << endl;
				// temp
				//AABB aabb = findAABB(*m_viewedDMOMesh);
				//cout << "AABB" << aabb.minPos[0] << " " << aabb.minPos[1] << " " << aabb.minPos[2] << " " << aabb.maxPos[0] << " " << aabb.maxPos[1] << " " << aabb.maxPos[2] << endl;
				if (m_viewedDMOMesh != nullptr) {
					options->selectedVertex = findClosestVertex(*m_viewedDMOMesh, Vec3f(mousePos.x, mousePos.y, mousePos.z));
					cout << "selected vertex " << options->selectedVertex << endl;
					updateSurfaceEstimation();
				}
				
			}
		}

		// handle camera movement
		if (!ImGui::GetIO().WantCaptureMouse) {
			Camera& cam = *(CameraManager::getInstance().getActiveCamera());
			cam.handleEvent(event);
		}
	}

	void Viewer::resize() {
		Camera& cam = *(CameraManager::getInstance().getActiveCamera());
		cam.zoom(1.f);
	}



	void Viewer::updateMeshPositionsDev(void* points, size_t size) {
		m_drawableMesh->updatePositionsDev(points, size);
	}

	void Viewer::updateMeshNormalsDev(void* normals, size_t size) {
		m_drawableMesh->updateNormalsDev(normals, size);
	}

	void Viewer::updateMeshIndicesDev(void* data, size_t size) {
		m_drawableMesh->setIboDev(data, size);
	}

	void Viewer::updateMeshCellIndicesDev(void* data, size_t size) {
		if (m_meshType == MeshType::TET) {
			dynamic_cast<DrawableInteropTetMesh&>(*m_drawableMesh).setIboCellsDev(data, size);
		}
		else if (m_meshType == MeshType::HEX) {
			dynamic_cast<DrawableInteropHexMesh&>(*m_drawableMesh).setIboCellsDev(data, size);
		}
	}

	void Viewer::updateQualities() {
		options->curr_quality = m_dmo->getQuality();
		options->qualities.push_back(options->curr_quality);
		m_dmo->getQualityHistogram(options->histogram, (int)options->histogram.size());

		Interop& interopTbo = m_drawableMesh->getTboInterop();
		interopTbo.map();
		void* dataptr = interopTbo.ptr();
		m_dmo->getElementQualities(dataptr);
		interopTbo.unmap();
	}


	void Viewer::updateMesh() {
		if (m_drawableMesh != nullptr) {
			m_drawableMesh->updatePositionsDev(m_viewedDMOMesh->getVertexPoints(), (size_t)m_viewedDMOMesh->nVertices * 3 * sizeof(float));
			m_drawableMesh->updateNormalsDev(m_viewedDMOMesh->getVertexNormals(), (size_t)m_viewedDMOMesh->nVerticesSurf * 3 * sizeof(float));
		}
	}

	void Viewer::updateSurfaceEstimation() {
		if (m_dmo != nullptr && m_drawableSurfaceEstimation != nullptr && options->showSurfaceEstimation) {
			int vid = options->selectedVertex;
			if (vid >= 0 && vid < m_viewedDMOMesh->nVerticesSurf) {
				Interop& interopPos = m_drawableSurfaceEstimation->getPosInterop();
				interopPos.map();
				void* dstSurfaceVertices = interopPos.ptr();
				if (options->showInterpolatedSurface) {
					m_dmo->getEstimateLocalSurfacePoints(options->selectedVertex, m_drawableSurfaceEstimation->nu(), m_drawableSurfaceEstimation->nv(), dstSurfaceVertices);
				}
				else {
					m_dmo->getLocalSurfacePoints(options->selectedVertex, m_drawableSurfaceEstimation->nu(), m_drawableSurfaceEstimation->nv(), dstSurfaceVertices, options->featureSurfaceID);
				}
				
				interopPos.unmap();
			}
		}
	}




	void Viewer::makeDMO() {
		if (m_meshType == MeshType::TET) {
			m_dmo = std::make_unique<DMO::DMOTetClass>(dynamic_cast<DMOMeshTet&>(*m_viewedDMOMesh), options->selectedMetric);
		}
		else if (m_meshType == MeshType::TRI) {
			if (m_viewedDMOMesh->isFlat()) {
				m_dmo = std::make_unique<DMO::DMOTriFlatClass>(dynamic_cast<DMOMeshTri&>(*m_viewedDMOMesh), options->selectedMetric);
			}
			else {
				m_dmo = std::make_unique<DMO::DMOTriClass>(dynamic_cast<DMOMeshTri&>(*m_viewedDMOMesh), options->selectedMetric);
			}
			
		}
		else if (m_meshType == MeshType::QUAD) {
			m_dmo = std::make_unique<DMO::DMOQuadClass>(dynamic_cast<DMOMeshQuad&>(*m_viewedDMOMesh), options->selectedMetric);
		}
		else if (m_meshType == MeshType::HEX) {
			m_dmo = std::make_unique<DMO::DMOHexClass>(dynamic_cast<DMOMeshHex&>(*m_viewedDMOMesh), options->selectedMetric);
		}
		m_dmoState = DMOExecutionState::PAUSED;
		m_inIteration = false;
		m_dataAvailable = false;
		m_activeWorker = true;
		m_worker = std::make_unique<std::thread>(&Viewer::dmoWorkerFunc, this);
		makeSurfaceEstimation();
	}

	void Viewer::openMesh(const std::string& file) {
		// kill old worker thread
		m_dmoState = DMOExecutionState::NONE;

		if (m_worker != nullptr) {
			m_activeWorker = false;
			//m_dmoState = DMOExecutionState::NONE;
			m_copyDone = true;
			m_worker->join();
			m_worker.reset();
		}

		m_inIteration = false;
		m_dataAvailable = false;
		m_copyDone = false;

		options->qualities.clear();
		
		if (m_dmo != nullptr)
			m_dmo.reset();

		std::size_t pointPos = file.find_last_of(".");
		std::string fileExtension;
		if (pointPos != std::string::npos) {
			fileExtension = file.substr(pointPos+1);
			cout << "file extension: " << fileExtension << endl;
		}
		
		

		if (fileExtension.compare("node") == 0 || fileExtension.compare("ele") == 0) {
			m_viewedDMOMesh = DMOMeshTetFactory::create(file.substr(0, pointPos));
			if (m_viewedDMOMesh == nullptr) {
				return;
			}
			DMOMeshTet& meshtet = dynamic_cast<DMOMeshTet&>(*m_viewedDMOMesh);
			m_meshType = MeshType::TET;
			m_drawableMesh = std::make_unique<DrawableInteropTetMesh>(glm::vec4(0, 0, 1, 1));
			DrawableInteropTetMesh& drawabletetmesh = dynamic_cast<DrawableInteropTetMesh&>(*m_drawableMesh);
			drawabletetmesh.init(meshtet.nVertices, meshtet.nTriangles, meshtet.nTetrahedra);
			//drawabletetmesh.initTbo(meshtet.nTetrahedra);

			updateMeshPositionsDev(meshtet.getVertexPoints(), meshtet.nVertices * 3 * sizeof(float));
			updateMeshNormalsDev(meshtet.getVertexNormals(), meshtet.nVerticesSurf * 3 * sizeof(float));
			updateMeshIndicesDev(meshtet.getTriangles(), meshtet.nTriangles * 3 * sizeof(int));

			updateMeshCellIndicesDev(meshtet.getTetrahedra(), meshtet.nTetrahedra * 4 * sizeof(int));
		}
		else if (fileExtension.compare("mesh") == 0) {
			m_viewedDMOMesh = DMOMeshHexFactory::create(file);
			if (m_viewedDMOMesh == nullptr) {
				return;
			}
			DMOMeshHex& meshhex = dynamic_cast<DMOMeshHex&>(*m_viewedDMOMesh);
			m_meshType = MeshType::HEX;
			m_drawableMesh = std::make_unique<DrawableInteropHexMesh>(glm::vec4(0, 0, 1, 1));
			DrawableInteropHexMesh& drawablehexmesh = dynamic_cast<DrawableInteropHexMesh&>(*m_drawableMesh);
			drawablehexmesh.init(meshhex.nVertices, meshhex.nQuads, meshhex.nHexahedra);
			//drawabletetmesh.initTbo(meshtet.nTetrahedra);

			updateMeshPositionsDev(meshhex.getVertexPoints(), meshhex.nVertices * 3 * sizeof(float));
			updateMeshNormalsDev(meshhex.getVertexNormals(), meshhex.nVerticesSurf * 3 * sizeof(float));
			updateMeshIndicesDev(meshhex.getQuads(), meshhex.nQuads * 4 * sizeof(int));

			updateMeshCellIndicesDev(meshhex.getHexahedra(), meshhex.nHexahedra * 8 * sizeof(int));
		}
		else {
			int nNodesFace = getFaceType(file);
			if (nNodesFace == 3) {
				m_viewedDMOMesh = DMOMeshTriFactory::create(file);
				if (m_viewedDMOMesh == nullptr) {
					return;
				}
				m_meshType = MeshType::TRI;
				m_drawableMesh = std::make_unique<DrawableInteropTriMesh>(glm::vec4(0, 0, 1, 1));
				m_drawableMesh->init(m_viewedDMOMesh->nVerticesSurf, m_viewedDMOMesh->nFaces(), m_viewedDMOMesh->nFaces());
				//m_drawableMesh->initTbo(m_viewedDMOMesh->nTriangles);

				updateMeshPositionsDev(m_viewedDMOMesh->getVertexPoints(), m_viewedDMOMesh->nVerticesSurf * 3 * sizeof(float));
				updateMeshNormalsDev(m_viewedDMOMesh->getVertexNormals(), m_viewedDMOMesh->nVerticesSurf * 3 * sizeof(float));
				updateMeshIndicesDev(m_viewedDMOMesh->getTriangles(), m_viewedDMOMesh->nFaces() * 3 * sizeof(int));
			}
			else if (nNodesFace == 4) {
				m_viewedDMOMesh = DMOMeshQuadFactory::create(file);
				if (m_viewedDMOMesh == nullptr) {
					return;
				}
				m_meshType = MeshType::QUAD;
				m_drawableMesh = std::make_unique<DrawableInteropQuadMesh>(glm::vec4(0, 0, 1, 1));
				m_drawableMesh->init(m_viewedDMOMesh->nVerticesSurf, m_viewedDMOMesh->nFaces(), m_viewedDMOMesh->nFaces());

				updateMeshPositionsDev(m_viewedDMOMesh->getVertexPoints(), m_viewedDMOMesh->nVerticesSurf * 3 * sizeof(float));
				updateMeshNormalsDev(m_viewedDMOMesh->getVertexNormals(), m_viewedDMOMesh->nVerticesSurf * 3 * sizeof(float));
				updateMeshIndicesDev(m_viewedDMOMesh->getQuads(), m_viewedDMOMesh->nFaces() * 4 * sizeof(int));
			}
			else {
				return;
			}
		}

		makeDMO();
		updateQualities();
		updateSurfaceEstimation();

		AABB aabb = findAABB(*m_viewedDMOMesh);
		//cout << "AABB" << aabb.minPos[0] << " " << aabb.minPos[1] << " " << aabb.minPos[2] << " " << aabb.maxPos[0] << " " << aabb.maxPos[1] << " " << aabb.maxPos[2] << endl;
		float sizeZ = aabb.maxPos.z - aabb.minPos.z;
		options->slicingZMin = aabb.minPos.z - sizeZ * 0.05f;
		options->slicingZMax = aabb.maxPos.z + sizeZ * 0.05f;
		// set mesh center for camera orbit
		Vec3f centerPos = 0.5f * (aabb.minPos + aabb.maxPos);
		Vec3f diagVec = aabb.maxPos - aabb.minPos;
		float meshScale = diagVec.norm();
		glm::vec3 centerPosGLM(centerPos[0], centerPos[1], centerPos[2]);
		m_drawableMesh->setCenter(glm::vec3(centerPos[0], centerPos[1], centerPos[2]));
		Camera3D& cam = dynamic_cast<Camera3D&>(*(CameraManager::getInstance().getActiveCamera()));
		cam.setViewLockOn(centerPosGLM, meshScale);
		
	}

	void Viewer::saveMesh(const std::string& file) {
		cout << "save " << file << endl;
		if (m_viewedDMOMesh == nullptr) return;
		while (m_inIteration); // dont save while worker thread is changing mesh
		if (m_meshType == MeshType::TET) {
			//m_dmo = std::make_unique<DMO::DMOTetClass>(dynamic_cast<DMOMeshTet&>(*m_viewedDMOMesh));
			writeTetgen(file, dynamic_cast<DMOMeshTet&>(*m_viewedDMOMesh));
		}
		else if (m_meshType == MeshType::TRI) {
			//m_dmo = std::make_unique<DMO::DMOTriClass>(dynamic_cast<DMOMeshTri&>(*m_viewedDMOMesh));
			writeOFF(file, dynamic_cast<DMOMeshTri&>(*m_viewedDMOMesh));
		}
		else if (m_meshType == MeshType::QUAD) {
			//m_dmo = std::make_unique<DMO::DMOTriClass>(dynamic_cast<DMOMeshTri&>(*m_viewedDMOMesh));
			writeOFF(file, dynamic_cast<DMOMeshQuad&>(*m_viewedDMOMesh));
		}
		else if (m_meshType == MeshType::HEX) {
			//m_dmo = std::make_unique<DMO::DMOTriClass>(dynamic_cast<DMOMeshTri&>(*m_viewedDMOMesh));
			writeHex(file, dynamic_cast<DMOMeshHex&>(*m_viewedDMOMesh));
		}
	}

	void Viewer::saveSurfaceMesh(const std::string& file) {
		if (m_viewedDMOMesh == nullptr) return;
		while (m_inIteration);
		if (m_meshType == MeshType::TET) {
			auto dmo_mesh_surf = DMOMeshTriFactory::create(dynamic_cast<DMOMeshTet&>(*m_viewedDMOMesh));
			writeOFF(file + "_surf", *dmo_mesh_surf);
		}
		else if (m_meshType == MeshType::HEX) {
			auto dmo_mesh_surf = DMOMeshQuadFactory::create(dynamic_cast<DMOMeshHex&>(*m_viewedDMOMesh));
			writeOFF(file + "_surf", *dmo_mesh_surf);
		}
	}

	void Viewer::startDMO() {
		//if (m_dmoState != NONE) {
		//	m_dmoState = RUNNING;
		//}
	}
	void Viewer::pauseDMO() {
		if (m_dmoState != DMOExecutionState::NONE) {
			m_dmoState = DMOExecutionState::PAUSED;
		}
	}
	void Viewer::continueDMO() {
		if (m_dmoState == DMOExecutionState::PAUSED) {
			m_dmoState = DMOExecutionState::RUNNING;
		}
	}
	void Viewer::stepDMO() {
		if (m_dmoState == DMOExecutionState::PAUSED) {
			m_dmoState = DMOExecutionState::STEP;
		}
	}

	void Viewer::printQuality() {
		if (m_dmo != nullptr)
			m_dmo->displayQualityGPU();
	}

	void Viewer::makeSurfaceEstimation() {
		m_drawableSurfaceEstimation = std::make_unique<DrawableSurfaceEstimation>(glm::vec4(0, 1, 1, 1), 16, 16);
		m_drawableSurfaceEstimation->init();
		//m_drawableSurfaceEstimation->calculateIbo(); // broken
	}

	void Viewer::dmoWorkerFunc() {
		while (m_activeWorker) {

			if (m_dmoState != DMOExecutionState::NONE) {
				if (m_dmoState == DMOExecutionState::RUNNING || m_dmoState == DMOExecutionState::STEP) {
					m_inIteration = true;
					Timer timer;
					m_dmo->doIteration();
					cout << "Iteration took " << timer.timeInSeconds() << "s" << endl;
					m_inIteration = false;

					m_dataAvailable = true;
					while (!m_copyDone);
					m_copyDone = false;
				}
			}
		}
	}




}
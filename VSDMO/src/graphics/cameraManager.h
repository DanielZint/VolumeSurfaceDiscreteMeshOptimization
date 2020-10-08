// cameraManager.h
#pragma once

#include "camera.h"
#include <memory>
#include <string>
#include <map>

namespace Graphics{


	// ##################################################################### //
	// ### CameraManager ################################################### //
	// ##################################################################### //
	// ## Singleton class to gain easy access to registered cameras.      ## //

	class CameraManager	{
	public:
		~CameraManager();

		static CameraManager& getInstance();

		void registerCamera(std::shared_ptr<Camera> camera, const std::string& name);
		std::shared_ptr<Camera> getCameraByName(const std::string& name) const;

		void setCameraActive(const std::string& name);
		std::shared_ptr<Camera> getActiveCamera() const;

		inline std::map<std::string, std::shared_ptr<Camera>>::iterator camerasBegin() { return m_cameraMap.begin(); }
		inline std::map<std::string, std::shared_ptr<Camera>>::iterator camerasEnd() { return m_cameraMap.end(); }
	protected:
		CameraManager();

	protected:
		std::string m_activeCamera;
		std::map<std::string, std::shared_ptr<Camera>> m_cameraMap;

	};

}

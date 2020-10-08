// cameraManager.cpp
#include "cameraManager.h"

#include <iostream>

namespace Graphics {

	CameraManager& CameraManager::getInstance() {
		static CameraManager instance;
		return instance;
	}

	CameraManager::CameraManager()
		: m_activeCamera()
		, m_cameraMap()
	{
		// do nothing
	}

	CameraManager::~CameraManager() {
		// do nothing
	}

	void CameraManager::registerCamera(std::shared_ptr<Camera> camera, const std::string& name) {
		m_cameraMap[name] = camera;
		m_activeCamera = name;
	}

	std::shared_ptr<Camera> CameraManager::getCameraByName(const std::string& name) const {
		std::shared_ptr<Camera> result = nullptr;
		try {
			result = m_cameraMap.at(name);
		}
		catch (const std::out_of_range& oor) {
			(void)oor;
			std::cerr << "There is no Camera registered with name " << name << std::endl;
			result = nullptr;
		}

		return result;
	}

	std::shared_ptr<Camera> CameraManager::getActiveCamera() const {
		return getCameraByName(m_activeCamera);
	}

	void CameraManager::setCameraActive(const std::string& name) {
		m_activeCamera = name;
	}

}
// camera.h
#pragma once

#include <SDL2/SDL.h>
#include <iostream>

#include "GlmConfig.h"

namespace Graphics {

	// Abstract superclass
	class Camera {
	public:
		Camera();
		virtual ~Camera();

		virtual void reset();
		virtual void update(float timePassed) = 0;
		virtual void handleEvent(SDL_Event event) = 0;
		virtual void toggleLock() { m_isLocked = !m_isLocked; }

		// --- camera movement ----------------------------------------------------
		virtual void forcePosition(const glm::vec3& position) = 0;
		virtual void translateBy(const glm::vec3& translate) = 0;
		virtual void forceOrientation(const glm::quat& orientation) = 0;

		/** zooms by factor. Behaviour is undefined for negative factors.
		 *  if 0 < factor < 1, the camera will zoom in. For a factor > 1 it will zoom out.
		 */
		virtual void zoom(float factor) = 0;
		virtual void forceZoom(float factor) = 0;

		void speedUp(float factor);

		// --- const functions ----------------------------------------------------
		const glm::mat4& getModelMatrix() const { return m_modelMatrix; }
		const glm::mat4& getViewMatrix() const { return m_viewMatrix; }
		const glm::mat4& getProjectionMatrix() const { return m_projectionMatrix; }
		bool isLocked() const { return m_isLocked; }
		float getZoom() const { return m_zoom; }
		float getSpeed() const { return m_speed; }
		const glm::vec3& getPosition() const { return m_position; }
		virtual const glm::quat& getOrientation() const = 0;
		const glm::vec3 getViewUp() const { return glm::vec3(m_modelMatrix[1]); }
		const glm::vec3 getViewDirection() const { return glm::vec3(m_modelMatrix[2]); }

	protected:
		glm::mat4 m_modelMatrix;
		glm::mat4 m_viewMatrix;
		glm::mat4 m_projectionMatrix;

		glm::vec3 m_position;

		glm::vec2 m_prevMousePosition;

		bool m_isLocked;
		float m_zoom;
		float m_speed;
	};

	// implements a 3d camera.
	class Camera3D : public Camera {
	public:
		Camera3D();
		~Camera3D();

		// --- Camera movement overrides ------------------------------------------
		virtual void forcePosition(const glm::vec3& position) override;
		virtual void translateBy(const glm::vec3& translate) override;
		virtual void forceOrientation(const glm::quat& orientation) override;
		virtual void zoom(float factor) override { std::cout << "TODO: Camera3D Zoom" << std::endl; }
		virtual void forceZoom(float factor) override { std::cout << "TODO: Camera3D forceZoom" << std::endl; }
		virtual const glm::quat& getOrientation() const override { return m_orientation; }

		// --- Camera3D specific functions ----------------------------------------
		void lookAt(const glm::vec3& eye, const glm::vec3& center, const glm::vec3& up);

		/*
		*	Locks the camera view to the position pivot.
		*	That means, it will only rotate around the pivot point if it is supposed to rotate.
		*/
		void setViewLockOn(const glm::vec3& pivot);
		void setViewLockOn(const glm::vec3& pivot, const float dist);
		void rotateAroundPivot(glm::vec3 xyz);

		void tiltView(float angle);

		// --- Camera overrides ---------------------------------------------------
		virtual void toggleLock() override;
		virtual void reset() override;
		virtual void update(float timePassed) override;
		virtual void handleEvent(SDL_Event event) override;

	protected:
		void updateViewMatrix();
		void updateFromSDLKeyboard(float dt);
		void updateFromSDLMouse(float dt);

		glm::quat m_orientation;
		glm::vec3 m_pivotPoint;
		float m_fovY;
	};

}
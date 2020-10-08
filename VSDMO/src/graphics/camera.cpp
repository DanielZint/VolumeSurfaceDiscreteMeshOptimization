// camera.cpp
#include "camera.h"

#include <algorithm>
#include <iostream>
#include "GlmConfig.h"
#include "imgui/imgui.h"

namespace Graphics {

	//#############################################################################
	//################################ Camera #####################################
	//#############################################################################

	Camera::Camera()
	{
		reset();
	}

	Camera::~Camera() {

	}

	void Camera::reset() {
		m_modelMatrix = glm::mat4(1.f);	//identity matrix
		m_viewMatrix = glm::mat4(1.f);
		m_projectionMatrix = glm::mat4(1.f);
		m_position = glm::vec3(0.f);
		m_prevMousePosition = glm::vec2(0.f);
		m_isLocked = false;
		m_zoom = 1.f;
		m_speed = 0.8f;
	}

	void Camera::speedUp(float factor) {
		m_speed *= factor;
	}

	//#############################################################################
	//############################### Camera3D ####################################
	//#############################################################################

	Camera3D::Camera3D()
		: Camera()
	{
		reset();
	}

	Camera3D::~Camera3D() {

	}

	void Camera3D::reset() {
		Camera::reset();
		m_orientation = glm::quat();
		m_position = glm::vec3(0.0f, 0.0f, 4.0f);
		m_projectionMatrix = glm::perspective(glm::radians(70.0f), 16.0f / 9.0f, 0.01f, 1000.0f);
		updateViewMatrix();

		m_isLocked = false;
		m_pivotPoint = glm::vec3(0.f, 0.f, 0.f);
	}

	void Camera3D::lookAt(const glm::vec3& eye, const glm::vec3& center, const glm::vec3& up)
	{
		m_viewMatrix = glm::lookAt(eye, center, up);
		m_modelMatrix = glm::inverse(m_viewMatrix);

		glm::mat3 rot = glm::mat3(m_modelMatrix);
		m_orientation = normalize(glm::quat_cast(rot));

		m_position = eye;
	}

	void Camera3D::setViewLockOn(const glm::vec3& pivot) {
		m_pivotPoint = pivot;
		lookAt(m_position, m_pivotPoint, getViewUp());
	}

	void Camera3D::setViewLockOn(const glm::vec3& pivot, const float dist) {
		m_pivotPoint = pivot;
		glm::vec3 viewVec = m_position - pivot;
		//float len = viewVec.length();
		m_position = pivot + dist * normalize(viewVec);
		m_speed = dist * 0.25f;
		lookAt(m_position, m_pivotPoint, getViewUp());
	}

	void Camera3D::forcePosition(const glm::vec3& position) {
		m_position = position;
		updateViewMatrix();
	}

	void Camera3D::forceOrientation(const glm::quat& orientation) {
		m_orientation = orientation;
		updateViewMatrix();
	}

	void Camera3D::toggleLock() {
		m_isLocked = !m_isLocked;
		setViewLockOn(m_pivotPoint);
	}

	void Camera3D::translateBy(const glm::vec3& translate)
	{
		if (m_isLocked) {
			//do not translate, instead just rotate around the pivotPoint
			rotateAroundPivot(translate);
		}
		else {
			auto d2 = m_orientation * glm::vec3(-translate.x, -translate.y, -translate.z);
			m_position += d2;
		}
	}

	void Camera3D::rotateAroundPivot(glm::vec3 xyz) {
		glm::vec3 direction = m_orientation * (-xyz);
		glm::vec3 toPivotPoint = m_pivotPoint - m_position;
		float distance = glm::length(toPivotPoint) + xyz.z;
		direction = direction * distance * .2f;
		glm::vec3 newOrientation = m_pivotPoint - (m_position + direction);
		newOrientation = glm::normalize(newOrientation);

		m_position = m_pivotPoint - newOrientation * distance;

		lookAt(m_position, m_pivotPoint, glm::normalize(glm::cross(m_orientation * glm::vec3(1.f, 0.f, 0.f), toPivotPoint)));
	}

	void Camera3D::tiltView(float angle) {
		m_orientation = glm::rotate(m_orientation, angle, glm::vec3(0,0,1));
	}

	void Camera3D::update(float dt)
	{
		updateFromSDLKeyboard(dt);
		updateFromSDLMouse(dt);
		updateViewMatrix();
	}

	void Camera3D::handleEvent(SDL_Event event) {
		if (event.type == SDL_MOUSEWHEEL && !ImGui::GetIO().WantCaptureMouse) {
			//zooms on mousewheel
			glm::vec3 toTarget = m_pivotPoint - m_position;
			//m_speed = std::min(1.f, glm::length(toTarget) / 4.f);
			float speed = std::max(0.001f, glm::length(toTarget) / 8.f);
			//float speed = m_speed * 1.f;
			if (m_isLocked) {
				auto d2 = m_orientation * glm::vec3(0,0, event.wheel.y * speed);
				m_position += d2;
				//translateBy(glm::vec3(0.f, 0.f, event.wheel.y * speed));
			}
			else {
				translateBy(glm::vec3(0.f, 0.f, -event.wheel.y * speed));
			}
		}
		if (event.type == SDL_KEYDOWN) {
			if (event.key.keysym.scancode == SDL_SCANCODE_LSHIFT) {
				m_isLocked = !m_isLocked;
			}
		}
	}


	void Camera3D::updateViewMatrix()
	{
		glm::mat4 translation = glm::translate(glm::mat4(1), glm::vec3(m_position));
		glm::mat4 rotation = glm::mat4_cast(m_orientation);
		m_modelMatrix = translation * rotation;
		m_viewMatrix = glm::inverse(m_modelMatrix);
	}

	void Camera3D::updateFromSDLKeyboard(float dt)
	{
		const Uint8 *keyBoardState = SDL_GetKeyboardState(NULL);
		float translateDelta = m_speed * dt * 10.0f;
		float rotateDelta = 0.1f;

		if (keyBoardState[SDL_SCANCODE_R]) { reset(); m_speed = 1.0f; };
		if (keyBoardState[SDL_SCANCODE_W]) {
			if (m_isLocked) rotateAroundPivot(glm::vec3(0.f, -rotateDelta, 0.f));
			else translateBy(glm::vec3(0.0f, 0.0f, +translateDelta));
		}
		if (keyBoardState[SDL_SCANCODE_S]) {
			if (m_isLocked) rotateAroundPivot(glm::vec3(0.f, +rotateDelta, 0.f));
			else translateBy(glm::vec3(0.0f, 0.0f, -translateDelta));
		}
		if (keyBoardState[SDL_SCANCODE_A] && !keyBoardState[SDL_SCANCODE_LCTRL]) {
			if (m_isLocked) rotateAroundPivot(glm::vec3(+rotateDelta, 0.f, 0.f));
			else translateBy(glm::vec3(+translateDelta, 0.0f, 0.0f));
		}
		else if (keyBoardState[SDL_SCANCODE_A]) {
			// tilt view
			tiltView(translateDelta * 0.3f);
		}
		if (keyBoardState[SDL_SCANCODE_D] && !keyBoardState[SDL_SCANCODE_LCTRL]) {
			if (m_isLocked) rotateAroundPivot(glm::vec3(-rotateDelta, 0.f, 0.f));
			else translateBy(glm::vec3(-translateDelta, 0.0f, 0.0f));
		}
		else if (keyBoardState[SDL_SCANCODE_D]) {
			// tilt view
			tiltView(-translateDelta * 0.3f);
		}
		if (keyBoardState[SDL_SCANCODE_Q]) {
			if (m_isLocked) translateBy(glm::vec3(0.f, 0.f, +translateDelta));
			else translateBy(glm::vec3(0.0f, +translateDelta, 0.0f));
		}
		if (keyBoardState[SDL_SCANCODE_E]) {
			if (m_isLocked) translateBy(glm::vec3(0.f, 0.f, -translateDelta));
			else translateBy(glm::vec3(0.0f, -translateDelta, 0.0f));
		}
		if (keyBoardState[SDL_SCANCODE_KP_PLUS]) {
			speedUp(1.0f / 0.9f);
		}
		if (keyBoardState[SDL_SCANCODE_KP_MINUS]) {
			speedUp(0.9f);
		}
	}

	void Camera3D::updateFromSDLMouse(float dt)
	{
		int mouseX, mouseY;
		Uint32 buttons = SDL_GetMouseState(&mouseX, &mouseY);

		glm::vec2 newMousePos = glm::vec2(mouseX, mouseY);
		glm::vec2 relMovement = m_prevMousePosition - newMousePos;

		if (SDL_BUTTON(SDL_BUTTON_LEFT) & buttons)
		{
			float thetaX = relMovement.x / 60.0f;
			float thetaY = relMovement.y / 60.0f;

			if (m_isLocked) {
				rotateAroundPivot(glm::vec3(-thetaX, thetaY, 0));
			}
			else {
				thetaX /= 6.f;
				thetaY /= 6.f;
				m_orientation = glm::rotate(m_orientation, thetaX, glm::vec3(0, 1, 0));
				m_orientation = glm::rotate(m_orientation, thetaY, glm::vec3(1, 0, 0));
				m_orientation = glm::normalize(m_orientation);
			}
		}
		m_prevMousePosition = newMousePos;
	}

}
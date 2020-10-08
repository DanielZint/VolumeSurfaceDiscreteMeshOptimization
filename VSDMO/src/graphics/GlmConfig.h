// glmConfig.h
#pragma once

#ifndef GLM_ENABLE_EXPERIMENTAL
#define GLM_ENABLE_EXPERIMENTAL
#endif

#include <glm/glm.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtx/transform.hpp>

#include <iostream>

namespace glm {
	std::ostream& operator<<(std::ostream& os, const vec4& vector);

	std::ostream& operator<<(std::ostream& os, const vec3& vector);

	std::ostream& operator<<(std::ostream& os, const vec2& vector);
}

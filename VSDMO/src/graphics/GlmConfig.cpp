// GlmConfig.cpp
#include "GlmConfig.h"


namespace glm {
	std::ostream& operator<<(std::ostream& os, const vec4& vector) {
		return os << "[" << vector.x << ", " << vector.y << ", " << vector.z << ", " << vector.w << "]";
	}

	std::ostream& operator<<(std::ostream& os, const vec3& vector) {
		return os << "[" << vector.x << ", " << vector.y << ", " << vector.z << "]";
	}

	std::ostream& operator<<(std::ostream& os, const vec2& vector) {
		return os << "[" << vector.x << ", " << vector.y << "]";
	}
}
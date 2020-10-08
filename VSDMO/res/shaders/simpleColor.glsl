---vertex
layout (location = 0) in vec3 in_position;
layout (location = 1) in vec4 in_color;

layout (location = 0) uniform mat4 projView;
layout (location = 1) uniform mat4 model;

out vec4 position_WorldCoords;

void main() {
    position_WorldCoords = model * vec4(in_position.xyz, 1);
    gl_Position = projView * position_WorldCoords;
}

---fragment
in vec4 position_WorldCoords;

layout (location = 2) uniform float depthOffset;
layout (location = 3) uniform vec4 color;

layout (location = 0) out vec4 out_color;

void main() {
	out_color = color;
	gl_FragDepth = gl_FragCoord.z - depthOffset;
}
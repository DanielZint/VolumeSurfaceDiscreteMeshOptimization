---vertex
layout (location = 0) in vec3 in_position;

layout (location = 0) uniform mat4 proj;
layout (location = 1) uniform mat4 view;
layout (location = 2) uniform mat4 model;
layout (location = 3) uniform vec4 color;
layout (location = 4) uniform float depthOffset;
layout (location = 14) uniform int startOffset;
layout (location = 15) uniform int endOffset;

//out vec4 position_WorldCoords;
out vec4 color_frag;

void main() {
    vec4 position_WorldCoords = model * vec4(in_position.xyz, 1);
	vec4 tmpSpace = position_WorldCoords;
	tmpSpace = view * tmpSpace;
	tmpSpace.xyz *= (1.f - depthOffset);
    color_frag = vec4(1, 1, 1, 0);
	if (gl_VertexID >= startOffset && gl_VertexID < endOffset) {
		color_frag = color; //vec4(1, 0.5, 0.5, 1);
	}
	//if (((gl_VertexID%8)+(gl_VertexID/8)) % 2 == 0) color_frag = vec4(1,0,0,1);//new
    gl_Position = proj * tmpSpace;
}

---fragment
//in vec4 position_WorldCoords;
in vec4 color_frag;

//layout (location = 12) uniform float depthOffset;

layout (location = 0) out vec4 out_color;

void main() {
	if (color_frag.a <= 0.f) discard;
	out_color = color_frag;
}

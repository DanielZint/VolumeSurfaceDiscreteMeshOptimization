---vertex
layout (location = 0) in vec3 in_position;
layout (location = 1) in vec3 in_normal;

layout (location = 0) uniform mat4 model;

out vec4 geo_normal;

void main() {
	geo_normal = model * vec4(in_normal.xyz, 0);
	gl_Position = model * vec4(in_position, 1);
}

---geometry
layout (points) in;
layout (line_strip, max_vertices=2) out;

layout (location = 1) uniform mat4 proj;
layout (location = 2) uniform mat4 view;
layout (location = 3) uniform float lineLength;

in vec4 geo_normal[];

void main() {
	gl_Position = view * (gl_in[0].gl_Position);
	//gl_Position.xyz *= 0.97;
	gl_Position = proj * gl_Position;
	EmitVertex();

	gl_Position = view * (gl_in[0].gl_Position + vec4(normalize(geo_normal[0].xyz) * lineLength, 0));
	//gl_Position.xyz *= 0.97;
	gl_Position = proj * gl_Position;
	EmitVertex();

	EndPrimitive();
}

---fragment
layout (location = 0) out vec4 out_color;

layout (location = 4) uniform vec4 color;

void main() {
	out_color = color;
}

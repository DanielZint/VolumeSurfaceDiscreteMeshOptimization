---vertex
layout (location = 0) in vec3 in_position;

layout (location = 0) uniform mat4 model;


void main() {
	gl_Position = model * vec4(in_position, 1);
}

---geometry
layout (triangles) in;
layout (triangle_strip, max_vertices=3) out;

layout (location = 1) uniform mat4 proj;
layout (location = 2) uniform mat4 view;

out vec4 position_WorldCoords;
out vec3 normal_WorldCoords;

void main() {
	normal_WorldCoords = normalize(cross(gl_in[1].gl_Position.xyz - gl_in[0].gl_Position.xyz, gl_in[2].gl_Position.xyz - gl_in[0].gl_Position.xyz));
	
	position_WorldCoords = gl_in[0].gl_Position;
	gl_Position = proj * view * (gl_in[0].gl_Position);
	EmitVertex();

	position_WorldCoords = gl_in[0].gl_Position;
	gl_Position = proj * view * (gl_in[1].gl_Position);
	EmitVertex();
	
	position_WorldCoords = gl_in[0].gl_Position;
	gl_Position = proj * view * (gl_in[2].gl_Position);
	EmitVertex();

	EndPrimitive();
}

---fragment
in vec4 position_WorldCoords;
in vec3 normal_WorldCoords;

layout (location = 0) out vec4 out_color;

layout (location = 4) uniform vec4 color;
layout (location = 6) uniform float ambient; 
layout (location = 7) uniform float diffuse; 
layout (location = 8) uniform float specular; 
layout (location = 9) uniform float shiny; 
layout (location = 10) uniform vec3 lightDirection; 
layout (location = 11) uniform vec4 lightColor; 
layout (location = 12) uniform vec3 cameraPosition;
layout (location = 13) uniform int inverseColor;

void main() {
	vec3 n = normalize(normal_WorldCoords);
	vec3 l = normalize(-lightDirection);
	vec3 v = normalize(cameraPosition - position_WorldCoords.xyz);
	vec3 r = 2.*dot(n, l)*n-l;
	vec3 ambient_color = ambient * color.rgb;
	vec3 diffuse_color = diffuse * color.rgb * clamp(dot(n,l), 0, 1);
	vec3 specular_color = specular * lightColor.rgb * pow(clamp(dot(v,r), 0, 1), shiny);
	out_color = vec4(ambient_color + diffuse_color + specular_color, color.a);
	if (inverseColor != 0) {
		out_color = vec4(out_color.rgb - vec3(0.6), 1.f);
	}
}

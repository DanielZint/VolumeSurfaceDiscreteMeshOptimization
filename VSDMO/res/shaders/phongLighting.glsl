---vertex
layout (location = 0) in vec3 in_position;
layout (location = 1) in vec4 in_color;

layout (location = 0) uniform mat4 projView;
layout (location = 1) uniform mat4 model;

out vec4 position_WorldCoords;
out vec4 color_frag;

void main() {
    position_WorldCoords = model * vec4(in_position.xyz, 1);
    color_frag = in_color;
    gl_Position = projView * position_WorldCoords;
}

---fragment
in vec4 position_WorldCoords;
in vec4 color_frag;

layout (location = 2) uniform float ambient; 
layout (location = 3) uniform float diffuse; 
layout (location = 4) uniform float specular; 
layout (location = 5) uniform float shiny; 
layout (location = 6) uniform vec3 lightDirection; 
layout (location = 7) uniform vec4 lightColor; 
layout (location = 8) uniform vec3 cameraPosition;
layout (location = 9) uniform float depthOffset;

layout (location = 0) out vec4 out_color;

void main() {
	vec3 n = vec3(0,0,1);
	vec3 l = normalize(-lightDirection);
	vec3 v = normalize(cameraPosition - position_WorldCoords.xyz);
	vec3 r = 2.*dot(n, l)*n-l;
	vec4 ambient_color = ambient * color_frag;
	vec4 diffuse_color = diffuse * color_frag * clamp(dot(n,l), 0, 1);
	vec4 specular_color = specular * lightColor * pow(clamp(dot(v,r), 0, 1), shiny);
	out_color = ambient_color + diffuse_color + specular_color;
	gl_FragDepth = .5f; //gl_FragCoord.z + depthOffset;
}

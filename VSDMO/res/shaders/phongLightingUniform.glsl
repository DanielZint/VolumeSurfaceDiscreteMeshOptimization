---vertex
layout (location = 0) in vec3 in_position;

layout (location = 0) uniform mat4 proj;
layout (location = 1) uniform mat4 view;
layout (location = 2) uniform mat4 model;
layout (location = 3) uniform vec4 color;
layout (location = 4) uniform float depthOffset;

out vec4 position_WorldCoords;
out vec4 color_frag;

void main() {
    position_WorldCoords = model * vec4(in_position.xyz, 1);
	vec4 tmpSpace = position_WorldCoords;
	tmpSpace = view * tmpSpace;
	tmpSpace.xyz *= (1.f - depthOffset);
    color_frag = color;
	//if (((gl_VertexID%8)+(gl_VertexID/8)) % 2 == 0) color_frag = vec4(1,0,0,1);//new
    gl_Position = proj * tmpSpace;
}

---fragment
in vec4 position_WorldCoords;
in vec4 color_frag;

layout (location = 5) uniform float ambient; 
layout (location = 6) uniform float diffuse; 
layout (location = 7) uniform float specular; 
layout (location = 8) uniform float shiny; 
layout (location = 9) uniform vec3 lightDirection; 
layout (location = 10) uniform vec4 lightColor; 
layout (location = 11) uniform vec3 cameraPosition;
layout (location = 12) uniform int inverseColor;
//layout (location = 12) uniform float depthOffset;

layout (location = 0) out vec4 out_color;

void main() {
	vec3 n = vec3(0,0,1);
	vec3 l = normalize(-lightDirection);
	vec3 v = normalize(cameraPosition - position_WorldCoords.xyz);
	vec3 r = 2.*dot(n, l)*n-l;
	vec3 ambient_color = ambient * color_frag.rgb;
	vec3 diffuse_color = diffuse * color_frag.rgb * clamp(dot(n,l), 0, 1);
	vec3 specular_color = specular * lightColor.rgb * pow(clamp(dot(v,r), 0, 1), shiny);
	out_color = vec4(ambient_color + diffuse_color + specular_color, color_frag.a);
	if (inverseColor != 0) {
		out_color = vec4(out_color.rgb - vec3(0.6), 1.f);
	}
}

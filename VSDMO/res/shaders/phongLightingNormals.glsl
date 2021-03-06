---vertex
layout (location = 0) in vec3 in_position;
layout (location = 1) in vec3 in_normal;

layout (location = 0) uniform mat4 proj;
layout (location = 1) uniform mat4 view;
layout (location = 2) uniform mat4 model;
layout (location = 3) uniform vec4 color;
layout (location = 4) uniform float depthOffset;
layout (location = 5) uniform float lineLength;

out vec4 position_WorldCoords;
out vec3 normal_WorldCoords;
out vec3 normal_ViewCoords;
out vec4 color_frag;

void main() {	
    position_WorldCoords = model * vec4(in_position.xyz, 1);
	normal_WorldCoords = (model * vec4(in_normal, 0)).xyz;
	normal_ViewCoords = (view * model * vec4(in_normal, 0)).xyz;
	vec4 tmpSpace = position_WorldCoords;
	tmpSpace = view * tmpSpace;
	tmpSpace.xyz *= (1.f - depthOffset);
    color_frag = color;
    gl_Position = proj * tmpSpace;
}

---fragment
in vec4 position_WorldCoords;
in vec3 normal_WorldCoords;
in vec3 normal_ViewCoords;
in vec4 color_frag;

layout (location = 1) uniform mat4 view;
layout (location = 6) uniform float ambient; 
layout (location = 7) uniform float diffuse; 
layout (location = 8) uniform float specular; 
layout (location = 9) uniform float shiny; 
layout (location = 10) uniform vec3 lightDirection; 
layout (location = 11) uniform vec4 lightColor; 
layout (location = 12) uniform vec3 cameraPosition;
layout (location = 13) uniform int inverseColor;
layout (location = 14) uniform int showElementQuality;
layout (location = 15) uniform samplerBuffer qualities;

layout (location = 0) out vec4 out_color;

void main() {
	if (showElementQuality != 0) {
		
		float q = texelFetch(qualities, gl_PrimitiveID).r;
		out_color.rgb = color_frag.rgb * q;
		out_color.a = 1.f;
		//out_color = vec4(f, 0, 0, 1);
		
		return;
	}
	//fixed light pos
	//vec3 n = normalize(normal_WorldCoords);
	//vec3 l = normalize(-lightDirection);
	
	//lit from camera
	vec3 n = normalize(normal_ViewCoords);
	vec3 l = normalize(vec3(0,0,1));
	
	vec3 v = normalize(cameraPosition - position_WorldCoords.xyz);
	vec3 r = 2.*dot(n, l)*n-l;
	vec3 ambient_color = ambient * color_frag.rgb;
	vec3 diffuse_color = diffuse * color_frag.rgb * clamp(dot(n,l), 0, 1);
	vec3 specular_color = 0.3f * lightColor.rgb * pow(clamp(dot(v,r), 0, 1), shiny);
	out_color = vec4(ambient_color + diffuse_color + specular_color, color_frag.a);
	if (inverseColor != 0) {
		out_color = vec4(out_color.rgb - vec3(0.6), 1.f);
	}
}

---vertex
layout (location = 0) in vec3 in_position;

layout (location = 0) uniform mat4 model;


void main() {
	gl_Position = model * vec4(in_position, 1);
}

---geometry
layout (lines_adjacency) in;
layout (triangle_strip, max_vertices=12) out;

layout (location = 1) uniform mat4 proj;
layout (location = 2) uniform mat4 view;
layout (location = 3) uniform float slicingZ;
layout (location = 4) uniform float sizeFactor;

//out vec4 position_WorldCoords;
out vec3 normal_WorldCoords;

void main() {
	float maxz = max(max(gl_in[0].gl_Position.z, gl_in[1].gl_Position.z), max(gl_in[2].gl_Position.z, gl_in[3].gl_Position.z));
	float minz = min(min(gl_in[0].gl_Position.z, gl_in[1].gl_Position.z), min(gl_in[2].gl_Position.z, gl_in[3].gl_Position.z));
	if (minz > slicingZ) {
		EndPrimitive();
		return;
	}
	
	vec4 pos[4] =
	{gl_in[0].gl_Position
	,gl_in[1].gl_Position
	,gl_in[2].gl_Position
	,gl_in[3].gl_Position
	};
	
	vec3 middle = 0.25f * (pos[0].xyz + pos[1].xyz + pos[2].xyz + pos[3].xyz);
	for (int i = 0; i < 4; ++i) {
		pos[i].xyz += sizeFactor * (middle.xyz - pos[i].xyz);
	}
	
	vec4 clipPos[4] =
	{proj * view * (pos[0])
	,proj * view * (pos[1])
	,proj * view * (pos[2])
	,proj * view * (pos[3])
	};
	
	vec3 edge01 = pos[1].xyz - pos[0].xyz;
	vec3 edge02 = pos[2].xyz - pos[0].xyz;
	vec3 edge03 = pos[3].xyz - pos[0].xyz;
	vec3 edge12 = pos[2].xyz - pos[1].xyz;
	vec3 edge13 = pos[3].xyz - pos[1].xyz;
	vec3 edge23 = pos[3].xyz - pos[2].xyz;
	
	vec3 faceNormals[4] =
	{
	normalize(cross(edge12, edge13))//1,2,3
	,normalize(cross(edge03, edge02))//0,3,2
	,normalize(cross(edge01, edge03))//0,1,3
	,normalize(cross(edge02, edge01))//0,2,1
	};
	
	
	normal_WorldCoords = faceNormals[0];
	gl_Position = clipPos[1];
	EmitVertex();
	gl_Position = clipPos[2];
	EmitVertex();
	gl_Position = clipPos[3];
	EmitVertex();
	EndPrimitive();
	
	normal_WorldCoords = faceNormals[1];
	gl_Position = clipPos[0];
	EmitVertex();
	gl_Position = clipPos[3];
	EmitVertex();
	gl_Position = clipPos[2];
	EmitVertex();
	EndPrimitive();
	
	normal_WorldCoords = faceNormals[2];
	gl_Position = clipPos[0];
	EmitVertex();
	gl_Position = clipPos[1];
	EmitVertex();
	gl_Position = clipPos[3];
	EmitVertex();
	EndPrimitive();
	
	normal_WorldCoords = faceNormals[3];
	gl_Position = clipPos[0];
	EmitVertex();
	gl_Position = clipPos[2];
	EmitVertex();
	gl_Position = clipPos[1];
	EmitVertex();
	EndPrimitive();
	
}

---fragment
//in vec4 position_WorldCoords;
in vec3 normal_WorldCoords;

layout (location = 0) out vec4 out_color;

layout (location = 5) uniform vec4 color;
layout (location = 6) uniform float ambient; 
layout (location = 7) uniform float diffuse; 
layout (location = 8) uniform float specular; 
layout (location = 9) uniform float shiny; 
layout (location = 10) uniform vec3 lightDirection; 
layout (location = 11) uniform vec4 lightColor; 
layout (location = 12) uniform vec3 cameraPosition;
layout (location = 13) uniform int inverseColor;

void main() {
	
	//out_color.rgb = vec3(1,1,1) * dot(normal_WorldCoords, lightDirection);
	out_color.rgb = normal_WorldCoords * vec3(0.5) + vec3(0.5);
	out_color.a = 1.f;
}

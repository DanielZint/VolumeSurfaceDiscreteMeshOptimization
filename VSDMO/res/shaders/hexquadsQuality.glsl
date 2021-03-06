---vertex
layout (location = 0) in vec3 in_position;
//layout (location = 1) in vec3 in_normal;

layout (location = 0) uniform mat4 proj;
layout (location = 1) uniform mat4 view;
layout (location = 2) uniform mat4 model;
layout (location = 3) uniform vec4 color;
layout (location = 4) uniform float depthOffset;
layout (location = 5) uniform float lineLength;

out vec4 position_WorldCoords;
out vec4 color_vg;

void main() {	
    position_WorldCoords = model * vec4(in_position.xyz, 1);
	//normal_WorldCoords_vg = (model * vec4(in_normal, 0)).xyz;
	//normal_ViewCoords_vg = (view * model * vec4(in_normal, 0)).xyz;
	vec4 tmpSpace = position_WorldCoords;
	tmpSpace = view * tmpSpace;
	tmpSpace.xyz *= (1.f - depthOffset);
    color_vg = color;
    //gl_Position = proj * tmpSpace;
	gl_Position = position_WorldCoords;
}

---geometry
layout (lines_adjacency) in;
layout (triangle_strip, max_vertices=4) out;

layout (location = 0) uniform mat4 proj;
layout (location = 1) uniform mat4 view;
layout (location = 6) uniform float slicingZ;
layout (location = 7) uniform float sizeFactor;

//out vec4 position_WorldCoords;
in vec4 color_vg[];
out vec4 color_frag;

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
	vec3 edge03 = pos[3].xyz - pos[0].xyz;
	vec3 edge21 = pos[1].xyz - pos[2].xyz;
	vec3 edge23 = pos[3].xyz - pos[2].xyz;
	
	gl_PrimitiveID = gl_PrimitiveIDIn;
	
	color_frag = color_vg[0];
	gl_Position = clipPos[0];
	EmitVertex();
	
	color_frag = color_vg[3];
	gl_Position = clipPos[3];
	EmitVertex();
	
	color_frag = color_vg[1];
	gl_Position = clipPos[1];
	EmitVertex();
	
	color_frag = color_vg[2];
	gl_Position = clipPos[2];
	EmitVertex();

	EndPrimitive();
}

---fragment
in vec4 position_WorldCoords;
in vec4 color_frag;

layout (location = 1) uniform mat4 view;
layout (location = 8) uniform float ambient; 
layout (location = 9) uniform float diffuse; 
layout (location = 10) uniform float specular; 
layout (location = 11) uniform float shiny; 
layout (location = 12) uniform vec3 lightDirection; 
layout (location = 13) uniform vec4 lightColor; 
layout (location = 14) uniform vec3 cameraPosition;
layout (location = 15) uniform int inverseColor;
layout (location = 16) uniform samplerBuffer qualities;

layout (location = 0) out vec4 out_color;

void main() {
	float q = texelFetch(qualities, gl_PrimitiveID / 6).r;
	//q = log2(q+1.0);
	out_color.rgb = color_frag.rgb * q;
	out_color.a = 1.f;
	//out_color = vec4(f, 0, 0, 1);
}

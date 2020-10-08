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
out vec4 color_vg;

void main() {	
    position_WorldCoords = model * vec4(in_position.xyz, 1);
	normal_WorldCoords = (model * vec4(in_normal, 0)).xyz;
	vec4 tmpSpace = position_WorldCoords;
	tmpSpace = view * tmpSpace;
	tmpSpace.xyz *= (1.f - depthOffset);
    color_vg = color;
    gl_Position = proj * tmpSpace;
}

---geometry
layout (lines_adjacency) in;
layout (triangle_strip, max_vertices=4) out;

//out vec4 position_WorldCoords;
in vec4 color_vg[];
out vec4 color_frag;

void main() {
	vec4 pos[4] =
	{gl_in[0].gl_Position
	,gl_in[1].gl_Position
	,gl_in[2].gl_Position
	,gl_in[3].gl_Position
	};
	
	gl_PrimitiveID = gl_PrimitiveIDIn;
	
	color_frag = color_vg[0];
	gl_Position = pos[0];
	EmitVertex();
	
	color_frag = color_vg[3];
	gl_Position = pos[3];
	EmitVertex();
	
	color_frag = color_vg[1];
	gl_Position = pos[1];
	EmitVertex();
	
	color_frag = color_vg[2];
	gl_Position = pos[2];
	EmitVertex();

	EndPrimitive();
}

---fragment
in vec4 position_WorldCoords;
in vec3 normal_WorldCoords;
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
layout (location = 14) uniform samplerBuffer qualities;

layout (location = 0) out vec4 out_color;

void main() {
	float q = texelFetch(qualities, gl_PrimitiveID).r;
	//q = log2(q+1.0);
	out_color.rgb = color_frag.rgb * q;
	out_color.a = 1.f;
	//out_color = vec4(f, 0, 0, 1);

}

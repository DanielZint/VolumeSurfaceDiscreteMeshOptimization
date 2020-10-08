#ifndef __COLORS_H__
#define __COLORS_H__

#define PI 3.14159265

vec3 rgb2hsv(vec3 color){
	vec4 K = vec4(0.0, -1.0/3.0, 2.0/3.0, -1.0);
	vec4 p = mix(vec4(color.bg, K.wz), vec4(color.gb, K.xy), step(color.b, color.g));
	vec4 q = mix(vec4(p.xyw, color.r), vec4(color.r, p.yzx), step(p.x, color.r));

	float d = q.x - min(q.w, q.y);
	float e = 1.0e-10;
	return vec3(abs(q.z + (q.w - q.y)/(6.0*d + e)), d/(q.x+e), q.x);
}

float h2rgb(float c, float t1, float t2) {
	if(c < 0.0) c += 1.0;
	else if (c > 1.0) c -= 1.0;
	float res;
	if(6.0 * c < 1.0) res = t2 + (t1 - t2) * 6.0 * c;
	else if(2.0 * c < 1.0) res = t1;
	else if(3.0 * c < 2.0) res = t2 + (t1 - t2) * ((2.0/3.0) - c) * 6.0;
	else res = t2;
	return res;
}

vec3 hsv2rgb(vec3 hsv){
	vec4 K = vec4(1.0, 2.0/3.0, 1.0/3.0, 3.0);
	vec3 p = abs(fract(hsv.xxx+K.xyz)*6.0 - K.www);
	vec3 rgb = hsv.z * mix(K.xxx, clamp(p-K.xxx, 0.0, 1.0), hsv.y);
	return rgb;
}

vec3 get_color_from_angle(float angle){
	float saturation = abs(PI*0.25f - abs(angle - (PI*0.25f))) / (PI*0.25);
	float value = 1 - saturation;	

	// color gradient:
	// 0 <= angle < 45 deg : black-blue-white
	// 45 <= angle <= 90 deg : white-red-black

	float hsAngle = 0.f;
	if(angle < PI*0.25f){
		hsAngle = PI*(4.4f/3.f);
	}

	return hsv2rgb(vec3(hsAngle, saturation, value));
}
#endif

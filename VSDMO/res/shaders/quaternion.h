#ifndef __QUATERNION_H__
#define __QUATERNION_H__

#define PI 3.14159265

vec4 create_quaternion_from_angle_and_axis(float angle, vec3 axis){
	vec4 result;
	float s = sin(angle * .5f);
	result.w = cos(angle * .5f);
	result.xyz = axis * s;
	return result;
}

vec4 multiply_quaternions(vec4 a, vec4 b){
	vec4 result = vec4(
		a.w*b.x + a.x*b.w + a.y*b.z + a.z*b.y,
		a.w*b.y + a.x*b.z + a.y*b.w + a.z*b.x,
		a.w*b.z + a.x*b.y + a.y*b.x + a.z*b.w,
		a.w*b.w + a.x*b.x + a.y*b.y + a.z*b.z);
	return result;

}

vec3 rotate_by_quaternion(vec3 point, vec4 quat){
	return point+2.f*cross(quat.xyz, cross(quat.xyz, point) + quat.w * point);
}

vec3 get_axis_from_quaternion(vec4 quat){
	float angle = 2.f * acos(quat.w);
	vec3 axis;
	if(angle < .000001){
		axis = vec3(0,1,0);
	} else {
		float brutal = sqrt(1-quat.w*quat.w);
		axis.x = quat.x / brutal;
		axis.y = quat.y / brutal;
		axis.z = quat.z / brutal;
	}
	return axis;
}

float get_angle_from_quaternion(vec4 quat){
	return 2.f * acos(quat.w);
}
#endif

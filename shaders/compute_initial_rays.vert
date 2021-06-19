#version 450 core

layout(location = 0) in vec3 pos;

layout(location = 0) out vec3 vray_dir;
layout(location = 1) flat out vec3 transformed_eye;

layout(set = 0, binding = 0, std140) uniform ViewParams
{
    mat4 proj_view;
    vec4 eye_pos;
    vec4 eye_dir;
    float near_plane;
};

layout(set = 0, binding = 2, std140) uniform VolumeParams
{
    uvec4 volume_dims;
    uvec4 padded_dims;
    vec4 volume_scale;
    uint max_bits;
    float isovalue;
    uint image_width;
};

void main(void) {
	vec3 volume_translation = vec3(0) - volume_scale.xyz * 0.5;
	gl_Position = proj_view * vec4(pos * volume_scale.xyz + volume_translation, 1);
	transformed_eye = (eye_pos.xyz - volume_translation) / volume_scale.xyz;
	vray_dir = pos - transformed_eye;
}

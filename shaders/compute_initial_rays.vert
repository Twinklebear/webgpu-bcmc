#version 450 core

layout(location = 0) in vec3 pos;

layout(location = 0) out vec3 vray_dir;
layout(location = 1) flat out vec3 transformed_eye;

layout(set = 0, binding = 0, std140) uniform ViewParams {
    mat4 proj_view;
    vec4 eye_pos;
};

layout(set = 0, binding = 2) uniform VolumeParams {
	ivec3 volume_dims;
    ivec3 padded_dims;
    uint image_width;
    vec3 volume_scale;
    float isovalue;
};

void main(void) {
	vec3 volume_translation = vec3(0) - volume_scale * 0.5;
	gl_Position = proj_view * vec4(pos * volume_scale + volume_translation, 1);
	transformed_eye = (eye_pos.xyz - volume_translation) / volume_scale;
	vray_dir = pos - transformed_eye;
}

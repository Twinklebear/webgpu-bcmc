#version 450 core

#include "util.glsl"

layout(location = 0) in vec3 vray_dir;
layout(location = 1) flat in vec3 transformed_eye;

layout(set = 0, binding = 1, std430) buffer RayInformation {
    RayInfo rays[];
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

layout(set = 0, binding = 3, std430) buffer RayBlockIDs
{
    uint block_ids[];
};

void main() {
    vec3 ray_dir = normalize(vray_dir);

    // Transform the ray into the dual grid space and intersect with the dual grid bounds
	const vec3 vol_eye = transformed_eye * volume_dims.xyz - vec3(0.5);
    const vec3 grid_ray_dir = normalize(ray_dir * volume_dims.xyz);

	vec2 t_hit = intersect_box(vol_eye, grid_ray_dir, vec3(0), volume_dims.xyz - 1);
    
	// We don't want to sample voxels behind the eye if it's
	// inside the volume, so keep the starting point at or in front
	// of the eye
	t_hit.x = max(t_hit.x, 0.0);
    
    const uint pixel = uint(gl_FragCoord.x) + image_width * uint(gl_FragCoord.y);
	if (t_hit.x < t_hit.y) {
        rays[pixel].ray_dir = grid_ray_dir;
        block_ids[pixel] = UINT_MAX;
        rays[pixel].t = t_hit.x;
    }
}

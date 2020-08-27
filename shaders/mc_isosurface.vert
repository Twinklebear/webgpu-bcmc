#version 450 core

#include "block_vertex_position.comp"

layout(location = 0) in uvec2 pos;

layout(location = 0) out vec4 world_pos;
#ifdef DRAW_FOG
layout(location = 1) out float t_enter;
#endif

layout(set = 0, binding = 0, std140) uniform ViewParams
{
    mat4 proj_view;
    vec4 eye_pos;
};

layout(set = 0, binding = 1, std140) uniform VolumeParams
{
    uvec4 volume_dims;
    uvec4 padded_dims;
    vec4 volume_scale;
    uint max_bits;
    float isovalue;
};

uvec3 block_id_to_pos(uint id)
{
    const uvec3 n_blocks = padded_dims.xyz / uvec3(4);
    return uvec3(id % n_blocks.x,
            (id / n_blocks.x) % n_blocks.y,
            id / (n_blocks.x * n_blocks.y));
}

#ifdef DRAW_FOG
vec2 intersect_box(vec3 orig, vec3 dir)
{
    const vec3 box_min = -vec3(volume_scale.xyz * 0.5);
    const vec3 box_max = vec3(volume_scale.xyz * 0.5);
    vec3 inv_dir = 1.0 / dir;
    vec3 tmin_tmp = (box_min - orig) * inv_dir;
    vec3 tmax_tmp = (box_max - orig) * inv_dir;
    vec3 tmin = min(tmin_tmp, tmax_tmp);
    vec3 tmax = max(tmin_tmp, tmax_tmp);
    float t0 = max(tmin.x, max(tmin.y, tmin.z));
    float t1 = min(tmax.x, min(tmax.y, tmax.z));
    return vec2(t0, t1);
}
#endif

void main(void)
{
    const vec3 block_pos = 4.0 * vec3(block_id_to_pos(uint(pos.y)));
    world_pos.xyz = block_pos + decompress_position(pos.x);
    world_pos.xyz = volume_scale.xyz * (world_pos.xyz / vec3(volume_dims.xyz) - 0.5);
#ifdef DRAW_FOG
    t_enter = max(intersect_box(eye_pos.xyz, normalize(world_pos.xyz - eye_pos.xyz)).x, 0);
#endif
    gl_Position = proj_view * vec4(world_pos.xyz, 1);
}

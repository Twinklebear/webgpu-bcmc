/*
// #include "util.glsl"
*/

const UINT_MAX: u32 = 0xffffffffu;
const FLT_MAX: f32 = 3.402823466e+38;

type float2 = vec2<f32>;
type float3 = vec3<f32>;
type float4 = vec4<f32>;
type uint2 = vec2<u32>;
type uint3 = vec3<u32>;
type uint4 = vec4<u32>;

struct RayInfo {
    ray_dir: float3,
    // block_id: u32,
    t: f32,
    // t_next: f32,
    // For WGSL we need to pad the struct up to 32 bytes so it matches
    // the GLSL struct alignment/padding rules we had before
    // @size(8) pad: f32
};

/*
layout(local_size_x = 8, local_size_y = 1, local_size_z = 1) in;
*/
/*
layout(set = 0, binding = 0, std140) uniform VolumeParams
{
    uvec4 volume_dims;
    uvec4 padded_dims;
    vec4 volume_scale;
    uint max_bits;
    float isovalue;
    uint image_width;
};
*/
struct VolumeParams {
  volume_dims: uint4,
  padded_dims: uint4,
  volume_scale: float4,
  max_bits: u32,
  isovalue: f32,
  image_width: u32,
}

@group(0) @binding(0) var<uniform> volume_params : VolumeParams;

/*
layout(set = 0, binding = 1, std140) uniform LOD
{
    uint LOD_threshold;
};
*/
struct LOD {
    threshold: f32,
}
@group(0) @binding(1) var<uniform> lod_threshold : LOD;

/*
layout(set = 0, binding = 2, std140) uniform ViewParams
{
    mat4 proj_view;
    vec4 eye_pos;
    vec4 eye_dir;
    float near_plane;
    uint current_pass_index;
};
*/
struct ViewParams {
  proj_view: mat4x4<f32>,
  eye_pos: float4,
  eye_dir: float4,
  near_plane : f32,
  current_pass_index: u32,
}
@group(0) @binding(2) var<uniform> view_params : ViewParams;

/*
layout(set = 0, binding = 3, std430) buffer BlockActive
{
    uint block_active[];
};
*/
// TODO: Is this valid WGSL? Try compiling with Tint
@group(0) @binding(3) var<storage, read_write> block_active : array<u32>;

/*
layout(set = 0, binding = 5, std430) buffer RayInformation
{
    RayInfo rays[];
};
*/
@group(0) @binding(4) var<storage, read_write> rays : array<RayInfo>;

/*
layout(set = 0, binding = 6, std430) buffer BlockVisible
{
    uint block_visible[];
};
*/
@group(0) @binding(5) var<storage, read_write> block_visible : array<atomic<u32>>;
@group(0) @binding(6) var<storage, read_write> block_ids : array<u32>;


//uniform layout(set = 1, binding = 0, rgba8) writeonly image2D render_target;
@group(1) @binding(0) var render_target : texture_storage_2d<rgba8unorm, write>;

fn block_id_to_pos(id: u32) -> uint3 {
    let n_blocks = volume_params.padded_dims.xyz / uint3(4u);
    return uint3(id % n_blocks.x,
            (id / n_blocks.x) % n_blocks.y,
            id / (n_blocks.x * n_blocks.y));
}

fn compute_block_id(block_pos: uint3) -> u32
{
    let n_blocks = volume_params.padded_dims.xyz / uint3(4u);
    return block_pos.x + n_blocks.x * (block_pos.y + n_blocks.y * block_pos.z);
}

@compute @workgroup_size(32, 1, 1)
fn main(@builtin(global_invocation_id) g_invocation_id : vec3<u32>) {
    if (g_invocation_id.x >= volume_params.image_width) {
        return;
    }

    let ray_index = g_invocation_id.x + g_invocation_id.y * volume_params.image_width;

    let block_id = block_ids[ray_index];
    if (block_id == UINT_MAX) {
        return;
    }
    let block_pos = block_id_to_pos(block_id);

    block_active[block_id] = 1u;
    //block_visible[block_id] = 1;
    let already_marked = atomicMax(&block_visible[block_id], 1u);

    // Count this ray for the block (this is now done in count_block_rays.wgsl
    //uint num_rays = atomicAdd(block_num_rays[block_id], uint(1)) + 1;
    //let num_rays = atomicAdd(&block_num_rays[block_id], 1u) + 1u;

    // Mark this ray's block's neighbors to the positive side as active
    // These blocks must be decompressed for neighbor data, but this ray
    // doesn't need to process them.
    if (already_marked == 0) {
        let n_blocks = volume_params.padded_dims.xyz / uint3(4u);
        for (var k = 0u; k < 2u; k += 1u) {
            for (var j = 0u; j < 2u; j += 1u) {
                for (var i = 0u; i < 2u; i += 1u) {
                    let neighbor = uint3(i, j, k);
                    let coords = block_pos + neighbor;
                    if (all(neighbor == uint3(0u)) || any(coords < uint3(0u)) || any(coords >= n_blocks)) {
                        continue;
                    }
                    let neighbor_id = compute_block_id(coords);
                    block_active[neighbor_id] = 1u;
                }               
            }
        }
    }
}

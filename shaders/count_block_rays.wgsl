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

struct VolumeParams {
  volume_dims: uint4,
  padded_dims: uint4,
  volume_scale: float4,
  max_bits: u32,
  isovalue: f32,
  image_width: u32,
}

@group(0) @binding(0) var<uniform> volume_params : VolumeParams;

@group(0) @binding(1) var<storage, read_write> block_num_rays : array<atomic<u32>>;

@group(0) @binding(2) var<storage, read_write> ray_block_ids : array<u32>;

@group(0) @binding(3) var<storage, read_write> block_compact_offsets : array<u32>;


@compute @workgroup_size(32, 1, 1)
fn main(@builtin(global_invocation_id) g_invocation_id : vec3<u32>) {
    if (g_invocation_id.x >= volume_params.image_width) {
        return;
    }

    let ray_index = g_invocation_id.x + g_invocation_id.y * volume_params.image_width;

    let block_id = ray_block_ids[ray_index];
    if (block_id == UINT_MAX) {
        return;
    }

    // Count this ray for the block
    let block_index = block_compact_offsets[block_id];
    atomicAdd(&block_num_rays[block_index], 1u);
}


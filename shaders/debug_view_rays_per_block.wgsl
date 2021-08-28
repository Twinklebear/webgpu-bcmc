//include util.wgsl
[[block]] struct UintArray {
    vals : array<u32>;
};
[[block]] struct RayInfos {
    rays : array<RayInfo>;
};
[[block]] struct VolumeParams {
    volume_dims : vec4<u32>;
    padded_dims : vec4<u32>;
    volume_scale : vec4<f32>;
    max_bits : u32;
    isovalue : f32;
    image_width : u32;
};

[[group(0), binding(0)]] var<uniform> volume_params : VolumeParams;
[[group(0), binding(1)]] var<storage, read> block_num_rays : UintArray;
[[group(0), binding(2)]] var<storage, read> ray_info : RayInfos;
// May not need to be uniform variable
[[group(0), binding(3)]] var<uniform> render_target : texture_storage_2d<rgba8unorm, write>;

[[stage(compute), workgroup_size(1, 1, 1)]]
fn main([[builtin(global_invocation_id)]] global_id : vec3<u32>) {
    var ray_index : u32 = global_id.x + global_id.y * volume_params.image_width;
    if (ray_info.rays[ray_index].t == FLT_MAX) {
        return;
    }

    let block_id : u32 = ray_info.rays[ray_index].block_id;
    var color : vec4<f32>;
    // We don't really bother to find the max, though we could do it in the
    // shader since this is just for debugging.
    color.rgb = vec3<f32>(block_num_rays.vals[block_id] / 256.0);
    color.a = 1.0;
    textureStore(render_target, vec2<i32>(global_id.xy), color);
}
//include util.wgsl
// Fragment shader
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

[[group(0), binding(1)]] var<storage, read_write> ray_info: RayInfos;
[[group(0), binding(2)]] var<uniform> volume_params : VolumeParams;

fn intersect_box(orig : vec3<f32>, dir : vec3<f32>, box_min : vec3<f32>, box_max : vec3<f32>) -> vec2<f32> {
    let inv_dir : vec3<f32> = 1.0 / dir;
    let tmin_tmp : vec3<f32> = (box_min - orig) * inv_dir;
    let tmax_tmp : vec3<f32> = (box_max - orig) * inv_dir;
    var tmin : vec3<f32> = min(tmin_tmp, tmax_tmp);
    var tmax : vec3<f32> = max(tmin_tmp, tmax_tmp);
    var t0 : f32 = max(tmin.x, max(tmin.y, tmin.z));
    var t1 : f32 = min(tmax.x, min(tmax.y, tmax.z));
    return vec2<f32>(t0, t1);
}

[[stage(fragment)]]
fn main(
  [[builtin(position)]] frag_coord : vec4<f32>,
  [[location(0)]] vray_dir : vec3<f32>, 
  [[location(1), interpolate(flat)]] transformed_eye : vec3<f32>
) {
    var ray_dir : vec3<f32> = normalize(vray_dir);

    // Transform the ray into the dual grid space and intersect with the dual grid bounds
    let vol_eye : vec3<f32> = transformed_eye * volume_params.volume_dims.xyz - vec3<f32>(0.5);
    let grid_ray_dir : vec3<f32> = normalize(ray_dir * volume_dims.xyz);

    var t_hit : vec2<f32> = intersect_box(vol_eye, grid_ray_dir, vec3<f32>(0.0), volume_dims.xyz - 1.0);

    // We don't want to sample voxels behind the eye if it's
	// inside the volume, so keep the starting point at or in front
	// of the eye
    t_hit.x = max(t_hit.x, 0.0);

    let pixel : u32 = u32(frag_coord.x) + volume_params.image_width * u32(frag_coord.y);
    if (t_hit.x < t_hit.y) { 
        ray_info.rays[pixel].ray_dir = ray_dir;
        ray_info.rays[pixel].block_id = UINT_MAX;
        ray_info.rays[pixel].t = t_hit.x;
    }
}


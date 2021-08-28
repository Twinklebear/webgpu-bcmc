// Vertex shader
struct VertexOutput {
  [[builtin(position)]] Position : vec4<f32>;
  [[location(0)]] vray_dir: vec3<f32>;
  [[location(1), interpolate(flat)]] transformed_eye: vec3<f32>;
};
[[block]] struct ViewParams {
  proj_view : mat4x4<f32>;
  eye_pos : vec4<f32>;
  eye_dir : vec4<f32>;
  near_plane : f32;
};
[[block]] struct VolumeParams {
    volume_dims : vec4<u32>;
    padded_dims : vec4<u32>;
    volume_scale : vec4<f32>;
    max_bits : u32;
    isovalue : f32;
    image_width : u32;
};
[[group(0), binding(0)]] var<uniform> view_params : ViewParams;
[[group(0), binding(2)]] var<uniform> volume_params : VolumeParams;

[[stage(vertex)]]
fn main([[location(0)]] position : vec3<f32>)
     -> VertexOutput {
    var output : VertexOutput;
    var volume_translation : vec3<f32> = vec3<f32>(0, 0, 0) - volume_params.volume_scale.xyz * 0.5;
    output.Position = view_params.proj_view * vec4<f32>(position * volume_params.volume_scale.xyz + volume_translation, 1.0);
    output.transformed_eye = (view_params.eye_pos.xyz - volume_translation) / volume_params.volume_scale.xyz;
    output.vray_dir = position - output.transformed_eye;
    return output;
}
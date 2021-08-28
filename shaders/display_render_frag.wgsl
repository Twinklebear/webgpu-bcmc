// Fragment shader
// May not need uniform declaration
[[group(0), binding(0)]] var output_texture : texture_2d<f32>;

[[stage(fragment)]]
fn main([[builtin(position)]] frag_coord : vec4<f32>) -> [[location(0)]] vec4<f32> {
    var color : vec4<f32> = textureLoad(output_texture, vec2<i32>(frag_coord.xy), 0);
    color.a = 1.0;
    return color;
}
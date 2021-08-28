// Draw a full screen quad using two triangles
struct VertexOutput {
  [[builtin(position)]] Position : vec4<f32>;
};
let pos : array<vec4<f32>, 6> = array<vec4<f32>, 6>(
	vec4<f32>(-1, 1, 0.5, 1),
	vec4<f32>(-1, -1, 0.5, 1),
	vec4<f32>(1, 1, 0.5, 1),
	vec4<f32>(-1, -1, 0.5, 1),
	vec4<f32>(1, 1, 0.5, 1),
	vec4<f32>(1, -1, 0.5, 1)
);

[[stage(vertex)]]
fn main([[builtin(vertex_index)]] vertex_index : u32)
     -> VertexOutput {
    var output : VertexOutput;
    output.Position = pos[vertex_index];
    return output;
}



[[block]] struct UintArray {
    vals : array<u32>;
};

[[group(0), binding(0)]] var<storage, read_write> vals : UintArray;
[[group(0), binding(1)]] var<storage, read> block_sums : UintArray;

[[stage(compute), workgroup_size(BLOCK_SIZE / 2.0)]]
fn main([[builtin(global_invocation_id)]] global_id : vec3<u32>, [[builtin(workgroup_id)]] workgroup_id : vec3<u32>) {
    let prev_sum : u32 = block_sums.vals[workgroup_id.x];
    vals.vals[2 * global_id.x] = vals.vals[2 * global_id.x] + prev_sum;
    vals.vals[2 * global_id.x + 1] = vals.vals[2 * global_id.x + 1] + prev_sum;
}
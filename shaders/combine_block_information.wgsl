//include util.wgsl
[[block]] struct UintArray {
    vals : array<u32>;
};
[[block]] struct BlockInfos {
    vals : array<BlockInfo>;
};

[[group(0), binding(0)]] var<storage, read_write> blocks : BlockInfos;
[[group(0), binding(1)]] var<storage, read_write> block_ids : UintArray;
[[group(0), binding(2)]] var<storage, read_write> block_ray_offsets : UintArray;
[[group(0), binding(3)]] var<storage, read_write> block_num_rays : UintArray;
[[group(0), binding(4)]] var<storage, read_write> block_active : UintArray;

[[stage(compute), workgroup_size(BLOCK_SIZE / 2.0)]]
fn main([[builtin(global_invocation_id)]] global_id : vec3<u32>) {
    // Combine the buffers to fit in fewer storage buffers until limits are removed
    // Note that 8 will be supported soon in Chromium so we could remove this
    // This data is compacted down as it's run on the compacted block ids
    let id : u32 = block_ids.vals[global_id.x];
    blocks.vals[global_id.x].id = id;
    blocks.vals[global_id.x].ray_offset = block_ray_offsets.vals[id];
    blocks.vals[global_id.x].num_rays = block_num_rays.vals[id];
    // If the block is running in this pipeline it must be visible,
    // so if it's not active, then it's an LOD block
    if (block_active.vals[id] == 0) {
        blocks.vals[global_id.x].lod = 1;
    } else {
        blocks.vals[global_id.x].lod = 0;
    }
}
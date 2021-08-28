let BLOCK_NUM_VOXELS : u32 = 64;

// For ghost voxels, we only need those in the positive dir,
// since verts for triangles ''behind'' us are the job of the neighboring
// block to that side. So our max size is 5^3 elements if we have a ghost
// layer on each side, which is rounded up to 128
var<workgroup> volume_block : array<f32, 128>;

[[block]] struct VolumeParams {
    volume_dims : vec4<u32>;
    padded_dims : vec4<u32>;
    volume_scale : vec4<f32>;
    max_bits : u32;
    isovalue : f32;
    image_width : u32;
};
[[block]] struct FloatArray {
    vals : array<f32>;
};
[[block]] struct IntArray {
    vals : array<i32>;
};

[[group(0), binding(0)]] var<uniform> volume_params : VolumeParams;
[[group(0), binding(1)]] var<storage, read> decompressed : FloatArray;
// Cached item slots in the cache
// this is lruCache.cachedItemSlots
[[group(0), binding(2)]] var<storage, read> block_locations : IntArray;

let index_to_vertex : array<vec3<i32>, 8> = array<vec3<i32>>(
    vec3<i32>(0, 0, 0), // v000 = 0
    vec3<i32>(1, 0, 0), // v100 = 1
    vec3<i32>(0, 1, 0), // v010 = 2
    vec3<i32>(1, 1, 0), // v110 = 3
    vec3<i32>(0, 0, 1), // v001 = 4
    vec3<i32>(1, 0, 1), // v101 = 5
    vec3<i32>(0, 1, 1), // v011 = 6
    vec3<i32>(1, 1, 1)  // v111 = 7
);

fn ray_id_to_pos(id : u32) -> vec2<u32> {
    return vec2<u32>(id % volume_params.image_width, id / volume_params.image_width);
}

fn block_id_to_pos(id : u32) -> vec3<u32> {
    var n_blocks : vec3<u32> = volume_params.padded_dims.xyz / vec3<u32>(4);
    return vec3<u32>(id % n_blocks.x, 
        (id / n_blocks.x) % n_blocks.y,
        id / (n_blocks.x * n_blocks.y));
}

fn compute_block_id(block_pos : vec3<u32>) -> u32 {
    var n_blocks : vec3<u32> = padded_dims.xyz / vec3<u32>(4);
    return block_pos.x + n_blocks.x * (block_pos.y + n_blocks.y * block_pos.z);
}

fn voxel_id_to_voxel(id : u32) -> vec3<u32> {
    return vec3<u32>(id % 4, (id / 4) % 4, id / 16);
}

fn compute_voxel_id(voxel_pos : vec3<u32>, block_dims : vec3<u32>) -> u32 {
    return voxel_pos.x + block_dims.x * (voxel_pos.y + block_dims.y * voxel_pos.z);
}

fn compute_vertex_values(voxel_pos : vec3<u32>, block_dims : vec3<u32>, values : array<f32, 8>, value_range : vec2<f32>) {
    
}
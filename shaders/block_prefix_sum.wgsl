[[block]] struct UintArray {
    vals : array<u32>;
};
[[block]] struct Carry {
    in : u32;
    out : u32;
};

[[group(0), binding(0)]] var<storage, read_write> vals : UintArray;
[[group(0), binding(1)]] var<storage, read_write> carry : Carry;

var<workgroup> chunk : array<u32, BLOCK_SIZE>;

[[stage(compute), workgroup_size(BLOCK_SIZE / 2.0)]]
fn main([[builtin(global_invocation_id)]] global_id : vec3<u32>, 
        [[builtin(workgroup_id)]] workgroup_id : vec3<u32>,
        [[builtin(local_invocation_id)]] local_id : vec3<u32>) {
    chunk[2 * local_id.x] = vals.vals[2 * global_id.x];
    chunk[2 * local_id.x + 1] = vals.vals[2 * global_id.x + 1];

    var offs : u32 = 1;
    // Reduce step up tree
    for (var d : i32 = BLOCK_SIZE >> 1; d > 0; d = d >> 1) {
        workgroupBarrier();
        if (local_id.x < d) {
            var a : u32 = offs * (2 * local_id.x + 1) - 1;
            var b : u32 = offs * (2 * local_id.x + 2) - 1;
            chunk[b] = chunk[b] + chunk[a];
        }
        offs = offs << 1;
    }

    if (local_id.x == 0) {
        carry.out = chunk[BLOCK_SIZE - 1] + carry.in;
        chunk[BLOCK_SIZE - 1] = 0;
    }

    // Sweep down the tree to finish the scan
    for (var d : i32 = 1; d < BLOCK_SIZE; d = d << 1) {
        offs = offs >> 1;
        workgroupBarrier();
        if (local_id.x < d) {
            var a : u32 = offs * (2 * local_id.x + 1) - 1;
            var b : u32 = offs * (2 * local_id.x + 2) - 1;
            let tmp : u32 = chunk[a];
            chunk[a] = chunk[b];
            chunk[b] = chunk[b] + tmp;
        }
    }

    workgroupBarrier();
    vals.vals[2 * global_id.x] = chunk[2 * local_id.x] + carry.in;
    vals.vals[2 * global_id.x + 1] = chunk[2 * local_id.x + 1] + carry.in;
}
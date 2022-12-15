#!/usr/bin/env python3

import sys
import os
import subprocess

if len(sys.argv) < 3:
    print("Usage <glslc> <tint>")

glslc = sys.argv[1]
tint = sys.argv[2]

output = "embedded_shaders.js"
shaders = [
    "prefix_sum.comp",
    "block_prefix_sum.comp",
    "add_block_sums.comp",
    "stream_compact.comp",
    "stream_compact_data.comp",
    "compute_initial_rays.vert",
    "compute_initial_rays.frag",
    "zfp_compute_block_range.comp",
    "zfp_decompress_block.comp",
    "lru_cache_init.comp",
    "lru_cache_mark_new_items.comp",
    "lru_cache_update.comp",
    "lru_copy_available_slot_age.comp",
    "lru_cache_age_slots.comp",
    "lru_cache_extract_slot_available.comp",
    "macro_traverse.comp",
    "radix_sort_chunk.comp",
    "reverse_buffer.comp",
    "merge_sorted_chunks.comp",
    "display_render.vert",
    "display_render.frag",
    "reset_rays.comp",
    # Must be manually ported to WGSL since it uses atomics
    # Tint cannot translate atomics from SPV -> WGSL due to
    # - https://bugs.chromium.org/p/tint/issues/detail?id=1207
    # - https://bugs.chromium.org/p/tint/issues/detail?id=1441
    #"mark_block_active.comp",
    "reset_block_active.comp",
    "reset_block_num_rays.comp",
    "debug_view_rays_per_block.comp",
    "write_ray_and_block_id.comp",
    "combine_block_information.comp",
    "raytrace_active_block.comp",
    "compute_voxel_range.comp",
    "compute_coarse_cell_range.comp",
    "reset_speculative_ids.comp",
    "depth_composite.comp",
    "mark_ray_active.comp"
]

try:
    os.stat(output)
    os.remove(output)
except:
    pass

block_size = 512
sort_chunk_size = 64
draw_fog = False
if "-fog" in sys.argv:
    draw_fog = True

compiled_shaders = ""
for shader in shaders:
    fname, ext = os.path.splitext(os.path.basename(shader))
    var_name = "{}_{}_spv".format(fname, ext[1:])
    print("Embedding {} as {}".format(shader, var_name))
    args = [
        "python3",
        "compile_shader.py",
        glslc,
        tint,
        shader,
        var_name,
        "-DBLOCK_SIZE={}".format(block_size),
        "-DSORT_CHUNK_SIZE={}".format(sort_chunk_size),
    ]
    if draw_fog:
        args.append("-DDRAW_FOG=1")
    compiled_shaders += subprocess.check_output(args).decode("utf-8")

# TODO: Read and append hand port of mark_block_active.comp to embed the WGSL shader
manual_wgsl_shaders = [
    "mark_block_active.wgsl",
    "count_block_rays.wgsl"
]
# TODO: Would also need to do a find/replace for the defines if we manually port any
# shaders that use BLOCK_SIZE or SORT_CHUNK_SIZE but I don't think it'll be needed
for shader in manual_wgsl_shaders:
    with open(shader, "r") as f:
        fname, ext = os.path.splitext(os.path.basename(shader))
        var_name = "{}_{}_spv".format(fname, ext[1:])
        print("Embedding manually WGSL'd shader {} as {}".format(shader, var_name))
        compiled_shaders += "const " + var_name + " = `" + "".join(f.readlines()) + "`;\n";

with open(output, "w") as f:
    f.write("const ScanBlockSize = {};\n".format(block_size))
    f.write("const SortChunkSize = {};\n".format(sort_chunk_size))
    f.write(compiled_shaders)


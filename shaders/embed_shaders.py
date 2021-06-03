#!/usr/bin/env python3

import sys
import os
import subprocess

if len(sys.argv) < 2:
    print("Usage <glslc>")

glslc = sys.argv[1]
output = "embedded_shaders.js"
shaders = [
    "prefix_sum.comp",
    "block_prefix_sum.comp",
    "add_block_sums.comp",
    "stream_compact.comp",
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
    print(shader)
    fname, ext = os.path.splitext(os.path.basename(shader))
    var_name = "{}_{}_spv".format(fname, ext[1:])
    print("Embedding {} as {}".format(shader, var_name))
    args = [
        "python",
        "compile_shader.py",
        glslc,
        shader,
        var_name,
        "-DBLOCK_SIZE={}".format(block_size),
        "-DSORT_CHUNK_SIZE={}".format(sort_chunk_size),
    ]
    if draw_fog:
        args.append("-DDRAW_FOG=1")
    compiled_shaders += subprocess.check_output(args).decode("utf-8")

with open(output, "w") as f:
    f.write("const ScanBlockSize = {};\n".format(block_size))
    f.write("const SortChunkSize = {};\n".format(sort_chunk_size))
    f.write(compiled_shaders)


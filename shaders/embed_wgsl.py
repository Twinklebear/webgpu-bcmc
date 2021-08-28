#!/usr/bin/env python3
shaders = [
    "add_block_sums.wgsl",
    "block_prefix_sum.wgsl",
    "combine_block_information.wgsl",
    "compute_initial_rays_frag.wgsl",
    "compute_initial_rays_vert.wgsl",
]
compiled_shaders = ""

with open("util.wgsl") as f:
    utils_code = f.read()

for shader in shaders:
    with open(shader, "r") as f:
        compiled_code = f.read()
        if compiled_code.startswith("//include util.wgsl"):
            compiled_code = utils_code + compiled_code
        compiled_shaders += f"const  {shader[:-5]} = `{compiled_code}`;\n"

with open("../js/wgsl.js", "w") as f:
    f.write(compiled_shaders)


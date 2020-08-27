#!/usr/bin/env python3

import sys
import os
import subprocess

if len(sys.argv) < 4:
    print("Usage <glslc> <shader> <var_name> [glslc_args...]")
    sys.exit(1)

glslc = sys.argv[1]
shader = sys.argv[2]
var_name = sys.argv[3]

compiled_shader = ""
args = [glslc, shader, "-mfmt=c"]
if len(sys.argv) > 4:
    args.extend(sys.argv[4:])

subprocess.check_output(args)
with open("a.spv", "r") as f:
    compiled_code = f.read()
    compiled_shader = "const " + var_name + " = new Uint32Array([" + compiled_code[1:-2] + "]);\n"

os.remove("a.spv")
print(compiled_shader)


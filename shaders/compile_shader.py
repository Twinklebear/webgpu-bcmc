#!/usr/bin/env python3

import sys
import os
import subprocess

if len(sys.argv) < 5:
    print("Usage <glslc> <tint> <shader> <var_name> [glslc_args...]")
    sys.exit(1)

glslc = sys.argv[1]
tint = sys.argv[2]
shader = sys.argv[3]
var_name = sys.argv[4]

compiled_shader = ""
args = [glslc, shader]
if len(sys.argv) > 5:
    args.extend(sys.argv[5:])

subprocess.check_output(args)

# Now compile the SPV file to WGSL with Tint
subprocess.check_output([tint, "a.spv", "-o", "a.wgsl"])

with open("a.wgsl", "r") as f:
    compiled_code = f.read()
    compiled_shader = "const " + var_name + " = `" + compiled_code + "`;\n"

os.remove("a.spv")
os.remove("a.wgsl")
print(compiled_shader)


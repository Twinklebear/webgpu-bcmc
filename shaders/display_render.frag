#version 450 core

layout(location = 0) out vec4 color;

uniform layout(binding = 0, rgba8) readonly image2D output_texture;
layout(set = 0, binding = 1) uniform Resolution
{
    uint half_resolution;
};

void main(void) {
    if (half_resolution == 1) {
        vec2 coords = gl_FragCoord.xy / 2;
        color = imageLoad(output_texture, ivec2(coords.xy));
    } else{
        color = imageLoad(output_texture, ivec2(gl_FragCoord.xy));
    }
    color.a = 1.f;
}

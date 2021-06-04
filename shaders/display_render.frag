#version 450 core

layout(location = 0) out vec4 color;

uniform layout(binding = 0, rgba8) readonly image2D output_texture;

void main(void) {
    color = imageLoad(output_texture, ivec2(gl_FragCoord.xy));
    color.a = 1.f;
}

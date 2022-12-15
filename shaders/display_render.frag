#version 450 core

layout(location = 0) out vec4 color;

layout(set = 0, binding = 0) uniform texture2D output_texture;
layout(set = 0, binding = 2) uniform sampler u_sampler;
layout(set = 0, binding = 1) uniform Resolution
{
    uint width;
    uint height;
};

void main(void) {
    color = texture(sampler2D(output_texture, u_sampler), gl_FragCoord.xy / vec2(width, height));
    color.a = 1.f;
}

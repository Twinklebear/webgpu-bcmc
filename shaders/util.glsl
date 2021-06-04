#ifndef UTIL_GLSL
#define UTIL_GLSL

#define UINT_MAX uint(0xffffffff)
#define FLT_MAX ( 3.402823466e+38f )

struct RayInfo {
    vec3 ray_dir;
    uint block_id;
    float t;
    // NOTE: std430 layout rules dictate the struct alignment is that of its
    // largest member, which is the vec3 ray dir (whose alignment is same as vec4).
    // This results in the struct size rounding up to 32, since it has to start
    // on 16 byte boundaries.
    // So we have 3 free 4 byte values to use if needed.
    //vec2 pad;
};

#endif


#ifndef UTIL_GLSL
#define UTIL_GLSL

#define UINT_MAX uint(0xffffffff)
#define FLT_MAX ( 3.402823466e+38f )

struct RayInfo {
    vec3 ray_dir;
    uint block_id;
    float t;
    float t_next;
    // NOTE: std430 layout rules dictate the struct alignment is that of its
    // largest member, which is the vec3 ray dir (whose alignment is same as vec4).
    // This results in the struct size rounding up to 32, since it has to start
    // on 16 byte boundaries.
    // So we have a free 4 byte value to use if needed.
};

struct BlockInfo {
    uint id;
    uint ray_offset;
    uint num_rays;
};

bool outside_grid(const vec3 p, const vec3 grid_dims) {
    return any(lessThan(p, vec3(0))) || any(greaterThanEqual(p, grid_dims));
}

bool outside_dual_grid(const vec3 p, const vec3 grid_dims) {
    return any(lessThan(p, vec3(0))) || any(greaterThanEqual(p, grid_dims - vec3(1)));
}


#endif


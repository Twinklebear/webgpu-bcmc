#ifndef UTIL_GLSL
#define UTIL_GLSL

#define UINT_MAX uint(0xffffffff)
#define FLT_MAX ( 3.402823466e+38f )

struct RayInfo {
    vec3 ray_dir;
    // uint block_id;
    float t;
    // float t_next;
    // NOTE: std430 layout rules dictate the struct alignment is that of its
    // largest member, which is the vec3 ray dir (whose alignment is same as vec4).
    // This results in the struct size rounding up to 32, since it has to start
    // on 16 byte boundaries.
    // So we have two free 4 byte values to use if needed.
};

struct BlockInfo {
    uint id;
    uint ray_offset;
    uint num_rays;
    uint lod;
};

struct GridIterator {
    ivec3 grid_dims;
    ivec3 grid_step;
    vec3 t_delta;

    ivec3 cell;
    vec3 t_max;
    float t;
};

// The state we save for saving/restoring the grid iterator state
struct GridIteratorState {
    // TODO: store single int for ID
    ivec3 cell;
    vec3 t_max;
    // TODO: remove t from saved state
    float t;
};

bool outside_grid(const vec3 p, const vec3 grid_dims) {
    return any(lessThan(p, vec3(0))) || any(greaterThanEqual(p, grid_dims));
}

bool outside_dual_grid(const vec3 p, const vec3 grid_dims) {
    return any(lessThan(p, vec3(0))) || any(greaterThanEqual(p, grid_dims - vec3(1)));
}

bool outside_grid(const ivec3 p, const ivec3 grid_dims) {
    return any(lessThan(p, ivec3(0))) || any(greaterThanEqual(p, grid_dims));
}

bool outside_dual_grid(const ivec3 p, const ivec3 grid_dims) {
    return any(lessThan(p, ivec3(0))) || any(greaterThanEqual(p, grid_dims - ivec3(1)));
}

// Initialize the grid traversal state. All positions/directions passed must be in the
// grid coordinate system where a grid cell is 1^3 in size.
GridIterator init_grid_iterator(vec3 ray_org, vec3 ray_dir, float t, ivec3 grid_dims) {
    GridIterator grid_iter;
    grid_iter.grid_dims = grid_dims;
    grid_iter.grid_step = ivec3(sign(ray_dir));

    const vec3 inv_ray_dir = 1.0 / ray_dir;
    grid_iter.t_delta = abs(inv_ray_dir);

	vec3 p = (ray_org + t * ray_dir);
    p = clamp(p, vec3(0), vec3(grid_dims - 1));
    vec3 cell = floor(p);
    const vec3 t_max_neg = (cell - ray_org) * inv_ray_dir;
    const vec3 t_max_pos = (cell + vec3(1) - ray_org) * inv_ray_dir;

    // Pick between positive/negative t_max based on the ray sign
    const bvec3 is_neg_dir = lessThan(ray_dir, vec3(0));
    grid_iter.t_max = mix(t_max_pos, t_max_neg, is_neg_dir);

    grid_iter.cell = ivec3(cell);

    grid_iter.t = t;

    return grid_iter;
}

GridIterator restore_grid_iterator(vec3 ray_org,
                                   vec3 ray_dir,
                                   ivec3 grid_dims,
                                   in GridIteratorState state)
{
    GridIterator grid_iter;
    grid_iter.grid_dims = grid_dims;
    grid_iter.grid_step = ivec3(sign(ray_dir));

    const vec3 inv_ray_dir = 1.0 / ray_dir;
    grid_iter.t_delta = abs(inv_ray_dir);

    grid_iter.cell = state.cell;
    grid_iter.t_max = state.t_max;
    // We don't really care about this value when restoring
    grid_iter.t = min(state.t_max.x, min(state.t_max.y, state.t_max.z));

    return grid_iter;
}

// Get the current cell the iterator is in and its t interval. Returns false if the iterator is
// outside the grid or the t interval is empty, indicating traversal should stop.
bool grid_iterator_get_cell(inout GridIterator iter, out vec2 cell_t_range, out ivec3 cell_id) {
    if (outside_grid(iter.cell, iter.grid_dims)) {
        return false;
    }
    // Return the current cell range and ID to the caller
    cell_t_range.x = iter.t;
    cell_t_range.y = min(iter.t_max.x, min(iter.t_max.y, iter.t_max.z));
    cell_id = iter.cell;
    if (cell_t_range.y < cell_t_range.x) {
        return false;
    }
    return true;
}

// Advance the iterator to the next cell in the grid.
void grid_iterator_advance(inout GridIterator iter) {
    // Move the iterator to the next cell we'll traverse
    iter.t = min(iter.t_max.x, min(iter.t_max.y, iter.t_max.z));
    if (iter.t == iter.t_max.x) {
        iter.cell.x += iter.grid_step.x;
        iter.t_max.x += iter.t_delta.x;
    } else if (iter.t == iter.t_max.y) {
        iter.cell.y += iter.grid_step.y;
        iter.t_max.y += iter.t_delta.y;
    } else {
        iter.cell.z += iter.grid_step.z;
        iter.t_max.z += iter.t_delta.z;
    }
}

vec2 intersect_box(vec3 orig, vec3 dir, const vec3 box_min, const vec3 box_max) {
    vec3 inv_dir = 1.0 / dir;
    vec3 tmin_tmp = (box_min - orig) * inv_dir;
    vec3 tmax_tmp = (box_max - orig) * inv_dir;
    vec3 tmin = min(tmin_tmp, tmax_tmp);
    vec3 tmax = max(tmin_tmp, tmax_tmp);
    float t0 = max(tmin.x, max(tmin.y, tmin.z));
    float t1 = min(tmax.x, min(tmax.y, tmax.z));
    return vec2(t0, t1);
}

// Pass a float color in [0,1] to pack
int pack_color(vec3 rgb) {
    ivec3 rbg256 = clamp(ivec3(rgb * 255), ivec3(0), ivec3(255));
    int c = 0;
    c |= (rbg256.x << 24) & 0xff000000;
    c |= (rbg256.y << 16) & 0x00ff0000;
    c |= (rbg256.z << 8) & 0x0000ff00;
    return c;
}

// Returns a float color in [0,1]
vec3 unpack_color(int rgb8) {
    vec3 rgb = vec3(0);
    rgb.x = ((rgb8 >> 24) & 0x000000ff) / 255.0;
    rgb.y = ((rgb8 >> 16) & 0x000000ff) / 255.0;
    rgb.z = ((rgb8 >> 8) & 0x000000ff) / 255.0;
    return rgb;
}

#endif


// Metal port of the image-gs gsplat CUDA rasterization kernels.
// Mirrors gsplat/cuda/csrc/forward.cu and backward.cu (2D, N-channel variants).
#include <metal_stdlib>
using namespace metal;

#define BLOCK_X 16
#define BLOCK_Y 16
#define BLOCK_SIZE (BLOCK_X * BLOCK_Y)
#define MAX_CHANNELS 12
#define TOP_K 10

// params layout (int32): [img_w, img_h, tiles_x, channels]
// fparams layout (float32): [eps]

inline float gaussian_sigma(float3 conic, float2 delta) {
    return 0.5f * (conic.x * delta.x * delta.x + conic.z * delta.y * delta.y) +
           conic.y * delta.x * delta.y;
}

inline bool bad_sigma(float sigma) {
    return sigma < 0.f || isnan(sigma) || isinf(sigma);
}

// ---------------------------------------------------------------------------
// Top-K normalized forward (also used for the no-tiles path, where every tile's
// bin spans all Gaussians). One threadgroup per 16x16 tile, one thread per pixel.
// ---------------------------------------------------------------------------
kernel void nd_rasterize_forward_topk_norm(
    device const int*   gaussian_ids_sorted [[buffer(0)]],
    device const int*   tile_bins           [[buffer(1)]],
    device const float* xys                 [[buffer(2)]],
    device const float* conics              [[buffer(3)]],
    device const float* colors              [[buffer(4)]],
    device float*       out_img             [[buffer(5)]],
    device int*         pixel_topk          [[buffer(6)]],
    device const int*   params              [[buffer(7)]],
    device const float* fparams             [[buffer(8)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint2 local [[thread_position_in_threadgroup]],
    uint  tr    [[thread_index_in_threadgroup]])
{
    const int img_w = params[0], img_h = params[1], tiles_x = params[2];
    const int channels = params[3];
    const float eps = fparams[0];

    const int tile_id = int(group.y) * tiles_x + int(group.x);
    const uint i = group.y * BLOCK_Y + local.y;
    const uint j = group.x * BLOCK_X + local.x;
    const float px = float(j), py = float(i);
    const bool inside = (i < uint(img_h) && j < uint(img_w));
    const int pix_id = int(i) * img_w + int(j);

    const int range_x = tile_bins[2 * tile_id];
    const int range_y = tile_bins[2 * tile_id + 1];
    const int num_batches = (range_y - range_x + BLOCK_SIZE - 1) / BLOCK_SIZE;

    threadgroup int    id_batch[BLOCK_SIZE];
    threadgroup float2 xy_batch[BLOCK_SIZE];
    threadgroup float3 conic_batch[BLOCK_SIZE];

    // topk holds Gaussian ids (for colors); topk_isect holds the position in
    // gaussian_ids_sorted, which the tile-cooperative backward pass keys on.
    int   topk[TOP_K];
    int   topk_isect[TOP_K];
    float topk_vals[TOP_K];
    for (int k = 0; k < TOP_K; ++k) { topk[k] = -1; topk_isect[k] = -1; topk_vals[k] = 0.f; }

    for (int b = 0; b < num_batches; ++b) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const int batch_start = range_x + BLOCK_SIZE * b;
        const int idx = batch_start + int(tr);
        if (idx < range_y) {
            const int g_id = gaussian_ids_sorted[idx];
            id_batch[tr] = g_id;
            xy_batch[tr] = float2(xys[2 * g_id], xys[2 * g_id + 1]);
            conic_batch[tr] = float3(conics[3 * g_id], conics[3 * g_id + 1], conics[3 * g_id + 2]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (!inside) continue;
        const int batch_size = min(BLOCK_SIZE, range_y - batch_start);
        for (int t = 0; t < batch_size; ++t) {
            const float2 delta = xy_batch[t] - float2(px, py);
            const float sigma = gaussian_sigma(conic_batch[t], delta);
            if (bad_sigma(sigma)) continue;
            const float alpha = exp(-sigma);
            const int g = id_batch[t];

            int min_topk_id = -1;
            float min_topk_val = 1e30f;
            for (int k = 0; k < TOP_K; ++k) {
                if (topk[k] < 0) {
                    min_topk_id = k;
                    min_topk_val = -1.0f;
                    break;
                } else if (topk_vals[k] < min_topk_val) {
                    min_topk_id = k;
                    min_topk_val = topk_vals[k];
                }
            }
            if (alpha > min_topk_val) {
                topk[min_topk_id] = g;
                topk_isect[min_topk_id] = batch_start + t;
                topk_vals[min_topk_id] = alpha;
            }
        }
    }

    if (!inside) return;

    float sum_val = 0.f;
    for (int k = 0; k < TOP_K; ++k)
        if (topk[k] >= 0) sum_val += topk_vals[k];

    for (int c = 0; c < channels; ++c) {
        float acc = 0.f;
        for (int k = 0; k < TOP_K; ++k) {
            const int g = topk[k];
            if (g < 0) continue;
            acc += colors[g * channels + c] * (topk_vals[k] / (sum_val + eps));
        }
        out_img[pix_id * channels + c] = acc;
    }
    for (int k = 0; k < TOP_K; ++k)
        pixel_topk[pix_id * TOP_K + k] = topk_isect[k];
}

// ---------------------------------------------------------------------------
// Top-K normalized backward. One threadgroup per tile, one thread per pixel.
// The tile walks its bin of Gaussians in order; each pixel contributes only if
// that Gaussian is among its top-K. Contributions are reduced per simdgroup
// with simd_sum and scattered with one atomic per simdgroup, which avoids the
// heavy per-pixel float-atomic contention that is slow on Apple GPUs.
// pixel_topk holds positions in gaussian_ids_sorted (-1 = empty slot).
// ---------------------------------------------------------------------------
kernel void nd_rasterize_backward_topk_norm(
    device const int*   gaussian_ids_sorted [[buffer(0)]],
    device const int*   tile_bins           [[buffer(1)]],
    device const float* xys                 [[buffer(2)]],
    device const float* conics              [[buffer(3)]],
    device const float* rgbs                [[buffer(4)]],
    device const float* v_output            [[buffer(5)]],
    device const int*   pixel_topk          [[buffer(6)]],
    device atomic_float* v_xy               [[buffer(7)]],
    device atomic_float* v_conic            [[buffer(8)]],
    device atomic_float* v_rgb              [[buffer(9)]],
    device const int*   params              [[buffer(10)]],
    device const float* fparams             [[buffer(11)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint2 local [[thread_position_in_threadgroup]],
    uint  lane  [[thread_index_in_simdgroup]])
{
    const int img_w = params[0], img_h = params[1], tiles_x = params[2];
    const int channels = params[3];
    const float eps = fparams[0];

    const int tile_id = int(group.y) * tiles_x + int(group.x);
    const uint i = group.y * BLOCK_Y + local.y;
    const uint j = group.x * BLOCK_X + local.x;
    const float px = float(j), py = float(i);
    const bool inside = (i < uint(img_h) && j < uint(img_w));
    const int pix_id = inside ? int(i) * img_w + int(j) : 0;

    int   isect[TOP_K];
    float v_sigma[TOP_K];
    float norm_d[TOP_K];
    float2 delta[TOP_K];
    float v_out[MAX_CHANNELS];
    for (int c = 0; c < MAX_CHANNELS; ++c)
        v_out[c] = (inside && c < channels) ? v_output[channels * pix_id + c] : 0.f;

    float d_local[TOP_K];
    float v_dot[TOP_K];
    float denom = eps;
    float weighted = 0.f;
    for (int k = 0; k < TOP_K; ++k) {
        isect[k] = inside ? pixel_topk[pix_id * TOP_K + k] : -1;
        d_local[k] = 0.f; v_dot[k] = 0.f; delta[k] = float2(0.f);
        if (isect[k] < 0) continue;
        const int g = gaussian_ids_sorted[isect[k]];
        delta[k] = float2(xys[2 * g], xys[2 * g + 1]) - float2(px, py);
        const float3 conic = float3(conics[3 * g], conics[3 * g + 1], conics[3 * g + 2]);
        const float d = exp(-gaussian_sigma(conic, delta[k]));
        float dot = 0.f;
        for (int c = 0; c < channels; ++c) dot += rgbs[channels * g + c] * v_out[c];
        d_local[k] = d; v_dot[k] = dot;
        denom += d;
        weighted += d * dot;
    }
    // out_c = sum_k d_k rgb_kc / denom  =>  dL/dd_k = dot_k / denom - sum_l d_l dot_l / denom^2
    const float shared_term = -weighted / (denom * denom);
    for (int k = 0; k < TOP_K; ++k) {
        norm_d[k] = d_local[k] / denom;
        v_sigma[k] = isect[k] < 0 ? 0.f : -d_local[k] * (v_dot[k] / denom + shared_term);
    }

    const int range_x = tile_bins[2 * tile_id];
    const int range_y = tile_bins[2 * tile_id + 1];
    for (int idx = range_x; idx < range_y; ++idx) {
        int kk = -1;
        for (int k = 0; k < TOP_K; ++k)
            if (isect[k] == idx) kk = k;
        const bool has = kk >= 0;
        if (!simd_any(has)) continue;

        const int g = gaussian_ids_sorted[idx];
        float vs = 0.f, nd = 0.f;
        float2 dl = float2(0.f);
        for (int k = 0; k < TOP_K; ++k) {
            if (k == kk) { vs = v_sigma[k]; nd = norm_d[k]; dl = delta[k]; }
        }
        const float3 conic = float3(conics[3 * g], conics[3 * g + 1], conics[3 * g + 2]);
        const float3 vc = simd_sum(float3(0.5f * vs * dl.x * dl.x, vs * dl.x * dl.y, 0.5f * vs * dl.y * dl.y));
        const float2 vx = simd_sum(float2(vs * (conic.x * dl.x + conic.y * dl.y),
                                          vs * (conic.y * dl.x + conic.z * dl.y)));
        for (int c = 0; c < channels; ++c) {
            const float s = simd_sum(nd * v_out[c]);
            if (lane == 0) atomic_fetch_add_explicit(&v_rgb[channels * g + c], s, memory_order_relaxed);
        }
        if (lane == 0) {
            atomic_fetch_add_explicit(&v_conic[3 * g + 0], vc.x, memory_order_relaxed);
            atomic_fetch_add_explicit(&v_conic[3 * g + 1], vc.y, memory_order_relaxed);
            atomic_fetch_add_explicit(&v_conic[3 * g + 2], vc.z, memory_order_relaxed);
            atomic_fetch_add_explicit(&v_xy[2 * g + 0], vx.x, memory_order_relaxed);
            atomic_fetch_add_explicit(&v_xy[2 * g + 1], vx.y, memory_order_relaxed);
        }
    }
}

// ---------------------------------------------------------------------------
// Plain summation forward (used with --disable_topk_norm).
// ---------------------------------------------------------------------------
kernel void nd_rasterize_forward_sum(
    device const int*   gaussian_ids_sorted [[buffer(0)]],
    device const int*   tile_bins           [[buffer(1)]],
    device const float* xys                 [[buffer(2)]],
    device const float* conics              [[buffer(3)]],
    device const float* colors              [[buffer(4)]],
    device float*       out_img             [[buffer(5)]],
    device const int*   params              [[buffer(6)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint2 local [[thread_position_in_threadgroup]],
    uint  tr    [[thread_index_in_threadgroup]])
{
    const int img_w = params[0], img_h = params[1], tiles_x = params[2];
    const int channels = params[3];

    const int tile_id = int(group.y) * tiles_x + int(group.x);
    const uint i = group.y * BLOCK_Y + local.y;
    const uint j = group.x * BLOCK_X + local.x;
    const float px = float(j), py = float(i);
    const bool inside = (i < uint(img_h) && j < uint(img_w));
    const int pix_id = int(i) * img_w + int(j);

    const int range_x = tile_bins[2 * tile_id];
    const int range_y = tile_bins[2 * tile_id + 1];
    const int num_batches = (range_y - range_x + BLOCK_SIZE - 1) / BLOCK_SIZE;

    threadgroup int    id_batch[BLOCK_SIZE];
    threadgroup float2 xy_batch[BLOCK_SIZE];
    threadgroup float3 conic_batch[BLOCK_SIZE];

    float pix_out[MAX_CHANNELS];
    for (int c = 0; c < MAX_CHANNELS; ++c) pix_out[c] = 0.f;

    for (int b = 0; b < num_batches; ++b) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const int batch_start = range_x + BLOCK_SIZE * b;
        const int idx = batch_start + int(tr);
        if (idx < range_y) {
            const int g_id = gaussian_ids_sorted[idx];
            id_batch[tr] = g_id;
            xy_batch[tr] = float2(xys[2 * g_id], xys[2 * g_id + 1]);
            conic_batch[tr] = float3(conics[3 * g_id], conics[3 * g_id + 1], conics[3 * g_id + 2]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (!inside) continue;
        const int batch_size = min(BLOCK_SIZE, range_y - batch_start);
        for (int t = 0; t < batch_size; ++t) {
            const float2 delta = xy_batch[t] - float2(px, py);
            const float sigma = gaussian_sigma(conic_batch[t], delta);
            if (bad_sigma(sigma)) continue;
            const float alpha = exp(-sigma);
            const int g = id_batch[t];
            for (int c = 0; c < channels; ++c)
                pix_out[c] += colors[g * channels + c] * alpha;
        }
    }

    if (inside)
        for (int c = 0; c < channels; ++c)
            out_img[pix_id * channels + c] = pix_out[c];
}

// ---------------------------------------------------------------------------
// Plain summation backward. Per-tile, reduces over each simdgroup before the
// atomic scatter (mirrors the CUDA warpSum + atomicAdd pattern).
// ---------------------------------------------------------------------------
kernel void nd_rasterize_backward_sum(
    device const int*   gaussian_ids_sorted [[buffer(0)]],
    device const int*   tile_bins           [[buffer(1)]],
    device const float* xys                 [[buffer(2)]],
    device const float* conics              [[buffer(3)]],
    device const float* rgbs                [[buffer(4)]],
    device const float* v_output            [[buffer(5)]],
    device atomic_float* v_xy               [[buffer(6)]],
    device atomic_float* v_conic            [[buffer(7)]],
    device atomic_float* v_rgb              [[buffer(8)]],
    device const int*   params              [[buffer(9)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint2 local [[thread_position_in_threadgroup]],
    uint  tr    [[thread_index_in_threadgroup]],
    uint  lane  [[thread_index_in_simdgroup]])
{
    const int img_w = params[0], img_h = params[1], tiles_x = params[2];
    const int channels = params[3];

    const int tile_id = int(group.y) * tiles_x + int(group.x);
    const uint i = group.y * BLOCK_Y + local.y;
    const uint j = group.x * BLOCK_X + local.x;
    const float px = float(j), py = float(i);
    const bool inside = (i < uint(img_h) && j < uint(img_w));
    const int pix_id = inside ? int(i) * img_w + int(j) : 0;

    const int range_x = tile_bins[2 * tile_id];
    const int range_y = tile_bins[2 * tile_id + 1];
    const int num_batches = (range_y - range_x + BLOCK_SIZE - 1) / BLOCK_SIZE;

    threadgroup int    id_batch[BLOCK_SIZE];
    threadgroup float2 xy_batch[BLOCK_SIZE];
    threadgroup float3 conic_batch[BLOCK_SIZE];
    threadgroup float  rgbs_batch[BLOCK_SIZE * MAX_CHANNELS];

    float v_out[MAX_CHANNELS];
    for (int c = 0; c < MAX_CHANNELS; ++c)
        v_out[c] = (inside && c < channels) ? v_output[channels * pix_id + c] : 0.f;

    for (int b = 0; b < num_batches; ++b) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const int batch_start = range_x + BLOCK_SIZE * b;
        const int idx = batch_start + int(tr);
        if (idx < range_y) {
            const int g_id = gaussian_ids_sorted[idx];
            id_batch[tr] = g_id;
            xy_batch[tr] = float2(xys[2 * g_id], xys[2 * g_id + 1]);
            conic_batch[tr] = float3(conics[3 * g_id], conics[3 * g_id + 1], conics[3 * g_id + 2]);
            for (int c = 0; c < channels; ++c)
                rgbs_batch[tr * MAX_CHANNELS + c] = rgbs[channels * g_id + c];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const int batch_size = min(BLOCK_SIZE, range_y - batch_start);
        for (int t = 0; t < batch_size; ++t) {
            const float3 conic = conic_batch[t];
            const float2 delta = xy_batch[t] - float2(px, py);
            const float sigma = gaussian_sigma(conic, delta);
            const bool valid = inside && !bad_sigma(sigma);
            const float d = valid ? exp(-sigma) : 0.f;

            float v_sigma = 0.f;
            for (int c = 0; c < channels; ++c)
                v_sigma += rgbs_batch[t * MAX_CHANNELS + c] * v_out[c];
            v_sigma *= -d;

            const float3 v_conic_local = float3(0.5f * v_sigma * delta.x * delta.x,
                                                v_sigma * delta.x * delta.y,
                                                0.5f * v_sigma * delta.y * delta.y);
            const float2 v_xy_local = float2(v_sigma * (conic.x * delta.x + conic.y * delta.y),
                                             v_sigma * (conic.y * delta.x + conic.z * delta.y));

            const int g = id_batch[t];
            for (int c = 0; c < channels; ++c) {
                const float s = simd_sum(d * v_out[c]);
                if (lane == 0) atomic_fetch_add_explicit(&v_rgb[channels * g + c], s, memory_order_relaxed);
            }
            const float3 sc = simd_sum(v_conic_local);
            const float2 sx = simd_sum(v_xy_local);
            if (lane == 0) {
                atomic_fetch_add_explicit(&v_conic[3 * g + 0], sc.x, memory_order_relaxed);
                atomic_fetch_add_explicit(&v_conic[3 * g + 1], sc.y, memory_order_relaxed);
                atomic_fetch_add_explicit(&v_conic[3 * g + 2], sc.z, memory_order_relaxed);
                atomic_fetch_add_explicit(&v_xy[2 * g + 0], sx.x, memory_order_relaxed);
                atomic_fetch_add_explicit(&v_xy[2 * g + 1], sx.y, memory_order_relaxed);
            }
        }
    }
}

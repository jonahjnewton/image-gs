"""Apple Silicon (MPS) backend for the image-gs gsplat rasterizer.

Projection and tile binning are expressed as PyTorch ops (they are cheap and
elementwise, and autograd reproduces the CUDA backward exactly). The per-pixel
rasterization forward/backward passes are Metal kernels compiled at runtime via
torch.mps.compile_shader, so no Xcode toolchain or C++ build step is needed.
"""

import os
from functools import lru_cache
from typing import Tuple

import torch
from torch import Tensor
from torch.autograd import Function

BLOCK_X = 16
BLOCK_Y = 16
TOP_K = 10
MAX_CHANNELS = 12
EPS = 1e-4
EPS_NO_TILES = 1e-7


@lru_cache(maxsize=1)
def _lib():
    if not hasattr(torch.mps, "compile_shader"):
        raise RuntimeError("gsplat MPS backend requires PyTorch >= 2.6 (torch.mps.compile_shader)")
    with open(os.path.join(os.path.dirname(__file__), "kernels.metal")) as f:
        return torch.mps.compile_shader(f.read())


def _iparams(device, *vals):
    return torch.tensor(vals, dtype=torch.int32, device=device)


def _fparams(device, *vals):
    return torch.tensor(vals, dtype=torch.float32, device=device)


def _tile_grid(tile_bounds):
    return (tile_bounds[0] * BLOCK_X, tile_bounds[1] * BLOCK_Y)


# The binning tensors change size every step as Gaussians move, and the MPS
# caching allocator keeps a buffer for every size it has seen. Left alone, the
# cache grows by ~1 GB per 100 steps on 2K texture stacks until it runs out of
# memory, so release it periodically.
_EMPTY_CACHE_EVERY = 100
_render_calls = 0


def _maybe_empty_cache():
    global _render_calls
    _render_calls += 1
    if _render_calls % _EMPTY_CACHE_EVERY == 0:
        torch.mps.empty_cache()


# ---------------------------------------------------------------------------
# Projection (port of project_gaussians_2d_scale_rot_forward_kernel)
# ---------------------------------------------------------------------------
def project_gaussians_2d_scale_rot(
    means2d: Tensor,
    scales2d: Tensor,
    rotation: Tensor,
    img_height: int,
    img_width: int,
    tile_bounds: Tuple[int, int, int],
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    num_points = means2d.shape[-2]
    if num_points < 1 or means2d.shape[-1] != 2:
        raise ValueError(f"Invalid shape for means2d: {means2d.shape}")

    rot = rotation.reshape(-1)
    c, s = torch.cos(rot), torch.sin(rot)
    sx2, sy2 = scales2d[:, 0] ** 2, scales2d[:, 1] ** 2
    # cov = R S S R^T with R = [[c, s], [-s, c]] (glm column-major in the CUDA code)
    cov_a = c * c * sx2 + s * s * sy2
    cov_b = c * s * (sy2 - sx2)
    cov_c = s * s * sx2 + c * c * sy2

    det = cov_a * cov_c - cov_b * cov_b
    ok = det != 0
    safe_det = torch.where(ok, det, torch.ones_like(det))
    inv_det = 1.0 / safe_det
    conics = torch.stack([cov_c * inv_det, -cov_b * inv_det, cov_a * inv_det], dim=-1)
    conics = torch.where(ok[:, None], conics, torch.zeros_like(conics))

    center = means2d * means2d.new_tensor([img_width, img_height])
    xys = torch.where(ok[:, None], center, torch.zeros_like(center))

    with torch.no_grad():
        b = 0.5 * (cov_a + cov_c)
        disc = torch.sqrt(torch.clamp(b * b - det, min=0.1))
        v = torch.maximum(b + disc, b - disc)
        radius = torch.ceil(3.0 * torch.sqrt(v))
        radii = torch.where(ok, radius, torch.zeros_like(radius)).to(torch.int32)

        tmin_x, tmax_x, tmin_y, tmax_y = _tile_bbox(center.detach(), radius, tile_bounds)
        area = (tmax_x - tmin_x) * (tmax_y - tmin_y)
        num_tiles_hit = torch.where(ok & (area > 0), area, torch.zeros_like(area)).to(torch.int32)

    return xys, radii, conics, num_tiles_hit


def _tile_bbox(center: Tensor, radius: Tensor, tile_bounds):
    """get_tile_bbox from helpers.cuh: inclusive min, exclusive max, clamped to the tile grid."""
    tcx, tcy = center[:, 0] / BLOCK_X, center[:, 1] / BLOCK_Y
    rx, ry = radius / BLOCK_X, radius / BLOCK_Y

    def clamp_trunc(x, hi):
        # (int) cast truncates toward zero, then clamps to [0, hi]
        return torch.clamp(torch.trunc(x), 0, hi).to(torch.int32)

    tmin_x = clamp_trunc(tcx - rx, tile_bounds[0])
    tmax_x = clamp_trunc(tcx + rx + 1, tile_bounds[0])
    tmin_y = clamp_trunc(tcy - ry, tile_bounds[1])
    tmax_y = clamp_trunc(tcy + ry + 1, tile_bounds[1])
    return tmin_x, tmax_x, tmin_y, tmax_y


# ---------------------------------------------------------------------------
# Binning (port of map_gaussian_to_intersects + sort + get_tile_bin_edges)
# ---------------------------------------------------------------------------
@torch.no_grad()
def bin_gaussians(xys: Tensor, radii: Tensor, num_tiles_hit: Tensor, tile_bounds) -> Tuple[Tensor, Tensor]:
    device = xys.device
    num_points = xys.shape[0]
    num_tiles = tile_bounds[0] * tile_bounds[1]

    counts = torch.where(radii > 0, num_tiles_hit, torch.zeros_like(num_tiles_hit)).long()
    tmin_x, tmax_x, tmin_y, _ = _tile_bbox(xys, radii.float(), tile_bounds)

    gid = torch.repeat_interleave(torch.arange(num_points, device=device), counts)
    if gid.numel() == 0:
        return torch.zeros(0, dtype=torch.int32, device=device), torch.zeros(num_tiles, 2, dtype=torch.int32, device=device)
    starts = torch.cumsum(counts, 0) - counts
    local = torch.arange(gid.numel(), device=device) - starts[gid]
    width = (tmax_x - tmin_x).long()[gid]
    tx = tmin_x.long()[gid] + local % width
    ty = tmin_y.long()[gid] + local // width
    tile_ids = ty * tile_bounds[0] + tx

    tile_sorted, order = torch.sort(tile_ids, stable=True)
    gaussian_ids_sorted = gid[order].to(torch.int32).contiguous()

    per_tile = torch.bincount(tile_sorted, minlength=num_tiles)
    ends = torch.cumsum(per_tile, 0)
    tile_bins = torch.stack([ends - per_tile, ends], dim=-1).to(torch.int32).contiguous()
    return gaussian_ids_sorted, tile_bins


# ---------------------------------------------------------------------------
# Rasterization autograd functions
# ---------------------------------------------------------------------------
def _check_channels(colors: Tensor):
    if colors.shape[-1] > MAX_CHANNELS:
        raise ValueError(f"MPS rasterizer supports at most {MAX_CHANNELS} channels, got {colors.shape[-1]}")


def _topk_forward(gaussian_ids_sorted, tile_bins, xys, conics, colors, img_h, img_w, tile_bounds, eps):
    device = xys.device
    channels = colors.shape[-1]
    out_img = torch.zeros(img_h, img_w, channels, dtype=torch.float32, device=device)
    pixel_topk = torch.zeros(img_h * img_w, TOP_K, dtype=torch.int32, device=device)
    _lib().nd_rasterize_forward_topk_norm(
        gaussian_ids_sorted, tile_bins, xys, conics, colors, out_img, pixel_topk,
        _iparams(device, img_w, img_h, tile_bounds[0], channels), _fparams(device, eps),
        threads=_tile_grid(tile_bounds), group_size=(BLOCK_X, BLOCK_Y),
    )
    return out_img, pixel_topk


def _topk_backward(gaussian_ids_sorted, tile_bins, xys, conics, colors, v_out, pixel_topk, img_h, img_w, tile_bounds, eps):
    device = xys.device
    channels = colors.shape[-1]
    v_xy = torch.zeros_like(xys)
    v_conic = torch.zeros_like(conics)
    v_colors = torch.zeros_like(colors)
    _lib().nd_rasterize_backward_topk_norm(
        gaussian_ids_sorted, tile_bins, xys, conics, colors, v_out.contiguous(), pixel_topk,
        v_xy, v_conic, v_colors,
        _iparams(device, img_w, img_h, tile_bounds[0], channels), _fparams(device, eps),
        threads=_tile_grid(tile_bounds), group_size=(BLOCK_X, BLOCK_Y),
    )
    return v_xy, v_conic, v_colors


class _RasterizeGaussiansSum(Function):
    @staticmethod
    def forward(ctx, xys, radii, conics, num_tiles_hit, colors, img_height, img_width, BLOCK_H=16, BLOCK_W=16, topk_norm=False):
        if (BLOCK_H, BLOCK_W) != (BLOCK_Y, BLOCK_X):
            raise ValueError("MPS rasterizer is hardcoded to 16x16 tiles")
        _check_channels(colors)
        tile_bounds = ((img_width + BLOCK_X - 1) // BLOCK_X, (img_height + BLOCK_Y - 1) // BLOCK_Y, 1)
        xys, conics, colors = xys.float().contiguous(), conics.float().contiguous(), colors.float().contiguous()
        gaussian_ids_sorted, tile_bins = bin_gaussians(xys, radii, num_tiles_hit, tile_bounds)

        ctx.img_height, ctx.img_width, ctx.topk_norm, ctx.tile_bounds = img_height, img_width, topk_norm, tile_bounds
        if topk_norm:
            out_img, pixel_topk = _topk_forward(gaussian_ids_sorted, tile_bins, xys, conics, colors,
                                                img_height, img_width, tile_bounds, EPS)
            ctx.save_for_backward(xys, conics, colors, gaussian_ids_sorted, tile_bins, pixel_topk)
        else:
            device = xys.device
            out_img = torch.zeros(img_height, img_width, colors.shape[-1], dtype=torch.float32, device=device)
            _lib().nd_rasterize_forward_sum(
                gaussian_ids_sorted, tile_bins, xys, conics, colors, out_img,
                _iparams(device, img_width, img_height, tile_bounds[0], colors.shape[-1]),
                threads=_tile_grid(tile_bounds), group_size=(BLOCK_X, BLOCK_Y),
            )
            ctx.save_for_backward(xys, conics, colors, gaussian_ids_sorted, tile_bins)
        return out_img

    @staticmethod
    def backward(ctx, v_out_img):
        h, w = ctx.img_height, ctx.img_width
        if ctx.topk_norm:
            xys, conics, colors, gaussian_ids_sorted, tile_bins, pixel_topk = ctx.saved_tensors
            v_xy, v_conic, v_colors = _topk_backward(gaussian_ids_sorted, tile_bins, xys, conics, colors, v_out_img,
                                                     pixel_topk, h, w, ctx.tile_bounds, EPS)
        else:
            xys, conics, colors, gaussian_ids_sorted, tile_bins = ctx.saved_tensors
            device = xys.device
            v_xy, v_conic, v_colors = torch.zeros_like(xys), torch.zeros_like(conics), torch.zeros_like(colors)
            _lib().nd_rasterize_backward_sum(
                gaussian_ids_sorted, tile_bins, xys, conics, colors, v_out_img.contiguous(),
                v_xy, v_conic, v_colors,
                _iparams(device, w, h, ctx.tile_bounds[0], colors.shape[-1]),
                threads=_tile_grid(ctx.tile_bounds), group_size=(BLOCK_X, BLOCK_Y),
            )
        return v_xy, None, v_conic, None, v_colors, None, None, None, None, None


class _RasterizeGaussiansNoTiles(Function):
    @staticmethod
    def forward(ctx, xys, conics, colors, img_height, img_width):
        _check_channels(colors)
        device = xys.device
        xys, conics, colors = xys.float().contiguous(), conics.float().contiguous(), colors.float().contiguous()
        tile_bounds = ((img_width + BLOCK_X - 1) // BLOCK_X, (img_height + BLOCK_Y - 1) // BLOCK_Y, 1)
        num_points = xys.shape[0]
        # Every tile sees every Gaussian, in index order: identical to the CUDA no-tiles loop.
        gaussian_ids = torch.arange(num_points, dtype=torch.int32, device=device)
        tile_bins = _iparams(device, 0, num_points).repeat(tile_bounds[0] * tile_bounds[1], 1).contiguous()
        out_img, pixel_topk = _topk_forward(gaussian_ids, tile_bins, xys, conics, colors,
                                            img_height, img_width, tile_bounds, EPS_NO_TILES)
        ctx.img_height, ctx.img_width, ctx.tile_bounds = img_height, img_width, tile_bounds
        ctx.save_for_backward(xys, conics, colors, gaussian_ids, tile_bins, pixel_topk)
        return out_img

    @staticmethod
    def backward(ctx, v_out_img):
        xys, conics, colors, gaussian_ids, tile_bins, pixel_topk = ctx.saved_tensors
        v_xy, v_conic, v_colors = _topk_backward(gaussian_ids, tile_bins, xys, conics, colors, v_out_img, pixel_topk,
                                                 ctx.img_height, ctx.img_width, ctx.tile_bounds, EPS_NO_TILES)
        return v_xy, v_conic, v_colors, None, None


def rasterize_gaussians_sum(xys, radii, conics, num_tiles_hit, colors, img_height, img_width,
                            BLOCK_H=16, BLOCK_W=16, topk_norm=False):
    _maybe_empty_cache()
    return _RasterizeGaussiansSum.apply(xys, radii, conics, num_tiles_hit, colors, img_height, img_width,
                                        BLOCK_H, BLOCK_W, topk_norm)


def rasterize_gaussians_no_tiles(xys, conics, colors, img_height, img_width):
    _maybe_empty_cache()
    return _RasterizeGaussiansNoTiles.apply(xys, conics, colors, img_height, img_width)

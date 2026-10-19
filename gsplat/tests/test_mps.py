"""Checks the MPS backend against a dense float64 PyTorch reference of the CUDA kernels.

Run: python gsplat/tests/test_mps.py
"""
import math

import torch

from gsplat import project_gaussians_2d_scale_rot, rasterize_gaussians_no_tiles, rasterize_gaussians_sum

TOP_K = 10


def reference(xy, scale, rot, feat, H, W, tiled, topk_norm, eps):
    c, s = torch.cos(rot[:, 0]), torch.sin(rot[:, 0])
    R = torch.stack([torch.stack([c, s], -1), torch.stack([-s, c], -1)], -2)
    S2 = torch.diag_embed(scale ** 2)
    cov = R @ S2 @ R.transpose(-1, -2)
    a, b, cc = cov[:, 0, 0], cov[:, 0, 1], cov[:, 1, 1]
    det = a * cc - b * b
    conic = torch.stack([cc / det, -b / det, a / det], -1)
    center = xy * torch.tensor([W, H], dtype=xy.dtype)

    py, px = torch.meshgrid(torch.arange(H, dtype=xy.dtype), torch.arange(W, dtype=xy.dtype), indexing="ij")
    pix = torch.stack([px, py], -1).reshape(-1, 2)
    d = center[None] - pix[:, None]
    sigma = 0.5 * (conic[:, 0] * d[..., 0] ** 2 + conic[:, 2] * d[..., 1] ** 2) + conic[:, 1] * d[..., 0] * d[..., 1]
    valid = (sigma >= 0) & torch.isfinite(sigma)

    if tiled:
        with torch.no_grad():
            bb = 0.5 * (a + cc)
            v = bb + torch.sqrt(torch.clamp(bb * bb - det, min=0.1))
            radius = torch.ceil(3 * torch.sqrt(v)).float()
            cf = center.float()
            tminx = torch.clamp(torch.trunc(cf[:, 0] / 16 - radius / 16), 0, math.ceil(W / 16))
            tmaxx = torch.clamp(torch.trunc(cf[:, 0] / 16 + radius / 16 + 1), 0, math.ceil(W / 16))
            tminy = torch.clamp(torch.trunc(cf[:, 1] / 16 - radius / 16), 0, math.ceil(H / 16))
            tmaxy = torch.clamp(torch.trunc(cf[:, 1] / 16 + radius / 16 + 1), 0, math.ceil(H / 16))
            ptx, pty = (pix[:, 0] // 16)[:, None], (pix[:, 1] // 16)[:, None]
            valid &= (ptx >= tminx) & (ptx < tmaxx) & (pty >= tminy) & (pty < tmaxy)

    alpha = torch.exp(-torch.where(valid, sigma, torch.zeros_like(sigma)))
    if not topk_norm:
        w = torch.where(valid, alpha, torch.zeros_like(alpha))
        return (w @ feat).reshape(H, W, -1)
    masked = torch.where(valid, alpha, torch.full_like(alpha, -1.0))
    _, idx = torch.topk(masked.detach(), TOP_K, dim=-1)
    vals = torch.gather(alpha, 1, idx)
    keep = torch.gather(valid, 1, idx)
    vals = torch.where(keep, vals, torch.zeros_like(vals))
    w = vals / (vals.sum(-1, keepdim=True) + eps)
    return (w[..., None] * feat[idx]).sum(1).reshape(H, W, -1)


def run_case(tiled, topk_norm, N=300, H=45, W=70, C=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    xy = torch.rand(N, 2, generator=g, dtype=torch.float64)
    scale = 1.0 + 6.0 * torch.rand(N, 2, generator=g, dtype=torch.float64)
    rot = (torch.rand(N, 1, generator=g, dtype=torch.float64) - 0.5) * 2 * math.pi
    feat = torch.rand(N, C, generator=g, dtype=torch.float64)
    v_out = torch.randn(H, W, C, generator=g, dtype=torch.float64)
    eps = 1e-4 if tiled else 1e-7

    ref_params = [p.clone().requires_grad_() for p in (xy, scale, rot, feat)]
    ref = reference(*ref_params, H, W, tiled, topk_norm, eps)
    (ref * v_out).sum().backward()

    mps_params = [p.float().to("mps").requires_grad_() for p in (xy, scale, rot, feat)]
    tile_bounds = ((W + 15) // 16, (H + 15) // 16, 1)
    xys, radii, conics, num_tiles_hit = project_gaussians_2d_scale_rot(mps_params[0], mps_params[1], mps_params[2], H, W, tile_bounds)
    if tiled:
        out = rasterize_gaussians_sum(xys, radii, conics, num_tiles_hit, mps_params[3], H, W, 16, 16, topk_norm)
    else:
        out = rasterize_gaussians_no_tiles(xys, conics, mps_params[3], H, W)
    (out * v_out.float().to("mps")).sum().backward()

    name = f"tiled={tiled} topk_norm={topk_norm}"
    ok = True
    for label, a, b in [("image", out.detach(), ref.detach())] + [
        (f"grad_{n}", p.grad, r.grad) for n, p, r in zip(("xy", "scale", "rot", "feat"), mps_params, ref_params)
    ]:
        a = a.cpu().double()
        err = (a - b).abs().max().item() / (b.abs().max().item() + 1e-12)
        status = "ok" if err < 2e-3 else "MISMATCH"
        ok &= status == "ok"
        print(f"  {name:28s} {label:11s} rel max err {err:.2e}  {status}")
    return ok


if __name__ == "__main__":
    results = [run_case(True, True), run_case(True, False), run_case(False, True)]
    print("ALL PASS" if all(results) else "FAILURES")

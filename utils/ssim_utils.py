import torch
import torch.nn.functional as F

_WINDOW_SIZE = 11
_SIGMA = 1.5


def _gaussian_1d(channels, device, dtype):
    coords = torch.arange(_WINDOW_SIZE, device=device, dtype=dtype) - _WINDOW_SIZE // 2
    g = torch.exp(-(coords ** 2) / (2 * _SIGMA ** 2))
    g = g / g.sum()
    return g.expand(channels, 1, 1, _WINDOW_SIZE).contiguous()


def fused_ssim(img1, img2, padding="same", train=True):
    """Drop-in PyTorch replacement for fused_ssim.fused_ssim (11x11 Gaussian window, sigma 1.5, zero padding)."""
    if padding not in ("same", "valid"):
        raise ValueError(f"Unsupported padding: {padding}")
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    channels = img1.shape[1]
    # All five blurs in one separable (horizontal then vertical) grouped conv.
    stacked = torch.cat([img1, img2, img1 * img1, img2 * img2, img1 * img2], dim=1)
    g = _gaussian_1d(stacked.shape[1], img1.device, img1.dtype)
    pad = _WINDOW_SIZE // 2
    blurred = F.conv2d(stacked, g, padding=(0, pad), groups=stacked.shape[1])
    blurred = F.conv2d(blurred, g.transpose(2, 3), padding=(pad, 0), groups=stacked.shape[1])
    mu1, mu2, e11, e22, e12 = blurred.split(channels, dim=1)

    mu1_sq, mu2_sq, mu12 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    sigma1_sq = e11 - mu1_sq
    sigma2_sq = e22 - mu2_sq
    sigma12 = e12 - mu12
    ssim_map = ((2 * mu12 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    if padding == "valid":
        ssim_map = ssim_map[:, :, 5:-5, 5:-5]
    return ssim_map.mean()

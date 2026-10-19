# Running Image-GS on Apple Silicon (MPS)

The CUDA rasterizer in `gsplat/` has a Metal port in `gsplat/gsplat/mps/`. The Metal kernels are compiled at runtime with `torch.mps.compile_shader`, so you don't need Xcode, a C++ toolchain, or CUDA.

## Setup

```bash
conda env create -f environment-mps.yml
conda activate image-gs-mps
BUILD_NO_CUDA=1 pip install -e ./gsplat --no-build-isolation --config-settings editable_mode=compat
```

`editable_mode=compat` matters here. Without it, running `main.py` from the repo root makes Python pick up the outer `gsplat/` folder as an empty namespace package, which shadows the installed one.

`fused-ssim` is optional. If it isn't installed, `model.py` falls back to `utils/ssim_utils.py`, a PyTorch implementation with the same output that runs about 4× slower. Building the upstream Metal version needs Xcode with its license accepted. Recent PyTorch headers (verified with 2.14) require C++20, so you also have to change its `-std=c++17` flag to `-std=c++20`:

```bash
git clone --depth 1 https://github.com/rahul-goel/fused-ssim.git
sed -i '' 's/"-std=c++17", "-ObjC++"/"-std=c++20", "-ObjC++"/' fused-ssim/setup.py
pip install ./fused-ssim --no-build-isolation
```

## Usage

Use the same commands as the main README. `device` defaults to `auto`, which picks `cuda:0`, then `mps`, then `cpu`.

```bash
python main.py --input_path="images/anime-1_2k.png" --exp_name="test/anime-1_2k" --num_gaussians=10000 --quantize
```

## What was ported

| CUDA (`gsplat/gsplat/cuda/csrc`) | MPS (`gsplat/gsplat/mps`) |
|---|---|
| `project_gaussians_2d_scale_rot_{forward,backward}` | PyTorch ops; autograd gives the same gradients |
| `map_gaussian_to_intersects`, sort, `get_tile_bin_edges` | PyTorch ops (`repeat_interleave`, stable sort, `bincount`) |
| `nd_rasterize_forward_topk_norm` | Metal, 16×16 threadgroups with shared-memory batches |
| `nd_rasterize_backward_topk_norm` | Metal. Each tile walks its bin, with `simd_sum` and one atomic per simdgroup |
| `nd_rasterize_forward` / `nd_rasterize_backward` (`--disable_topk_norm`) | Metal |
| `nd_rasterize_{forward,backward}_no_tiles` (`--disable_tiles`) | The top-K kernels, with every tile's bin covering all Gaussians |

`rasterize_gaussians_simple` isn't ported because `model.py` doesn't use it.

The rasterizer calls `torch.mps.empty_cache()` every 100 renders. The tile-binning tensors change size every step, and without these calls the MPS caching allocator holds onto a buffer for every size. On 2K texture stacks that grew past 40 GB and ran out of memory around step 3300.

Also expect slower stretches after each progressive Gaussian addition. Some newly added Gaussians briefly get very large radii, which can roughly double the tile intersections. The math is the same as on CUDA, but it costs more time here.

There's one deliberate difference from CUDA. In the CUDA sum backward (`--disable_topk_norm` only), one invalid Gaussian stops a pixel's gradients from all later Gaussians too. The Metal version skips only the invalid Gaussian. The forward output is the same.

## Verification

```bash
python gsplat/tests/test_mps.py
```

This compares the forward image and the gradients for xy, scale, rotation and features against a dense float64 PyTorch reference. It covers the tiled top-K, tiled sum, and no-tiles modes.

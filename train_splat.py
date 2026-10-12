#!/usr/bin/env python3
"""Train a Gaussian splat from infer_visit.py outputs and write it as a .splat (antimatter15 format).

    python train_splat.py --inp /work/out --out /work/out/scene.splat [--iters 7000]

Inputs: frames (keyframes/*.jpg, model resolution), poses.npz (c2w + K per frame, OpenCV axes),
cloud.npz (xyz, rgb, conf) as the starting Gaussians. gsplat (Apache-2.0) rasterises; its
DefaultStrategy grows and prunes. Colour is view-independent (SH degree 0): the .splat format
stores one RGBA per Gaussian anyway.

.splat: 32 bytes per Gaussian, most important first — xyz f32x3, scale f32x3, rgba u8x4,
quaternion u8x4 ((q * 128 + 128), w first). Capped at --max_gaussians (Storage files <= 50 MB).
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from gsplat import rasterization
from gsplat.strategy import DefaultStrategy

ap = argparse.ArgumentParser()
ap.add_argument("--inp", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--iters", type=int, default=7000)
ap.add_argument("--init_points", type=int, default=300_000, help="most confident cloud points to start from")
ap.add_argument("--max_gaussians", type=int, default=1_400_000, help="written to .splat (32 B each)")
a = ap.parse_args()
t0 = time.time()
dev = torch.device("cuda")
inp = Path(a.inp)


def log(*m):
    print(f"[splat {time.time() - t0:6.1f}s]", *m, flush=True)


# ---------------------------------------------------------------- data
P = np.load(inp / "poses.npz")
names = [str(n) for n in P["names"]]
imgs = torch.stack([torch.from_numpy(np.asarray(Image.open(inp / "keyframes" / n).convert("RGB")))
                    for n in names]).to(dev)                                  # [F,H,W,3] uint8
Fn, H, W = imgs.shape[:3]
viewmats = torch.from_numpy(P["w2c"]).float().to(dev)                       # [F,4,4]
Ks = torch.from_numpy(P["K"]).float().to(dev)                               # [F,3,3], at H x W
log(f"{Fn} frames at {W}x{H}")

C = np.load(inp / "cloud.npz")
xyz, rgb, conf = C["xyz"].astype(np.float32), C["rgb"].astype(np.float32) / 255, C["conf"].astype(np.float32)
keep = np.argsort(-conf)[: a.init_points]
xyz, rgb = torch.from_numpy(xyz[keep]).to(dev), torch.from_numpy(rgb[keep]).to(dev)
N = len(xyz)
extent = float(torch.quantile(torch.linalg.norm(xyz - xyz.median(0).values, dim=1), 0.95))
# initial size: mean distance to the 3 nearest neighbours (chunked, exact)
nn = torch.empty(N, device=dev)
for i in range(0, N, 512):          # 512 x N distances per chunk (~0.6 GB at 300k)
    d = torch.cdist(xyz[i:i + 512], xyz)
    nn[i:i + 512] = d.topk(4, largest=False).values[:, 1:].mean(1)
params = torch.nn.ParameterDict({
    "means": torch.nn.Parameter(xyz),
    "scales": torch.nn.Parameter(torch.log(nn.clamp(min=1e-4))[:, None].repeat(1, 3)),
    "quats": torch.nn.Parameter(torch.tensor([1.0, 0, 0, 0], device=dev).repeat(N, 1)),
    "opacities": torch.nn.Parameter(torch.logit(torch.full((N,), 0.1, device=dev))),
    "colors": torch.nn.Parameter(torch.logit(rgb.clamp(0.02, 0.98))),
})
lrs = {"means": 1.6e-4 * extent, "scales": 5e-3, "quats": 1e-3, "opacities": 5e-2, "colors": 2.5e-3}
optimizers = {k: torch.optim.Adam([{"params": params[k], "lr": lrs[k], "name": k}], eps=1e-15) for k in params}
strategy = DefaultStrategy(refine_start_iter=500, refine_stop_iter=int(a.iters * 0.7), reset_every=3000,
                           refine_every=100, verbose=False)
strategy.check_sanity(params, optimizers)
state = strategy.initialize_state(scene_scale=extent)
log(f"{N} starting Gaussians, scene extent {extent:.2f}")


def ssim(x, y):
    """x, y: [1,3,H,W] in [0,1]; 11x11 Gaussian window."""
    g = torch.exp(-((torch.arange(11, device=dev) - 5.0) ** 2) / (2 * 1.5 ** 2))
    w = (g[:, None] * g[None, :] / g.sum() ** 2).expand(3, 1, 11, 11)
    f = lambda t: F.conv2d(t, w, padding=5, groups=3)
    mx, my = f(x), f(y)
    sx, sy, sxy = f(x * x) - mx ** 2, f(y * y) - my ** 2, f(x * y) - mx * my
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    return (((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx ** 2 + my ** 2 + c1) * (sx + sy + c2))).mean()


# ---------------------------------------------------------------- train
mean_decay = 0.01 ** (1 / a.iters)   # means lr decays 100x over the run
for step in range(a.iters):
    i = int(torch.randint(Fn, (1,)))
    gt = imgs[i].float()[None] / 255
    render, alpha, info = rasterization(
        params["means"], F.normalize(params["quats"], dim=-1), torch.exp(params["scales"]),
        torch.sigmoid(params["opacities"]), torch.sigmoid(params["colors"]),
        viewmats[i:i + 1], Ks[i:i + 1], W, H, packed=False)
    strategy.step_pre_backward(params, optimizers, state, step, info)
    l1 = (render - gt).abs().mean()
    loss = 0.8 * l1 + 0.2 * (1 - ssim(render.permute(0, 3, 1, 2), gt.permute(0, 3, 1, 2)))
    loss.backward()
    for o in optimizers.values():
        o.step()
        o.zero_grad(set_to_none=True)
    for g in optimizers["means"].param_groups:
        g["lr"] *= mean_decay
    strategy.step_post_backward(params, optimizers, state, step, info, packed=False)
    if step % 1000 == 0 or step == a.iters - 1:
        log(f"step {step} l1 {l1.item():.4f} gaussians {len(params['means'])}")

# ---------------------------------------------------------------- export .splat
with torch.no_grad():
    s = torch.exp(params["scales"])
    o = torch.sigmoid(params["opacities"])
    order = torch.argsort(-(o * s.prod(1)), descending=False)[: a.max_gaussians]   # big, opaque first
    means = params["means"][order].cpu().numpy().astype(np.float32)
    scales = s[order].cpu().numpy().astype(np.float32)
    rgba = torch.cat([torch.sigmoid(params["colors"][order]), o[order, None]], 1)
    rgba = (rgba * 255).clamp(0, 255).byte().cpu().numpy()
    q = F.normalize(params["quats"][order], dim=-1)
    q = ((q * 128) + 128).clamp(0, 255).byte().cpu().numpy()
rec = np.zeros(len(means), dtype=[("p", "<f4", 3), ("s", "<f4", 3), ("c", "u1", 4), ("q", "u1", 4)])
rec["p"], rec["s"], rec["c"], rec["q"] = means, scales, rgba, q
Path(a.out).write_bytes(rec.tobytes())
# the walk-through opens where the cleaner started (viewer takes a column-major world-to-camera)
(Path(a.out).parent / "splat_start.json").write_text(json.dumps(P["w2c"][0].T.flatten().round(5).tolist()))
log(f"wrote {a.out}: {len(means)} Gaussians, {Path(a.out).stat().st_size / 1e6:.1f} MB")
(Path(a.out).parent / "SPLAT_DONE").touch()

#!/usr/bin/env python3
"""Train a Gaussian splat from infer_visit.py outputs and write it as a .splat (antimatter15 format).

    python train_splat.py --inp /work/out --out /work/out/scene.splat [--iters 7000]

Inputs: lingbot's keyframes (keyframes/*.jpg at full resolution; the other frames are model-size and
unused), poses.npz (w2c + K per frame, OpenCV axes), cloud.npz (xyz, rgb, conf) as the starting Gaussians.
Each frame's pose and exposure are refined while training: lingbot estimated poses under its own
(wrong, non-square) focal, and a few pixels of disagreement between frames trains into blobs. gsplat (Apache-2.0) rasterises; its
DefaultStrategy grows and prunes. Colour is view-independent (SH degree 0): the .splat format
stores one RGBA per Gaussian anyway.

.splat: 32 bytes per Gaussian, most important first — xyz f32x3, scale f32x3, rgba u8x4,
quaternion u8x4 ((q * 128 + 128), w first). Capped at --max_gaussians (Storage files <= 50 MB).
"""
import argparse
import json
import math
import os
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
ap.add_argument("--iters", type=int, default=30000)
ap.add_argument("--init_points", type=int, default=300_000, help="cloud points to start from (uniform sample)")
ap.add_argument("--long_side", type=int, default=960, help="training resolution (keyframes are 1920)")
ap.add_argument("--pose_lr", type=float, default=1e-3, help="per-frame pose refinement (rad; x extent for translation)")
ap.add_argument("--max_gaussians", type=int, default=1_400_000, help="written to .splat (32 B each)")
ap.add_argument("--progressive", action="store_true",
                help="train the calmest, sharpest stretch first, then add neighbours a few at a time, each pose "
                     "locked onto the scene before it may change it; frames that never fit are dropped")
ap.add_argument("--seed_frames", type=int, default=30, help="keyframes in the starting stretch")
ap.add_argument("--seed_iters", type=int, default=3000, help="steps on the starting stretch alone")
ap.add_argument("--grow_every", type=int, default=200, help="steps between additions")
ap.add_argument("--grow_by", type=int, default=2, help="frames added on each side per addition")
ap.add_argument("--locate_iters", type=int, default=60, help="pose-only steps for each new frame")
ap.add_argument("--reject", type=float, default=1.8, help="drop a new frame whose l1 is this x the active median")
a = ap.parse_args()
t0 = time.time()
dev = torch.device("cuda")
inp = Path(a.inp)
(Path(a.out).parent / "splat.pid").write_text(str(os.getpid()))   # worker.py stops a run that overruns


def log(*m):
    print(f"[splat {time.time() - t0:6.1f}s]", *m, flush=True)


# ---------------------------------------------------------------- data
P = np.load(inp / "poses.npz")
names = [str(n) for n in P["names"]]
mh, mw = json.loads((inp / "meta.json").read_text())["image_hw"]   # model resolution, what K is for
sizes = [Image.open(inp / "keyframes" / n).size for n in names]
sw, sh = max(sizes)                                                 # source resolution (lingbot keyframes)
sel = [i for i, s in enumerate(sizes) if s == (sw, sh)]
s = a.long_side / max(sw, sh)
W, H = round(sw * s), round(sh * s)
imgs = torch.stack([torch.from_numpy(np.array(Image.open(inp / "keyframes" / names[i]).convert("RGB").resize((W, H), Image.LANCZOS)))
                    for i in sel]).to(dev)                                    # [F,H,W,3] uint8
Fn = len(imgs)
viewmats = torch.from_numpy(P["w2c"][sel]).float().to(dev)                  # [F,4,4]
Ks = torch.from_numpy(P["K"][sel]).float().to(dev)                          # [F,3,3] at mw x mh: rescale per axis
Ks[:, 0] *= W / mw                                                          # (model frames are a plain resize)
Ks[:, 1] *= H / mh
log(f"{Fn} keyframes of {len(names)} at {W}x{H}")

C = np.load(inp / "cloud.npz")
xyz, rgb, conf = C["xyz"].astype(np.float32), C["rgb"].astype(np.float32) / 255, C["conf"].astype(np.float32)
# uniform, not most-confident: lingbot is surest on near textured surfaces, and plain walls/floors
# (what turnover QA looks at) would start empty; the cloud is already filtered at conf > 2.5
keep = np.random.default_rng(0).choice(len(xyz), min(a.init_points, len(xyz)), replace=False)
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
# per-frame corrections, zero = lingbot's pose / unchanged colour; frame 0 stays fixed (anchors the world)
pose_d = torch.zeros(Fn, 6, device=dev, requires_grad=True)                # rotation (axis-angle), translation
expo = torch.zeros(Fn, 2, 3, device=dev, requires_grad=True)               # log gain, bias per channel
extra_opt = torch.optim.Adam([{"params": [pose_d], "lr": a.pose_lr}, {"params": [expo], "lr": 1e-3}])
anchor = torch.ones(Fn, 1, device=dev)
active = list(range(Fn))
if a.progressive:
    # sharpness (variance of the Laplacian) and camera turn per keyframe; the seed is the window with the best
    # sum of z(sharp) - z(turn): slow and sharp is where lingbot's poses are most likely self-consistent
    with torch.no_grad():
        lap = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], device=dev, dtype=torch.float32)[None, None]
        sharp = torch.stack([F.conv2d(imgs[i].float().mean(-1)[None, None], lap).var() for i in range(Fn)]).cpu().numpy()
    R = P["w2c"][sel][:, :3, :3]
    step_deg = np.degrees(np.arccos(np.clip((np.einsum("nij,nij->n", R[1:], R[:-1]) - 1) / 2, -1, 1)))
    turn = np.maximum(np.r_[step_deg[:1], step_deg], np.r_[step_deg, step_deg[-1:]])
    z = lambda v: (v - v.mean()) / (v.std() + 1e-9)
    score = z(np.log(sharp)) - z(turn)
    k = min(a.seed_frames, Fn)
    s0 = int(np.argmax(np.convolve(score, np.ones(k), "valid")))
    active = list(range(s0, s0 + k))
    lo, hi = s0 - 1, s0 + k                 # next frames to add on either side
    log(f"seed keyframes {s0}..{s0 + k - 1}: turn median {np.median(turn[s0:s0 + k]):.2f} deg/keyframe "
        f"(all {np.median(turn):.2f}), sharpness {np.median(sharp[s0:s0 + k]):.0f} (all {np.median(sharp):.0f})")
    anchor[s0 + k // 2] = 0                 # the middle of the seed anchors the world
else:
    anchor[0] = 0
rejected = []


def refined(i):
    d = pose_d[i] * anchor[i]
    w = d[:3]
    skew = torch.zeros(3, 3, device=dev)
    skew[0, 1], skew[0, 2], skew[1, 2] = -w[2], w[1], -w[0]
    skew = skew - skew.T
    delta = torch.eye(4, device=dev)
    delta[:3, :3] = torch.linalg.matrix_exp(skew)
    delta[:3, 3] = d[3:] * extent
    return delta @ viewmats[i]


lrs = {"means": 1.6e-4 * extent, "scales": 5e-3, "quats": 1e-3, "opacities": 5e-2, "colors": 2.5e-3}
optimizers = {k: torch.optim.Adam([{"params": params[k], "lr": lrs[k], "name": k}], eps=1e-15) for k in params}
strategy = DefaultStrategy(refine_start_iter=500, refine_stop_iter=int(a.iters * (0.75 if a.progressive else 0.5)), reset_every=3000,
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


def render_frame(i):
    render, alpha, info = rasterization(
        params["means"], F.normalize(params["quats"], dim=-1), torch.exp(params["scales"]),
        torch.sigmoid(params["opacities"]), torch.sigmoid(params["colors"]),
        refined(i)[None], Ks[i:i + 1], W, H, packed=False)
    return render * torch.exp(expo[i, 0]) + expo[i, 1], info


def frame_l1(i):
    with torch.no_grad():
        return (render_frame(i)[0] - imgs[i].float()[None] / 255).abs().mean().item()


def locate(i):
    """Fit frame i's pose and exposure to the current scene (scene frozen), starting from its neighbour's
    correction; keep it only if it then matches about as well as the frames already in."""
    nb = i + 1 if i < active[0] else i - 1
    with torch.no_grad():
        pose_d[i] = pose_d[nb]
        expo[i] = expo[nb]
    opt = torch.optim.Adam([{"params": [pose_d], "lr": 2e-3}, {"params": [expo], "lr": 1e-2}])
    for _ in range(a.locate_iters):
        r, _ = render_frame(i)
        gt = imgs[i].float()[None] / 255
        loss = 0.8 * (r - gt).abs().mean() + 0.2 * (1 - ssim(r.permute(0, 3, 1, 2), gt.permute(0, 3, 1, 2)))
        loss.backward()
        pose_d.grad[torch.arange(Fn, device=dev) != i] = 0   # only this frame moves
        expo.grad[torch.arange(Fn, device=dev) != i] = 0
        opt.step()
        opt.zero_grad(set_to_none=True)
    for p in params.values():
        p.grad = None
    l1, ref = frame_l1(i), float(np.median([frame_l1(j) for j in active[::max(1, len(active) // 20)]]))
    if l1 > a.reject * ref:
        rejected.append(i)
        return False
    active.append(i)
    return True


# ---------------------------------------------------------------- train
mean_decay = 0.01 ** (1 / a.iters)   # means and pose lr decay 100x over the run
l1_sum = 0.0
for step in range(a.iters):
    if a.progressive and step == a.seed_iters:
        log(f"seed alone after {step} steps: l1 {np.mean([frame_l1(j) for j in active]):.4f} "
            f"(if this is not near 0.02, the footage, not the poses, is the limit)")
    if a.progressive and step >= a.seed_iters and (step - a.seed_iters) % a.grow_every == 0 and (lo >= 0 or hi < Fn):
        for _ in range(a.grow_by):
            if hi < Fn:
                locate(hi)
                hi += 1
            if lo >= 0:
                locate(lo)
                lo -= 1
        if lo < 0 and hi >= Fn:
            log(f"step {step}: all frames placed, {len(active)} kept, {len(rejected)} dropped")
    i = active[int(torch.randint(len(active), (1,)))]
    gt = imgs[i].float()[None] / 255
    render, info = render_frame(i)
    strategy.step_pre_backward(params, optimizers, state, step, info)
    l1 = (render - gt).abs().mean()
    loss = 0.8 * l1 + 0.2 * (1 - ssim(render.permute(0, 3, 1, 2), gt.permute(0, 3, 1, 2)))
    loss.backward()
    for o in [*optimizers.values(), extra_opt]:
        o.step()
        o.zero_grad(set_to_none=True)
    for g in [*optimizers["means"].param_groups, extra_opt.param_groups[0]]:
        g["lr"] *= mean_decay
    l1_sum += l1.item()
    strategy.step_post_backward(params, optimizers, state, step, info, packed=False)
    if step < strategy.refine_stop_iter and len(params["means"]) > a.max_gaussians:
        strategy.refine_stop_iter = step   # DefaultStrategy grows without a cap: stop at what the export keeps
        log(f"step {step}: {len(params['means'])} Gaussians, densification stopped at the export cap")
    if (step + 1) % 1000 == 0:   # mean over the last 1000 steps (one frame's l1 is noise); ~0.02 is a good splat
        log(f"step {step + 1} l1 {l1_sum / 1000:.4f} gaussians {len(params['means'])}"
            + (f" frames {len(active)}/{Fn} dropped {len(rejected)}" if a.progressive else ""))
        l1_sum = 0.0
with torch.no_grad():
    d = pose_d[active].abs()
    if a.progressive:   # comparable with the plain run's l1: every kept frame, final scene
        log(f"final l1 over {len(active)} kept frames {np.mean([frame_l1(j) for j in active]):.4f}; dropped {sorted(rejected)}")
    log(f"pose refinement: median {d[:, :3].median().item() * 57.3:.2f} deg / {d[:, 3:].median().item() * extent:.3f}, "
        f"max {d[:, :3].max().item() * 57.3:.2f} deg / {d[:, 3:].max().item() * extent:.3f} (world units)")

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
(Path(a.out).parent / "splat_start.json").write_text(json.dumps(P["w2c"][sel[0]].T.flatten().round(5).tolist()))
log(f"wrote {a.out}: {len(means)} Gaussians, {Path(a.out).stat().st_size / 1e6:.1f} MB")
(Path(a.out).parent / "SPLAT_DONE").touch()

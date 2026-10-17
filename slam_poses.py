#!/usr/bin/env python3
"""Replace lingbot's poses with DROID-SLAM's (BSD-3) and rebuild the cloud from lingbot's depth under them.

    python slam_poses.py --inp /work/out --weights /opt/ckpt/droid.pth

Test of whether a SLAM front end (dense BA over every frame, proximity edges = loop closure) gives
splat-grade poses where lingbot's feed-forward ones (median 1.3 deg, max 13 deg off) do not.
DROID runs on every processed frame at ~384x512 area with lingbot's square-pixel K. Its trajectory is
Sim(3)-aligned to lingbot's camera centres so lingbot's metric depth still fits. Keeps poses_lingbot.npz.
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np
import torch

sys.path.append("/opt/droid/droid_slam")
from droid import Droid                     # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--inp", required=True)
ap.add_argument("--weights", required=True)
ap.add_argument("--vis_threshold", type=float, default=2.5)   # infer_visit.py's cloud settings
ap.add_argument("--max_depth", type=float, default=10.0)
ap.add_argument("--cloud_step", type=int, default=2)
ap.add_argument("--cloud_max_points", type=int, default=2_000_000)
a = ap.parse_args()
t0 = time.time()


def log(*m):
    print(f"[slam {time.time() - t0:6.1f}s]", *m, flush=True)


P = dict(np.load(os.path.join(a.inp, "poses.npz")))
names, K = [str(n) for n in P["names"]], P["K"]
N = len(names)
D =np.load(os.path.join(a.inp, "depth.npz"))
depth, conf = D["depth"].astype(np.float32), D["conf"].astype(np.float32)
H, W = depth.shape[1:]                                       # model resolution, what K is for
h1, w1 = int(H * np.sqrt(384 * 512 / (H * W))), int(W * np.sqrt(384 * 512 / (H * W)))
h1, w1 = h1 - h1 % 8, w1 - w1 % 8


def stream():
    for t, n in enumerate(names):
        im = cv2.resize(cv2.imread(os.path.join(a.inp, "keyframes", n)), (W, H), interpolation=cv2.INTER_AREA)
        im = cv2.resize(im, (w1, h1), interpolation=cv2.INTER_AREA)
        k = K[t]
        yield t, torch.as_tensor(im).permute(2, 0, 1)[None], torch.tensor(
            [k[0, 0] * w1 / W, k[1, 1] * h1 / H, k[0, 2] * w1 / W, k[1, 2] * h1 / H], dtype=torch.float32)


args = argparse.Namespace(weights=a.weights, buffer=1024, image_size=[h1, w1], disable_vis=True, stereo=False,
                          beta=0.3, filter_thresh=2.4, warmup=8, keyframe_thresh=4.0, frontend_thresh=16.0,
                          frontend_window=25, frontend_radius=2, frontend_nms=1, backend_thresh=22.0,
                          backend_radius=2, backend_nms=3, upsample=False, asynchronous=False,
                          frontend_device="cuda", backend_device="cuda")
torch.multiprocessing.set_start_method("spawn")
droid = None
for t, im, intr in stream():
    if droid is None:
        droid = Droid(args)
    droid.track(t, im, intrinsics=intr)
log(f"tracked {N} frames at {w1}x{h1}; {droid.video.counter.value} keyframes")
traj = droid.terminate(stream())                             # [N,7] c2w: tx ty tz qx qy qz qw
log(f"global BA + fill done; peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
assert len(traj) == N, (len(traj), N)

Rd = Rotation.from_quat(traj[:, 3:]).as_matrix()
td = traj[:, :3]
# Sim(3) (Umeyama) taking DROID's camera centres onto lingbot's: keeps lingbot's scale and world frame
tl = P["c2w"][:, :3, 3]
mu_d, mu_l = td.mean(0), tl.mean(0)
U, S, Vt = np.linalg.svd((tl - mu_l).T @ (td - mu_d) / N)
E = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
R = U @ E @ Vt
s = (S * np.diag(E)).sum() / ((td - mu_d) ** 2).sum(1).mean()
c2w = np.tile(np.eye(4, dtype=np.float32), (N, 1, 1))
c2w[:, :3, :3] = R @ Rd
c2w[:, :3, 3] = (s * (R @ td.T)).T + (mu_l - s * R @ mu_d)
res = np.linalg.norm(c2w[:, :3, 3] - tl, axis=1)
ang = np.degrees(np.arccos(np.clip((np.einsum("nij,nij->n", c2w[:, :3, :3], P["c2w"][:, :3, :3]) - 1) / 2, -1, 1)))
ext = np.linalg.norm(tl - np.median(tl, 0), axis=1).max()
log(f"DROID vs lingbot after Sim3: centres median {np.median(res):.3f} / max {res.max():.3f} (cams span {ext:.2f}); "
    f"rotation median {np.median(ang):.2f} / p95 {np.percentile(ang, 95):.2f} / max {ang.max():.2f} deg")

os.replace(os.path.join(a.inp, "poses.npz"), os.path.join(a.inp, "poses_lingbot.npz"))
P["c2w"], P["w2c"] = c2w, np.linalg.inv(c2w).astype(np.float32)
np.savez(os.path.join(a.inp, "poses.npz"), **P)

# cloud: lingbot's depth, DROID's poses (same as infer_visit.py's)
v, u = np.mgrid[0:H, 0:W]
xyz, rgb, cf = [], [], []
for i in range(0, N, a.cloud_step):
    m = (conf[i] > a.vis_threshold) & (depth[i] > 0) & (depth[i] < a.max_depth)
    z = depth[i][m]
    pc = np.stack([(u[m] - K[i, 0, 2]) * z / K[i, 0, 0], (v[m] - K[i, 1, 2]) * z / K[i, 1, 1], z, np.ones_like(z)], 1) @ c2w[i].T
    im = cv2.cvtColor(cv2.resize(cv2.imread(os.path.join(a.inp, "keyframes", names[i])), (W, H), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    xyz.append(pc[:, :3].astype(np.float32))
    rgb.append(im[m])
    cf.append(conf[i][m])
xyz, rgb, cf = map(np.concatenate, (xyz, rgb, cf))
if len(xyz) > a.cloud_max_points:
    sel = np.sort(np.random.default_rng(0).choice(len(xyz), a.cloud_max_points, replace=False))
    xyz, rgb, cf = xyz[sel], rgb[sel], cf[sel]
os.replace(os.path.join(a.inp, "cloud.npz"), os.path.join(a.inp, "cloud_lingbot.npz"))
np.savez_compressed(os.path.join(a.inp, "cloud.npz"), xyz=xyz, rgb=rgb, conf=cf, cams=c2w[:, :3, 3])
log(f"cloud {len(xyz)} points; done")

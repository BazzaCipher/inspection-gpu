#!/usr/bin/env python
"""Headless lingbot-map streaming inference for ONE room clip.

Replicates demo_render/batch_demo.py's streaming path without importing demo.py or anything
under lingbot_map.vis / demo_render (those pull viser/kaolin/open3d). Verified recipe
(2026-10-08, IMG_3035): dense frames, streaming mode, keyframe_interval = ceil(N/320).

usage:
  infer_visit.py --video in.mov --out /work/out [--fps 10] [--max_frames 960]
  infer_visit.py --frames dir  --out /work/out

Outputs in --out:
  keyframes/%06d.jpg   every processed frame: keyframes at --scale_w (full), in-between at model res
  poses.npz            c2w [N,4,4], w2c [N,4,4], K [N,3,3], is_keyframe [N], frame_type [N], names
  depth.npz            depth [N,H,W] f16, conf [N,H,W] f16
  cloud.npz            xyz f32, rgb u8, conf f32, cams f32  (for build_explorer.py)
  meta.json            timings, gpu, frame counts, settings
  DONE                 marker written last

Pose convention: pose_encoding_to_extri_intri() output, inverted, is exactly what batch_demo
saves as `extrinsic`, and that array behaves as W2C (the repo's own renderer inverts it:
demo_render/rgbd_render/data/loader.py:93-106). So: w2c = inv(decoded), c2w = inv(w2c).
"""
import argparse
import glob
import json
import math
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import torch
from PIL import Image

t_start = time.time()
ap = argparse.ArgumentParser()
src = ap.add_mutually_exclusive_group(required=True)
src.add_argument("--video")
src.add_argument("--frames")
ap.add_argument("--out", required=True)
ap.add_argument("--fps", type=float, default=10.0, help="extraction rate from --video")
ap.add_argument("--max_frames", type=int, default=960, help="cap after extraction (uniform subsample)")
ap.add_argument("--scale_w", type=int, default=1440, help="ffmpeg scale width; keyframes are kept at this size for crops (model input is 518 regardless)")
ap.add_argument("--model_path", default="/opt/ckpt/lingbot-map.pt")
ap.add_argument("--keyframe_interval", type=int, default=0, help="0 = ceil(N/320)")
ap.add_argument("--num_scale_frames", type=int, default=8)
ap.add_argument("--camera_num_iterations", type=int, default=4)
ap.add_argument("--vis_threshold", type=float, default=2.5, help="depth_conf cut for cloud.npz")
ap.add_argument("--max_depth", type=float, default=10.0)
ap.add_argument("--long_side", type=int, default=728, help="model input long side (multiple of 14); the "
                "whole frame is kept; 728 (portrait 406x728) beat 518 on confidence and drift, 2026-10-12")
ap.add_argument("--cloud_max_points", type=int, default=2_000_000)
ap.add_argument("--cloud_step", type=int, default=2, help="use every Nth frame for cloud.npz")
args = ap.parse_args()

out = args.out
os.makedirs(out, exist_ok=True)
for stale in ("DONE", "FAILED"):
    p = os.path.join(out, stale)
    if os.path.exists(p):
        os.remove(p)


def log(*a):
    print(f"[infer {time.time() - t_start:7.1f}s]", *a, flush=True)


# ---------------------------------------------------------------- frames
if args.video:
    frames_dir = os.path.join(out, "frames_raw")
    shutil.rmtree(frames_dir, ignore_errors=True)
    os.makedirs(frames_dir)
    cmd = ["ffmpeg", "-loglevel", "error", "-i", args.video,
           "-vf", f"fps={args.fps},scale='min({args.scale_w},iw)':-2", "-q:v", "2",   # never upscale a 720p clip
           os.path.join(frames_dir, "%06d.jpg")]
    log("extracting:", " ".join(cmd))
    subprocess.run(cmd, check=True)
else:
    frames_dir = args.frames

paths = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")) + glob.glob(os.path.join(frames_dir, "*.png")))
if not paths:
    sys.exit(f"no frames in {frames_dir}")
if len(paths) > args.max_frames:
    idx = np.linspace(0, len(paths) - 1, args.max_frames).round().astype(int)
    paths = [paths[i] for i in idx]
N = len(paths)
kf_interval = args.keyframe_interval or max(1, math.ceil(N / 320))
log(f"{N} frames, keyframe_interval={kf_interval} (~{math.ceil(N / kf_interval)} keyframes in KV cache)")

# ---------------------------------------------------------------- model
sys.path.insert(0, "/opt/lingbot-map")
from lingbot_map.models.gct_stream import GCTStream  # noqa: E402
from lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri  # noqa: E402

try:
    import flashinfer  # noqa: F401
    use_sdpa = False
except Exception as e:  # pragma: no cover
    log(f"flashinfer unavailable ({e.__class__.__name__}); using SDPA")
    use_sdpa = True

device = torch.device("cuda")
cap = torch.cuda.get_device_capability()
amp_dtype = torch.bfloat16 if cap[0] >= 8 else torch.float16
gpu_name = torch.cuda.get_device_name()
log(f"gpu={gpu_name} capability={cap} amp={amp_dtype} sdpa={use_sdpa}")

# mirrors demo.load_model(); batch_demo uses the same constructor
model = GCTStream(
    img_size=518, patch_size=14, enable_3d_rope=True, max_frame_num=1024,
    kv_cache_sliding_window=64, kv_cache_scale_frames=args.num_scale_frames,
    kv_cache_cross_frame_special=True, kv_cache_include_scale_frames=True,
    use_sdpa=use_sdpa, camera_num_iterations=args.camera_num_iterations,
)
ckpt = torch.load(args.model_path, map_location="cpu", weights_only=False)
missing, unexpected = model.load_state_dict(ckpt.get("model", ckpt), strict=False)
log(f"checkpoint loaded; missing={len(missing)} unexpected={len(unexpected)}")
del ckpt
model = model.to(device).eval()
# demo.py casts the aggregator to the AMP dtype and keeps the heads fp32 (saves ~2-3 GB)
model.aggregator = model.aggregator.to(dtype=amp_dtype)

# ---------------------------------------------------------------- inference
# Whole frame, long side 518, no crop: lingbot's "crop" mode centre-cropped portrait phone video
# to a square and threw away the top and bottom 44% (floor and ceiling). Portrait also costs 57% of
# the tokens of 518x518, so it is faster and lighter than before.
def fit(path):
    with Image.open(path) as im:
        w, h = im.size
        s = args.long_side / max(w, h)
        size = (max(14, round(w * s / 14) * 14), max(14, round(h * s / 14) * 14))
        return torch.from_numpy(np.asarray(im.convert("RGB").resize(size, Image.BICUBIC), dtype=np.float32) / 255).permute(2, 0, 1)
images = torch.stack([fit(p) for p in paths])  # [S,3,H,W] in [0,1]
H, W = images.shape[-2:]
images = images.pin_memory()
log(f"preprocessed to {W}x{H}; running streaming inference")


# lingbot guesses the FoV along with the pose and refines both over camera_num_iterations rounds, each
# round conditioned on the previous pose. On portrait video it guesses a near-square FoV (see K below),
# so rotations were refined (and streamed forward) under the wrong lens. Pin square pixels, from its
# short-side FoV, on the pose fed back each round: poses are then estimated under the real lens.
def pin_square_pixels(head, inp):
    x = inp[0]
    if x.data_ptr() == head.empty_pose_tokens.data_ptr():     # round 1: learned empty pose, leave it
        return None
    x = x.clone()                                             # [.., 9]: T(3) quat(4) fov_h fov_w
    if W <= H:
        x[..., 7] = 2 * torch.atan(torch.tan(x[..., 8] / 2) * H / W)
    else:
        x[..., 8] = 2 * torch.atan(torch.tan(x[..., 7] / 2) * W / H)
    return (x,)


cam = model.camera_head
cam.embed_pose.register_forward_pre_hook(lambda m, inp: pin_square_pixels(cam, inp))
t0 = time.time()
with torch.amp.autocast("cuda", dtype=amp_dtype):
    pred = model.inference_streaming(
        images, num_scale_frames=args.num_scale_frames,
        keyframe_interval=kf_interval, output_device=torch.device("cpu"),
    )
t_infer = time.time() - t0
log(f"inference {t_infer:.1f}s = {N / t_infer:.2f} FPS; peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")

pose_enc = pred["pose_enc"].float()                       # [1,S,9]
decoded, K = pose_encoding_to_extri_intri(pose_enc, (H, W))   # [1,S,3,4], [1,S,3,3]
decoded4 = torch.eye(4).repeat(1, N, 1, 1)
decoded4[..., :3, :4] = decoded.cpu()
w2c = torch.linalg.inv(decoded4)                          # == batch_demo's saved `extrinsic`
c2w = torch.linalg.inv(w2c)
w2c, c2w, K = w2c[0].numpy(), c2w[0].numpy(), K[0].cpu().numpy()
# lingbot predicts near-square FOVs on portrait video (44 x 47 deg), so fy came out 1.7x fx and the
# cloud was squashed vertically. Pixels are square, and the short-side focal is the right one: it
# matched the iPhone's own metadata (26 mm equiv -> 500.9 px at 406x728; lingbot 502).
f = (K[:, 0, 0] if W <= H else K[:, 1, 1]).copy()
K[:, 0, 0] = K[:, 1, 1] = f
depth = pred["depth"][0, ..., 0].float().numpy()          # [S,H,W]
conf = pred["depth_conf"][0].float().numpy()              # [S,H,W]
is_kf = pred["is_keyframe"][0].cpu().numpy().astype(bool)
frame_type = pred["frame_type"][0].cpu().numpy().astype(np.uint8)
del pred
torch.cuda.empty_cache()

# ---------------------------------------------------------------- outputs
kf_dir = os.path.join(out, "keyframes")
shutil.rmtree(kf_dir, ignore_errors=True)
os.makedirs(kf_dir)
names = []
imgs = (images.numpy().transpose(0, 2, 3, 1) * 255).round().astype(np.uint8)
# Keyframes keep the full extracted frame (pairing, VLM triage and the before/after crops read
# these; at model resolution every crop was a thumbnail). In-between frames stay at model
# resolution. Findings are anchored in the full frame; meta["crop"] lets the report map an
# anchor back into the model-resolution view where the depth lives.
with Image.open(paths[0]) as im0:
    src_w, src_h = im0.size
resized_h, crop_y0 = H, 0                                   # fit(): whole frame, nothing cropped
for i in range(N):
    name = f"{i:06d}.jpg"
    if is_kf[i]:
        shutil.copyfile(paths[i], os.path.join(kf_dir, name))
    else:
        Image.fromarray(imgs[i]).save(os.path.join(kf_dir, name), quality=92)
    names.append(name)
crop_info = dict(model_w=int(W), model_h=int(H), src_w=src_w, src_h=src_h, resized_h=int(resized_h), crop_y0=int(crop_y0))

np.savez(os.path.join(out, "poses.npz"), c2w=c2w.astype(np.float32), w2c=w2c.astype(np.float32),
         K=K.astype(np.float32), is_keyframe=is_kf, frame_type=frame_type,
         names=np.array(names), source=np.array([os.path.basename(p) for p in paths]))
np.savez_compressed(os.path.join(out, "depth.npz"), depth=depth.astype(np.float16), conf=conf.astype(np.float16))

# cloud for the explorer: unproject every cloud_step-th frame where conf > threshold
v, u = np.mgrid[0:H, 0:W]
xyz, rgb, cf = [], [], []
for i in range(0, N, args.cloud_step):
    m = (conf[i] > args.vis_threshold) & (depth[i] > 0) & (depth[i] < args.max_depth)
    z = depth[i][m]
    x = (u[m] - K[i, 0, 2]) * z / K[i, 0, 0]
    y = (v[m] - K[i, 1, 2]) * z / K[i, 1, 1]
    pc = np.stack([x, y, z, np.ones_like(z)], 1) @ c2w[i].T
    xyz.append(pc[:, :3].astype(np.float32))
    rgb.append(imgs[i][m])
    cf.append(conf[i][m].astype(np.float32))
xyz, rgb, cf = map(np.concatenate, (xyz, rgb, cf))
if len(xyz) > args.cloud_max_points:
    sel = np.sort(np.random.default_rng(0).choice(len(xyz), args.cloud_max_points, replace=False))
    xyz, rgb, cf = xyz[sel], rgb[sel], cf[sel]
np.savez_compressed(os.path.join(out, "cloud.npz"), xyz=xyz, rgb=rgb, conf=cf, cams=c2w[:, :3, 3].astype(np.float32))

lo, hi = np.percentile(xyz, 1, 0), np.percentile(xyz, 99, 0)
meta = {
    "frames": N, "keyframes": int(is_kf.sum()), "keyframe_interval": kf_interval,
    "image_hw": [int(H), int(W)], "crop": crop_info, "gpu": gpu_name, "amp": str(amp_dtype), "sdpa": use_sdpa,
    "infer_seconds": round(t_infer, 1), "fps": round(N / t_infer, 2),
    "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
    "cloud_points": int(len(xyz)), "cloud_extent_p1p99": np.round(hi - lo, 3).tolist(),
    "cam_path_len": round(float(np.linalg.norm(np.diff(c2w[:, :3, 3], axis=0), axis=1).sum()), 3),
    "total_seconds": round(time.time() - t_start, 1),
    "args": vars(args),
}
with open(os.path.join(out, "meta.json"), "w") as f:
    json.dump(meta, f, indent=2)
if args.video:
    shutil.rmtree(frames_dir, ignore_errors=True)
open(os.path.join(out, "DONE"), "w").close()
log("done", json.dumps({k: meta[k] for k in ("frames", "keyframes", "fps", "peak_vram_gib", "cloud_points", "cloud_extent_p1p99")}))

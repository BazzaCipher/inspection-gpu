#!/usr/bin/env python3
"""Pair a visit's keyframes against the REFERENCE visit's keyframes, confidence-gated.

    .venv-hloc/bin/python pair_visit.py --ref work/visit_ref/kitchen --new work/visit_new/kitchen \
        --out work/pairs/kitchen [--device cuda] [--max_queries 40]

Inputs are infer_visit.py output dirs (keyframes/*.jpg, optional poses.npz). Three stages,
cheapest first, each only narrowing or scoring — nothing here ever *places* anything:

  1. retrieval  NetVLAD global descriptors (hloc) → top-k reference candidates per new keyframe
  2. sim3       if both visits have poses: a robust Sim3 between the two camera paths from the
                top-1 candidates, then each candidate's residual = how far the new camera lands
                from the candidate's camera. Used to re-rank and to flag wrong-wall lookalikes.
  3. roma       RoMa v1 dense match on the best candidates → certainty + RANSAC inlier ratio →
                match_confidence; `aligned` only above --aligned_min, otherwise `unaligned`.
                For aligned pairs, the reference image is warped into the new keyframe's view
                (the "before" that lines up pixel-for-pixel with the "after").

Writes <out>/pairs.json, <out>/warps/<new>__<ref>.jpg (after | warped before | raw before) and
<out>/contact_sheet.jpg (random aligned + unaligned sample, for the eyeball gate).

Hard rule (specs/turnover-qa.md §6/§9): when in doubt, `unaligned`. A wrong-wall pair is worse
than no pair.
"""
import argparse
import json
import os
import random
import sys
import tempfile
import time
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from importlib import import_module  # noqa: E402
umeyama_sim3 = import_module("03_ab_compare").umeyama_sim3  # module name starts with a digit

PIPELINE_VERSION = "pair_visit/0.1"

ap = argparse.ArgumentParser()
ap.add_argument("--ref", required=True)
ap.add_argument("--new", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
ap.add_argument("--topk", type=int, default=5, help="retrieval candidates per query")
ap.add_argument("--verify_k", type=int, default=2, help="candidates sent to RoMa per query")
ap.add_argument("--max_queries", type=int, default=0, help="0 = all new keyframes")
ap.add_argument("--keyframes_only", action="store_true", help="use only lingbot KV keyframes (needs poses.npz)")
ap.add_argument("--aligned_min", type=float, default=0.55, help="match_confidence floor for `aligned`")
ap.add_argument("--min_inliers", type=int, default=150)
ap.add_argument("--roma_res", type=int, default=560, help="RoMa coarse res (default 560; 448 is faster)")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
random.seed(a.seed)
t_start = time.time()
out = Path(a.out)
(out / "warps").mkdir(parents=True, exist_ok=True)


def log(*m):
    print(f"[pair {time.time() - t_start:6.1f}s]", *m, flush=True)


def load_visit(d, keyframes_only):
    d = Path(d).resolve()   # absolute: hloc gets these via symlinks in a temp dir
    names = sorted(p.name for p in (d / "keyframes").glob("*.jpg"))
    poses = None
    if (d / "poses.npz").exists():
        p = np.load(d / "poses.npz")
        poses = {"c2w": p["c2w"], "names": list(p["names"]), "is_keyframe": p["is_keyframe"]}
        if keyframes_only:
            keep = {n for n, k in zip(poses["names"], poses["is_keyframe"]) if k}
            names = [n for n in names if n in keep]
    return d, names, poses


ref_dir, ref_names, ref_poses = load_visit(a.ref, a.keyframes_only)
new_dir, new_names, new_poses = load_visit(a.new, a.keyframes_only)
if a.max_queries and len(new_names) > a.max_queries:
    idx = np.linspace(0, len(new_names) - 1, a.max_queries).round().astype(int)
    new_names = [new_names[i] for i in idx]
log(f"ref {len(ref_names)} keyframes, new {len(new_names)} queries, device={a.device}")

# ---------------------------------------------------------------- 1. retrieval
from hloc import extract_features  # noqa: E402  (slow import)

with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    (tmp / "img").mkdir()
    os.symlink(ref_dir / "keyframes", tmp / "img" / "ref")
    os.symlink(new_dir / "keyframes", tmp / "img" / "new")
    image_list = [f"ref/{n}" for n in ref_names] + [f"new/{n}" for n in new_names]
    conf = dict(extract_features.confs["netvlad"])
    conf["preprocessing"] = {"resize_max": 640}  # 518-wide inputs; no point upscaling
    feats = extract_features.main(conf, tmp / "img", tmp, image_list=image_list, as_half=False)
    with h5py.File(feats, "r") as h:
        def desc(key):
            return np.asarray(h[key]["global_descriptor"], dtype=np.float32)
        D_ref = np.stack([desc(f"ref/{n}") for n in ref_names])
        D_new = np.stack([desc(f"new/{n}") for n in new_names])
D_ref /= np.linalg.norm(D_ref, axis=1, keepdims=True) + 1e-9
D_new /= np.linalg.norm(D_new, axis=1, keepdims=True) + 1e-9
sim = D_new @ D_ref.T                                   # [Q, R], cosine in [-1,1]
cand = np.argsort(-sim, axis=1)[:, :a.topk]
log(f"retrieval done; top-1 sim median {np.median(sim[np.arange(len(cand)), cand[:, 0]]):.3f}")

# ---------------------------------------------------------------- 2. sim3 score (optional)
sim3_res = np.full(cand.shape, np.nan, dtype=np.float32)
if ref_poses is not None and new_poses is not None:
    def centres(poses, names):
        lut = {n: i for i, n in enumerate(poses["names"])}
        return np.stack([poses["c2w"][lut[n], :3, 3] for n in names])
    C_ref, C_new = centres(ref_poses, ref_names), centres(new_poses, new_names)
    src, dst = C_new, C_ref[cand[:, 0]]
    # robust Sim3: RANSAC over top-1 correspondences, keep the model with most inliers
    best = None
    rng = np.random.default_rng(a.seed)
    scale_ref = np.linalg.norm(np.diff(C_ref, axis=0), axis=1).sum() or 1.0
    for _ in range(300):
        pick = rng.choice(len(src), 4, replace=False)
        try:
            s, R, t = umeyama_sim3(src[pick], dst[pick])
        except Exception:
            continue
        pred = (s * (R @ src.T)).T + t
        err = np.linalg.norm(pred - dst, axis=1) / scale_ref
        inl = err < 0.02
        if best is None or inl.sum() > best[0]:
            best = (inl.sum(), inl)
    if best and best[0] >= 6:
        s, R, t = umeyama_sim3(src[best[1]], dst[best[1]])
        pred = (s * (R @ C_new.T)).T + t
        for q in range(len(cand)):
            for j, r in enumerate(cand[q]):
                sim3_res[q, j] = np.linalg.norm(pred[q] - C_ref[r]) / scale_ref
        # re-rank candidates by residual when it clearly disagrees with retrieval
        order = np.argsort(sim3_res, axis=1)
        cand = np.take_along_axis(cand, order, axis=1)
        sim3_res = np.take_along_axis(sim3_res, order, axis=1)
        log(f"sim3 fitted on {best[0]}/{len(src)} top-1 pairs; candidates re-ranked by path residual")
    else:
        log("sim3: too few consistent top-1 pairs; skipping (retrieval order kept)")
else:
    log("sim3: no poses on one side; skipping")

# ---------------------------------------------------------------- 3. RoMa verify + warp
from romatch import roma_indoor  # noqa: E402

dev = torch.device(a.device)
# use_custom_corr needs RoMa's compiled `local_corr` extension; the plain PyTorch path is fine
roma = roma_indoor(device=dev, coarse_res=a.roma_res, upsample_res=864, use_custom_corr=False,
                   amp_dtype=torch.float16 if dev.type == "cuda" else torch.float32)
roma.sample_thresh = 0.05


def verify(new_path, ref_path):
    """RoMa A=new (after), B=ref (before). Returns metrics + warped-before image (HxWx3 uint8)."""
    im_A = Image.open(new_path).convert("RGB")
    im_B = Image.open(ref_path).convert("RGB")
    W_A, H_A = im_A.size
    W_B, H_B = im_B.size
    with torch.no_grad():
        warp, cert = roma.match(str(new_path), str(ref_path), device=dev)   # symmetric: [1,H,2W,4], [1,H,2W]
    warp, cert = warp[0], cert[0]
    Hs, Ws2 = cert.shape
    Ws = Ws2 // 2
    warp_A, cert_A = warp[:, :Ws], cert[:, :Ws]          # A pixels → B coords
    cert_frac = float((cert_A > 0.5).float().mean())
    cert_mean = float(cert_A.mean())
    # geometric check on sampled matches
    matches, mc = roma.sample(warp_A, cert_A, num=3000)
    kA, kB = roma.to_pixel_coordinates(matches, H_A, W_A, H_B, W_B)
    kA, kB = kA.cpu().numpy(), kB.cpu().numpy()
    inlier_ratio, n_inl = 0.0, 0
    if len(kA) >= 8:
        _, mask = cv2.findFundamentalMat(kA, kB, cv2.USAC_MAGSAC, 1.5, 0.999, 10000)
        if mask is not None:
            n_inl = int(mask.sum())
            inlier_ratio = n_inl / len(kA)
    # warp B into A's frame: upsample the flow field to A's full resolution, then sample B (full
    # resolution) once — upsampling the warped *image* instead threw away B's detail and made the
    # before/after crops look soft
    B_t = torch.from_numpy(np.asarray(im_B)).permute(2, 0, 1)[None].float().to(dev) / 255
    grid = warp_A[None, ..., 2:].permute(0, 3, 1, 2)       # [1,2,Hs,Ws] in [-1,1]
    grid = F.interpolate(grid, size=(H_A, W_A), mode="bilinear", align_corners=False).permute(0, 2, 3, 1)
    warped = F.grid_sample(B_t, grid, mode="bilinear", padding_mode="zeros", align_corners=False)[0]
    mask = F.interpolate(cert_A[None, None].float(), size=(H_A, W_A), mode="bilinear", align_corners=False)[0, 0]
    # 0.3 left big holes on plain walls in the first smoke test; the warp is still right there
    warped = (warped * (mask > 0.15)).permute(1, 2, 0).cpu().numpy()
    warped = (np.clip(warped, 0, 1) * 255).astype(np.uint8)
    return dict(roma_certainty=cert_mean, cert_frac=cert_frac, roma_inlier_ratio=inlier_ratio,
                n_inliers=n_inl, n_samples=int(len(kA))), warped, np.asarray(im_A), np.asarray(im_B)


pairs = []
for q, new_name in enumerate(new_names):
    best = None
    for j in range(min(a.verify_k, cand.shape[1])):
        r = int(cand[q, j])
        ref_name = ref_names[r]
        m, warped, im_A, im_B = verify(new_dir / "keyframes" / new_name, ref_dir / "keyframes" / ref_name)
        # confidence: dense certainty coverage and epipolar inliers must both agree
        conf_ = 0.5 * m["cert_frac"] + 0.5 * m["roma_inlier_ratio"]
        rec = dict(new=new_name, ref=ref_name, retrieval_score=float(sim[q, r]),
                   sim3_residual=None if np.isnan(sim3_res[q, j]) else float(sim3_res[q, j]),
                   match_confidence=round(conf_, 4), **m)
        if best is None or conf_ > best[0]["match_confidence"]:
            best = (rec, warped, im_A, im_B)
    rec, warped, im_A, im_B = best
    aligned = (rec["match_confidence"] >= a.aligned_min and rec["n_inliers"] >= a.min_inliers
               and (rec["sim3_residual"] is None or rec["sim3_residual"] < 0.05))
    rec["match_state"] = "aligned" if aligned else "unaligned"
    rec["pipeline_version"] = PIPELINE_VERSION
    trip = np.concatenate([im_A, warped, cv2.resize(im_B, (im_A.shape[1], im_A.shape[0]))], axis=1)
    rec["warp_file"] = f"warps/{Path(new_name).stem}__{Path(rec['ref']).stem}.jpg"
    Image.fromarray(trip).save(out / rec["warp_file"], quality=88)
    pairs.append(rec)
    log(f"{q + 1}/{len(new_names)} {new_name}↔{rec['ref']} conf={rec['match_confidence']:.2f} "
        f"inl={rec['n_inliers']} {rec['match_state']}")

n_al = sum(p["match_state"] == "aligned" for p in pairs)
summary = dict(ref=str(ref_dir), new=str(new_dir), queries=len(pairs), aligned=n_al,
               aligned_frac=round(n_al / max(1, len(pairs)), 3),
               median_confidence=float(np.median([p["match_confidence"] for p in pairs])),
               sim3_used=bool(np.isfinite(sim3_res).any()), seconds=round(time.time() - t_start, 1),
               args=vars(a))
json.dump(dict(summary=summary, pairs=pairs), open(out / "pairs.json", "w"), indent=1)

# contact sheet: up to 10 aligned + 10 unaligned, labelled — the eyeball gate
sample = [p for p in pairs if p["match_state"] == "aligned"]
random.shuffle(sample)
sample = sample[:10]
un = [p for p in pairs if p["match_state"] == "unaligned"]
random.shuffle(un)
sample += un[:10]
tiles = []
for p in sample:
    im = cv2.imread(str(out / p["warp_file"]))
    im = cv2.resize(im, (900, int(900 * im.shape[0] / im.shape[1])))
    col = (80, 200, 80) if p["match_state"] == "aligned" else (60, 60, 230)
    cv2.putText(im, f"{p['match_state']} conf={p['match_confidence']:.2f} inl={p['n_inliers']} "
                    f"{p['new']}<->{p['ref']}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
    tiles.append(im)
if tiles:
    cv2.imwrite(str(out / "contact_sheet.jpg"), np.concatenate(tiles, axis=0))
log("summary", json.dumps({k: summary[k] for k in ("queries", "aligned", "aligned_frac", "median_confidence", "sim3_used", "seconds")}))

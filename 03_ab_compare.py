#!/usr/bin/env python3
"""
Stage 2 A/B — COLMAP vs. lingbot-map, same frames, poses compared.

Purpose: decide whether the feed-forward front-end (lingbot-map,
github.com/robbyant/lingbot-map, Apache-2.0 code, DINOv2-Apache backbone — Gate 1
cleared) can replace COLMAP as the SfM front-end (pipeline Stage 2). This does the
apples-to-apples pose comparison. It does NOT compare the pretty point clouds: the
COLMAP sparse cloud is triangulated (measurement-grade); lingbot's dense cloud is a
feed-forward depth prediction that hallucinates geometry (fine for the cosmetic
splat, unusable as a backbone). Poses are the only common, meaningful axis.

WHY THIS IS NOT "video in -> map out":
  - Feed lingbot the SAME extracted frames COLMAP got (`demo.py --image_folder ...`),
    NOT the raw video. Raw video lets lingbot pick its own fps/stride and you are no
    longer comparing the same inputs.
  - lingbot is streaming/stateful: its KV cache degrades past ~320 frames. For a
    whole-house walk you MUST run windowed (`--mode windowed --window_size 128
    --overlap_keyframes 8`, tune `--keyframe_interval`) or it pose-collapses. That
    config surface is itself part of what you are evaluating — a front-end that only
    works with hand-tuned windows is not a free win over COLMAP's global BA.
  - Neither is metrically scaled from monocular video, and the two live in different
    coordinate frames. So the comparison is trajectory SHAPE after a similarity
    (Sim3) alignment, never absolute coordinates.

THE CARDINAL SIN (this whole project's hard rule): never silently align. A low ATE
is only meaningful if enough frames overlap and the fit is not being flattered by
scale. This script always prints the alignment residual, the frame count it aligned
on, and the recovered scale. If the overlap is thin it says INCONCLUSIVE rather than
handing back a confident-looking number. A wrong "they agree" is worse than "cannot
tell".

INPUTS
  COLMAP:  a sparse model as TEXT. Convert the binary model first:
             colmap model_converter \
               --input_path  work/full/sparse/0 \
               --output_path work/full/sparse/0_txt \
               --output_type TXT
           We read images.txt (world-to-camera q,t per registered image).
  lingbot: the NPZ(s) written by `demo.py ... --save_predictions`. Schema confirmed
           from the repo's own consumer (interactive_viewer/npz_to_glb.py):
             c2w        [N,4,4] camera-to-world  (camera centre = c2w[:, :3, 3])
             K          [N,3,3] per-frame intrinsics (feed-forward estimated)
             depth      [N,H,W]
             confidence [N,H,W]  -- used here for the blank-wall failure report
             images     [N,H,W,3]
           A single combined .npz or a directory of per-frame npz both work.
  frames:  the directory of extracted frames both engines were fed (Stage 1 output).
           Its sorted filenames define frame index <-> COLMAP image NAME. lingbot
           emits a pose for EVERY input frame in order; COLMAP drops the ones it
           could not register — so we compare on the intersection and report the gap.

USAGE
  .venv/bin/python 03_ab_compare.py \
      --colmap work/full/sparse/0_txt/images.txt \
      --lingbot work/lingbot/predictions.npz \
      --frames work/frames

Depends on numpy only (already in .venv). No pycolmap, no torch.
"""

import argparse
import glob
import os
import sys

import numpy as np

# Overlap below this fraction of COLMAP-registered frames -> refuse to render a
# verdict. Umeyama on a handful of near-collinear camera centres will "fit" almost
# anything; a small ATE there is noise, not agreement.
MIN_OVERLAP_FRAMES = 12


# --------------------------------------------------------------------------- COLMAP

def quat_wxyz_to_R(qw, qx, qy, qz):
    """COLMAP stores rotation as a unit quaternion (w, x, y, z), world-to-camera."""
    n = np.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw),     2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw),     1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw),     2 * (qy * qz + qx * qw),     1 - 2 * (qx * qx + qy * qy)],
    ])


def read_colmap_images_txt(path):
    """Return {image_name: camera_centre_world[3]} for every registered image.

    images.txt lists two lines per image; the first is
        IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
    and encodes world-to-camera: X_cam = R X_world + t. The camera centre in world
    coordinates is therefore C = -R^T t. The second line (2D points) is skipped.
    """
    centres = {}
    with open(path) as fh:
        lines = [ln for ln in fh if not ln.startswith("#")]
    # Registered images are the odd-indexed content lines (0: pose, 1: points, ...).
    for i in range(0, len(lines), 2):
        parts = lines[i].split()
        if len(parts) < 10:
            continue
        qw, qx, qy, qz = map(float, parts[1:5])
        tx, ty, tz = map(float, parts[5:8])
        name = parts[9]
        R = quat_wxyz_to_R(qw, qx, qy, qz)
        t = np.array([tx, ty, tz])
        centres[name] = -R.T @ t
    return centres


# -------------------------------------------------------------------------- lingbot

def load_lingbot(path):
    """Return (c2w [N,4,4], confidence [N,H,W] or None) from --save_predictions output.

    Accepts a single .npz (combined, N stacked) or a directory of per-frame npz which
    we sort by filename and stack. Keys per the repo's npz_to_glb.py consumer.
    """
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.npz")))
        if not files:
            sys.exit(f"no .npz files in {path}")
        c2w, conf = [], []
        for f in files:
            d = np.load(f)
            c2w.append(d["c2w"])
            conf.append(d["confidence"] if "confidence" in d.files else None)
        c2w = np.stack([c[0] if c.ndim == 3 else c for c in c2w])
        conf = None if any(c is None for c in conf) else np.stack(
            [c[0] if c.ndim == 3 else c for c in conf])
        return c2w, conf
    d = np.load(path)
    c2w = d["c2w"]
    conf = d["confidence"] if "confidence" in d.files else None
    return c2w, conf


# ------------------------------------------------------------------------- geometry

def umeyama_sim3(src, dst):
    """Least-squares similarity (scale s, rotation R, translation t) mapping src->dst.

    Umeyama (1991). Minimises sum || dst_i - (s R src_i + t) ||^2. We need scale
    because monocular video fixes neither trajectory's absolute size, so without
    solving s the two are trivially "far apart" for reasons that say nothing about
    pose quality. Returns (s, R, t).
    """
    src = np.asarray(src, float)
    dst = np.asarray(dst, float)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    sc, dc = src - mu_s, dst - mu_d
    cov = dc.T @ sc / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:  # reflection guard
        S[2, 2] = -1
    R = U @ S @ Vt
    var_s = (sc ** 2).sum() / len(src)
    s = np.trace(np.diag(D) @ S) / var_s
    t = mu_d - s * R @ mu_s
    return s, R, t


# ---------------------------------------------------------------------------- report

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--colmap", required=True, help="COLMAP images.txt (TEXT model)")
    ap.add_argument("--lingbot", required=True, help="lingbot --save_predictions npz or dir")
    ap.add_argument("--frames", required=True, help="dir of extracted frames both engines saw")
    args = ap.parse_args()

    colmap = read_colmap_images_txt(args.colmap)
    c2w, conf = load_lingbot(args.lingbot)

    frame_names = [os.path.basename(p) for p in sorted(
        glob.glob(os.path.join(args.frames, "*")))
        if p.lower().endswith((".jpg", ".jpeg", ".png"))]
    n_frames = len(frame_names)

    if len(c2w) != n_frames:
        print(f"WARNING: lingbot has {len(c2w)} poses but {n_frames} frames on disk. "
              f"lingbot emits one pose per input frame — a mismatch means it was fed a "
              f"different set (raw video? different stride?). Comparison unsafe until fixed.")

    # Intersection: frames COLMAP registered AND that map to a lingbot pose by index.
    name_to_idx = {name: i for i, name in enumerate(frame_names)}
    pairs = [(name, name_to_idx[name]) for name in colmap
             if name in name_to_idx and name_to_idx[name] < len(c2w)]
    pairs.sort(key=lambda p: p[1])

    src = np.array([c2w[i][:3, 3] for _, i in pairs])      # lingbot camera centres
    dst = np.array([colmap[name] for name, _ in pairs])    # COLMAP camera centres

    print("=" * 68)
    print("A/B: COLMAP (reference) vs lingbot-map")
    print("=" * 68)
    print(f"frames on disk .............. {n_frames}")
    print(f"COLMAP registered ........... {len(colmap)}  ({len(colmap)/max(n_frames,1):.0%})")
    print(f"lingbot poses emitted ....... {len(c2w)}  ({len(c2w)/max(n_frames,1):.0%})")
    print(f"overlap compared on ......... {len(pairs)}")

    if len(pairs) < MIN_OVERLAP_FRAMES:
        print()
        print(f"INCONCLUSIVE: only {len(pairs)} overlapping frames (< {MIN_OVERLAP_FRAMES}). "
              f"Too few to trust a Sim3 fit — it would flatter any trajectory. "
              f"Not reporting an ATE. Fix registration or frame correspondence first.")
        return

    s, R, t = umeyama_sim3(src, dst)
    aligned = (s * (R @ src.T).T) + t
    err = np.linalg.norm(aligned - dst, axis=1)
    ate = np.sqrt((err ** 2).mean())
    span = np.linalg.norm(dst.max(0) - dst.min(0))  # COLMAP trajectory extent

    # Translational RPE on consecutive compared frames: drift, not global offset.
    # This is the metric that exposes lingbot's KV-cache degradation.
    d_al = np.diff(aligned, axis=0)
    d_gt = np.diff(dst, axis=0)
    rpe = np.linalg.norm(d_al - d_gt, axis=1)

    print()
    print(f"Sim3 scale (lingbot->COLMAP)  {s:.4f}   (far from steady across runs = unstable)")
    print(f"ATE  RMSE ................... {ate:.4f}  ({ate/span:.1%} of trajectory span)")
    print(f"ATE  median / max ........... {np.median(err):.4f} / {err.max():.4f}")
    print(f"RPE  translational median ... {np.median(rpe):.4f}")
    print(f"RPE  translational max ...... {rpe.max():.4f}   (drift spikes flag pose collapse)")

    # Where does the worst disagreement land? Correlate with lingbot's own confidence
    # if we have it — the blank-wall / low-texture failure mode should show up as
    # high error AND low confidence together (lingbot honest about its own weakness),
    # or high error with HIGH confidence (lingbot confidently wrong — the dangerous case).
    worst = np.argsort(err)[-5:][::-1]
    print()
    print("worst 5 frames (COLMAP name : ATE" + (" : mean conf" if conf is not None else "") + "):")
    for k in worst:
        name, idx = pairs[k]
        line = f"  {name:<28} {err[k]:.4f}"
        if conf is not None and idx < len(conf):
            c = float(np.nanmean(conf[idx]))
            flag = "  <- confidently wrong" if c > np.nanmean(conf) else "  (also low-confidence)"
            line += f"   conf={c:.3f}{flag}"
        print(line)

    print()
    print("Read it like this: ATE/span under ~2-3% and no RPE spikes = lingbot tracks")
    print("COLMAP and is a viable front-end. RPE spikes mid-sequence = KV drift; retry")
    print("with a shorter --window_size before concluding. 'Confidently wrong' frames")
    print("are the disqualifier for a MEASUREMENT backbone even if the average looks fine.")


if __name__ == "__main__":
    main()

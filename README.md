# inspection-gpu

GPU job image for a property-inspection pipeline: runs
[lingbot-map](https://github.com/robbyant/lingbot-map) (Apache-2.0, code and weights) streaming
inference on one room clip and writes posed keyframes, depth, and a point cloud. Headless — no
render stack. Built by GitHub Actions and published to `ghcr.io/<owner>/inspection-gpu`.

Contains only Apache-2.0 code and weights. No property data.

```
infer_visit.py --video in.mov --out /work/out      # see the docstring for outputs
start.sh                                            # pod entrypoint: sshd + self-terminate watchdog
```

Runs on any 24 GB CUDA card (A5000/3090/4090). Peak VRAM ~24 GB at 320 keyframes.

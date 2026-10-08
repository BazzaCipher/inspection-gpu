# GPU job image for Inspect3D turnover reconstruction: lingbot-map streaming inference, headless.
# No render stack (kaolin/open3d/warp) on purpose — the MP4 fly-through is not part of the product.
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_DISABLE_TELEMETRY=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        git ffmpeg curl openssh-server libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /run/sshd /root/.ssh && chmod 700 /root/.ssh \
    && sed -i 's/^#\?PermitRootLogin .*/PermitRootLogin prohibit-password/; s/^#\?PasswordAuthentication .*/PasswordAuthentication no/' /etc/ssh/sshd_config

# lingbot-map, pinned. Base deps only (Pillow, huggingface_hub, einops, safetensors, opencv, tqdm, scipy);
# torch 2.8 / torchvision 0.23 come from the base image. flashinfer is the attention backend the
# README recommends; infer_visit.py falls back to SDPA if it fails to import.
ARG LINGBOT_COMMIT=8fdf984
RUN git clone https://github.com/robbyant/lingbot-map /opt/lingbot-map \
    && cd /opt/lingbot-map && git checkout ${LINGBOT_COMMIT} \
    && pip install -e . \
    && pip install flashinfer-python opencv-python-headless

# Bake the checkpoint (Apache-2.0, 4.6 GB) so a job never depends on Hugging Face being up.
RUN python -c "from huggingface_hub import hf_hub_download as d; \
    d('robbyant/lingbot-map', 'lingbot-map.pt', local_dir='/opt/ckpt')" \
    && ls -la /opt/ckpt

COPY infer_visit.py start.sh /app/
RUN chmod +x /app/start.sh /app/infer_visit.py

WORKDIR /work
CMD ["/app/start.sh"]

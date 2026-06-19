# ── Stage: runtime ──────────────────────────────────────────────────────────
# CPU-only PyTorch keeps the image ~1.5 GB smaller than the CUDA variant.
# For GPU inference use ECS EC2 launch type + a CUDA base image instead.
FROM python:3.11-slim

WORKDIR /app

# ── System deps ──────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# ── Python deps (heavy, cached as its own layer) ─────────────────────────────
# Install CPU-only torch first so pip doesn't pull the ~2 GB CUDA wheel.
RUN pip install --no-cache-dir \
        torch==2.3.1 torchvision==0.18.1 \
        --index-url https://download.pytorch.org/whl/cpu

# PerspectiveFields + ML stack
COPY PerspectiveFields/ ./PerspectiveFields/
RUN pip install --no-cache-dir -e ./PerspectiveFields/

# Ultralytics (YOLO) + Flask
RUN pip install --no-cache-dir \
        ultralytics \
        flask>=3.0 \
        gunicorn

# ── Pre-cache PerspectiveFields weights ──────────────────────────────────────
# Run a tiny import so torch.hub downloads the backbone weights at build time
# (avoids a slow cold-start on the first container boot).
RUN python - << 'EOF'
import os, sys
os.environ.setdefault("TORCH_HOME", "/app/.cache/torch")
try:
    sys.path.insert(0, "/app/PerspectiveFields")
    from perspective2d import PerspectiveFields
    PerspectiveFields("Paramnet-360Cities-edina-centered")
    print("PerspectiveFields weights cached OK")
except Exception as e:
    print(f"Pre-cache skipped ({e}); weights will download on first request.")
EOF

# ── Application code ──────────────────────────────────────────────────────────
COPY infer.py train.py server.py ./
COPY web/ ./web/

# ── Trained model weights (gitignored, copied separately) ────────────────────
# These are small enough (~168 MB total) to bake into the image.
# If you prefer S3: remove these COPY lines and add S3 download logic to server.py.
COPY runs/segment/0405-road/weights/best.pt      ./runs/segment/0405-road/weights/best.pt
COPY runs/segment/0407-crosswalk/weights/best.pt ./runs/segment/0407-crosswalk/weights/best.pt
COPY runs/segment/0401-object/weights/best.pt    ./runs/segment/0401-object/weights/best.pt

# ── Runtime ──────────────────────────────────────────────────────────────────
ENV TORCH_HOME=/app/.cache/torch \
    PYTHONUNBUFFERED=1

# uploads/ and output/ are ephemeral inside the container; mount EFS if you
# need persistent storage across task restarts.
RUN mkdir -p uploads output

EXPOSE 5000

# 1 worker: inference is serialised inside server.py anyway.
CMD ["gunicorn", \
     "--bind", "0.0.0.0:5000", \
     "--workers", "1", \
     "--timeout", "600", \
     "--access-logfile", "-", \
     "server:app"]

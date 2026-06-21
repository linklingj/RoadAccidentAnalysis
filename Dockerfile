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
        torch==2.4.1 torchvision==0.19.1 \
        --index-url https://download.pytorch.org/whl/cpu

# PerspectiveFields + ML stack
COPY PerspectiveFields/ ./PerspectiveFields/
RUN pip install --no-cache-dir -e ./PerspectiveFields/

# Ultralytics (YOLO) + Flask + ONNX Runtime + RF-DETR + supervision
RUN pip install --no-cache-dir \
        ultralytics \
        flask>=3.0 \
        gunicorn \
        imageio \
        onnxruntime \
        supervision

# rfdetr: install its non-torch deps first, then rfdetr itself with --no-deps
# so pip cannot overwrite the CPU-only torch wheel we already have.
RUN pip install --no-cache-dir \
        transformers \
        pydantic \
        pyDeprecate \
        tqdm \
        requests
RUN pip install --no-cache-dir --no-deps rfdetr

# Verify rfdetr is importable at build time so failures surface here, not at runtime.
RUN python -c "from rfdetr import RFDETRLarge; print('rfdetr OK')"

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
COPY smp_road/ ./smp_road/

# ── Trained model weights (gitignored, copied separately) ────────────────────
# These are small enough (~168 MB total) to bake into the image.
# If you prefer S3: remove these COPY lines and add S3 download logic to server.py.
COPY runs/segment/0405-road/weights/best.pt           ./runs/segment/0405-road/weights/best.pt
COPY runs/segment/0407-crosswalk/weights/best.pt      ./runs/segment/0407-crosswalk/weights/best.pt
COPY runs/segment/0407-crosswalk/weights/best.onnx    ./runs/segment/0407-crosswalk/weights/best.onnx
COPY runs/segment/0401-object/weights/best.pt         ./runs/segment/0401-object/weights/best.pt
COPY runs/smp-road/best.onnx                          ./runs/smp-road/best.onnx
COPY runs/smp-road/best.onnx.data                     ./runs/smp-road/best.onnx.data
COPY runs/detect/rfdetr-object-0519/best_checkpoint.pth ./runs/detect/rfdetr-object-0519/best_checkpoint.pth

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

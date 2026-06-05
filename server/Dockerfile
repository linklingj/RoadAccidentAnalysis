# GPU-ready image. On CPU-only hosts (e.g. dev), it still runs — ultralytics
# auto-falls back to CPU. For an AWS g4dn/g5/p3 instance use the CUDA base
# image variant: nvidia/cuda:12.1.1-runtime-ubuntu22.04 + matching torch wheels.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    TORCH_HOME=/opt/torch_cache

# System deps: ffmpeg for video decode, libgl for OpenCV headless fallback.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first to cache the layer.
COPY server/requirements.txt /app/server/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/server/requirements.txt

# Copy the inference pipeline + PerspectiveFields fallback package.
COPY infer.py /app/infer.py
COPY util /app/util
COPY PerspectiveFields /app/PerspectiveFields

# Server code.
COPY server /app/server

# Model weights are mounted at runtime via volume or downloaded by an entrypoint;
# the image stays small. See server/README.md.
ENV UPLOAD_DIR=/var/lib/road-accident/uploads \
    RESULT_DIR=/var/lib/road-accident/results
RUN mkdir -p ${UPLOAD_DIR} ${RESULT_DIR}

EXPOSE 5000
WORKDIR /app/server
CMD ["gunicorn", "-c", "gunicorn_config.py", "app:app"]

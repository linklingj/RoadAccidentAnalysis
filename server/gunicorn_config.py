"""Gunicorn config for the inference server.

A single worker with threads is the right shape: model state lives in process
memory and you don't want N copies of YOLO+PerspectiveFields on a single GPU.
Use ``--threads`` to absorb upload concurrency while one job inferences.
"""

import os

bind = f"0.0.0.0:{os.environ.get('PORT', '5000')}"
workers = int(os.environ.get("GUNICORN_WORKERS", "1"))
threads = int(os.environ.get("GUNICORN_THREADS", "4"))
worker_class = os.environ.get("GUNICORN_WORKER_CLASS", "gthread")
# Video processing can take minutes; allow long requests for the polling endpoints
# (uploads themselves are bounded by MAX_UPLOAD_MB).
timeout = int(os.environ.get("GUNICORN_TIMEOUT", "300"))
graceful_timeout = 60
keepalive = 5
accesslog = "-"
errorlog = "-"
loglevel = os.environ.get("LOG_LEVEL", "info").lower()

"""Flask server that fronts the BEV inference pipeline.

API surface (all JSON unless noted):

* ``POST /api/upload``  — multipart upload (form field ``file``). Returns ``{job_id}``.
* ``GET  /api/status/<job_id>`` — ``{job_id, status, progress, message, ...}``.
* ``GET  /api/result/<job_id>`` — scene JSON when status==done; 404 otherwise.
* ``GET  /api/health`` — liveness probe.

Designed for a single instance behind gunicorn; jobs run on a background thread
(see ``jobs.py``). Swap ``registry`` for a Redis-backed implementation if you
need to scale horizontally.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from flask import Flask, abort, jsonify, request, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename


REPO_ROOT = Path(__file__).resolve().parent.parent
# So ``import infer`` resolves to the repo's pipeline module.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from jobs import registry  # noqa: E402 — must come after sys.path manipulation
from worker import handle_job  # noqa: E402


ALLOWED_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ALLOWED_VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
MAX_CONTENT_LENGTH = int(os.environ.get("MAX_UPLOAD_MB", "512")) * 1024 * 1024


def _classify(filename: str) -> str | None:
    suffix = Path(filename).suffix.lower()
    if suffix in ALLOWED_IMAGE_EXT:
        return "image"
    if suffix in ALLOWED_VIDEO_EXT:
        return "video"
    return None


def create_app() -> Flask:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

    origins = [o.strip() for o in os.environ.get("CORS_ORIGINS", "*").split(",") if o.strip()]
    CORS(app, resources={r"/api/*": {"origins": origins or "*"}})

    upload_dir = Path(os.environ.get("UPLOAD_DIR", "/tmp/road-accident/uploads"))
    upload_dir.mkdir(parents=True, exist_ok=True)

    registry.configure(handle_job)

    @app.get("/api/health")
    def health():
        return jsonify({"status": "ok"})

    @app.post("/api/upload")
    def upload():
        if "file" not in request.files:
            return jsonify({"error": "missing file"}), 400
        file = request.files["file"]
        if not file.filename:
            return jsonify({"error": "empty filename"}), 400

        kind = _classify(file.filename)
        if kind is None:
            return jsonify({"error": f"unsupported file type: {file.filename}"}), 415

        safe = secure_filename(file.filename) or "input"
        # Use a sub-dir per-upload so re-uploads of the same name don't collide.
        # The job_id is assigned later; for the input file, prefix with a temp id.
        from uuid import uuid4

        slot = upload_dir / uuid4().hex
        slot.mkdir(parents=True, exist_ok=True)
        target = slot / safe
        file.save(target)

        job = registry.submit(filename=safe, media_kind=kind, input_path=str(target))
        return jsonify({"job_id": job.job_id, "status": job.status, "kind": kind}), 202

    @app.get("/api/status/<job_id>")
    def status(job_id: str):
        job = registry.get(job_id)
        if job is None:
            return jsonify({"error": "not found"}), 404
        return jsonify(job.to_public())

    @app.get("/api/result/<job_id>")
    def result(job_id: str):
        job = registry.get(job_id)
        if job is None:
            return jsonify({"error": "not found"}), 404
        if job.status != "done":
            return jsonify({
                "error": "not ready",
                "status": job.status,
                "progress": job.progress,
            }), 409
        if not job.result_path or not Path(job.result_path).exists():
            return jsonify({"error": "result missing"}), 410
        return send_file(job.result_path, mimetype="application/json")

    @app.errorhandler(413)
    def too_large(_):
        return jsonify({
            "error": "file too large",
            "max_bytes": MAX_CONTENT_LENGTH,
        }), 413

    return app


app = create_app()


if __name__ == "__main__":
    # Dev-only entry point; production uses gunicorn (see Dockerfile/gunicorn_config.py).
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")), debug=False)

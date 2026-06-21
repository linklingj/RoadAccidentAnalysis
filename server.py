"""Flask server: serves the three.js viewer and runs inference on uploaded media.

Flow: the browser uploads a video/image to ``POST /api/infer``; the server runs
the BEV pipeline (``infer.py``) in a background worker and returns the scene JSON
the viewer renders. Inference is serialized through a single queue/worker because
the YOLO/torch models are not safe to run concurrently, and a single
``RoadSceneProjector`` is reused across requests so models load only once.

Run inside the `dl` conda env (cv2/torch/ultralytics live there):

    conda activate dl
    pip install flask
    python server.py                  # http://localhost:5000

`infer` is imported lazily (inside the worker) so the server still starts and
serves the static viewer in environments without the ML dependencies.
"""
from __future__ import annotations

import argparse
import queue
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from flask import Flask, jsonify, request, send_from_directory

REPO_ROOT = Path(__file__).resolve().parent
WEB_DIR = REPO_ROOT / "web"
UPLOAD_DIR = REPO_ROOT / "uploads"
OUTPUT_DIR = REPO_ROOT / "output"

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

# Pre-computed scene JSONs served instantly for the landing-page sample cards.
# Add more entries here when additional pre-inferred JSONs become available.
SAMPLE_META = [
    {
        "id": "sample1",
        "name": "교차로1 씬",
        "desc": "교차로 추돌 사고",
        "file": WEB_DIR / "data" / "sample_scene1.json",
    },
    {
        "id": "sample2",
        "name": "교차로2 씬",
        "desc": "교차로 추돌 사고",
        "file": WEB_DIR / "data" / "sample_scene2.json",
    },
    {
        "id": "sample3",
        "name": "도로 씬",
        "desc": "일반 도로 추돌 사고",
        "file": WEB_DIR / "data" / "sample_scene3.json",
    },
]
_SAMPLE_BY_ID = {s["id"]: s for s in SAMPLE_META}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 512 * 1024 * 1024  # 512 MB upload cap


# ── Pipeline singleton (models load lazily on the first job) ────────────────
_projector = None
_projector_error: Optional[str] = None


def _build_config():
    from infer import PipelineConfig

    device = ARGS.device
    if isinstance(device, str) and device.isdigit():
        device = int(device)
    return PipelineConfig(
        road_model_path=ARGS.road_model,
        crosswalk_model_path=ARGS.crosswalk_model,
        object_model_path=ARGS.object_model,
        road_detector_type="unet",
        object_detector_type="rfdetr",
        rfdetr_object_model_path=ARGS.rfdetr_object_model,
        road_conf=ARGS.road_conf,
        object_conf=ARGS.object_conf,
        camera_height_m=ARGS.camera_height,
        pixels_per_meter=ARGS.ppm,
        bev_width=ARGS.bev_width,
        bev_height=ARGS.bev_height,
        use_clahe=not ARGS.no_clahe,
        device=device,
        use_onnx=True,
    )


def _get_projector():
    global _projector
    if _projector is None:
        from infer import RoadSceneProjector

        _projector = RoadSceneProjector(config=_build_config())
    return _projector


# ── Job queue (one inference at a time) ─────────────────────────────────────
_jobs: Dict[str, Dict[str, Any]] = {}
_jobs_lock = threading.Lock()
_job_queue: "queue.Queue[str]" = queue.Queue()


def _update_job(job_id: str, **fields: Any) -> None:
    with _jobs_lock:
        if job_id in _jobs:
            _jobs[job_id].update(fields)


def _worker() -> None:
    while True:
        job_id = _job_queue.get()
        try:
            with _jobs_lock:
                job = dict(_jobs.get(job_id, {}))
            if not job:
                continue
            _update_job(job_id, status="running", started_at=time.time())

            proj = _get_projector()
            # Always set both per job (the worker is serialized but shares one
            # config, so an unset field must reset to the default rather than
            # inherit the previous job's value).
            proj.config.camera_height_m = (
                float(job["camera_height"]) if job.get("camera_height") is not None else float(ARGS.camera_height)
            )
            proj.config.pixels_per_meter = (
                float(job["ppm"]) if job.get("ppm") is not None else float(ARGS.ppm)
            )

            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            if job["mode"] == "video":
                outputs = proj.run_video(video_path=job["path"], save_dir=str(OUTPUT_DIR))
            else:
                outputs = proj.run(image_path=job["path"], save_dir=str(OUTPUT_DIR))

            _update_job(
                job_id,
                status="done",
                finished_at=time.time(),
                scene=outputs.get("scene_data"),
            )
        except Exception as exc:  # noqa: BLE001 — report any failure to the client
            traceback.print_exc()
            _update_job(
                job_id,
                status="error",
                finished_at=time.time(),
                error=f"{type(exc).__name__}: {exc}",
            )
        finally:
            _job_queue.task_done()


_worker_thread = threading.Thread(target=_worker, name="inference-worker", daemon=True)
_worker_thread.start()


# ── API ─────────────────────────────────────────────────────────────────────
@app.post("/api/infer")
def api_infer():
    file = request.files.get("file")
    if file is None or not file.filename:
        return jsonify(error="업로드된 파일이 없습니다 (field 'file')."), 400

    ext = Path(file.filename).suffix.lower()
    if ext in VIDEO_EXTS:
        mode = "video"
    elif ext in IMAGE_EXTS:
        mode = "image"
    else:
        return jsonify(error=f"지원하지 않는 형식입니다: {ext or '(없음)'}"), 400

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    job_id = uuid.uuid4().hex[:12]
    dest = UPLOAD_DIR / f"{job_id}{ext}"
    file.save(str(dest))

    with _jobs_lock:
        _jobs[job_id] = {
            "status": "queued",
            "mode": mode,
            "filename": file.filename,
            "path": str(dest),
            "created_at": time.time(),
            "camera_height": request.form.get("camera_height", type=float),
            "ppm": request.form.get("ppm", type=float),
            "queue_position": _job_queue.qsize(),
        }
    _job_queue.put(job_id)
    return jsonify(job_id=job_id, mode=mode, status="queued"), 202


@app.get("/api/jobs/<job_id>")
def api_job(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
        job = dict(job) if job else None
    if job is None:
        return jsonify(error="알 수 없는 job_id"), 404

    resp: Dict[str, Any] = {"job_id": job_id, "status": job["status"], "mode": job.get("mode")}
    if job.get("started_at"):
        end = job.get("finished_at") or time.time()
        resp["elapsed_sec"] = round(end - job["started_at"], 1)
    if job["status"] == "done":
        resp["scene"] = job.get("scene")
    elif job["status"] == "error":
        resp["error"] = job.get("error")
    return jsonify(resp)


@app.get("/api/health")
def api_health():
    return jsonify(ok=True, projector_loaded=_projector is not None)


@app.get("/api/samples")
def api_samples():
    return jsonify([{"id": s["id"], "name": s["name"], "desc": s["desc"]} for s in SAMPLE_META])


@app.get("/api/samples/<sample_id>")
def api_sample(sample_id: str):
    sample = _SAMPLE_BY_ID.get(sample_id)
    if sample is None:
        return jsonify(error="알 수 없는 샘플 ID"), 404
    json_path: Path = sample["file"]
    if not json_path.exists():
        return jsonify(error=f"샘플 파일 없음: {json_path.name}"), 404
    import json as _json
    with open(json_path, encoding="utf-8") as f:
        return jsonify(_json.load(f))


# ── Static frontend (web/) ──────────────────────────────────────────────────
@app.get("/")
def index():
    return send_from_directory(WEB_DIR, "index.html")


@app.get("/<path:path>")
def static_files(path: str):
    return send_from_directory(WEB_DIR, path)


# ── Entry point ─────────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Serve the three.js viewer + run inference on uploads")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5000)
    p.add_argument("--road-model", default="runs/segment/0405-road/weights/best.pt")
    p.add_argument("--crosswalk-model", default="runs/segment/0407-crosswalk/weights/best.pt")
    p.add_argument("--object-model", default="runs/segment/0401-object/weights/best.pt")
    p.add_argument("--road-conf", type=float, default=0.25)
    p.add_argument("--object-conf", type=float, default=0.15)
    p.add_argument("--camera-height", type=float, default=2.5, help="Default camera height (m); overridable per upload")
    p.add_argument("--ppm", type=float, default=28.0, help="Default pixels-per-meter; overridable per upload")
    p.add_argument("--bev-width", type=int, default=960)
    p.add_argument("--bev-height", type=int, default=960)
    p.add_argument("--no-clahe", action="store_true")
    p.add_argument("--device", default=None, help="e.g. 'cpu', '0', 'cuda:0'")
    p.add_argument("--rfdetr-object-model", default="runs/detect/rfdetr-object-0519/best_checkpoint.pth",
                   help="RF-DETR object 탐지 모델 경로 (.pth)")
    return p


# Parsed with defaults at import so `flask run` works; overridden in __main__.
ARGS = build_parser().parse_args([])


def main() -> None:
    global ARGS
    ARGS = build_parser().parse_args()
    print(f"[server] serving web/ + inference API on http://{ARGS.host}:{ARGS.port}")
    print("[server] models load lazily on the first /api/infer request.")
    app.run(host=ARGS.host, port=ARGS.port, threaded=True)


if __name__ == "__main__":
    main()

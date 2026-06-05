"""Job handler that runs the existing BEV pipeline and writes the scene JSON.

The handler imports ``infer`` lazily so the Flask process boots even without
the heavy ML deps installed (helpful for unit tests / dev).
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Optional

from jobs import Job, registry


log = logging.getLogger("worker")


# ---- Lazy singletons --------------------------------------------------------
_projector = None  # cached RoadSceneProjector to avoid reloading models per job


def _load_projector():
    global _projector
    if _projector is not None:
        return _projector

    from infer import PipelineConfig, RoadSceneProjector  # noqa: WPS433

    cfg = PipelineConfig(
        road_model_path=os.environ.get("ROAD_MODEL_PATH", PipelineConfig.road_model_path),
        crosswalk_model_path=os.environ.get("CROSSWALK_MODEL_PATH", PipelineConfig.crosswalk_model_path),
        object_model_path=os.environ.get("OBJECT_MODEL_PATH", PipelineConfig.object_model_path),
        camera_height_m=float(os.environ.get("CAMERA_HEIGHT_M", PipelineConfig.camera_height_m)),
        pixels_per_meter=float(os.environ.get("PIXELS_PER_METER", PipelineConfig.pixels_per_meter)),
        device=_parse_device(os.environ.get("INFER_DEVICE")),
        # Server-mode pipeline does not need to mirror to Unity StreamingAssets.
        unity_streaming_assets_dir=None,
    )

    _projector = RoadSceneProjector(config=cfg)
    return _projector


def _parse_device(spec: Optional[str]):
    if not spec:
        return None
    if spec.isdigit():
        return int(spec)
    return spec


# ---- Public entry point: registry handler -----------------------------------
def handle_job(job: Job) -> None:
    """Process one queued job — runs in the worker thread."""
    projector = _load_projector()
    registry.set_progress(job.job_id, 0.05, "warming up models")

    save_dir = Path(os.environ.get("RESULT_DIR", "/tmp/road-accident/results")) / job.job_id
    save_dir.mkdir(parents=True, exist_ok=True)

    log.info("Job %s start (%s): %s", job.job_id, job.media_kind, job.input_path)

    if job.media_kind == "image":
        registry.set_progress(job.job_id, 0.2, "running inference")
        outputs = projector.run(image_path=job.input_path, save_dir=str(save_dir))
        scene = outputs.get("scene_data", {})
    elif job.media_kind == "video":
        registry.set_progress(job.job_id, 0.2, "running inference")
        outputs = projector.run_video(video_path=job.input_path, save_dir=str(save_dir))
        scene = outputs.get("scene_data", {})
    else:
        raise ValueError(f"Unsupported media kind: {job.media_kind}")

    registry.set_progress(job.job_id, 0.9, "writing result")

    result_path = save_dir / "scene_data.json"
    with result_path.open("w", encoding="utf-8") as f:
        json.dump(scene, f, ensure_ascii=False)

    log.info("Job %s done: %s", job.job_id, result_path)
    registry.set_result(job.job_id, str(result_path))

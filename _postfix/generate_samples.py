"""Generate sample scene JSONs via inference, replicating server.py's config.

Saves to web/data/sample_scene{1,2,3}.json. Reuses one projector (models load
once), matching the server's single-RoadSceneProjector behaviour.
"""
import json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from infer import PipelineConfig, RoadSceneProjector  # noqa: E402

JOBS = [
    (ROOT / "web/data/sample1.mp4", ROOT / "web/data/sample_scene1.json"),
    (ROOT / "web/data/sample2.mp4", ROOT / "web/data/sample_scene2.json"),
    (ROOT / "web/data/sample3.mp4", ROOT / "web/data/sample_scene3.json"),
]

cfg = PipelineConfig(
    road_model_path="runs/segment/0405-road/weights/best.pt",
    crosswalk_model_path="runs/segment/0407-crosswalk/weights/best.pt",
    object_model_path="runs/segment/0401-object/weights/best.pt",
    road_detector_type="unet",
    object_detector_type="rfdetr",
    rfdetr_object_model_path="runs/detect/rfdetr-object-0519/best_checkpoint.pth",
    road_conf=0.25,
    object_conf=0.15,
    camera_height_m=2.5,
    pixels_per_meter=28.0,
    bev_width=960,
    bev_height=960,
    use_clahe=True,
    use_onnx=True,
    web_data_dir=None,  # do not clobber the existing scene_data.json mirror
)

print("[gen] building projector (loading models)…", flush=True)
proj = RoadSceneProjector(config=cfg)
print("[gen] projector ready", flush=True)

for video, target in JOBS:
    if not video.exists():
        print(f"[gen] MISSING VIDEO: {video}", flush=True)
        continue
    t0 = time.time()
    print(f"[gen] >>> {video.name} -> {target.name}", flush=True)
    outputs = proj.run_video(video_path=str(video), save_dir=str(ROOT / "output"))
    scene = outputs.get("scene_data")
    if scene is None:
        print(f"[gen] !!! no scene_data for {video.name}", flush=True)
        continue
    with open(target, "w", encoding="utf-8") as f:
        json.dump(scene, f, ensure_ascii=False)
    dt = time.time() - t0
    print(f"[gen] done {video.name}: frames={len(scene.get('frames', []))} "
          f"tracks={len(scene.get('tracks', []))} fps={scene.get('fps')} "
          f"({dt:.1f}s) -> {target}", flush=True)

print("[gen] ALL DONE", flush=True)

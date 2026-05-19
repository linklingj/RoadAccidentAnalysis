from __future__ import annotations

import json
import math
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO


DEFAULT_ROAD_MODEL_PATH = "runs/segment/0401-road/weights/best.pt"
DEFAULT_CROSSWALK_MODEL_PATH = "runs/segment/0406-crosswalk/weights/best.pt"
DEFAULT_OBJECT_MODEL_PATH = "runs/segment/0401-object/weights/best.pt"
DEFAULT_RFDETR_OBJECT_MODEL_PATH = "runs/detect/rfdetr-object/best_checkpoint.pth"
DEFAULT_PERSPECTIVE_VERSION = "Paramnet-360Cities-edina-centered"

# cctv-object-dataset/data.yaml names 순서와 동일하게 유지
OBJECT_CLASS_NAMES: List[str] = ["attention", "bus", "car", "crosswalk", "person", "riders", "truck"]


@dataclass
class PipelineConfig:
    road_model_path: str = DEFAULT_ROAD_MODEL_PATH
    crosswalk_model_path: str = DEFAULT_CROSSWALK_MODEL_PATH
    object_model_path: str = DEFAULT_OBJECT_MODEL_PATH
    # "yolo" | "rfdetr"
    object_detector_type: str = "yolo"
    rfdetr_object_model_path: str = DEFAULT_RFDETR_OBJECT_MODEL_PATH
    perspective_version: str = DEFAULT_PERSPECTIVE_VERSION
    road_conf: float = 0.25
    object_conf: float = 0.15
    crosswalk_conf: float = 0.15
    camera_height_m: float = 6.5
    pixels_per_meter: float = 42.0
    bev_width: int = 960
    bev_height: int = 960
    max_polygon_points: int = 600
    track_max_distance_m: float = 3.0
    track_max_missed_frames: int = 10
    trajectory_max_length: int = 60
    video_recompute_camera_each_frame: bool = False
    use_clahe: bool = True
    device: Optional[Any] = None
    # Unity 연동: scene JSON을 StreamingAssets로 미러링하여 Unity SceneLoader가 바로 로드
    unity_streaming_assets_dir: Optional[str] = "Road3dReconstruction/Assets/StreamingAssets"
    unity_scene_filename: str = "scene_data.json"


def _resolve_device(device: Optional[Any]) -> Any:
    if device is not None:
        return device
    return 0 if torch.cuda.is_available() else "cpu"


def _perspective_torch_device(device: Any) -> str:
    if isinstance(device, int):
        return f"cuda:{device}" if torch.cuda.is_available() else "cpu"
    if isinstance(device, str):
        value = device.lower()
        if value == "cpu":
            return "cpu"
        if value.startswith("cuda"):
            return device if torch.cuda.is_available() else "cpu"
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _to_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    if value is None:
        return default
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return default
        return float(value.detach().reshape(-1)[0].cpu().item())
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return default
        return float(value.reshape(-1)[0])
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _load_perspective_model(version: str, device: Any):
    try:
        from perspective2d import PerspectiveFields  # type: ignore
    except ModuleNotFoundError:
        repo_root = Path(__file__).resolve().parent
        local_pkg = repo_root / "PerspectiveFields"
        if local_pkg.exists():
            sys.path.insert(0, str(local_pkg))
        from perspective2d import PerspectiveFields  # type: ignore

    pf_model = PerspectiveFields(version).eval()
    torch_device = _perspective_torch_device(device)
    pf_model = pf_model.to(torch_device)
    return pf_model


def _apply_clahe(image_bgr: np.ndarray, clip_limit: float = 2.0, tile_size: int = 8) -> np.ndarray:
    """야간·역광 CCTV 영상의 명도 불균일을 보정한다 (CLAHE on L channel)."""
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def _extract_road_polygons(result: Any) -> List[np.ndarray]:
    polygons: List[np.ndarray] = []
    if result is None or result.masks is None or result.masks.xy is None:
        return polygons
    for poly in result.masks.xy:
        if poly is None or len(poly) < 3:
            continue
        polygons.append(np.asarray(poly, dtype=np.float32))
    return polygons

def _extract_crosswalk_polygons(result: Any) -> List[np.ndarray]:
    polygons: List[np.ndarray] = []
    if result is None or result.masks is None or result.masks.xy is None:
        return polygons
    for poly in result.masks.xy:
        if poly is None or len(poly) < 3:
            continue
        polygons.append(np.asarray(poly, dtype=np.float32))
    return polygons

def _extract_object_detections(result: Any) -> List[Dict[str, Any]]:
    detections: List[Dict[str, Any]] = []
    if result is None or result.boxes is None:
        return detections

    names = result.names if hasattr(result, "names") else {}
    masks_xy = result.masks.xy if result.masks is not None else None
    track_ids = result.boxes.id  # None when using predict(), Tensor when using track()

    for i, box in enumerate(result.boxes):
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = float(box.conf.item())
        cls_id = int(box.cls.item())
        class_name = names[cls_id] if isinstance(names, dict) and cls_id in names else str(cls_id)
        # skip 'attention' & 'crosswalk'
        if class_name == "attention" or class_name == "crosswalk":
            continue

        foot_x = (x1 + x2) * 0.5
        foot_y = y2
        if masks_xy is not None and i < len(masks_xy):
            contour = np.asarray(masks_xy[i], dtype=np.float32)
            if contour.size > 0:
                lowest_idx = int(np.argmax(contour[:, 1]))
                foot_x = float(contour[lowest_idx, 0])
                foot_y = float(contour[lowest_idx, 1])

        track_id = int(track_ids[i].item()) if track_ids is not None else None

        detections.append(
            {
                "detection_id": i,
                "class_id": cls_id,
                "class_name": class_name,
                "confidence": conf,
                "bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)],
                "footpoint_uv": [float(foot_x), float(foot_y)],
                "track_id": track_id,
            }
        )
    return detections


def infer_road_model(
    image_bgr: np.ndarray,
    model: YOLO,
    conf: float,
    device: Any,
) -> Dict[str, Any]:
    results = model.predict(
        source=image_bgr,
        save=False,
        conf=conf,
        device=device,
        verbose=False,
    )
    result = results[0] if results else None
    return {
        "raw_result": result,
        "road_polygons_uv": _extract_road_polygons(result),
    }

def infer_crosswalk_model(
    image_bgr: np.ndarray,
    model: YOLO,
    conf: float,
    device: Any,
) -> Dict[str, Any]:
    results = model.predict(
        source=image_bgr,
        save=False,
        conf=conf,
        device=device,
        verbose=False,
        classes=[1]
    )
    result = results[0] if results else None
    return {
        "raw_result": result,
        "crosswalk_polygons_uv": _extract_crosswalk_polygons(result),
    }

def infer_object_model(
    image_bgr: np.ndarray,
    model: YOLO,
    conf: float,
    device: Any,
    use_tracker: bool = False,
) -> Dict[str, Any]:
    common_kwargs = dict(source=image_bgr, save=False, conf=conf, device=device, verbose=False)
    if use_tracker:
        results = model.track(**common_kwargs, tracker="bytetrack.yaml", persist=True)
    else:
        results = model.predict(**common_kwargs)
    result = results[0] if results else None
    return {
        "raw_result": result,
        "detections": _extract_object_detections(result),
    }


def _load_rfdetr_model(model_path: str) -> Any:
    try:
        from rfdetr import RFDETRLarge  # type: ignore
    except ImportError:
        raise ImportError("rfdetr 패키지가 없습니다. `pip install rfdetr supervision` 을 실행하세요.")
    return RFDETRLarge(pretrain_weights=model_path)


def _extract_rfdetr_detections(
    detections: Any,
    class_names: List[str],
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    if detections is None or len(detections) == 0:
        return results

    xyxy = detections.xyxy          # (N, 4) numpy
    confs = detections.confidence   # (N,)
    cls_ids = detections.class_id   # (N,)
    track_ids = getattr(detections, "tracker_id", None)

    for i in range(len(xyxy)):
        x1, y1, x2, y2 = float(xyxy[i][0]), float(xyxy[i][1]), float(xyxy[i][2]), float(xyxy[i][3])
        cls_id = int(cls_ids[i])
        conf = float(confs[i])
        class_name = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)

        if class_name in ("attention", "crosswalk"):
            continue

        # 마스크 없음 → bbox 하단 중앙을 footpoint로 사용
        foot_x = (x1 + x2) * 0.5
        foot_y = y2

        track_id = int(track_ids[i]) if track_ids is not None else None

        results.append({
            "detection_id": i,
            "class_id": cls_id,
            "class_name": class_name,
            "confidence": conf,
            "bbox_xyxy": [x1, y1, x2, y2],
            "footpoint_uv": [foot_x, foot_y],
            "track_id": track_id,
        })
    return results


def infer_object_model_rfdetr(
    image_bgr: np.ndarray,
    model: Any,
    conf: float,
    class_names: List[str],
    sv_tracker: Optional[Any] = None,
) -> Dict[str, Any]:
    """RF-DETR 모델로 객체를 탐지하고 (선택적으로) supervision ByteTrack으로 추적한다."""
    import PIL.Image  # type: ignore

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_pil = PIL.Image.fromarray(image_rgb)

    detections = model.predict(image_pil, threshold=conf)

    if sv_tracker is not None:
        detections = sv_tracker.update_with_detections(detections)

    return {
        "raw_result": detections,
        "detections": _extract_rfdetr_detections(detections, class_names),
    }


def estimate_camera_params(
    image_bgr: np.ndarray,
    perspective_model: Any,
) -> Dict[str, float]:
    pred = perspective_model.inference(img_bgr=image_bgr)

    h, w = image_bgr.shape[:2]
    roll_deg = _to_float(pred.get("pred_roll"), 0.0) or 0.0
    pitch_deg = _to_float(pred.get("pred_pitch"), 0.0) or 0.0
    vfov_deg = _to_float(pred.get("pred_general_vfov"), None)
    if vfov_deg is None:
        vfov_deg = _to_float(pred.get("pred_vfov"), 65.0) or 65.0

    rel_cx = _to_float(pred.get("pred_rel_cx"), 0.0) or 0.0
    rel_cy = _to_float(pred.get("pred_rel_cy"), 0.0) or 0.0
    rel_focal = _to_float(pred.get("pred_rel_focal"), None)
    if rel_focal is None:
        rel_focal = 0.5 / math.tan(math.radians(vfov_deg) * 0.5)

    focal_px = rel_focal * h
    cx = (0.5 + rel_cx) * w
    cy = (0.5 + rel_cy) * h

    return {
        "roll_deg": float(roll_deg),
        "pitch_deg": float(pitch_deg),
        "vfov_deg": float(vfov_deg),
        "rel_cx": float(rel_cx),
        "rel_cy": float(rel_cy),
        "rel_focal": float(rel_focal),
        "focal_px": float(focal_px),
        "cx": float(cx),
        "cy": float(cy),
    }


def _camera_ray_to_world(x_norm: float, y_norm: float, roll_rad: float, pitch_rad: float) -> np.ndarray:
    # Coordinate convention follows PerspectiveFields/PanoCam:
    # x: right, y: down, z: forward.
    cr, sr = math.cos(roll_rad), math.sin(roll_rad)
    cp, sp = math.cos(pitch_rad), math.sin(pitch_rad)

    dir_x = x_norm * cr - y_norm * sr
    dir_y = x_norm * cp * sr + y_norm * cp * cr - sp
    dir_z = x_norm * sp * sr + y_norm * sp * cr + cp
    return np.array([dir_x, dir_y, dir_z], dtype=np.float32)


def project_uv_to_ground(
    u: float,
    v: float,
    camera: Dict[str, float],
    camera_height_m: float,
) -> Optional[Tuple[float, float]]:
    focal = camera["focal_px"]
    if focal <= 1e-6:
        return None

    x_norm = (u - camera["cx"]) / focal
    y_norm = (v - camera["cy"]) / focal

    ray = _camera_ray_to_world(
        x_norm=x_norm,
        y_norm=y_norm,
        roll_rad=math.radians(camera["roll_deg"]),
        pitch_rad=math.radians(camera["pitch_deg"]),
    )
    denom = float(ray[1])
    if denom <= 1e-6:
        return None

    t = camera_height_m / denom
    if t <= 0:
        return None

    x_m = float(t * ray[0])
    z_m = float(t * ray[2])
    if not (math.isfinite(x_m) and math.isfinite(z_m)):
        return None
    if z_m <= 0:
        return None

    return x_m, z_m


def world_to_bev(
    x_m: float,
    z_m: float,
    bev_w: int,
    bev_h: int,
    pixels_per_meter: float,
) -> Tuple[int, int]:
    x_px = int(round(bev_w * 0.5 + x_m * pixels_per_meter))
    y_px = int(round(bev_h - z_m * pixels_per_meter))
    return x_px, y_px


def bev_to_world(
    x_px: float,
    y_px: float,
    bev_w: int,
    bev_h: int,
    pixels_per_meter: float,
) -> Tuple[float, float]:
    x_m = (float(x_px) - bev_w * 0.5) / pixels_per_meter
    z_m = (bev_h - float(y_px)) / pixels_per_meter
    return float(x_m), float(z_m)


def _draw_bev_grid(canvas: np.ndarray, pixels_per_meter: float) -> None:
    h, w = canvas.shape[:2]
    max_depth_m = int(h / pixels_per_meter)
    max_side_m = int((w * 0.5) / pixels_per_meter)

    for z in range(0, max_depth_m + 1, 5):
        y = int(round(h - z * pixels_per_meter))
        color = (60, 60, 60) if z % 10 else (80, 80, 80)
        cv2.line(canvas, (0, y), (w, y), color, 1, cv2.LINE_AA)

    for x in range(-max_side_m, max_side_m + 1, 5):
        x_px = int(round(w * 0.5 + x * pixels_per_meter))
        color = (60, 60, 60) if x % 10 else (80, 80, 80)
        cv2.line(canvas, (x_px, 0), (x_px, h), color, 1, cv2.LINE_AA)

    cv2.circle(canvas, (w // 2, h - 1), 6, (40, 190, 255), -1)
    cv2.putText(canvas, "camera", (w // 2 + 10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (170, 220, 255), 1, cv2.LINE_AA)


def _project_poly_to_bev(
    polygons_uv: List[np.ndarray],
    camera: Dict[str, float],
    cfg: PipelineConfig,
) -> List[np.ndarray]:
    projected: List[np.ndarray] = []
    for poly in polygons_uv:
        if len(poly) > cfg.max_polygon_points:
            step = max(1, len(poly) // cfg.max_polygon_points)
            poly = poly[::step]

        bev_pts: List[List[int]] = []
        for u, v in poly:
            proj = project_uv_to_ground(float(u), float(v), camera, cfg.camera_height_m)
            if proj is None:
                continue
            x_m, z_m = proj
            x_px, y_px = world_to_bev(x_m, z_m, cfg.bev_width, cfg.bev_height, cfg.pixels_per_meter)
            bev_pts.append([x_px, y_px])

        if len(bev_pts) >= 3:
            projected.append(np.asarray(bev_pts, dtype=np.int32))
    return projected


def _project_objects_to_bev(
    detections: List[Dict[str, Any]],
    camera: Dict[str, float],
    cfg: PipelineConfig,
) -> List[Dict[str, Any]]:
    projected: List[Dict[str, Any]] = []
    for det in detections:
        foot_u, foot_v = det["footpoint_uv"]
        proj = project_uv_to_ground(float(foot_u), float(foot_v), camera, cfg.camera_height_m)
        if proj is None:
            continue

        x_m, z_m = proj
        x_px, y_px = world_to_bev(x_m, z_m, cfg.bev_width, cfg.bev_height, cfg.pixels_per_meter)

        item = dict(det)
        item["world_position_m"] = [x_m, z_m]
        item["bev_xy"] = [x_px, y_px]
        projected.append(item)
    return projected


def _track_color(track_id: int) -> Tuple[int, int, int]:
    # Deterministic BGR color from track id.
    r = (37 * track_id + 53) % 255
    g = (17 * track_id + 101) % 255
    b = (67 * track_id + 191) % 255
    return int(b), int(g), int(r)


class BEVObjectTracker:
    """Accumulates BEV trajectories using track_ids supplied by ByteTrack (via YOLO.track)."""

    def __init__(self, cfg: PipelineConfig):
        self.max_missed_frames = cfg.track_max_missed_frames
        self.max_history = cfg.trajectory_max_length
        self.tracks: Dict[int, Dict[str, Any]] = {}

    def _new_track(self, track_id: int, obj: Dict[str, Any]) -> None:
        history: Deque[Tuple[int, int]] = deque(maxlen=self.max_history)
        history.append(tuple(obj["bev_xy"]))
        self.tracks[track_id] = {
            "track_id": track_id,
            "class_name": obj["class_name"],
            "world_position_m": np.asarray(obj["world_position_m"], dtype=np.float32),
            "bev_xy": tuple(obj["bev_xy"]),
            "missed": 0,
            "history": history,
        }

    def update(self, objects_bev: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[int, List[Tuple[int, int]]]]:
        seen_track_ids: set[int] = set()

        tracked_objects: List[Dict[str, Any]] = []
        for obj in objects_bev:
            track_id = obj.get("track_id")
            if track_id is None:
                tracked_objects.append(obj)
                continue

            seen_track_ids.add(track_id)
            if track_id not in self.tracks:
                self._new_track(track_id, obj)
            else:
                track = self.tracks[track_id]
                track["world_position_m"] = np.asarray(obj["world_position_m"], dtype=np.float32)
                track["bev_xy"] = tuple(obj["bev_xy"])
                track["missed"] = 0
                track["history"].append(tuple(obj["bev_xy"]))

            item = dict(obj)
            item["track_id"] = track_id
            tracked_objects.append(item)

        to_delete: List[int] = []
        for track_id, track in self.tracks.items():
            if track_id in seen_track_ids:
                continue
            track["missed"] += 1
            if track["missed"] > self.max_missed_frames:
                to_delete.append(track_id)
        for track_id in to_delete:
            del self.tracks[track_id]

        trajectories = {
            track_id: list(track["history"])
            for track_id, track in self.tracks.items()
            if len(track["history"]) >= 2
        }
        return tracked_objects, trajectories


def render_bev_scene(
    road_polygons_uv: List[np.ndarray],
    crosswalk_polygons_uv: List[np.ndarray],
    detections: List[Dict[str, Any]],
    camera: Dict[str, float],
    cfg: PipelineConfig,
    projected_objects: Optional[List[Dict[str, Any]]] = None,
    trajectories: Optional[Dict[int, List[Tuple[int, int]]]] = None,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    canvas = np.full((cfg.bev_height, cfg.bev_width, 3), 24, dtype=np.uint8)
    _draw_bev_grid(canvas, cfg.pixels_per_meter)

    road_bev_polygons = _project_poly_to_bev(road_polygons_uv, camera, cfg)
    for poly in road_bev_polygons:
        cv2.fillPoly(canvas, [poly], color=(44, 92, 44))
        cv2.polylines(canvas, [poly], isClosed=True, color=(86, 160, 86), thickness=1, lineType=cv2.LINE_AA)

    crosswalk_bev_polygons = _project_poly_to_bev(crosswalk_polygons_uv, camera, cfg)
    for poly in crosswalk_bev_polygons:
        cv2.fillPoly(canvas, [poly], color=(0, 220, 220))
        #cv2.polylines(canvas, [poly], isClosed=True, color=(255, 255, 0), thickness=1, lineType=cv2.LINE_AA)

    if projected_objects is None:
        projected_objects = _project_objects_to_bev(detections, camera, cfg)

    if trajectories:
        for track_id, points in trajectories.items():
            if len(points) < 2:
                continue
            line = np.asarray(points, dtype=np.int32)
            cv2.polylines(
                canvas,
                [line],
                isClosed=False,
                color=_track_color(track_id),
                thickness=2,
                lineType=cv2.LINE_AA,
            )

    for obj in projected_objects:
        x_px, y_px = obj["bev_xy"]
        if 0 <= x_px < cfg.bev_width and 0 <= y_px < cfg.bev_height:
            track_id = obj.get("track_id")
            if track_id is None:
                color = (60, 160, 255)
                label = f"{obj['class_name']}:{obj['confidence']:.2f}"
            else:
                color = _track_color(int(track_id))
                label = f"T{int(track_id)} {obj['class_name']}:{obj['confidence']:.2f}"
            cv2.circle(canvas, (x_px, y_px), 5, color, -1)
            cv2.putText(
                canvas,
                label,
                (x_px + 6, y_px - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (225, 235, 245),
                1,
                cv2.LINE_AA,
            )

    cv2.putText(
        canvas,
        f"roll={camera['roll_deg']:.2f} pitch={camera['pitch_deg']:.2f} vfov={camera['vfov_deg']:.2f}",
        (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    return canvas, projected_objects


def render_image_overlay(
    image_bgr: np.ndarray,
    road_polygons_uv: List[np.ndarray],
    crosswalk_polygons_uv: List[np.ndarray],
    detections: List[Dict[str, Any]],
    track_id_map: Optional[Dict[int, int]] = None,
) -> np.ndarray:
    overlay = image_bgr.copy()

    if road_polygons_uv:
        mask_layer = overlay.copy()
        int_polys = [np.asarray(poly, dtype=np.int32) for poly in road_polygons_uv if len(poly) >= 3]
        if int_polys:
            cv2.fillPoly(mask_layer, int_polys, color=(0, 140, 0))
            overlay = cv2.addWeighted(mask_layer, 0.3, overlay, 0.7, 0.0)

    if crosswalk_polygons_uv:
        mask_layer = overlay.copy()
        int_polys = [np.asarray(poly, dtype=np.int32) for poly in crosswalk_polygons_uv if len(poly) >= 3]
        if int_polys:
            cv2.fillPoly(mask_layer, int_polys, color=(0, 220, 220))
            overlay = cv2.addWeighted(mask_layer, 0.3, overlay, 0.7, 0.0)

    for det in detections:
        x1, y1, x2, y2 = det["bbox_xyxy"]
        foot_u, foot_v = det["footpoint_uv"]
        det_id = int(det.get("detection_id", -1))
        track_id = track_id_map.get(det_id) if track_id_map else None
        if track_id is None:
            color = (0, 220, 255)
            label = f"{det['class_name']}:{det['confidence']:.2f}"
        else:
            color = _track_color(int(track_id))
            label = f"T{int(track_id)} {det['class_name']}:{det['confidence']:.2f}"

        cv2.rectangle(overlay, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.circle(overlay, (int(foot_u), int(foot_v)), 4, (20, 20, 255), -1)
        cv2.putText(
            overlay,
            label,
            (int(x1), max(20, int(y1) - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
    return overlay


def export_scene_json(
    camera: Dict[str, float],
    road_polygons_uv: List[np.ndarray],
    crosswalk_polygons_uv: List[np.ndarray],
    projected_objects: List[Dict[str, Any]],
    trajectories: Optional[Dict[int, List[Tuple[int, int]]]] = None,
    cfg: Optional[PipelineConfig] = None,
    frame_index: int = 0,
) -> Dict[str, Any]:
    """Build a Unity-friendly scene description in real-world (meter) coordinates.

    Schema is shaped for `JsonUtility.FromJson<T>` consumption (no dictionaries).
    All polygons are projected from UV to ground plane using the same homography
    used for BEV rendering. Trajectory points stored in BEV pixel space are
    inverse-projected back to world meters.
    """
    cfg = cfg or PipelineConfig()
    cam_h = float(cfg.camera_height_m)

    def _poly_to_world(polygons_uv: List[np.ndarray]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for poly in polygons_uv:
            if len(poly) > cfg.max_polygon_points:
                step = max(1, len(poly) // cfg.max_polygon_points)
                poly = poly[::step]
            pts: List[Dict[str, float]] = []
            for u, v in poly:
                proj = project_uv_to_ground(float(u), float(v), camera, cam_h)
                if proj is None:
                    continue
                pts.append({"x": round(float(proj[0]), 3), "z": round(float(proj[1]), 3)})
            if len(pts) >= 3:
                out.append({"points": pts})
        return out

    objects_out: List[Dict[str, Any]] = []
    for obj in projected_objects:
        wp = obj.get("world_position_m")
        if wp is None:
            continue
        track_id = obj.get("track_id")
        objects_out.append(
            {
                "track_id": int(track_id) if track_id is not None else -1,
                "class_name": str(obj.get("class_name", "")),
                "confidence": round(float(obj.get("confidence", 0.0)), 3),
                "x_m": round(float(wp[0]), 3),
                "z_m": round(float(wp[1]), 3),
            }
        )

    trajectories_out: List[Dict[str, Any]] = []
    if trajectories:
        for tid, points in trajectories.items():
            world_pts: List[Dict[str, float]] = []
            for p in points:
                wx, wz = bev_to_world(p[0], p[1], cfg.bev_width, cfg.bev_height, cfg.pixels_per_meter)
                world_pts.append({"x": round(wx, 3), "z": round(wz, 3)})
            if len(world_pts) >= 2:
                trajectories_out.append({"track_id": int(tid), "points": world_pts})

    return {
        "camera": {
            "height_m": round(cam_h, 4),
            "pitch_deg": round(float(camera.get("pitch_deg", 0.0)), 4),
            "roll_deg": round(float(camera.get("roll_deg", 0.0)), 4),
            "vfov_deg": round(float(camera.get("vfov_deg", 0.0)), 4),
        },
        "road_polygons": _poly_to_world(road_polygons_uv),
        "crosswalk_polygons": _poly_to_world(crosswalk_polygons_uv),
        "objects": objects_out,
        "trajectories": trajectories_out,
        "frame_index": int(frame_index),
    }


def write_scene_json(
    scene: Dict[str, Any],
    primary_path: Path,
    cfg: Optional[PipelineConfig] = None,
) -> List[Path]:
    """Write the scene JSON to ``primary_path`` and mirror to Unity StreamingAssets."""
    written: List[Path] = []
    primary_path.parent.mkdir(parents=True, exist_ok=True)
    with primary_path.open("w", encoding="utf-8") as f:
        json.dump(scene, f, ensure_ascii=False)
    written.append(primary_path)

    cfg = cfg or PipelineConfig()
    if cfg.unity_streaming_assets_dir:
        # Resolve relative path against this file's repo root so it works
        # regardless of the caller's CWD.
        repo_root = Path(__file__).resolve().parent
        unity_dir = Path(cfg.unity_streaming_assets_dir)
        if not unity_dir.is_absolute():
            unity_dir = repo_root / unity_dir
        try:
            unity_dir.mkdir(parents=True, exist_ok=True)
            unity_path = unity_dir / cfg.unity_scene_filename
            with unity_path.open("w", encoding="utf-8") as f:
                json.dump(scene, f, ensure_ascii=False)
            written.append(unity_path)
        except OSError:
            # Unity 프로젝트가 없을 수도 있으므로 mirror 실패는 치명적이지 않음
            pass
    return written


class RoadSceneProjector:
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.device = _resolve_device(self.config.device)
        self.road_model = YOLO(self.config.road_model_path)
        self.crosswalk_model = YOLO(self.config.crosswalk_model_path)
        self.perspective_model = _load_perspective_model(self.config.perspective_version, self.device)

        self._use_rfdetr = self.config.object_detector_type == "rfdetr"
        if self._use_rfdetr:
            self.object_model = _load_rfdetr_model(self.config.rfdetr_object_model_path)
            try:
                import supervision as sv  # type: ignore
                self._sv_tracker = sv.ByteTrack()
            except ImportError:
                raise ImportError("supervision 패키지가 없습니다. `pip install supervision` 을 실행하세요.")
        else:
            self.object_model = YOLO(self.config.object_model_path)
            self._sv_tracker = None

    def run(self, image_path: str, save_dir: Optional[str] = None) -> Dict[str, Any]:
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            raise FileNotFoundError(f"Failed to read image: {image_path}")

        if self.config.use_clahe:
            image_bgr = _apply_clahe(image_bgr)

        road_out = infer_road_model(
            image_bgr=image_bgr,
            model=self.road_model,
            conf=self.config.road_conf,
            device=self.device,
        )
        crosswalk_out = infer_crosswalk_model(
            image_bgr=image_bgr,
            model=self.crosswalk_model,
            conf=self.config.crosswalk_conf,
            device=self.device,
        )
        if self._use_rfdetr:
            obj_out = infer_object_model_rfdetr(
                image_bgr=image_bgr,
                model=self.object_model,
                conf=self.config.object_conf,
                class_names=OBJECT_CLASS_NAMES,
            )
        else:
            obj_out = infer_object_model(
                image_bgr=image_bgr,
                model=self.object_model,
                conf=self.config.object_conf,
                device=self.device,
            )

        camera = estimate_camera_params(image_bgr, self.perspective_model)
        bev_image, projected_objects = render_bev_scene(
            road_polygons_uv=road_out["road_polygons_uv"],
            crosswalk_polygons_uv=crosswalk_out["crosswalk_polygons_uv"],
            detections=obj_out["detections"],
            camera=camera,
            cfg=self.config,
        )
        overlay_image = render_image_overlay(
            image_bgr=image_bgr,
            road_polygons_uv=road_out["road_polygons_uv"],
            crosswalk_polygons_uv=crosswalk_out["crosswalk_polygons_uv"],
            detections=obj_out["detections"],
        )

        scene_data = export_scene_json(
            camera=camera,
            road_polygons_uv=road_out["road_polygons_uv"],
            crosswalk_polygons_uv=crosswalk_out["crosswalk_polygons_uv"],
            projected_objects=projected_objects,
            trajectories=None,
            cfg=self.config,
            frame_index=0,
        )

        outputs: Dict[str, Any] = {
            "image_path": image_path,
            "camera": camera,
            "road_polygons_uv": road_out["road_polygons_uv"],
            "crosswalk_polygons_uv": crosswalk_out["crosswalk_polygons_uv"],
            "detections_2d": obj_out["detections"],
            "detections_bev": projected_objects,
            "overlay_image": overlay_image,
            "bev_image": bev_image,
            "scene_data": scene_data,
        }

        if save_dir is not None:
            save_root = Path(save_dir)
            save_root.mkdir(parents=True, exist_ok=True)
            stem = Path(image_path).stem

            overlay_path = save_root / f"{stem}_overlay.png"
            bev_path = save_root / f"{stem}_bev.png"
            cv2.imwrite(str(overlay_path), overlay_image)
            cv2.imwrite(str(bev_path), bev_image)

            scene_path = save_root / f"{stem}_scene.json"
            written = write_scene_json(scene_data, scene_path, cfg=self.config)

            outputs["saved_overlay_path"] = str(overlay_path)
            outputs["saved_bev_path"] = str(bev_path)
            outputs["saved_scene_paths"] = [str(p) for p in written]

        return outputs

    def _process_video_frame(
        self,
        frame_bgr: np.ndarray,
        road_polygons_uv: List[np.ndarray],
        crosswalk_polygons_uv: List[np.ndarray],
        camera: Dict[str, float],
        tracker: BEVObjectTracker,
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]], List[Dict[str, Any]], Dict[int, List[Tuple[int, int]]]]:
        if self._use_rfdetr:
            obj_out = infer_object_model_rfdetr(
                image_bgr=frame_bgr,
                model=self.object_model,
                conf=self.config.object_conf,
                class_names=OBJECT_CLASS_NAMES,
                sv_tracker=self._sv_tracker,
            )
        else:
            obj_out = infer_object_model(
                image_bgr=frame_bgr,
                model=self.object_model,
                conf=self.config.object_conf,
                device=self.device,
                use_tracker=True,
            )
        projected = _project_objects_to_bev(obj_out["detections"], camera, self.config)
        tracked_objects, trajectories = tracker.update(projected)
        track_id_map = {
            int(item["detection_id"]): int(item["track_id"])
            for item in tracked_objects
            if "detection_id" in item and item.get("track_id") is not None
        }

        bev_image, _ = render_bev_scene(
            road_polygons_uv=road_polygons_uv,
            crosswalk_polygons_uv=crosswalk_polygons_uv,
            detections=obj_out["detections"],
            camera=camera,
            cfg=self.config,
            projected_objects=tracked_objects,
            trajectories=trajectories,
        )
        overlay_image = render_image_overlay(
            image_bgr=frame_bgr,
            road_polygons_uv=road_polygons_uv,
            crosswalk_polygons_uv=crosswalk_polygons_uv,
            detections=obj_out["detections"],
            track_id_map=track_id_map,
        )
        return overlay_image, bev_image, tracked_objects, obj_out["detections"], trajectories

    def run_video(self, video_path: str, save_dir: Optional[str] = None) -> Dict[str, Any]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Failed to open video: {video_path}")

        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if fps <= 1e-3:
            fps = 30.0
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        ok, first_frame = cap.read()
        if not ok or first_frame is None:
            cap.release()
            raise RuntimeError(f"Video has no readable frame: {video_path}")

        if self.config.use_clahe:
            first_frame = _apply_clahe(first_frame)

        # Road model is intentionally run once for static-camera video.
        road_out = infer_road_model(
            image_bgr=first_frame,
            model=self.road_model,
            conf=self.config.road_conf,
            device=self.device,
        )
        road_polygons_uv = road_out["road_polygons_uv"]
        crosswalk_out = infer_crosswalk_model(
            image_bgr=first_frame,
            model=self.crosswalk_model,
            conf=self.config.crosswalk_conf,
            device=self.device,
        )
        crosswalk_polygons_uv = crosswalk_out["crosswalk_polygons_uv"]

        camera = estimate_camera_params(first_frame, self.perspective_model)
        tracker = BEVObjectTracker(self.config)

        save_root = Path(save_dir or "output")
        save_root.mkdir(parents=True, exist_ok=True)
        stem = Path(video_path).stem
        output_video_path = save_root / f"{stem}_tracked_bev.mp4"

        out_w = frame_w * 2
        out_h = frame_h
        writer = cv2.VideoWriter(
            str(output_video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (out_w, out_h),
        )
        if not writer.isOpened():
            cap.release()
            raise RuntimeError(f"Failed to create output video writer: {output_video_path}")

        frame_idx = 0
        total_detections_2d = 0
        total_tracked = 0
        last_overlay = None
        last_bev = None
        last_trajectories: Dict[int, List[Tuple[int, int]]] = {}
        last_tracked_objects: List[Dict[str, Any]] = []

        while True:
            if frame_idx == 0:
                frame = first_frame
            else:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                if self.config.use_clahe:
                    frame = _apply_clahe(frame)

            if self.config.video_recompute_camera_each_frame and frame_idx > 0:
                camera = estimate_camera_params(frame, self.perspective_model)

            overlay, bev, tracked_objects, detections_2d, trajectories = self._process_video_frame(
                frame_bgr=frame,
                road_polygons_uv=road_polygons_uv,
                crosswalk_polygons_uv=crosswalk_polygons_uv,
                camera=camera,
                tracker=tracker,
            )

            bev_resized = cv2.resize(bev, (frame_w, frame_h), interpolation=cv2.INTER_LINEAR)
            vis = np.hstack([overlay, bev_resized])
            cv2.putText(
                vis,
                f"frame={frame_idx}",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (220, 220, 220),
                2,
                cv2.LINE_AA,
            )

            writer.write(vis)
            total_detections_2d += len(detections_2d)
            total_tracked += len(tracked_objects)
            frame_idx += 1
            last_overlay = overlay
            last_bev = bev
            last_trajectories = trajectories
            last_tracked_objects = tracked_objects

        cap.release()
        writer.release()

        scene_data = export_scene_json(
            camera=camera,
            road_polygons_uv=road_polygons_uv,
            crosswalk_polygons_uv=crosswalk_polygons_uv,
            projected_objects=last_tracked_objects,
            trajectories=last_trajectories,
            cfg=self.config,
            frame_index=max(0, frame_idx - 1),
        )
        scene_path = save_root / f"{stem}_scene.json"
        written_scene_paths = write_scene_json(scene_data, scene_path, cfg=self.config)

        outputs: Dict[str, Any] = {
            "video_path": video_path,
            "saved_video_path": str(output_video_path),
            "frames_processed": frame_idx,
            "camera": camera,
            "road_polygons_uv": road_polygons_uv,
            "crosswalk_polygons_uv": crosswalk_polygons_uv,
            "total_detections_2d": total_detections_2d,
            "avg_detections_2d_per_frame": (total_detections_2d / frame_idx) if frame_idx else 0.0,
            "avg_tracked_per_frame": (total_tracked / frame_idx) if frame_idx else 0.0,
            "last_overlay_image": last_overlay,
            "last_bev_image": last_bev,
            "last_trajectories": last_trajectories,
            "scene_data": scene_data,
            "saved_scene_paths": [str(p) for p in written_scene_paths],
        }
        return outputs


def run_pipeline(
    image_path: str,
    save_dir: str = "output",
    config: Optional[PipelineConfig] = None,
) -> Dict[str, Any]:
    projector = RoadSceneProjector(config=config)
    return projector.run(image_path=image_path, save_dir=save_dir)


def run_video_pipeline(
    video_path: str,
    save_dir: str = "output",
    config: Optional[PipelineConfig] = None,
) -> Dict[str, Any]:
    projector = RoadSceneProjector(config=config)
    return projector.run_video(video_path=video_path, save_dir=save_dir)

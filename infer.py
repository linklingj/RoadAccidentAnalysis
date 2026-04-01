from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO


DEFAULT_ROAD_MODEL_PATH = "runs/segment/0401-road/weights/best.pt"
DEFAULT_OBJECT_MODEL_PATH = "runs/segment/0401-object/weights/best.pt"
DEFAULT_PERSPECTIVE_VERSION = "Paramnet-360Cities-edina-centered"


@dataclass
class PipelineConfig:
    road_model_path: str = DEFAULT_ROAD_MODEL_PATH
    object_model_path: str = DEFAULT_OBJECT_MODEL_PATH
    perspective_version: str = DEFAULT_PERSPECTIVE_VERSION
    road_conf: float = 0.25
    object_conf: float = 0.15
    camera_height_m: float = 3.5
    pixels_per_meter: float = 42.0
    bev_width: int = 960
    bev_height: int = 960
    max_polygon_points: int = 600
    device: Optional[Any] = None


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


def _extract_road_polygons(result: Any) -> List[np.ndarray]:
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

    for i, box in enumerate(result.boxes):
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = float(box.conf.item())
        cls_id = int(box.cls.item())
        class_name = names[cls_id] if isinstance(names, dict) and cls_id in names else str(cls_id)
        # skip 'attention'
        if class_name == "attention":
            continue

        foot_x = (x1 + x2) * 0.5
        foot_y = y2
        if masks_xy is not None and i < len(masks_xy):
            contour = np.asarray(masks_xy[i], dtype=np.float32)
            if contour.size > 0:
                lowest_idx = int(np.argmax(contour[:, 1]))
                foot_x = float(contour[lowest_idx, 0])
                foot_y = float(contour[lowest_idx, 1])

        detections.append(
            {
                "class_id": cls_id,
                "class_name": class_name,
                "confidence": conf,
                "bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)],
                "footpoint_uv": [float(foot_x), float(foot_y)],
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


def infer_object_model(
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
        "detections": _extract_object_detections(result),
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


def _project_road_to_bev(
    road_polygons_uv: List[np.ndarray],
    camera: Dict[str, float],
    cfg: PipelineConfig,
) -> List[np.ndarray]:
    projected: List[np.ndarray] = []
    for poly in road_polygons_uv:
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


def render_bev_scene(
    road_polygons_uv: List[np.ndarray],
    detections: List[Dict[str, Any]],
    camera: Dict[str, float],
    cfg: PipelineConfig,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    canvas = np.full((cfg.bev_height, cfg.bev_width, 3), 24, dtype=np.uint8)
    _draw_bev_grid(canvas, cfg.pixels_per_meter)

    road_bev_polygons = _project_road_to_bev(road_polygons_uv, camera, cfg)
    for poly in road_bev_polygons:
        cv2.fillPoly(canvas, [poly], color=(44, 92, 44))
        cv2.polylines(canvas, [poly], isClosed=True, color=(86, 160, 86), thickness=1, lineType=cv2.LINE_AA)

    projected_objects = _project_objects_to_bev(detections, camera, cfg)
    for obj in projected_objects:
        x_px, y_px = obj["bev_xy"]
        if 0 <= x_px < cfg.bev_width and 0 <= y_px < cfg.bev_height:
            cv2.circle(canvas, (x_px, y_px), 5, (60, 160, 255), -1)
            label = f"{obj['class_name']}:{obj['confidence']:.2f}"
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
    detections: List[Dict[str, Any]],
) -> np.ndarray:
    overlay = image_bgr.copy()

    if road_polygons_uv:
        mask_layer = overlay.copy()
        int_polys = [np.asarray(poly, dtype=np.int32) for poly in road_polygons_uv if len(poly) >= 3]
        if int_polys:
            cv2.fillPoly(mask_layer, int_polys, color=(0, 140, 0))
            overlay = cv2.addWeighted(mask_layer, 0.3, overlay, 0.7, 0.0)

    for det in detections:
        x1, y1, x2, y2 = det["bbox_xyxy"]
        foot_u, foot_v = det["footpoint_uv"]
        cv2.rectangle(overlay, (int(x1), int(y1)), (int(x2), int(y2)), (0, 220, 255), 2)
        cv2.circle(overlay, (int(foot_u), int(foot_v)), 4, (20, 20, 255), -1)
        label = f"{det['class_name']}:{det['confidence']:.2f}"
        cv2.putText(
            overlay,
            label,
            (int(x1), max(20, int(y1) - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 220, 255),
            2,
            cv2.LINE_AA,
        )
    return overlay


class RoadSceneProjector:
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.device = _resolve_device(self.config.device)

        self.road_model = YOLO(self.config.road_model_path)
        self.object_model = YOLO(self.config.object_model_path)
        self.perspective_model = _load_perspective_model(self.config.perspective_version, self.device)

    def run(self, image_path: str, save_dir: Optional[str] = None) -> Dict[str, Any]:
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            raise FileNotFoundError(f"Failed to read image: {image_path}")

        road_out = infer_road_model(
            image_bgr=image_bgr,
            model=self.road_model,
            conf=self.config.road_conf,
            device=self.device,
        )
        obj_out = infer_object_model(
            image_bgr=image_bgr,
            model=self.object_model,
            conf=self.config.object_conf,
            device=self.device,
        )

        camera = estimate_camera_params(image_bgr, self.perspective_model)
        bev_image, projected_objects = render_bev_scene(
            road_polygons_uv=road_out["road_polygons_uv"],
            detections=obj_out["detections"],
            camera=camera,
            cfg=self.config,
        )
        overlay_image = render_image_overlay(
            image_bgr=image_bgr,
            road_polygons_uv=road_out["road_polygons_uv"],
            detections=obj_out["detections"],
        )

        outputs: Dict[str, Any] = {
            "image_path": image_path,
            "camera": camera,
            "road_polygons_uv": road_out["road_polygons_uv"],
            "detections_2d": obj_out["detections"],
            "detections_bev": projected_objects,
            "overlay_image": overlay_image,
            "bev_image": bev_image,
        }

        if save_dir is not None:
            save_root = Path(save_dir)
            save_root.mkdir(parents=True, exist_ok=True)
            stem = Path(image_path).stem

            overlay_path = save_root / f"{stem}_overlay.png"
            bev_path = save_root / f"{stem}_bev.png"
            cv2.imwrite(str(overlay_path), overlay_image)
            cv2.imwrite(str(bev_path), bev_image)

            outputs["saved_overlay_path"] = str(overlay_path)
            outputs["saved_bev_path"] = str(bev_path)

        return outputs


def run_pipeline(
    image_path: str,
    save_dir: str = "output",
    config: Optional[PipelineConfig] = None,
) -> Dict[str, Any]:
    projector = RoadSceneProjector(config=config)
    return projector.run(image_path=image_path, save_dir=save_dir)

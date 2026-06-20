from __future__ import annotations

import json
import math
import sys
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO


DEFAULT_ROAD_MODEL_PATH = "runs/segment/0401-road/weights/best.pt"
DEFAULT_ROAD_UNET_MODEL_PATH = "runs/smp-road/best.pt"
DEFAULT_CROSSWALK_MODEL_PATH = "runs/segment/0406-crosswalk/weights/best.pt"
DEFAULT_OBJECT_MODEL_PATH = "runs/segment/0619-object/weights/best.pt"
DEFAULT_RFDETR_OBJECT_MODEL_PATH = "runs/detect/rfdetr-object-0519/best_checkpoint.pth"
DEFAULT_PERSPECTIVE_VERSION = "Paramnet-360Cities-edina-centered"

# cctv-object-dataset/data.yaml names 순서와 동일하게 유지
OBJECT_CLASS_NAMES: List[str] = ["bus", "car", "person", "riders", "truck"]


@dataclass
class PipelineConfig:
    road_model_path: str = DEFAULT_ROAD_MODEL_PATH
    crosswalk_model_path: str = DEFAULT_CROSSWALK_MODEL_PATH
    object_model_path: str = DEFAULT_OBJECT_MODEL_PATH
    # 도로 segmentation 백엔드: "yolo"(YOLO-seg) | "unet"(SMP U-Net)
    # U-Net 비교/통합은 docs/unet_vs_yolo_roadseg.md 참고
    road_detector_type: str = "yolo"
    road_unet_model_path: str = DEFAULT_ROAD_UNET_MODEL_PATH
    road_unet_imgsz: int = 512
    road_unet_threshold: float = 0.5
    road_unet_min_area: int = 1500  # 이보다 작은 도로 마스크 조각(노이즈)은 폐기
    # "yolo" | "rfdetr"
    object_detector_type: str = "yolo"
    rfdetr_object_model_path: str = DEFAULT_RFDETR_OBJECT_MODEL_PATH
    perspective_version: str = DEFAULT_PERSPECTIVE_VERSION
    road_conf: float = 0.25
    object_conf: float = 0.1
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
    # 웹 뷰어 연동: scene JSON을 web/data로 미러링하여 three.js 뷰어가 바로 로드
    web_data_dir: Optional[str] = "web/data"
    web_scene_filename: str = "scene_data.json"


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

def _footpoint_from_mask_robust(
    contour: Optional[np.ndarray],
    mask_raster: Optional[np.ndarray],
    bbox_xyxy: Tuple[float, float, float, float],
) -> Tuple[float, float]:
    """Estimate the ground contact point of a vehicle mask with sub-pixel precision.

    Picking a single argmax(y) pixel makes the footpoint twitchy because mask edges
    fluctuate by 1-3 px between frames. Instead, sample the bottom rows of the mask,
    take the median x per row, then fit a 2nd-order polynomial x(y) and return the
    point at the deepest y. Falls back gracefully when the mask is too thin or
    missing.
    """
    x1, y1, x2, y2 = bbox_xyxy
    bbox_h = max(1.0, float(y2 - y1))
    k = int(np.clip(round(bbox_h * 0.08), 3, 12))

    fallback_x = (x1 + x2) * 0.5
    fallback_y = y2

    if mask_raster is not None and mask_raster.size > 0:
        rows_with_pixels = np.flatnonzero(mask_raster.any(axis=1))
        if rows_with_pixels.size == 0:
            return float(fallback_x), float(fallback_y)
        y_bot = int(rows_with_pixels[-1])
        y_lo = max(0, y_bot - k + 1)
        ys: List[float] = []
        xs: List[float] = []
        for y in range(y_lo, y_bot + 1):
            row = mask_raster[y]
            xs_active = np.flatnonzero(row)
            if xs_active.size == 0:
                continue
            ys.append(float(y))
            xs.append(float(np.median(xs_active)))
        if len(ys) >= 3:
            ys_arr = np.asarray(ys, dtype=np.float64)
            xs_arr = np.asarray(xs, dtype=np.float64)
            try:
                a, b, c = np.polyfit(ys_arr, xs_arr, 2)
                y_star = float(ys_arr.max())
                x_star = float(a * y_star * y_star + b * y_star + c)
                return x_star, y_star
            except (np.linalg.LinAlgError, ValueError):
                pass
        if len(ys) >= 1:
            return float(xs[-1]), float(ys[-1])
        return float(fallback_x), float(fallback_y)

    if contour is not None and contour.size > 0:
        cy = contour[:, 1]
        y_bot = float(cy.max())
        band = contour[cy >= y_bot - max(2.0, bbox_h * 0.05)]
        if band.shape[0] >= 3:
            ys_arr = band[:, 1].astype(np.float64)
            xs_arr = band[:, 0].astype(np.float64)
            try:
                a, b, c = np.polyfit(ys_arr, xs_arr, 2)
                y_star = float(ys_arr.max())
                x_star = float(a * y_star * y_star + b * y_star + c)
                return x_star, y_star
            except (np.linalg.LinAlgError, ValueError):
                pass
        return float(np.median(band[:, 0])), float(y_bot)

    return float(fallback_x), float(fallback_y)


# 차량을 'white' / 'black'으로 가르는 명도(median luma, 0-255) 임계값. CCTV 크롭 기준.
CAR_COLOR_BRIGHTNESS_THRESH = 95.0


def _estimate_car_color(
    image_bgr: Optional[np.ndarray],
    raster: Optional[np.ndarray],
    bbox: Tuple[float, float, float, float],
) -> Optional[str]:
    """차량 픽셀의 명도로 'white' / 'black'을 분류한다.

    가능하면 segmentation 마스크 안쪽 픽셀을 쓰고, 마스크가 없으면 도로/배경이
    덜 섞이도록 bbox 중앙 영역을 사용한다. 판단 불가 시 None.
    명도는 창문·그림자에 흔들리지 않도록 luma의 median을 쓴다.
    """
    if image_bgr is None or image_bgr.size == 0:
        return None
    h, w = image_bgr.shape[:2]
    x1, y1, x2, y2 = (int(round(v)) for v in bbox)
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(x1 + 1, min(w, x2))
    y2 = max(y1 + 1, min(h, y2))

    pixels: Optional[np.ndarray] = None
    if raster is not None and raster.shape[:2] == (h, w):
        sub_mask = raster[y1:y2, x1:x2].astype(bool)
        if int(sub_mask.sum()) >= 25:
            pixels = image_bgr[y1:y2, x1:x2][sub_mask]
    if pixels is None:
        bw, bh = x2 - x1, y2 - y1
        cx1, cx2 = x1 + int(bw * 0.25), x2 - int(bw * 0.25)
        cy1, cy2 = y1 + int(bh * 0.25), y2 - int(bh * 0.25)
        if cx2 <= cx1 or cy2 <= cy1:
            cx1, cy1, cx2, cy2 = x1, y1, x2, y2
        pixels = image_bgr[cy1:cy2, cx1:cx2].reshape(-1, 3)
    if pixels.size == 0:
        return None

    luma = 0.114 * pixels[:, 0] + 0.587 * pixels[:, 1] + 0.299 * pixels[:, 2]
    return "white" if float(np.median(luma)) >= CAR_COLOR_BRIGHTNESS_THRESH else "black"


def _extract_object_detections(
    result: Any,
    color_image_bgr: Optional[np.ndarray] = None,
) -> List[Dict[str, Any]]:
    detections: List[Dict[str, Any]] = []
    if result is None or result.boxes is None:
        return detections

    names = result.names if hasattr(result, "names") else {}
    masks_xy = result.masks.xy if result.masks is not None else None
    masks_data = result.masks.data if result.masks is not None else None
    track_ids = result.boxes.id  # None when using predict(), Tensor when using track()

    img_h, img_w = (None, None)
    if hasattr(result, "orig_shape") and result.orig_shape is not None:
        img_h, img_w = int(result.orig_shape[0]), int(result.orig_shape[1])

    for i, box in enumerate(result.boxes):
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = float(box.conf.item())
        cls_id = int(box.cls.item())
        class_name = names[cls_id] if isinstance(names, dict) and cls_id in names else str(cls_id)
        # skip 'attention' & 'crosswalk'
        if class_name == "attention" or class_name == "crosswalk":
            continue

        contour_i: Optional[np.ndarray] = None
        if masks_xy is not None and i < len(masks_xy):
            contour_i = np.asarray(masks_xy[i], dtype=np.float32)

        raster_i: Optional[np.ndarray] = None
        if masks_data is not None and i < len(masks_data):
            mask_tensor = masks_data[i]
            if hasattr(mask_tensor, "cpu"):
                raster_i = mask_tensor.detach().cpu().numpy()
            else:
                raster_i = np.asarray(mask_tensor)
            raster_i = (raster_i > 0.5).astype(np.uint8)
            if img_h is not None and img_w is not None:
                if raster_i.shape != (img_h, img_w):
                    raster_i = cv2.resize(raster_i, (img_w, img_h), interpolation=cv2.INTER_NEAREST)

        foot_x, foot_y = _footpoint_from_mask_robust(contour_i, raster_i, (x1, y1, x2, y2))

        track_id = int(track_ids[i].item()) if track_ids is not None else None

        det: Dict[str, Any] = {
            "detection_id": i,
            "class_id": cls_id,
            "class_name": class_name,
            "confidence": conf,
            "bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)],
            "footpoint_uv": [float(foot_x), float(foot_y)],
            "track_id": track_id,
        }
        # 색 분류는 'car'에만 적용한다 (요청 범위).
        if class_name == "car" and color_image_bgr is not None:
            color = _estimate_car_color(color_image_bgr, raster_i, (x1, y1, x2, y2))
            if color is not None:
                det["color"] = color
        detections.append(det)
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
    color_image_bgr: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    common_kwargs = dict(source=image_bgr, save=False, conf=conf, device=device, verbose=False)
    if use_tracker:
        results = model.track(**common_kwargs, tracker="bytetrack.yaml", persist=True)
    else:
        results = model.predict(**common_kwargs)
    result = results[0] if results else None
    return {
        "raw_result": result,
        "detections": _extract_object_detections(result, color_image_bgr=color_image_bgr),
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
    color_image_bgr: Optional[np.ndarray] = None,
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

        det: Dict[str, Any] = {
            "detection_id": i,
            "class_id": cls_id,
            "class_name": class_name,
            "confidence": conf,
            "bbox_xyxy": [x1, y1, x2, y2],
            "footpoint_uv": [foot_x, foot_y],
            "track_id": track_id,
        }
        # RF-DETR는 마스크가 없어 bbox 기반으로 'car' 색을 분류한다.
        if class_name == "car" and color_image_bgr is not None:
            color = _estimate_car_color(color_image_bgr, None, (x1, y1, x2, y2))
            if color is not None:
                det["color"] = color
        results.append(det)
    return results


def infer_object_model_rfdetr(
    image_bgr: np.ndarray,
    model: Any,
    conf: float,
    class_names: List[str],
    sv_tracker: Optional[Any] = None,
    color_image_bgr: Optional[np.ndarray] = None,
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
        "detections": _extract_rfdetr_detections(
            detections, class_names, color_image_bgr=color_image_bgr
        ),
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


def project_uv_to_ground_with_cov(
    u: float,
    v: float,
    camera: Dict[str, float],
    camera_height_m: float,
    sigma_uv_px: float = 1.5,
    ray_y_min: float = 0.05,
) -> Optional[Tuple[float, float, float, float, float]]:
    """Project a pixel to the ground plane and propagate UV noise to (x, z) covariance.

    Returns (x_m, z_m, cov_xx, cov_xz, cov_zz). Returns None when the ray points too
    close to horizontal (ground projection is then very unstable; clamp via ray_y_min).
    """
    focal = camera["focal_px"]
    if focal <= 1e-6:
        return None
    if sigma_uv_px <= 0.0:
        sigma_uv_px = 1.0

    cx = camera["cx"]
    cy = camera["cy"]
    roll_rad = math.radians(camera["roll_deg"])
    pitch_rad = math.radians(camera["pitch_deg"])

    x_norm = (u - cx) / focal
    y_norm = (v - cy) / focal
    ray = _camera_ray_to_world(x_norm, y_norm, roll_rad, pitch_rad)

    ry = float(ray[1])
    if ry < ray_y_min:
        return None

    t = camera_height_m / ry
    if t <= 0:
        return None

    x_m = float(t * ray[0])
    z_m = float(t * ray[2])
    if not (math.isfinite(x_m) and math.isfinite(z_m)):
        return None
    if z_m <= 0:
        return None

    # Jacobian of (x_m, z_m) w.r.t. (u, v):
    #   ray = R(x_norm, y_norm), with x_norm = (u-cx)/f, y_norm = (v-cy)/f
    #   x_m = camera_height * ray_x / ray_y
    #   z_m = camera_height * ray_z / ray_y
    cr, sr = math.cos(roll_rad), math.sin(roll_rad)
    cp, sp = math.cos(pitch_rad), math.sin(pitch_rad)

    # dray/dx_norm and dray/dy_norm (from _camera_ray_to_world)
    drx_dxn, drx_dyn = cr, -sr
    dry_dxn, dry_dyn = cp * sr, cp * cr
    drz_dxn, drz_dyn = sp * sr, sp * cr

    inv_f = 1.0 / focal
    h = float(camera_height_m)
    rx, rz = float(ray[0]), float(ray[2])

    # d(x_m)/dray_x = h/ry, d(x_m)/dray_y = -h*rx/ry^2
    dxm_drx = h / ry
    dxm_dry = -h * rx / (ry * ry)
    dzm_drx = 0.0
    dzm_drz = h / ry
    dzm_dry = -h * rz / (ry * ry)

    # Chain to (x_norm, y_norm) then to (u, v) via inv_f.
    dxm_du = (dxm_drx * drx_dxn + dxm_dry * dry_dxn) * inv_f
    dxm_dv = (dxm_drx * drx_dyn + dxm_dry * dry_dyn) * inv_f
    dzm_du = (dzm_drx * drx_dxn + dzm_drz * drz_dxn + dzm_dry * dry_dxn) * inv_f
    dzm_dv = (dzm_drx * drx_dyn + dzm_drz * drz_dyn + dzm_dry * dry_dyn) * inv_f

    s2 = float(sigma_uv_px) * float(sigma_uv_px)
    cov_xx = s2 * (dxm_du * dxm_du + dxm_dv * dxm_dv)
    cov_zz = s2 * (dzm_du * dzm_du + dzm_dv * dzm_dv)
    cov_xz = s2 * (dxm_du * dzm_du + dxm_dv * dzm_dv)

    return x_m, z_m, float(cov_xx), float(cov_xz), float(cov_zz)


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
        proj = project_uv_to_ground_with_cov(
            float(foot_u), float(foot_v), camera, cfg.camera_height_m
        )
        if proj is None:
            continue

        x_m, z_m, cxx, cxz, czz = proj
        x_px, y_px = world_to_bev(x_m, z_m, cfg.bev_width, cfg.bev_height, cfg.pixels_per_meter)

        item = dict(det)
        item["world_position_m"] = [x_m, z_m]
        item["world_cov"] = [cxx, cxz, czz]
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


def smooth_tracks_rts(
    track_observations: Dict[int, List[Dict[str, Any]]],
    fps: float,
    q_pos: float = 0.5,
    q_vel: float = 0.5,
    r_floor: float = 1e-3,
) -> Dict[int, Dict[int, Dict[str, float]]]:
    """Forward-backward (RTS) smoothing of per-track BEV positions.

    Each observation is a dict with keys: ``frame_index``, ``x_m``, ``z_m``,
    and optional ``cov_xx``, ``cov_xz``, ``cov_zz``. Returns a nested mapping
    ``{track_id: {frame_index: {x, z, vx, vz, x_var, z_var}}}`` with smoothed
    state mean + marginal position variance.

    State vector: [x, z, vx, vz]. Constant-velocity dynamics, white-noise jerk
    process model. Q is integrated over dt between consecutive observations,
    so gaps are handled naturally.
    """
    if fps is None or fps <= 1e-6:
        fps = 30.0
    base_dt = 1.0 / float(fps)
    qp = float(q_pos)
    qv = float(q_vel)
    smoothed_out: Dict[int, Dict[int, Dict[str, float]]] = {}

    for track_id, obs_list in track_observations.items():
        if len(obs_list) == 0:
            continue
        obs_sorted = sorted(obs_list, key=lambda o: int(o["frame_index"]))
        n = len(obs_sorted)

        means: List[np.ndarray] = []
        covs: List[np.ndarray] = []
        Fs: List[np.ndarray] = []
        Qs: List[np.ndarray] = []
        priors_mean: List[np.ndarray] = []
        priors_cov: List[np.ndarray] = []

        first = obs_sorted[0]
        x0 = float(first["x_m"])
        z0 = float(first["z_m"])
        rxx0 = max(float(first.get("cov_xx", 0.0)), r_floor)
        rzz0 = max(float(first.get("cov_zz", 0.0)), r_floor)
        # Initial state: position from measurement, velocity unknown.
        m_post = np.array([x0, z0, 0.0, 0.0], dtype=np.float64)
        P_post = np.diag([rxx0, rzz0, 25.0, 25.0])  # 5 m/s std velocity prior
        means.append(m_post.copy())
        covs.append(P_post.copy())

        prev_frame = int(first["frame_index"])
        for i in range(1, n):
            obs = obs_sorted[i]
            cur_frame = int(obs["frame_index"])
            frames_gap = max(1, cur_frame - prev_frame)
            dt = frames_gap * base_dt

            F = np.array([
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ])
            # Discrete white-noise acceleration model along each axis (decoupled).
            dt2 = dt * dt
            dt3 = dt2 * dt
            dt4 = dt2 * dt2
            qx = np.array([
                [dt4 / 4.0, dt3 / 2.0],
                [dt3 / 2.0, dt2],
            ]) * qp
            qv_block = np.array([
                [dt4 / 4.0, dt3 / 2.0],
                [dt3 / 2.0, dt2],
            ]) * qv
            Q = np.zeros((4, 4))
            Q[0:2, 0:2] = np.array([[qx[0, 0], 0.0], [0.0, qx[0, 0]]])
            Q[0, 2] = qx[0, 1]
            Q[2, 0] = qx[1, 0]
            Q[1, 3] = qv_block[0, 1]
            Q[3, 1] = qv_block[1, 0]
            Q[2, 2] = qx[1, 1]
            Q[3, 3] = qv_block[1, 1]

            m_pred = F @ m_post
            P_pred = F @ P_post @ F.T + Q

            priors_mean.append(m_pred.copy())
            priors_cov.append(P_pred.copy())
            Fs.append(F.copy())
            Qs.append(Q.copy())

            cxx = max(float(obs.get("cov_xx", 0.0)), r_floor)
            cxz = float(obs.get("cov_xz", 0.0))
            czz = max(float(obs.get("cov_zz", 0.0)), r_floor)
            R = np.array([[cxx, cxz], [cxz, czz]])
            H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
            z_meas = np.array([float(obs["x_m"]), float(obs["z_m"])])

            y = z_meas - H @ m_pred
            S = H @ P_pred @ H.T + R
            try:
                S_inv = np.linalg.inv(S)
            except np.linalg.LinAlgError:
                S_inv = np.linalg.pinv(S)
            K = P_pred @ H.T @ S_inv

            m_post = m_pred + K @ y
            P_post = (np.eye(4) - K @ H) @ P_pred

            means.append(m_post.copy())
            covs.append(P_post.copy())
            prev_frame = cur_frame

        # RTS backward sweep.
        sm_means = [m.copy() for m in means]
        sm_covs = [P.copy() for P in covs]
        for i in range(n - 2, -1, -1):
            F_next = Fs[i]
            P_pred_next = priors_cov[i]
            m_pred_next = priors_mean[i]
            try:
                P_pred_inv = np.linalg.inv(P_pred_next)
            except np.linalg.LinAlgError:
                P_pred_inv = np.linalg.pinv(P_pred_next)
            C = covs[i] @ F_next.T @ P_pred_inv
            sm_means[i] = means[i] + C @ (sm_means[i + 1] - m_pred_next)
            sm_covs[i] = covs[i] + C @ (sm_covs[i + 1] - P_pred_next) @ C.T

        per_frame: Dict[int, Dict[str, float]] = {}
        for i, obs in enumerate(obs_sorted):
            fi = int(obs["frame_index"])
            m = sm_means[i]
            P = sm_covs[i]
            per_frame[fi] = {
                "x": float(m[0]),
                "z": float(m[1]),
                "vx": float(m[2]),
                "vz": float(m[3]),
                "x_var": float(max(P[0, 0], 0.0)),
                "z_var": float(max(P[1, 1], 0.0)),
            }
        smoothed_out[int(track_id)] = per_frame

    return smoothed_out


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
    frames_history: Optional[List[Dict[str, Any]]] = None,
    tracks_registry: Optional[Dict[int, str]] = None,
    track_colors: Optional[Dict[int, str]] = None,
    fps: float = 0.0,
) -> Dict[str, Any]:
    """Build a web/three.js-friendly scene description in real-world (meter) coordinates.

    Emitted as plain JSON consumed by the three.js viewer (web/). All polygons are
    projected from UV to ground plane using the same homography used for BEV
    rendering. Trajectory points stored in BEV pixel space are inverse-projected
    back to world meters.
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

    def _bbox_dims_m(
        bbox: Optional[List[float]],
    ) -> Tuple[Optional[float], Optional[float]]:
        """Project bbox bottom edge → width_m; top–bottom centres → length_m."""
        if bbox is None:
            return None, None
        x1, y1, x2, y2 = bbox
        left = project_uv_to_ground(x1, y2, camera, cam_h)
        right = project_uv_to_ground(x2, y2, camera, cam_h)
        w = None
        if left is not None and right is not None:
            w = round(math.hypot(right[0] - left[0], right[1] - left[1]), 3)
        cx = (x1 + x2) * 0.5
        top_g = project_uv_to_ground(cx, y1, camera, cam_h)
        bot_g = project_uv_to_ground(cx, y2, camera, cam_h)
        l = None
        if top_g is not None and bot_g is not None:
            l = round(math.hypot(top_g[0] - bot_g[0], top_g[1] - bot_g[1]), 3)
        return w, l

    objects_out: List[Dict[str, Any]] = []
    for obj in projected_objects:
        wp = obj.get("world_position_m")
        if wp is None:
            continue
        track_id = obj.get("track_id")
        cov = obj.get("world_cov") or [0.0, 0.0, 0.0]
        entry: Dict[str, Any] = {
            "track_id": int(track_id) if track_id is not None else -1,
            "class_name": str(obj.get("class_name", "")),
            "confidence": round(float(obj.get("confidence", 0.0)), 3),
            "x_m": round(float(wp[0]), 3),
            "z_m": round(float(wp[1]), 3),
            "x_m_smoothed": round(float(wp[0]), 3),
            "z_m_smoothed": round(float(wp[1]), 3),
            "vx_m": 0.0,
            "vz_m": 0.0,
            "x_var": round(float(cov[0]), 6),
            "z_var": round(float(cov[2]), 6),
        }
        w_m, l_m = _bbox_dims_m(obj.get("bbox_xyxy"))
        if w_m is not None:
            entry["width_m"] = w_m
        if l_m is not None:
            entry["length_m"] = l_m
        if obj.get("color"):
            entry["color"] = str(obj["color"])
        objects_out.append(entry)

    trajectories_out: List[Dict[str, Any]] = []
    if trajectories:
        for tid, points in trajectories.items():
            world_pts: List[Dict[str, float]] = []
            for p in points:
                wx, wz = bev_to_world(p[0], p[1], cfg.bev_width, cfg.bev_height, cfg.pixels_per_meter)
                world_pts.append({"x": round(wx, 3), "z": round(wz, 3)})
            if len(world_pts) >= 2:
                trajectories_out.append({"track_id": int(tid), "points": world_pts})

    tracks_out: List[Dict[str, Any]] = []
    if tracks_registry:
        for tid, cname in tracks_registry.items():
            entry_t: Dict[str, Any] = {"track_id": int(tid), "class_name": str(cname)}
            if track_colors and tid in track_colors:
                entry_t["color"] = str(track_colors[tid])
            tracks_out.append(entry_t)

    frames_out: List[Dict[str, Any]] = []
    if frames_history:
        for entry in frames_history:
            frame_objs: List[Dict[str, Any]] = []
            for o in entry.get("objects", []):
                raw_x = float(o.get("x_m", 0.0))
                raw_z = float(o.get("z_m", 0.0))
                sm_x = float(o.get("x_m_smoothed", raw_x))
                sm_z = float(o.get("z_m_smoothed", raw_z))
                fobj: Dict[str, Any] = {
                    "track_id": int(o.get("track_id", -1)),
                    "class_name": str(o.get("class_name", "")),
                    "confidence": round(float(o.get("confidence", 0.0)), 3),
                    "x_m": round(raw_x, 3),
                    "z_m": round(raw_z, 3),
                    "x_m_smoothed": round(sm_x, 3),
                    "z_m_smoothed": round(sm_z, 3),
                    "vx_m": round(float(o.get("vx_m", 0.0)), 4),
                    "vz_m": round(float(o.get("vz_m", 0.0)), 4),
                    "x_var": round(float(o.get("x_var", 0.0)), 6),
                    "z_var": round(float(o.get("z_var", 0.0)), 6),
                }
                fw_m, fl_m = _bbox_dims_m(o.get("bbox_xyxy"))
                if fw_m is not None:
                    fobj["width_m"] = fw_m
                if fl_m is not None:
                    fobj["length_m"] = fl_m
                # 트랙 다수결 색을 우선 적용해 프레임 간 깜빡임을 막는다.
                color = (track_colors or {}).get(int(o.get("track_id", -1))) or o.get("color")
                if color:
                    fobj["color"] = str(color)
                frame_objs.append(fobj)
            frames_out.append(
                {
                    "frame_index": int(entry.get("frame_index", 0)),
                    "objects": frame_objs,
                }
            )

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
        "fps": round(float(fps), 4),
        "frame_count": len(frames_out),
        "tracks": tracks_out,
        "frames": frames_out,
    }


def write_scene_json(
    scene: Dict[str, Any],
    primary_path: Path,
    cfg: Optional[PipelineConfig] = None,
) -> List[Path]:
    """Write the scene JSON to ``primary_path`` and mirror to web/data for the viewer."""
    written: List[Path] = []
    primary_path.parent.mkdir(parents=True, exist_ok=True)
    with primary_path.open("w", encoding="utf-8") as f:
        json.dump(scene, f, ensure_ascii=False)
    written.append(primary_path)

    cfg = cfg or PipelineConfig()
    if cfg.web_data_dir:
        # Resolve relative path against this file's repo root so it works
        # regardless of the caller's CWD.
        repo_root = Path(__file__).resolve().parent
        web_dir = Path(cfg.web_data_dir)
        if not web_dir.is_absolute():
            web_dir = repo_root / web_dir
        try:
            web_dir.mkdir(parents=True, exist_ok=True)
            web_path = web_dir / cfg.web_scene_filename
            with web_path.open("w", encoding="utf-8") as f:
                json.dump(scene, f, ensure_ascii=False)
            written.append(web_path)
        except OSError:
            # web/ 디렉토리가 없을 수도 있으므로 mirror 실패는 치명적이지 않음
            pass
    return written


class RoadSceneProjector:
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.device = _resolve_device(self.config.device)

        # 도로 백엔드 선택: YOLO-seg(기본) 또는 SMP U-Net.
        # U-Net 의존성(segmentation_models_pytorch)은 "unet" 일 때만 지연 로드한다.
        self._use_road_unet = self.config.road_detector_type == "unet"
        if self._use_road_unet:
            from smp_road.model import load_checkpoint  # lazy import
            self._road_torch_device = _perspective_torch_device(self.device)
            self.road_model, _ = load_checkpoint(
                self.config.road_unet_model_path, device=self._road_torch_device
            )
        else:
            self._road_torch_device = None
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

    def _infer_road(self, image_bgr: np.ndarray) -> Dict[str, Any]:
        """도로 백엔드 추상화: YOLO-seg / U-Net 어느 쪽이든 {'road_polygons_uv': [...]} 반환."""
        if self._use_road_unet:
            from smp_road.model import infer_road_polygons  # lazy import
            return infer_road_polygons(
                self.road_model,
                image_bgr,
                imgsz=self.config.road_unet_imgsz,
                threshold=self.config.road_unet_threshold,
                min_area=self.config.road_unet_min_area,
                device=self._road_torch_device,
            )
        return infer_road_model(
            image_bgr=image_bgr,
            model=self.road_model,
            conf=self.config.road_conf,
            device=self.device,
        )

    def run(self, image_path: str, save_dir: Optional[str] = None) -> Dict[str, Any]:
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            raise FileNotFoundError(f"Failed to read image: {image_path}")

        # 색 분류는 CLAHE(명도 평활화) 이전의 원본 색을 써야 정확하다.
        color_src = image_bgr
        if self.config.use_clahe:
            image_bgr = _apply_clahe(image_bgr)

        road_out = self._infer_road(image_bgr)
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
                color_image_bgr=color_src,
            )
        else:
            obj_out = infer_object_model(
                image_bgr=image_bgr,
                model=self.object_model,
                conf=self.config.object_conf,
                device=self.device,
                color_image_bgr=color_src,
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
        color_image_bgr: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]], List[Dict[str, Any]], Dict[int, List[Tuple[int, int]]]]:
        if self._use_rfdetr:
            obj_out = infer_object_model_rfdetr(
                image_bgr=frame_bgr,
                model=self.object_model,
                conf=self.config.object_conf,
                class_names=OBJECT_CLASS_NAMES,
                sv_tracker=self._sv_tracker,
                color_image_bgr=color_image_bgr,
            )
        else:
            obj_out = infer_object_model(
                image_bgr=frame_bgr,
                model=self.object_model,
                conf=self.config.object_conf,
                device=self.device,
                use_tracker=True,
                color_image_bgr=color_image_bgr,
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

        # 색 분류용 원본(첫 프레임)을 CLAHE 적용 전에 보관한다.
        first_frame_color = first_frame
        if self.config.use_clahe:
            first_frame = _apply_clahe(first_frame)

        # Road model is intentionally run once for static-camera video.
        road_out = self._infer_road(first_frame)
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
        frames_history: List[Dict[str, Any]] = []
        tracks_registry: Dict[int, str] = {}

        while True:
            if frame_idx == 0:
                frame = first_frame
                frame_color = first_frame_color
            else:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                frame_color = frame
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
                color_image_bgr=frame_color,
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

            frame_objs: List[Dict[str, Any]] = []
            for o in tracked_objects:
                tid = o.get("track_id")
                if tid is None:
                    continue
                wp = o.get("world_position_m")
                if wp is None:
                    continue
                cls = str(o.get("class_name", ""))
                tracks_registry.setdefault(int(tid), cls)
                cov = o.get("world_cov") or [0.0, 0.0, 0.0]
                fobj: Dict[str, Any] = {
                    "track_id": int(tid),
                    "class_name": cls,
                    "confidence": float(o.get("confidence", 0.0)),
                    "x_m": float(wp[0]),
                    "z_m": float(wp[1]),
                    "cov_xx": float(cov[0]),
                    "cov_xz": float(cov[1]),
                    "cov_zz": float(cov[2]),
                }
                bbox = o.get("bbox_xyxy")
                if bbox is not None:
                    fobj["bbox_xyxy"] = [float(v) for v in bbox]
                color = o.get("color")
                if color:
                    fobj["color"] = color
                frame_objs.append(fobj)
            frames_history.append({"frame_index": frame_idx, "objects": frame_objs})

            frame_idx += 1
            last_overlay = overlay
            last_bev = bev
            last_trajectories = trajectories
            last_tracked_objects = tracked_objects

        cap.release()
        writer.release()

        # Offline forward-backward (RTS) smoothing per track.
        track_observations: Dict[int, List[Dict[str, Any]]] = {}
        for entry in frames_history:
            fi = int(entry.get("frame_index", 0))
            for o in entry.get("objects", []):
                tid = int(o.get("track_id", -1))
                if tid < 0:
                    continue
                track_observations.setdefault(tid, []).append({
                    "frame_index": fi,
                    "x_m": float(o["x_m"]),
                    "z_m": float(o["z_m"]),
                    "cov_xx": float(o.get("cov_xx", 0.0)),
                    "cov_xz": float(o.get("cov_xz", 0.0)),
                    "cov_zz": float(o.get("cov_zz", 0.0)),
                })
        smoothed_tracks = smooth_tracks_rts(track_observations, fps=fps)
        for entry in frames_history:
            fi = int(entry.get("frame_index", 0))
            for o in entry.get("objects", []):
                tid = int(o.get("track_id", -1))
                sm = smoothed_tracks.get(tid, {}).get(fi)
                if sm is None:
                    o["x_m_smoothed"] = float(o["x_m"])
                    o["z_m_smoothed"] = float(o["z_m"])
                    o["vx_m"] = 0.0
                    o["vz_m"] = 0.0
                    o["x_var"] = float(o.get("cov_xx", 0.0))
                    o["z_var"] = float(o.get("cov_zz", 0.0))
                else:
                    o["x_m_smoothed"] = sm["x"]
                    o["z_m_smoothed"] = sm["z"]
                    o["vx_m"] = sm["vx"]
                    o["vz_m"] = sm["vz"]
                    o["x_var"] = sm["x_var"]
                    o["z_var"] = sm["z_var"]

        # 트랙별 색을 다수결로 확정한다 (프레임마다 흔들리지 않도록).
        color_votes: Dict[int, Counter] = {}
        for entry in frames_history:
            for o in entry.get("objects", []):
                color = o.get("color")
                if not color:
                    continue
                tid = int(o.get("track_id", -1))
                if tid < 0:
                    continue
                color_votes.setdefault(tid, Counter())[color] += 1
        track_colors = {tid: votes.most_common(1)[0][0] for tid, votes in color_votes.items()}

        scene_data = export_scene_json(
            camera=camera,
            road_polygons_uv=road_polygons_uv,
            crosswalk_polygons_uv=crosswalk_polygons_uv,
            projected_objects=last_tracked_objects,
            trajectories=last_trajectories,
            cfg=self.config,
            frame_index=max(0, frame_idx - 1),
            frames_history=frames_history,
            tracks_registry=tracks_registry,
            track_colors=track_colors,
            fps=fps,
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

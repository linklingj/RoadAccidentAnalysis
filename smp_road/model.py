"""U-Net 빌드/로드 + BGR 이미지 추론 헬퍼.

추론 헬퍼는 기존 `infer.infer_road_model` 의 반환 형식과 호환되도록
도로 polygon 리스트(원본 픽셀 좌표, float32 (N,2))를 돌려준다.
이렇게 하면 파이프라인의 나머지(BEV 투영, 마스크 필터)는 수정 없이 동작한다.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def build_unet(encoder_name: str = "resnet34", encoder_weights: str | None = "imagenet") -> smp.Unet:
    return smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=1,
        activation=None,  # logits; sigmoid 는 추론 시 적용
    )


def save_checkpoint(path: str, model: smp.Unet, meta: Dict[str, Any]) -> None:
    torch.save({"state_dict": model.state_dict(), "meta": meta}, path)


def load_checkpoint(path: str, device: Any = "cpu") -> Tuple[smp.Unet, Dict[str, Any]]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    meta = ckpt.get("meta", {})
    model = build_unet(
        encoder_name=meta.get("encoder_name", "resnet34"),
        encoder_weights=None,
    )
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, meta


def _preprocess(image_rgb: np.ndarray, imgsz: int) -> torch.Tensor:
    resized = cv2.resize(image_rgb, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
    arr = resized.astype(np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0)
    return tensor


@torch.no_grad()
def predict_mask(
    model: smp.Unet,
    image_bgr: np.ndarray,
    imgsz: int = 512,
    threshold: float = 0.5,
    device: Any = "cpu",
) -> np.ndarray:
    """BGR 이미지 -> 원본 해상도 0/1 binary mask(H,W)."""
    h, w = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    tensor = _preprocess(image_rgb, imgsz).to(device)
    logits = model(tensor)
    prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
    prob = cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)
    return (prob >= threshold).astype(np.uint8)


def mask_to_polygons(
    mask: np.ndarray,
    min_area: int = 200,
    approx_eps_frac: float = 0.002,
) -> List[np.ndarray]:
    """binary mask -> polygon 리스트(float32 (N,2), 원본 픽셀 좌표).

    작은 노이즈 영역은 min_area 로 제거하고, Douglas-Peucker 로 점 수를 줄인다.
    """
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons: List[np.ndarray] = []
    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue
        eps = approx_eps_frac * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, eps, True)
        if len(approx) < 3:
            continue
        polygons.append(approx.reshape(-1, 2).astype(np.float32))
    return polygons


@torch.no_grad()
def infer_road_polygons(
    model: smp.Unet,
    image_bgr: np.ndarray,
    imgsz: int = 512,
    threshold: float = 0.5,
    min_area: int = 1500,
    device: Any = "cpu",
) -> Dict[str, Any]:
    """`infer.infer_road_model` 호환 반환: {'road_polygons_uv': [...], 'mask': ...}.

    min_area: 이 픽셀 면적 미만의 작은 노이즈 조각 폴리곤은 버린다(메인 도로만 유지).
    """
    mask = predict_mask(model, image_bgr, imgsz=imgsz, threshold=threshold, device=device)
    return {"road_polygons_uv": mask_to_polygons(mask, min_area=min_area), "mask": mask}

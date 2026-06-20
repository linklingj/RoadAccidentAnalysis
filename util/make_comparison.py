"""두 RF-DETR 모델(baseline / augmented)의 추론 결과를 나란히 비교하는 이미지를 생성한다.

출력: docs/assets/comparison/
  - rf_compare_*.jpg      : RF 도메인 비교
  - aihub_compare_*.jpg   : AI-Hub 도메인 비교
  - grid_rf.jpg / grid_aihub.jpg : 여러 이미지 그리드

사용법:
    conda activate dl
    python util/make_comparison.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ── 설정 ──────────────────────────────────────────────────────────────────────
ROOT = Path(".")
OUT_DIR = ROOT / "docs" / "assets" / "comparison"
BASELINE_CKPT = "runs/rfdetr/baseline/checkpoint_best_total.pth"
AUGMENTED_CKPT = "runs/rfdetr/augmented/checkpoint_best_total.pth"
TEST_DIR = ROOT / "datasets-rfdetr" / "baseline" / "test"
THRESHOLD = 0.35

CLASS_COLORS = {
    "_background_": (128, 128, 128),
    "bus":    (255, 140,   0),
    "car":    ( 30, 144, 255),
    "person": (220,  20,  60),
    "riders": (148,   0, 211),
    "truck":  ( 34, 139,  34),
}

# 비교할 이미지 파일명 (rf 3장, aihub 3장)
RF_IMAGES = [
    "rf_-8-mp4_20_jpg.rf.04355885f90b336d9e94cd7c8f691867.jpg",
    "rf_frame_Xia0468_jpg.rf.6e9acd4612886c5e3e8d4e6ca29dfd6c.jpg",
    "rf_dji_fly_20240303_195840_107_1709468120439_photo_optimized_jpg.rf.e2916cb5bd80ad813624d4bf555c005e.jpg",
]
AIHUB_IMAGES = [
    "aihub_BC2000701_20201008_065959_S_4455.jpg",
    "aihub_BC2000204_20201016_125958_S_6039.jpg",
    "aihub_BC1000401_20201016_060000-075749_S_596.jpg",
]


def load_model(ckpt: str):
    import rfdetr
    return rfdetr.RFDETRNano(pretrain_weights=str(ckpt))


def get_class_names(model) -> list[str]:
    for getter in (lambda m: m.model.class_names,
                   lambda m: m.class_names):
        try:
            v = getter(model)
            if v:
                return list(v)
        except Exception:
            pass
    return []


def draw_boxes(img: Image.Image, det, class_names: list[str], title: str) -> Image.Image:
    img = img.copy().convert("RGB")
    draw = ImageDraw.Draw(img)
    W, H = img.size

    xyxy = np.asarray(det.xyxy).reshape(-1, 4) if len(det.xyxy) else np.zeros((0, 4))
    conf = np.asarray(det.confidence).reshape(-1) if len(det.confidence) else np.zeros(0)
    cids = np.asarray(det.class_id).reshape(-1).astype(int) if len(det.class_id) else np.zeros(0, dtype=int)

    for (x1, y1, x2, y2), s, c in zip(xyxy, conf, cids):
        name = class_names[c] if 0 <= c < len(class_names) else str(c)
        color = CLASS_COLORS.get(name, (255, 255, 255))
        lw = max(2, int(min(W, H) * 0.003))
        draw.rectangle([x1, y1, x2, y2], outline=color, width=lw)
        label = f"{name} {s:.2f}"
        fs = max(12, int(min(W, H) * 0.022))
        try:
            font = ImageFont.truetype("arial.ttf", fs)
        except Exception:
            font = ImageFont.load_default()
        bb = draw.textbbox((x1, y1 - fs - 2), label, font=font)
        draw.rectangle(bb, fill=color)
        draw.text((x1, y1 - fs - 2), label, fill=(255, 255, 255), font=font)

    # Title bar
    bar_h = max(32, int(H * 0.045))
    bar = Image.new("RGB", (W, bar_h), (30, 30, 30))
    bd = ImageDraw.Draw(bar)
    tfs = max(14, int(bar_h * 0.55))
    try:
        tfont = ImageFont.truetype("arial.ttf", tfs)
    except Exception:
        tfont = ImageFont.load_default()
    bd.text((8, (bar_h - tfs) // 2), title, fill=(255, 255, 255), font=tfont)
    out = Image.new("RGB", (W, H + bar_h))
    out.paste(bar, (0, 0))
    out.paste(img, (0, bar_h))
    return out


def make_comparison(img_path: Path, baseline_model, augmented_model,
                    bl_names: list[str], aug_names: list[str],
                    out_path: Path, gt_ann=None):
    img = Image.open(img_path).convert("RGB")

    bl_det = baseline_model.predict(str(img_path), threshold=THRESHOLD)
    aug_det = augmented_model.predict(str(img_path), threshold=THRESHOLD)

    bl_drawn = draw_boxes(img, bl_det, bl_names, f"Baseline  ({len(bl_det.xyxy)} dets)")
    aug_drawn = draw_boxes(img, aug_det, aug_names, f"Augmented ({len(aug_det.xyxy)} dets)")

    # GT 패널 추가
    if gt_ann is not None:
        gt_img = _draw_gt(img, gt_ann)
        combined = _hstack([gt_img, bl_drawn, aug_drawn])
    else:
        combined = _hstack([bl_drawn, aug_drawn])

    combined.save(out_path, quality=90)
    print(f"  saved → {out_path}")
    return combined


def _draw_gt(img: Image.Image, boxes_labels: list) -> Image.Image:
    """GT boxes 시각화."""
    img = img.copy()
    draw = ImageDraw.Draw(img)
    W, H = img.size
    lw = max(2, int(min(W, H) * 0.003))
    fs = max(12, int(min(W, H) * 0.022))
    try:
        font = ImageFont.truetype("arial.ttf", fs)
    except Exception:
        font = ImageFont.load_default()
    for (x, y, w, h), name in boxes_labels:
        color = CLASS_COLORS.get(name, (200, 200, 200))
        draw.rectangle([x, y, x + w, y + h], outline=color, width=lw)
        bb = draw.textbbox((x, y - fs - 2), name, font=font)
        draw.rectangle(bb, fill=color)
        draw.text((x, y - fs - 2), name, fill=(255, 255, 255), font=font)
    bar_h = max(32, int(H * 0.045))
    bar = Image.new("RGB", (W, bar_h), (30, 30, 30))
    bd = ImageDraw.Draw(bar)
    tfs = max(14, int(bar_h * 0.55))
    try:
        tfont = ImageFont.truetype("arial.ttf", tfs)
    except Exception:
        tfont = ImageFont.load_default()
    bd.text((8, (bar_h - tfs) // 2), "Ground Truth", fill=(255, 255, 0), font=tfont)
    out = Image.new("RGB", (W, H + bar_h))
    out.paste(bar, (0, 0))
    out.paste(img, (0, bar_h))
    return out


def _hstack(imgs: list[Image.Image]) -> Image.Image:
    max_h = max(i.height for i in imgs)
    resized = []
    for im in imgs:
        if im.height != max_h:
            w = int(im.width * max_h / im.height)
            im = im.resize((w, max_h), Image.LANCZOS)
        resized.append(im)
    total_w = sum(i.width for i in resized) + 4 * (len(resized) - 1)
    canvas = Image.new("RGB", (total_w, max_h), (50, 50, 50))
    x = 0
    for im in resized:
        canvas.paste(im, (x, 0))
        x += im.width + 4
    return canvas


def make_grid(panels: list[Image.Image], out_path: Path, n_cols: int = 1):
    """panels 를 세로로 쌓아 grid 이미지 생성."""
    max_w = max(p.width for p in panels)
    resized = []
    for p in panels:
        if p.width != max_w:
            h = int(p.height * max_w / p.width)
            p = p.resize((max_w, h), Image.LANCZOS)
        resized.append(p)
    gap = 8
    total_h = sum(p.height for p in resized) + gap * (len(resized) - 1)
    canvas = Image.new("RGB", (max_w, total_h), (20, 20, 20))
    y = 0
    for p in resized:
        canvas.paste(p, (0, y))
        y += p.height + gap
    canvas.save(out_path, quality=88)
    print(f"  grid saved → {out_path}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ann_data = json.loads((TEST_DIR / "_annotations.coco.json").read_text(encoding="utf-8"))
    cats = {c["id"]: c["name"] for c in ann_data["categories"]}
    img_id_map = {i["file_name"]: i["id"] for i in ann_data["images"]}
    # img_id → list of (xywh, class_name)
    gt_map: dict[int, list] = {}
    for a in ann_data["annotations"]:
        gt_map.setdefault(a["image_id"], []).append((a["bbox"], cats[a["category_id"]]))

    print("Loading baseline model…")
    bl_model = load_model(BASELINE_CKPT)
    bl_names = get_class_names(bl_model)
    print(f"  class_names: {bl_names}")

    print("Loading augmented model…")
    aug_model = load_model(AUGMENTED_CKPT)
    aug_names = get_class_names(aug_model)

    rf_panels, aihub_panels = [], []

    print("\n── RF 도메인 ──")
    for fn in RF_IMAGES:
        img_path = TEST_DIR / fn
        if not img_path.exists():
            print(f"  SKIP (not found): {fn}")
            continue
        iid = img_id_map.get(fn)
        gt = gt_map.get(iid, []) if iid else []
        out = OUT_DIR / f"compare_{fn}"
        panel = make_comparison(img_path, bl_model, aug_model, bl_names, aug_names, out, gt)
        rf_panels.append(panel)

    print("\n── AI-Hub 도메인 ──")
    for fn in AIHUB_IMAGES:
        img_path = TEST_DIR / fn
        if not img_path.exists():
            print(f"  SKIP (not found): {fn}")
            continue
        iid = img_id_map.get(fn)
        gt = gt_map.get(iid, []) if iid else []
        out = OUT_DIR / f"compare_{fn}"
        panel = make_comparison(img_path, bl_model, aug_model, bl_names, aug_names, out, gt)
        aihub_panels.append(panel)

    if rf_panels:
        make_grid(rf_panels, OUT_DIR / "grid_rf.jpg")
    if aihub_panels:
        make_grid(aihub_panels, OUT_DIR / "grid_aihub.jpg")
    print("\n완료. 출력 폴더:", OUT_DIR)


if __name__ == "__main__":
    main()

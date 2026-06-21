"""AI-Hub 데이터셋 추가 전(0519 모델) vs 후(augmented 모델) 추론 비교 이미지 생성.

- 이전 모델: runs/detect/rfdetr-object-0519/checkpoint_best_ema.pth
  (cctv-object-dataset-coco 학습, 8클래스: attention/bus/car/crosswalk/person/riders/truck)
- 이후 모델: runs/rfdetr/augmented/checkpoint_best_total.pth
  (RF + AI-Hub 2× 학습, 6클래스: _background_/bus/car/person/riders/truck)

출력: docs/assets/before_after/
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(".")
OUT_DIR = ROOT / "docs" / "assets" / "before_after"
BEFORE_CKPT = "runs/detect/rfdetr-object-0519/checkpoint_best_ema.pth"
AFTER_CKPT  = "runs/rfdetr/augmented/checkpoint_best_total.pth"
TEST_DIR    = ROOT / "datasets-rfdetr" / "baseline" / "test"
THRESHOLD   = 0.30

CLASS_COLORS = {
    "_background_": (128, 128, 128),
    "attention":    (200, 200,   0),
    "crosswalk":    (180, 180, 180),
    "bus":    (255, 140,   0),
    "car":    ( 30, 144, 255),
    "person": (220,  20,  60),
    "riders": (148,   0, 211),
    "truck":  ( 34, 139,  34),
}
SKIP_CLASSES = {"attention", "crosswalk", "_background_"}

# RF 도메인: 다양한 클래스가 있는 장면
RF_IMAGES = [
    "rf_frame_Xia0468_jpg.rf.6e9acd4612886c5e3e8d4e6ca29dfd6c.jpg",    # car+person+riders+truck
    "rf_-8-mp4_20_jpg.rf.04355885f90b336d9e94cd7c8f691867.jpg",         # car+person+riders+bus
    "rf_frame_Xia0540_jpg.rf.3fd46873353ad7d9ff2f9a64de97b1c4.jpg",     # car+person+riders+truck
]
# AI-Hub 도메인: bus/truck 이 많아 개선 폭이 큰 장면
AIHUB_IMAGES = [
    "aihub_BC2000701_20201008_065959_S_4455.jpg",                       # car+person+bus+truck+riders
    "aihub_BC2000204_20201016_125958_S_6039.jpg",                       # car+person+truck+bus
    "aihub_BC1000401_20201016_060000-075749_S_596.jpg",                 # 야간, car+truck+bus
]


def load_model(ckpt: str):
    import rfdetr
    return rfdetr.RFDETRNano(pretrain_weights=str(ckpt))


def get_class_names(model) -> list[str]:
    for getter in (lambda m: m.model.class_names, lambda m: m.class_names):
        try:
            v = getter(model)
            if v:
                return list(v)
        except Exception:
            pass
    return []


def draw_boxes(img: Image.Image, det, class_names: list[str], title: str,
               skip_classes: set[str] = frozenset()) -> Image.Image:
    img = img.copy().convert("RGB")
    draw = ImageDraw.Draw(img)
    W, H = img.size
    lw = max(2, int(min(W, H) * 0.003))

    xyxy = np.asarray(det.xyxy).reshape(-1, 4) if len(det.xyxy) else np.zeros((0, 4))
    conf = np.asarray(det.confidence).reshape(-1) if len(det.confidence) else np.zeros(0)
    cids = np.asarray(det.class_id).reshape(-1).astype(int) if len(det.class_id) else np.zeros(0, int)

    kept = 0
    for (x1, y1, x2, y2), s, c in zip(xyxy, conf, cids):
        name = class_names[c] if 0 <= c < len(class_names) else str(c)
        if name in skip_classes:
            continue
        kept += 1
        color = CLASS_COLORS.get(name, (255, 255, 255))
        draw.rectangle([x1, y1, x2, y2], outline=color, width=lw)
        fs = max(11, int(min(W, H) * 0.020))
        try:
            font = ImageFont.truetype("arial.ttf", fs)
        except Exception:
            font = ImageFont.load_default()
        label = f"{name} {s:.2f}"
        bb = draw.textbbox((x1, y1 - fs - 2), label, font=font)
        draw.rectangle(bb, fill=color)
        draw.text((x1, y1 - fs - 2), label, fill=(255, 255, 255), font=font)

    # 타이틀 바
    bar_h = max(36, int(H * 0.048))
    bar = Image.new("RGB", (W, bar_h), (20, 20, 20))
    bd = ImageDraw.Draw(bar)
    tfs = max(14, int(bar_h * 0.52))
    try:
        tfont = ImageFont.truetype("arialbd.ttf", tfs)
    except Exception:
        try:
            tfont = ImageFont.truetype("arial.ttf", tfs)
        except Exception:
            tfont = ImageFont.load_default()
    bd.text((8, (bar_h - tfs) // 2), f"{title}  ({kept} dets)", fill=(255, 255, 255), font=tfont)
    out = Image.new("RGB", (W, H + bar_h))
    out.paste(bar, (0, 0))
    out.paste(img, (0, bar_h))
    return out


def hstack(imgs: list[Image.Image], gap: int = 6) -> Image.Image:
    max_h = max(i.height for i in imgs)
    panels = []
    for im in imgs:
        if im.height != max_h:
            w = int(im.width * max_h / im.height)
            im = im.resize((w, max_h), Image.LANCZOS)
        panels.append(im)
    total_w = sum(p.width for p in panels) + gap * (len(panels) - 1)
    canvas = Image.new("RGB", (total_w, max_h), (40, 40, 40))
    x = 0
    for p in panels:
        canvas.paste(p, (x, 0))
        x += p.width + gap
    return canvas


def vstack(imgs: list[Image.Image], gap: int = 10) -> Image.Image:
    max_w = max(i.width for i in imgs)
    panels = []
    for im in imgs:
        if im.width != max_w:
            h = int(im.height * max_w / im.width)
            im = im.resize((max_w, h), Image.LANCZOS)
        panels.append(im)
    total_h = sum(p.height for p in panels) + gap * (len(panels) - 1)
    canvas = Image.new("RGB", (max_w, total_h), (20, 20, 20))
    y = 0
    for p in panels:
        canvas.paste(p, (0, y))
        y += p.height + gap
    return canvas


def section_label(width: int, text: str, bg=(50, 50, 80)) -> Image.Image:
    h = 40
    bar = Image.new("RGB", (width, h), bg)
    d = ImageDraw.Draw(bar)
    fs = 18
    try:
        font = ImageFont.truetype("arialbd.ttf", fs)
    except Exception:
        try:
            font = ImageFont.truetype("arial.ttf", fs)
        except Exception:
            font = ImageFont.load_default()
    d.text((12, (h - fs) // 2), text, fill=(220, 220, 255), font=font)
    return bar


def process_image(img_path: Path, before_model, after_model,
                  before_names, after_names) -> Image.Image | None:
    if not img_path.exists():
        print(f"  SKIP (not found): {img_path.name}")
        return None
    img = Image.open(img_path).convert("RGB")
    before_det = before_model.predict(str(img_path), threshold=THRESHOLD)
    after_det  = after_model.predict(str(img_path),  threshold=THRESHOLD)
    before_panel = draw_boxes(img, before_det, before_names,
                              "Before (RF only, 0519)", skip_classes=SKIP_CLASSES)
    after_panel  = draw_boxes(img, after_det,  after_names,
                              "After  (RF + AI-Hub)", skip_classes=SKIP_CLASSES)
    return hstack([before_panel, after_panel])


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading BEFORE model (rfdetr-object-0519)…")
    before_model = load_model(BEFORE_CKPT)
    before_names = get_class_names(before_model)
    print(f"  class_names: {before_names}")

    print("Loading AFTER model (augmented)…")
    after_model = load_model(AFTER_CKPT)
    after_names = get_class_names(after_model)
    print(f"  class_names: {after_names}")

    rf_rows, aihub_rows = [], []

    print("\n── RF 도메인 ──")
    for fn in RF_IMAGES:
        panel = process_image(TEST_DIR / fn, before_model, after_model,
                              before_names, after_names)
        if panel:
            rf_rows.append(panel)
            print(f"  OK: {fn[:50]}…")

    print("\n── AI-Hub 도메인 ──")
    for fn in AIHUB_IMAGES:
        panel = process_image(TEST_DIR / fn, before_model, after_model,
                              before_names, after_names)
        if panel:
            aihub_rows.append(panel)
            print(f"  OK: {fn[:50]}…")

    all_rows = []
    if rf_rows:
        lbl = section_label(rf_rows[0].width, "▶ RF 도메인 (Roboflow CCTV)")
        all_rows += [lbl] + rf_rows
    if aihub_rows:
        lbl = section_label(aihub_rows[0].width, "▶ AI-Hub 도메인 (한국 시내도로 CCTV)")
        all_rows += [lbl] + aihub_rows

    if all_rows:
        grid = vstack(all_rows, gap=6)
        out_path = OUT_DIR / "before_after_grid.jpg"
        grid.save(out_path, quality=90)
        print(f"\n  grid saved → {out_path}  ({grid.width}×{grid.height}px)")

    # 도메인별 개별 저장
    if rf_rows:
        vstack(rf_rows, gap=6).save(OUT_DIR / "before_after_rf.jpg", quality=90)
    if aihub_rows:
        vstack(aihub_rows, gap=6).save(OUT_DIR / "before_after_aihub.jpg", quality=90)

    print("완료. 출력 폴더:", OUT_DIR)


if __name__ == "__main__":
    main()

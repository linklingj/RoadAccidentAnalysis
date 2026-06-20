"""U-Net(SMP) vs YOLO-seg 도로 segmentation 비교 평가.

같은 test 셋(cctv-roadseg-dataset/test, 135장)에 대해 두 모델을 동일한
픽셀 단위 지표(IoU / Dice / pixel accuracy)로 비교한다. GT 는 YOLO polygon
라벨을 래스터화해 만든다(학습 때와 동일). 추론 지연시간도 측정한다.

사용:
    conda activate dl
    python eval_road_smp.py \
        --yolo-model runs/segment/0401-road/weights/best.pt \
        --smp-ckpt runs/smp-road/best.pt
결과:
    runs/smp-road/metrics.json     # 집계 지표(보고서용)
    runs/smp-road/compare/*.png    # 정성 비교 패널
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from smp_road.dataset import list_image_paths, label_path_for, rasterize_yolo_polygons
from smp_road.model import load_checkpoint, predict_mask


def mask_metrics(pred: np.ndarray, gt: np.ndarray):
    p = pred.astype(bool)
    g = gt.astype(bool)
    inter = np.logical_and(p, g).sum()
    union = np.logical_or(p, g).sum()
    psum, gsum = p.sum(), g.sum()
    iou = inter / union if union > 0 else 1.0
    dice = 2 * inter / (psum + gsum) if (psum + gsum) > 0 else 1.0
    acc = (p == g).mean()
    return float(iou), float(dice), float(acc)


def yolo_mask(model, image_bgr, conf, device, imgsz=None):
    """YOLO-seg 결과 polygon 을 원본 해상도 binary mask 로 래스터화(파이프라인과 동일 소비방식)."""
    h, w = image_bgr.shape[:2]
    kw = dict(source=image_bgr, conf=conf, device=device, verbose=False, save=False)
    if imgsz is not None:
        kw["imgsz"] = imgsz
    results = model.predict(**kw)
    res = results[0] if results else None
    mask = np.zeros((h, w), dtype=np.uint8)
    if res is None or res.masks is None or res.masks.xy is None:
        return mask
    for poly in res.masks.xy:
        if poly is None or len(poly) < 3:
            continue
        cv2.fillPoly(mask, [np.round(poly).astype(np.int32)], 1)
    return mask


def overlay(image_bgr, mask, color):
    out = image_bgr.copy()
    layer = out.copy()
    layer[mask.astype(bool)] = color
    return cv2.addWeighted(layer, 0.45, out, 0.55, 0)


def label(img, text):
    cv2.rectangle(img, (0, 0), (img.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(img, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="cctv-roadseg-dataset")
    ap.add_argument("--split", default="test")
    ap.add_argument("--yolo-model", default="runs/segment/0401-road/weights/best.pt")
    ap.add_argument("--smp-ckpt", default="runs/smp-road/best.pt")
    ap.add_argument("--yolo-conf", type=float, default=0.25)
    ap.add_argument("--yolo-imgsz", type=int, default=None, help="YOLO predict 해상도(미지정=기본 640). 학습 해상도(1280) 매칭 비교용")
    ap.add_argument("--smp-imgsz", type=int, default=512)
    ap.add_argument("--out-dir", default="runs/smp-road")
    ap.add_argument("--num-vis", type=int, default=8)
    ap.add_argument("--extra-images", nargs="*", default=["input/image1.png", "input/image2.png", "input/image3.png"])
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir)
    vis_dir = out_dir / "compare"
    vis_dir.mkdir(parents=True, exist_ok=True)

    print(f"[eval] device={device}")
    from ultralytics import YOLO
    yolo = YOLO(args.yolo_model)
    smp_model, smp_meta = load_checkpoint(args.smp_ckpt, device=device)
    print(f"[eval] yolo={args.yolo_model}  smp={args.smp_ckpt} (meta={smp_meta})")

    images_dir = Path(args.data_dir) / args.split / "images"
    image_paths = list_image_paths(images_dir)
    print(f"[eval] {args.split} images: {len(image_paths)}")

    agg = {"yolo": {"iou": [], "dice": [], "acc": [], "ms": []},
           "smp": {"iou": [], "dice": [], "acc": [], "ms": []}}

    # GPU warmup (latency 측정 공정성)
    warm = cv2.imread(str(image_paths[0]))
    for _ in range(2):
        yolo_mask(yolo, warm, args.yolo_conf, device, imgsz=args.yolo_imgsz)
        predict_mask(smp_model, warm, imgsz=args.smp_imgsz, device=device)

    vis_count = 0
    for idx, ip in enumerate(image_paths):
        img = cv2.imread(str(ip))
        if img is None:
            continue
        h, w = img.shape[:2]
        gt = rasterize_yolo_polygons(label_path_for(ip), h, w)

        t0 = time.perf_counter()
        ym = yolo_mask(yolo, img, args.yolo_conf, device, imgsz=args.yolo_imgsz)
        agg["yolo"]["ms"].append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        sm = predict_mask(smp_model, img, imgsz=args.smp_imgsz, device=device)
        agg["smp"]["ms"].append((time.perf_counter() - t0) * 1000)

        for key, m in (("yolo", ym), ("smp", sm)):
            iou, dice, acc = mask_metrics(m, gt)
            agg[key]["iou"].append(iou)
            agg[key]["dice"].append(dice)
            agg[key]["acc"].append(acc)

        if vis_count < args.num_vis and idx % max(1, len(image_paths) // args.num_vis) == 0:
            panel = np.hstack([
                label(img.copy(), "input"),
                label(overlay(img, gt, (0, 200, 0)), "GT"),
                label(overlay(img, ym, (0, 160, 255)), f"YOLO IoU={mask_metrics(ym,gt)[0]:.2f}"),
                label(overlay(img, sm, (255, 120, 0)), f"U-Net IoU={mask_metrics(sm,gt)[0]:.2f}"),
            ])
            cv2.imwrite(str(vis_dir / f"cmp_{vis_count:02d}_{ip.stem[:20]}.png"), panel)
            vis_count += 1

    def summarize(d):
        return {k: round(float(np.mean(v)), 4) for k, v in d.items()}

    summary = {
        "split": args.split,
        "num_images": len(image_paths),
        "yolo_model": args.yolo_model,
        "smp_ckpt": args.smp_ckpt,
        "smp_meta": {k: smp_meta.get(k) for k in ("encoder_name", "imgsz", "epoch", "iou")},
        "device": device,
        "yolo": summarize(agg["yolo"]),
        "smp": summarize(agg["smp"]),
    }
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2))

    print("\n==== ROAD SEGMENTATION: U-Net(SMP) vs YOLO-seg ====")
    print(f"{'metric':<14}{'YOLO-seg':>12}{'U-Net':>12}")
    for m, label_m in (("iou", "mean IoU"), ("dice", "mean Dice"),
                       ("acc", "pixel acc"), ("ms", "latency ms")):
        print(f"{label_m:<14}{summary['yolo'][m]:>12.4f}{summary['smp'][m]:>12.4f}")

    # input/ 등 GT 없는 이미지: YOLO vs U-Net 정성 비교만
    for ep in args.extra_images:
        p = Path(ep)
        if not p.exists():
            continue
        img = cv2.imread(str(p))
        if img is None:
            continue
        ym = yolo_mask(yolo, img, args.yolo_conf, device, imgsz=args.yolo_imgsz)
        sm = predict_mask(smp_model, img, imgsz=args.smp_imgsz, device=device)
        panel = np.hstack([
            label(img.copy(), "input"),
            label(overlay(img, ym, (0, 160, 255)), "YOLO-seg"),
            label(overlay(img, sm, (255, 120, 0)), "U-Net"),
        ])
        cv2.imwrite(str(vis_dir / f"extra_{p.stem}.png"), panel)

    print(f"\n[eval] metrics -> {out_dir/'metrics.json'}")
    print(f"[eval] panels  -> {vis_dir}")


if __name__ == "__main__":
    main()

"""SMP U-Net 기반 도로 segmentation 학습 스크립트.

기존 YOLO-seg 도로 모델(`train.py --model road`)과 동일한 데이터셋
(cctv-roadseg-dataset)으로 U-Net 을 학습해 비교 실험을 한다.

사용:
    conda activate dl
    python train_road_smp.py --epochs 30 --batch 12 --imgsz 512
결과:
    runs/smp-road/best.pt        # valid IoU 최고 체크포인트
    runs/smp-road/last.pt
    runs/smp-road/history.json   # epoch 별 loss/IoU/Dice (보고서용)
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

from util.smp_road.dataset import RoadSegDataset
from util.smp_road.model import build_unet, save_checkpoint

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(imgsz: int):
    train_tf = A.Compose([
        A.Resize(imgsz, imgsz),
        A.HorizontalFlip(p=0.5),
        # 야간·역광 등 CCTV 조도 변화 대응
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        # 고정 카메라이므로 소폭 기하 변형만
        A.Affine(scale=(0.9, 1.1), translate_percent=(0.0, 0.05), rotate=(-5, 5), p=0.4),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])
    val_tf = A.Compose([
        A.Resize(imgsz, imgsz),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])
    return train_tf, val_tf


@torch.no_grad()
def evaluate(model, loader, device, threshold=0.5):
    """valid 셋에 대한 평균 IoU / Dice / pixel accuracy."""
    model.eval()
    inter = union = dice_num = dice_den = correct = total = 0.0
    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=(device != "cpu")):
            logits = model(images)
        preds = (torch.sigmoid(logits) >= threshold).float()
        p = preds.bool()
        g = masks.bool()
        inter += (p & g).sum().item()
        union += (p | g).sum().item()
        dice_num += 2.0 * (p & g).sum().item()
        dice_den += (p.sum() + g.sum()).item()
        correct += (preds == masks).sum().item()
        total += masks.numel()
    iou = inter / union if union > 0 else 0.0
    dice = dice_num / dice_den if dice_den > 0 else 0.0
    acc = correct / total if total > 0 else 0.0
    return {"iou": iou, "dice": dice, "pixel_acc": acc}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="cctv-roadseg-dataset")
    ap.add_argument("--encoder", default="resnet34")
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--output-dir", default="runs/smp-road")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[smp-road] device={device} encoder={args.encoder} imgsz={args.imgsz} "
          f"batch={args.batch} epochs={args.epochs}")

    train_tf, val_tf = build_transforms(args.imgsz)
    data_dir = Path(args.data_dir)
    train_ds = RoadSegDataset(data_dir / "train" / "images", transform=train_tf)
    val_ds = RoadSegDataset(data_dir / "valid" / "images", transform=val_tf)
    print(f"[smp-road] train={len(train_ds)} valid={len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers,
        pin_memory=True, drop_last=True, persistent_workers=args.workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False, num_workers=args.workers,
        pin_memory=True, persistent_workers=args.workers > 0,
    )

    model = build_unet(encoder_name=args.encoder, encoder_weights="imagenet").to(device)

    dice_loss = smp.losses.DiceLoss(mode="binary", from_logits=True)
    bce_loss = torch.nn.BCEWithLogitsLoss()

    def criterion(logits, target):
        return dice_loss(logits, target) + bce_loss(logits, target)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    history = []
    best_iou = -1.0
    meta_base = {"encoder_name": args.encoder, "imgsz": args.imgsz,
                 "mean": IMAGENET_MEAN, "std": IMAGENET_STD}

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        running = 0.0
        for images, masks in train_loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                logits = model(images)
                loss = criterion(logits, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item()
        scheduler.step()

        train_loss = running / max(1, len(train_loader))
        val_metrics = evaluate(model, val_loader, device)
        dt = time.time() - t0
        row = {"epoch": epoch, "train_loss": round(train_loss, 5),
               "val_iou": round(val_metrics["iou"], 5),
               "val_dice": round(val_metrics["dice"], 5),
               "val_pixel_acc": round(val_metrics["pixel_acc"], 5),
               "lr": optimizer.param_groups[0]["lr"], "sec": round(dt, 1)}
        history.append(row)
        print(f"[ep {epoch:02d}/{args.epochs}] loss={train_loss:.4f} "
              f"IoU={val_metrics['iou']:.4f} Dice={val_metrics['dice']:.4f} "
              f"acc={val_metrics['pixel_acc']:.4f} ({dt:.0f}s)")

        save_checkpoint(str(out_dir / "last.pt"), model, {**meta_base, "epoch": epoch, **val_metrics})
        if val_metrics["iou"] > best_iou:
            best_iou = val_metrics["iou"]
            save_checkpoint(str(out_dir / "best.pt"), model, {**meta_base, "epoch": epoch, **val_metrics})
            print(f"        ↑ new best IoU={best_iou:.4f} -> best.pt")

        (out_dir / "history.json").write_text(json.dumps(
            {"args": vars(args), "best_val_iou": best_iou, "history": history}, indent=2))

    print(f"[smp-road] done. best val IoU={best_iou:.4f}  -> {out_dir/'best.pt'}")


if __name__ == "__main__":
    main()

"""YOLO segmentation format dataset을 RF-DETR 학습용 COCO JSON format으로 변환한다.

Usage:
    python util/convert_yolo_to_coco.py \
        --src cctv-object-dataset \
        --dst cctv-object-dataset-coco

YOLO seg label 형식: class_id x1 y1 x2 y2 ... xn yn  (정규화된 polygon 좌표)
COCO 출력 형식: _annotations.coco.json (bbox = polygon의 bounding box)
"""

import argparse
import json
import shutil
from pathlib import Path

import cv2


SPLITS = [("train", "train"), ("valid", "valid"), ("test", "test")]


def _polygon_to_bbox(coords: list[float], img_w: int, img_h: int) -> tuple[float, float, float, float]:
    """정규화된 polygon 좌표 목록에서 COCO bbox [x, y, width, height]를 반환한다."""
    xs = [coords[i] * img_w for i in range(0, len(coords), 2)]
    ys = [coords[i] * img_h for i in range(1, len(coords), 2)]
    x1, y1 = min(xs), min(ys)
    x2, y2 = max(xs), max(ys)
    return x1, y1, x2 - x1, y2 - y1


def _convert_split(
    src_split: Path,
    dst_split: Path,
    class_names: list[str],
    split_name: str,
    copy_images: bool,
) -> dict:
    images_dir = src_split / "images"
    labels_dir = src_split / "labels"

    dst_images_dir = dst_split / "images"
    dst_images_dir.mkdir(parents=True, exist_ok=True)

    categories = [{"id": i, "name": name, "supercategory": "object"} for i, name in enumerate(class_names)]

    coco_images = []
    coco_annotations = []
    ann_id = 1

    image_files = sorted(images_dir.glob("*.[jp][pn][g]*")) + sorted(images_dir.glob("*.jpeg"))
    # deduplicate (glob pattern may overlap)
    seen = set()
    unique_image_files = []
    for f in image_files:
        if f.name not in seen:
            seen.add(f.name)
            unique_image_files.append(f)
    image_files = unique_image_files

    for img_id, img_path in enumerate(image_files, start=1):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  [skip] cannot read {img_path.name}")
            continue
        img_h, img_w = img.shape[:2]

        if copy_images:
            shutil.copy2(img_path, dst_images_dir / img_path.name)

        coco_images.append({
            "id": img_id,
            "file_name": img_path.name,
            "width": img_w,
            "height": img_h,
        })

        label_path = labels_dir / (img_path.stem + ".txt")
        if not label_path.exists():
            continue

        for line in label_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue

            cls_id = int(parts[0])
            coords = [float(v) for v in parts[1:]]

            if len(coords) == 4:
                # YOLO det: cx cy w h (정규화)
                cx, cy, bw, bh = coords
                x = (cx - bw / 2) * img_w
                y = (cy - bh / 2) * img_h
                w = bw * img_w
                h = bh * img_h
            else:
                # YOLO seg: polygon vertices (정규화)
                if len(coords) % 2 != 0:
                    coords = coords[:-1]
                x, y, w, h = _polygon_to_bbox(coords, img_w, img_h)

            area = w * h
            coco_annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": cls_id,
                "bbox": [round(x, 2), round(y, 2), round(w, 2), round(h, 2)],
                "area": round(area, 2),
                "iscrowd": 0,
            })
            ann_id += 1

    coco_json = {
        "info": {"description": f"CCTV Object Dataset - {split_name}"},
        "categories": categories,
        "images": coco_images,
        "annotations": coco_annotations,
    }

    ann_path = dst_split / "_annotations.coco.json"
    ann_path.write_text(json.dumps(coco_json, ensure_ascii=False, indent=2))
    print(f"  [{split_name}] {len(coco_images)} images, {len(coco_annotations)} annotations → {ann_path}")
    return coco_json


def convert(src_dir: str, dst_dir: str, copy_images: bool = True) -> None:
    src = Path(src_dir)
    dst = Path(dst_dir)

    import yaml
    data_yaml = src / "data.yaml"
    with open(data_yaml) as f:
        data = yaml.safe_load(f)
    class_names: list[str] = data["names"]
    print(f"Classes ({len(class_names)}): {class_names}")

    dst.mkdir(parents=True, exist_ok=True)

    for src_split_name, dst_split_name in SPLITS:
        src_split = src / src_split_name
        if not src_split.exists():
            print(f"  [skip] {src_split_name} split not found")
            continue
        dst_split = dst / dst_split_name
        _convert_split(src_split, dst_split, class_names, dst_split_name, copy_images)

    print(f"\nDone. COCO dataset saved to: {dst}")
    print("RF-DETR 학습 명령:")
    print(f"  python train.py --model rfdetr-object --dataset {dst}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="cctv-object-dataset", help="YOLO format dataset root")
    parser.add_argument("--dst", default="cctv-object-dataset-coco", help="COCO format output root")
    parser.add_argument("--no-copy-images", action="store_true", help="이미지 복사 대신 원본 참조 (심볼릭 링크 미지원 환경용)")
    args = parser.parse_args()

    convert(args.src, args.dst, copy_images=not args.no_copy_images)

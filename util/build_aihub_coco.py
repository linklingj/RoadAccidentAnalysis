"""AI-Hub `교통안전(Bbox)` 데이터셋을 기존 Roboflow 데이터셋과 합쳐 RF-DETR 학습용
COCO 데이터셋으로 빌드한다.

생성물 (모두 RF-DETR Roboflow-COCO 레이아웃: split 폴더에 이미지 + _annotations.coco.json):

    <out>/baseline/{train,valid,test}    기존 Roboflow 데이터만
    <out>/augmented/{train,valid,test}   Roboflow + AI-Hub 서브샘플
    (두 데이터셋의 test 스플릿은 **동일**하다 → 공정 비교)

비교 실험 설계
--------------
- baseline  : RF train(608) / RF valid(164) / SHARED test
- augmented : RF train + AIHub_tr / RF valid + AIHub_va / SHARED test
- SHARED test = RF test(83) + AIHub held-out(--n-test)
  · 이미지 파일명 prefix 로 도메인을 구분한다(`rf_*` / `aihub_*`)
    → 평가 스크립트가 도메인별 mAP 를 따로 집계할 수 있다.

AI-Hub 라벨 포맷(특이사항)
--------------------------
- 카메라(영상)당 JSON 1개. images[i] ↔ annotations[i] 가 1:1.
- 한 annotation 안에 bbox(여러 개, [x1,y1,x2,y2] 코너좌표)와 category_id 가
  병렬 리스트로 들어있다 (표준 COCO 와 다름).
- 이미지가 JSON 참조보다 적다(부분 다운로드) → 디스크 존재 확인 필수.
- 해상도가 FHD/HD 혼재 → 이미지마다 실제 크기를 읽어야 한다.
- 클래스 8종 → 기존 5종으로 매핑(`분류없음` 제외).

사용법
------
    conda activate dl
    python util/build_aihub_coco.py --ratio 2.0 --seed 42
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from PIL import Image  # noqa: E402

# ---------------------------------------------------------------------------
# 클래스 정의 (기존 Roboflow data.yaml 순서를 그대로 따른다)
#   YOLO idx:  0 bus, 1 car, 2 person, 3 riders, 4 truck
#   COCO cat id = yolo_idx + 1  (0 번은 Roboflow 관례상 placeholder 로 비워둠)
# ---------------------------------------------------------------------------
UNIFIED_NAMES = ["bus", "car", "person", "riders", "truck"]  # yolo idx 0..4

# AI-Hub category_id(1..8) -> 통합 yolo idx(0..4), None 이면 제외
AIHUB_TO_UNIFIED = {
    1: 1,  # 승용차          -> car
    2: 0,  # 소형버스        -> bus
    3: 0,  # 대형버스        -> bus
    4: 4,  # 트럭            -> truck
    5: 4,  # 대형 트레일러   -> truck
    6: 3,  # 오토바이(자전거)-> riders
    7: 2,  # 보행자          -> person
    8: None,  # 분류없음     -> 제외
}

AIHUB_ROOT = Path("cctv-object-dataset-aihub")
ROBOFLOW_ROOT = Path("cctv-object-dataset")


@dataclass
class ImageRecord:
    """COCO 한 장 + 그 박스들. coords 는 절대 픽셀(원본 해상도) 기준."""

    src_path: Path            # 원본 이미지 경로
    dst_name: str             # 대상 폴더에 복사될 파일명(고유)
    width: int
    height: int
    boxes: list[tuple[int, float, float, float, float]] = field(default_factory=list)
    # (unified_idx, x, y, w, h)  -- COCO xywh


# ---------------------------------------------------------------------------
# AI-Hub
# ---------------------------------------------------------------------------
def _aihub_candidates(split_label: str, split_dir: str) -> list[dict]:
    """한 트리(Training/Validation)의 모든 (이미지 메타 + 박스) 후보를 모은다.
    디스크 존재 확인은 하지 않는다(샘플링 단계에서 lazy 확인)."""
    base = AIHUB_ROOT / "label" / "01.데이터" / split_label / "교통안전(Bbox)"
    out: list[dict] = []
    for jf in sorted(base.rglob("*.json")):
        location = jf.parent.parent.name
        camera = jf.parent.name
        data_base = AIHUB_ROOT / "data" / split_dir / "교통안전(Bbox)" / f"[원천]{location}" / location
        d = json.load(open(jf, encoding="utf-8"))
        for img, ann in zip(d["images"], d["annotations"]):
            out.append({
                "src": data_base / img["file_name"],
                "camera": camera,
                "base": Path(img["file_name"]).name,
                "bboxes": ann["bbox"],
                "cats": ann["category_id"],
            })
    return out


def _build_aihub_record(cand: dict) -> ImageRecord | None:
    """후보 1건을 ImageRecord 로. 디스크에 없거나 유효 박스가 없으면 None."""
    src: Path = cand["src"]
    if not src.exists():
        return None
    try:
        with Image.open(src) as im:
            W, H = im.size
    except Exception:
        return None

    boxes: list[tuple[int, float, float, float, float]] = []
    for (x1, y1, x2, y2), cat in zip(cand["bboxes"], cand["cats"]):
        uni = AIHUB_TO_UNIFIED.get(int(cat))
        if uni is None:
            continue
        x1 = max(0.0, min(float(x1), W))
        y1 = max(0.0, min(float(y1), H))
        x2 = max(0.0, min(float(x2), W))
        y2 = max(0.0, min(float(y2), H))
        w, h = x2 - x1, y2 - y1
        if w <= 1.0 or h <= 1.0:
            continue
        boxes.append((uni, round(x1, 2), round(y1, 2), round(w, 2), round(h, 2)))

    if not boxes:
        return None
    dst = f"aihub_{cand['camera']}_{cand['base']}"
    return ImageRecord(src, dst, W, H, boxes)


def sample_aihub(split_label: str, split_dir: str, target: int, rng: random.Random,
                 exclude_dst: set[str] | None = None) -> list[ImageRecord]:
    """존재하고 유효 박스를 가진 이미지를 target 개수만큼 무작위 추출."""
    exclude_dst = exclude_dst or set()
    cands = _aihub_candidates(split_label, split_dir)
    rng.shuffle(cands)
    picked: list[ImageRecord] = []
    for c in cands:
        if len(picked) >= target:
            break
        rec = _build_aihub_record(c)
        if rec is None or rec.dst_name in exclude_dst:
            continue
        picked.append(rec)
    return picked


# ---------------------------------------------------------------------------
# Roboflow (YOLO seg/det, 정규화 좌표)
# ---------------------------------------------------------------------------
def load_roboflow_split(split: str) -> list[ImageRecord]:
    images_dir = ROBOFLOW_ROOT / split / "images"
    labels_dir = ROBOFLOW_ROOT / split / "labels"
    recs: list[ImageRecord] = []
    for img_path in sorted(images_dir.glob("*.jpg")):
        try:
            with Image.open(img_path) as im:
                W, H = im.size
        except Exception:
            continue
        boxes: list[tuple[int, float, float, float, float]] = []
        lp = labels_dir / (img_path.stem + ".txt")
        if lp.exists():
            for line in lp.read_text().splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                cls = int(parts[0])
                if cls < 0 or cls >= len(UNIFIED_NAMES):
                    continue
                coords = [float(v) for v in parts[1:]]
                if len(coords) == 4:  # det: cx cy w h
                    cx, cy, bw, bh = coords
                    x, y = (cx - bw / 2) * W, (cy - bh / 2) * H
                    w, h = bw * W, bh * H
                else:                 # seg polygon -> bbox
                    if len(coords) % 2:
                        coords = coords[:-1]
                    xs = [coords[i] * W for i in range(0, len(coords), 2)]
                    ys = [coords[i] * H for i in range(1, len(coords), 2)]
                    x, y = min(xs), min(ys)
                    w, h = max(xs) - x, max(ys) - y
                if w <= 1 or h <= 1:
                    continue
                boxes.append((cls, round(x, 2), round(y, 2), round(w, 2), round(h, 2)))
        recs.append(ImageRecord(img_path, f"rf_{img_path.name}", W, H, boxes))
    return recs


# ---------------------------------------------------------------------------
# COCO 출력
# ---------------------------------------------------------------------------
def write_split(records: list[ImageRecord], out_split: Path) -> dict:
    out_split.mkdir(parents=True, exist_ok=True)
    categories = [{"id": 0, "name": "_background_", "supercategory": "none"}]
    categories += [{"id": i + 1, "name": n, "supercategory": "object"}
                   for i, n in enumerate(UNIFIED_NAMES)]

    images, annotations = [], []
    ann_id = 1
    for img_id, rec in enumerate(records, start=1):
        shutil.copy2(rec.src_path, out_split / rec.dst_name)
        images.append({"id": img_id, "file_name": rec.dst_name,
                       "width": rec.width, "height": rec.height})
        for uni, x, y, w, h in rec.boxes:
            annotations.append({
                "id": ann_id, "image_id": img_id,
                "category_id": uni + 1,             # 1..5
                "bbox": [x, y, w, h], "area": round(w * h, 2), "iscrowd": 0,
            })
            ann_id += 1

    coco = {"info": {"description": f"CCTV object detection - {out_split.parent.name}/{out_split.name}"},
            "categories": categories, "images": images, "annotations": annotations}
    (out_split / "_annotations.coco.json").write_text(
        json.dumps(coco, ensure_ascii=False), encoding="utf-8")
    return coco


def _class_hist(records: list[ImageRecord]) -> Counter:
    c = Counter()
    for r in records:
        for uni, *_ in r.boxes:
            c[UNIFIED_NAMES[uni]] += 1
    return c


def _report(tag: str, recs: list[ImageRecord]) -> None:
    hist = _class_hist(recs)
    nbox = sum(hist.values())
    print(f"  {tag:28s} images={len(recs):5d}  boxes={nbox:7d}  {dict(hist)}")


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="datasets-rfdetr", help="출력 루트")
    ap.add_argument("--ratio", type=float, default=2.0,
                    help="AI-Hub:Roboflow train/valid 비율 (1.0~2.0 권장)")
    ap.add_argument("--n-test", type=int, default=250, help="AI-Hub held-out test 장수")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out = Path(args.out)

    # --- Roboflow ---
    print("[1/4] Roboflow 로드")
    rf_train = load_roboflow_split("train")
    rf_valid = load_roboflow_split("valid")
    rf_test = load_roboflow_split("test")
    _report("rf_train", rf_train)
    _report("rf_valid", rf_valid)
    _report("rf_test", rf_test)

    n_tr = round(len(rf_train) * args.ratio)
    n_va = round(len(rf_valid) * args.ratio)
    print(f"\n[2/4] AI-Hub 서브샘플  (ratio={args.ratio}: train≈{n_tr}, valid≈{n_va}, test={args.n_test})")
    # train 은 Training 트리, valid/test 는 Validation 트리(공식 split)에서 추출
    ah_train = sample_aihub("1.Training", "Training", n_tr, rng)
    ah_test = sample_aihub("2.Validation", "Validation", args.n_test, rng)
    ah_valid = sample_aihub("2.Validation", "Validation", n_va, rng,
                            exclude_dst={r.dst_name for r in ah_test})
    _report("aihub_train", ah_train)
    _report("aihub_valid", ah_valid)
    _report("aihub_test (held-out)", ah_test)

    # --- SHARED test = RF test + AIHub held-out (두 데이터셋 공통) ---
    shared_test = rf_test + ah_test

    print("\n[3/4] baseline 데이터셋 작성")
    write_split(rf_train, out / "baseline" / "train")
    write_split(rf_valid, out / "baseline" / "valid")
    write_split(shared_test, out / "baseline" / "test")

    print("[4/4] augmented 데이터셋 작성")
    write_split(rf_train + ah_train, out / "augmented" / "train")
    write_split(rf_valid + ah_valid, out / "augmented" / "valid")
    write_split(shared_test, out / "augmented" / "test")

    print("\n=== 요약 ===")
    _report("baseline/train", rf_train)
    _report("baseline/valid", rf_valid)
    _report("augmented/train", rf_train + ah_train)
    _report("augmented/valid", rf_valid + ah_valid)
    _report("SHARED/test (rf+aihub)", shared_test)
    print(f"\n완료 → {out}/baseline , {out}/augmented")
    print("학습:  python train.py --model rfdetr-object --dataset datasets-rfdetr/baseline ...")


if __name__ == "__main__":
    main()

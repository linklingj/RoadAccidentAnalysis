"""학습된 RF-DETR 체크포인트를 COCO test 셋에서 평가한다 (pycocotools mAP).

- 파일명 prefix(`rf_*` / `aihub_*`)로 도메인을 나눠 도메인별 mAP 를 따로 집계한다.
- 예측 class_id → model.class_names[class_id] → 이름 → GT category_id(이름 매칭)로
  안전하게 매핑한다(placeholder/offset 무관).

사용법:
    conda activate dl
    python util/eval_rfdetr.py \
        --ckpt runs/rfdetr/baseline/checkpoint_best_total.pth \
        --size nano --test-dir datasets-rfdetr/baseline/test \
        --tag baseline --out runs/rfdetr/eval_baseline.json
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np  # noqa: E402
from pycocotools.coco import COCO  # noqa: E402
from pycocotools.cocoeval import COCOeval  # noqa: E402

SIZE_TO_CLASS = {
    "nano": "RFDETRNano", "small": "RFDETRSmall", "medium": "RFDETRMedium",
    "base": "RFDETRBase", "large": "RFDETRLarge",
}


def load_model(ckpt: str, size: str):
    import rfdetr
    cls = getattr(rfdetr, SIZE_TO_CLASS[size])
    return cls(pretrain_weights=str(ckpt))


def get_class_names(model) -> list[str] | None:
    for getter in (lambda m: m.model.class_names,
                   lambda m: m.class_names,
                   lambda m: m.model.model.class_names):
        try:
            v = getter(model)
            if v:
                return list(v)
        except Exception:
            pass
    return None


def gather_predictions(model, test_dir: Path, names, name2catid, threshold: float):
    gt = COCO(str(test_dir / "_annotations.coco.json"))
    results = []
    domains = {"all": [], "rf": [], "aihub": []}
    for img_id in gt.getImgIds():
        info = gt.loadImgs(img_id)[0]
        fn = info["file_name"]
        det = model.predict(str(test_dir / fn), threshold=threshold)
        xyxy = np.asarray(det.xyxy).reshape(-1, 4)
        conf = np.asarray(det.confidence).reshape(-1)
        cid = np.asarray(det.class_id).reshape(-1)
        for (x1, y1, x2, y2), s, c in zip(xyxy, conf, cid):
            c = int(c)
            if names is not None and 0 <= c < len(names):
                catid = name2catid.get(names[c])
            else:
                catid = c
            if catid is None:
                continue
            results.append({
                "image_id": img_id, "category_id": int(catid),
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "score": float(s),
            })
        domains["all"].append(img_id)
        domains["aihub" if fn.startswith("aihub_") else "rf"].append(img_id)
    return gt, results, domains


def _summarize(gt, dt, img_ids, cat_ids=None):
    e = COCOeval(gt, dt, "bbox")
    e.params.imgIds = img_ids
    if cat_ids is not None:
        e.params.catIds = cat_ids
    e.evaluate()
    e.accumulate()
    with contextlib.redirect_stdout(io.StringIO()):
        e.summarize()
    # stats: [mAP, mAP50, mAP75, mAP_s, mAP_m, mAP_l, AR1, AR10, AR100, ...]
    return float(e.stats[0]), float(e.stats[1]), float(e.stats[2])


def evaluate(ckpt, size, test_dir, tag, threshold):
    test_dir = Path(test_dir)
    print(f"[eval] tag={tag} ckpt={ckpt}")
    model = load_model(ckpt, size)
    names = get_class_names(model)
    print(f"[eval] class_names={names}")

    gt = COCO(str(test_dir / "_annotations.coco.json"))
    name2catid = {c["name"]: c["id"] for c in gt.cats.values()}
    gt, results, domains = gather_predictions(model, test_dir, names, name2catid, threshold)
    print(f"[eval] {len(results)} detections over {len(domains['all'])} images "
          f"(rf={len(domains['rf'])}, aihub={len(domains['aihub'])})")

    out = {"tag": tag, "ckpt": str(ckpt), "threshold": threshold,
           "class_names": names, "domains": {}}
    if not results:
        print("[eval] WARNING: no detections!")
        return out

    dt = gt.loadRes(results)
    real_cats = [cid for cid, c in gt.cats.items() if c["name"] != "_background_"]
    catid2name = {c["id"]: c["name"] for c in gt.cats.values()}

    for dom, ids in domains.items():
        if not ids:
            continue
        mAP, mAP50, mAP75 = _summarize(gt, dt, ids, cat_ids=real_cats)
        per_class = {}
        for cid in real_cats:
            ap, ap50, _ = _summarize(gt, dt, ids, cat_ids=[cid])
            per_class[catid2name[cid]] = {"AP_50_95": round(ap, 4), "AP_50": round(ap50, 4)}
        out["domains"][dom] = {
            "n_images": len(ids),
            "mAP_50_95": round(mAP, 4), "mAP_50": round(mAP50, 4), "mAP_75": round(mAP75, 4),
            "per_class": per_class,
        }
        print(f"  [{dom:5s}] n={len(ids):4d}  mAP@50:95={mAP:.4f}  mAP@50={mAP50:.4f}  mAP@75={mAP75:.4f}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--size", default="nano", choices=list(SIZE_TO_CLASS))
    ap.add_argument("--test-dir", required=True, help="images + _annotations.coco.json 폴더")
    ap.add_argument("--tag", default="model")
    ap.add_argument("--threshold", type=float, default=0.001)
    ap.add_argument("--out", default=None, help="결과 JSON 저장 경로")
    args = ap.parse_args()

    out = evaluate(args.ckpt, args.size, args.test_dir, args.tag, args.threshold)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[eval] saved → {args.out}")


if __name__ == "__main__":
    main()

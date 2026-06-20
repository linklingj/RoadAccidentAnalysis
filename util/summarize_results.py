"""학습 metrics.csv + eval JSON 을 모아 보고서용 마크다운 표/요약을 출력한다.

사용법:
    python util/summarize_results.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(".")
RUNS = ROOT / "runs" / "rfdetr"


def last_val_row(metrics_csv: Path) -> dict | None:
    """마지막으로 val/mAP_50_95 가 기록된 행을 반환."""
    if not metrics_csv.exists():
        return None
    rows = list(csv.DictReader(open(metrics_csv, encoding="utf-8")))
    val_rows = [r for r in rows if r.get("val/mAP_50_95") not in (None, "")]
    return val_rows[-1] if val_rows else None


def best_val(metrics_csv: Path) -> tuple[float, int] | None:
    if not metrics_csv.exists():
        return None
    rows = list(csv.DictReader(open(metrics_csv, encoding="utf-8")))
    best = None
    for r in rows:
        v = r.get("val/ema_mAP_50_95") or r.get("val/mAP_50_95")
        if v in (None, ""):
            continue
        v = float(v)
        ep = int(float(r["epoch"]))
        if best is None or v > best[0]:
            best = (v, ep)
    return best


def main():
    print("# RF-DETR 실험 결과 요약\n")

    for tag in ("baseline", "augmented"):
        mc = RUNS / tag / "metrics.csv"
        bv = best_val(mc)
        lv = last_val_row(mc)
        print(f"## {tag}")
        if bv:
            print(f"- best val mAP@50:95 (EMA): {bv[0]:.4f} @ epoch {bv[1]}")
        if lv:
            print(f"- last epoch {lv['epoch']}: val mAP@50:95={float(lv['val/mAP_50_95']):.4f}, "
                  f"mAP@50={float(lv['val/mAP_50']):.4f}")
        print()

    # eval JSONs
    print("## Test 평가 (held-out)\n")
    evals = {}
    for tag in ("baseline", "augmented"):
        p = RUNS / f"eval_{tag}.json"
        if p.exists():
            evals[tag] = json.loads(p.read_text(encoding="utf-8"))

    if evals:
        for dom in ("all", "rf", "aihub"):
            print(f"### test 도메인: {dom}")
            print("| model | mAP@50:95 | mAP@50 | mAP@75 |")
            print("|---|---|---|---|")
            for tag, e in evals.items():
                d = e.get("domains", {}).get(dom)
                if d:
                    print(f"| {tag} | {d['mAP_50_95']:.4f} | {d['mAP_50']:.4f} | {d['mAP_75']:.4f} |")
            print()

        # per-class on aihub
        print("### per-class AP@50:95 (aihub 도메인)")
        cls = list(next(iter(evals.values()))["domains"].get("aihub", {}).get("per_class", {}))
        if cls:
            print("| class | " + " | ".join(evals) + " |")
            print("|---" * (len(evals) + 1) + "|")
            for c in cls:
                row = [c]
                for tag, e in evals.items():
                    pc = e["domains"]["aihub"]["per_class"].get(c, {})
                    row.append(f"{pc.get('AP_50_95', 0):.4f}")
                print("| " + " | ".join(row) + " |")
        print()


if __name__ == "__main__":
    main()

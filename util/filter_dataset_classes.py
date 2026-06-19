#!/usr/bin/env python3
"""cctv-object-dataset 라벨에서 추론에 사용하지 않는 클래스를 제거한다.

원본 클래스 매핑 (7개):
  0: attention  ← 제거 (학습 gradient의 ~8% 낭비)
  1: bus        → 0
  2: car        → 1
  3: crosswalk  ← 제거 (학습 gradient의 ~74% 낭비)
  4: person     → 2
  5: riders     → 3
  6: truck      → 4

attention+crosswalk 두 클래스가 전체 annotation의 약 82%를 차지하면서
실제 탐지에는 쓰이지 않아 차량·보행자 클래스의 학습 효율을 크게 떨어뜨린다.
이 스크립트 실행 후 `python train.py --model object`로 재학습하면 된다.

사용법:
    python util/filter_dataset_classes.py --dry-run   # 결과 미리 보기
    python util/filter_dataset_classes.py             # 실제 적용
    python util/filter_dataset_classes.py --dataset other-dataset-dir
"""

import argparse
import re
import shutil
from pathlib import Path

REMOVE_IDS = {0, 3}  # attention(0), crosswalk(3)
REMAP = {1: 0, 2: 1, 4: 2, 5: 3, 6: 4}
NEW_NAMES = ["bus", "car", "person", "riders", "truck"]


def filter_label_file(path: Path, dry_run: bool) -> tuple[int, int]:
    """라벨 파일에서 제거 클래스를 걸러내고 인덱스를 재번호한다.

    Returns:
        (original_line_count, kept_line_count)
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    kept: list[str] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        try:
            cls_id = int(parts[0])
        except (ValueError, IndexError):
            continue
        if cls_id in REMOVE_IDS:
            continue
        new_id = REMAP.get(cls_id)
        if new_id is None:
            continue
        parts[0] = str(new_id)
        kept.append(" ".join(parts))

    if not dry_run:
        path.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
    return len(lines), len(kept)


def update_data_yaml(yaml_path: Path, dry_run: bool) -> None:
    if not yaml_path.is_file():
        print(f"  경고: data.yaml 없음 ({yaml_path})")
        return

    content = yaml_path.read_text(encoding="utf-8")
    content = re.sub(r"nc\s*:\s*\d+", "nc: 5", content)
    names_str = str(NEW_NAMES).replace("'", '"')
    content = re.sub(r"names\s*:\s*\[.*?\]", f"names: {names_str}", content, flags=re.DOTALL)

    if dry_run:
        print(f"  [dry-run] data.yaml 업데이트 예정 → nc: 5, names: {NEW_NAMES}")
    else:
        backup = yaml_path.with_suffix(".yaml.bak")
        shutil.copy2(yaml_path, backup)
        yaml_path.write_text(content, encoding="utf-8")
        print(f"  data.yaml 업데이트 완료 (백업: {backup.name})")


def process_dataset(dataset_dir: Path, dry_run: bool) -> None:
    total_orig = total_kept = total_files = 0

    for split in ("train", "valid", "test"):
        labels_dir = dataset_dir / split / "labels"
        if not labels_dir.is_dir():
            continue
        files = sorted(labels_dir.glob("*.txt"))
        if not files:
            continue
        print(f"\n[{split}] {len(files)}개 라벨 파일")
        split_orig = split_kept = 0
        for f in files:
            o, k = filter_label_file(f, dry_run)
            split_orig += o
            split_kept += k
        print(f"  annotation {split_orig} → {split_kept} (제거 {split_orig - split_kept})")
        total_orig += split_orig
        total_kept += split_kept
        total_files += len(files)

    print(f"\n총계: 파일 {total_files}개, annotation {total_orig} → {total_kept} (제거 {total_orig - total_kept})")

    update_data_yaml(dataset_dir / "data.yaml", dry_run)

    if not dry_run:
        print("\n완료! 재학습 전 infer.py 상단의 OBJECT_CLASS_NAMES를 아래로 변경하세요:")
        print('  OBJECT_CLASS_NAMES = ["bus", "car", "person", "riders", "truck"]')
        print("\n재학습:")
        print("  python train.py --model object")


def main() -> None:
    parser = argparse.ArgumentParser(description="필터 클래스 제거 및 인덱스 재번호")
    parser.add_argument("--dataset", default="cctv-object-dataset", help="데이터셋 루트 디렉토리")
    parser.add_argument("--dry-run", action="store_true", help="실제 파일 수정 없이 결과만 확인")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.is_dir():
        raise SystemExit(f"오류: 데이터셋 디렉토리를 찾을 수 없습니다: {dataset_path.resolve()}")

    mode = "dry-run (미리 보기)" if args.dry_run else "실제 적용"
    print(f"데이터셋: {dataset_path.resolve()}")
    print(f"모드    : {mode}")
    print(f"제거 클래스: attention(0), crosswalk(3)")
    print(f"재번호: bus→0, car→1, person→2, riders→3, truck→4")

    process_dataset(dataset_path, dry_run=args.dry_run)


if __name__ == "__main__":
    main()

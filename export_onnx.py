"""ONNX 모델 변환 스크립트.

사용법:
    python export_onnx.py            # crosswalk + U-Net 전체 변환
    python export_onnx.py --crosswalk
    python export_onnx.py --unet

생성 위치:
    - crosswalk: runs/segment/0407-crosswalk/weights/best.onnx (또는 --crosswalk-model 지정 경로)
    - U-Net    : runs/smp-road/best.onnx (또는 --unet-model 지정 경로)

dl conda 환경에서 실행:
    conda activate dl
    python export_onnx.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _export_yolo(pt_path: str, label: str) -> str:
    from ultralytics import YOLO

    onnx_path = str(Path(pt_path).with_suffix(".onnx"))
    print(f"[{label}] {pt_path} → {onnx_path}")
    model = YOLO(pt_path)
    model.export(format="onnx", opset=18, simplify=True, dynamic=False, imgsz=640)
    _verify_yolo_onnx(onnx_path, label)
    print(f"[{label}] 완료: {onnx_path}")
    return onnx_path


def export_crosswalk(pt_path: str) -> str:
    return _export_yolo(pt_path, "crosswalk")


def export_object(pt_path: str) -> str:
    return _export_yolo(pt_path, "object")


def export_unet(pt_path: str) -> str:
    import torch
    from util.smp_road.model import load_checkpoint

    onnx_path = str(Path(pt_path).with_suffix(".onnx"))
    print(f"[U-Net] {pt_path} → {onnx_path}")

    model, _ = load_checkpoint(pt_path, device="cpu")
    model.eval()
    dummy = torch.zeros(1, 3, 512, 512)

    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        opset_version=18,
        input_names=["image"],
        output_names=["logits"],
    )

    _try_simplify_onnx(onnx_path)
    _verify_unet_onnx(onnx_path)
    print(f"[U-Net] 완료: {onnx_path}")
    return onnx_path


def _try_simplify_onnx(onnx_path: str) -> None:
    try:
        import onnx
        from onnxsim import simplify as onnx_simplify  # type: ignore

        model_proto = onnx.load(onnx_path)
        simplified, ok = onnx_simplify(model_proto)
        if ok:
            onnx.save(simplified, onnx_path)
            print(f"  onnx-simplifier 적용됨")
        else:
            print(f"  onnx-simplifier 실패 (원본 유지)")
    except ImportError:
        print("  onnxsim 미설치 (pip install onnxsim) — 단순화 건너뜀")


def _verify_yolo_onnx(onnx_path: str, label: str = "") -> None:
    """ultralytics가 .onnx를 로드해 predict 가능한지 확인."""
    import numpy as np
    from ultralytics import YOLO

    if not Path(onnx_path).exists():
        print(f"  [경고] {onnx_path} 파일이 없음 — 검증 건너뜀")
        return
    model = YOLO(onnx_path)
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    result = model.predict(source=dummy, device="cpu", verbose=False, conf=0.1)
    assert result is not None, "YOLO ONNX predict 반환값이 None"
    print(f"  sanity check 통과 (predict 정상)")


def _verify_unet_onnx(onnx_path: str) -> None:
    """onnxruntime으로 더미 입력을 실행해 shape/NaN을 확인한다."""
    import numpy as np
    import onnxruntime as ort

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    inp_name = sess.get_inputs()[0].name
    dummy = np.zeros((1, 3, 512, 512), dtype=np.float32)
    out = sess.run(None, {inp_name: dummy})[0]
    assert out.shape == (1, 1, 512, 512), f"예상 shape (1,1,512,512), 실제 {out.shape}"
    assert not bool(np.isnan(out).any()), "출력에 NaN 포함"
    print(f"  sanity check 통과 (shape={out.shape}, NaN 없음)")


def main() -> None:
    parser = argparse.ArgumentParser(description="ONNX 모델 변환")
    parser.add_argument("--crosswalk", action="store_true", help="crosswalk YOLO 모델만 변환")
    parser.add_argument("--unet", action="store_true", help="U-Net road 모델만 변환")
    parser.add_argument("--object", action="store_true", help="object YOLO 모델만 변환")
    parser.add_argument("--crosswalk-model", type=str,
                        default="runs/segment/0407-crosswalk/weights/best.pt",
                        help="crosswalk YOLO .pt 경로")
    parser.add_argument("--unet-model", type=str,
                        default="runs/smp-road/best.pt",
                        help="U-Net road .pt 경로")
    parser.add_argument("--object-model", type=str,
                        default="runs/segment/0619-object/weights/best.pt",
                        help="object YOLO .pt 경로")
    args = parser.parse_args()

    run_all = not args.crosswalk and not args.unet and not args.object

    errors: list[str] = []

    if run_all or args.crosswalk:
        if not Path(args.crosswalk_model).exists():
            print(f"[건너뜀] crosswalk 모델 없음: {args.crosswalk_model}", file=sys.stderr)
        else:
            try:
                export_crosswalk(args.crosswalk_model)
            except Exception as e:
                errors.append(f"crosswalk: {e}")
                print(f"[오류] crosswalk 변환 실패: {e}", file=sys.stderr)

    if run_all or args.unet:
        if not Path(args.unet_model).exists():
            print(f"[건너뜀] U-Net 모델 없음: {args.unet_model}", file=sys.stderr)
        else:
            try:
                export_unet(args.unet_model)
            except Exception as e:
                errors.append(f"unet: {e}")
                print(f"[오류] U-Net 변환 실패: {e}", file=sys.stderr)

    if run_all or args.object:
        if not Path(args.object_model).exists():
            print(f"[건너뜀] object 모델 없음: {args.object_model}", file=sys.stderr)
        else:
            try:
                export_object(args.object_model)
            except Exception as e:
                errors.append(f"object: {e}")
                print(f"[오류] object 변환 실패: {e}", file=sys.stderr)

    if errors:
        sys.exit(1)
    print("\n변환 완료. --use-onnx 플래그로 ONNX 추론을 활성화할 수 있습니다.")


if __name__ == "__main__":
    main()

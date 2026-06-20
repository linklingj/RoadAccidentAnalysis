"""SMP(segmentation_models_pytorch) 기반 도로 segmentation 실험 패키지.

기존 YOLO-seg(`infer.py` 도로 단계)와 비교하기 위한 U-Net 학습/추론 코드.
- dataset.py : YOLO polygon 라벨 -> binary mask 로더
- model.py   : U-Net 빌드/체크포인트 로드, BGR 이미지 -> polygon 추론 헬퍼
"""

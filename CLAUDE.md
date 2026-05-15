# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Goal

CCTV 영상을 기반으로 교통사고 현장을 3차원 공간에 재구성하는 딥러닝 실습 프로젝트 (22011807 최재현).  
2D 영상만으로는 파악하기 어려운 차량 간 거리, 충돌 지점, 차선 이탈 여부 등을 3D로 시각화하여 사고 원인 분석 및 시나리오 시뮬레이션에 활용한다.

최종 목표는 BEV(Bird's Eye View) 좌표계로 추출한 도로/객체 정보를 Unity 게임 엔진으로 전달해 다양한 시점에서 사고 상황을 재현하는 것이다. **Unity 연동 및 Flask 서버는 아직 미구현** 상태이며, 현재 구현은 BEV 영상 렌더링까지 완료된 상태다.

## Commands

### Python 환경

모든 Python 명령은 `dl` conda 환경에서 실행한다 (`cv2`, `torch`, `ultralytics` 등 의존성이 이 환경에만 설치되어 있음).

```bash
conda activate dl
```

기본 `python3` (`/opt/homebrew/anaconda3/bin/python3`)에는 `cv2`가 없으므로 활성화 없이 `python` / `python3` 호출 시 `ModuleNotFoundError: No module named 'cv2'`가 발생한다. 환경 활성화 없이 실행하려면 인터프리터를 명시한다: `/opt/homebrew/anaconda3/envs/dl/bin/python main.py ...`.

### Run inference
```bash
# Image mode
python main.py --image input/image1.png --output-dir output

# Video mode
python main.py --video input/video1.mp4 --output-dir output

# With custom models and camera settings
python main.py --image input/image1.png \
  --road-model runs/segment/0405-road/weights/best.pt \
  --crosswalk-model runs/segment/0407-crosswalk/weights/best.pt \
  --object-model runs/segment/0401-object/weights/best.pt \
  --camera-height 6.5 --ppm 42.0
```

### Train models
```bash
python train.py --model road        # 도로 segmentation (cctv-roadseg-dataset/data.yaml)
python train.py --model crosswalk   # 횡단보도 segmentation (cctv-crosswalk-dataset/data.yaml)
python train.py --model object      # 객체 탐지 (cctv-object-dataset/data.yaml)
```
학습된 가중치는 `runs/segment/<run-name>/weights/best.pt`에 저장된다.

### Fetch live CCTV video
```bash
# 위경도로 가장 가까운 CCTV 탐색 (ITS OpenAPI, .env의 cctv_api_key 필요)
python util/convert_live_video.py --lat 36.586 --lng 128.186

# 60초 영상 다운로드
python util/convert_live_video.py --lat 36.586 --lng 128.186 --download --output input/video.mp4 --seconds 60
```
HLS/m3u8 스트리밍 URL 처리에는 `ffmpeg`가 필요하다.

## Architecture

### 전체 파이프라인

```
CCTV 영상 입력
    │
    ├─ 도로/횡단보도 Segmentation (YOLO-seg)
    │      └─ UV 공간의 polygon mask
    │
    ├─ 객체 탐지 (YOLO-seg)
    │      └─ bounding box + 발끝 좌표(footpoint)
    │
    ├─ 카메라 파라미터 추정 (PerspectiveFields)
    │      └─ roll, pitch, vFOV, 주점(principal point)
    │
    ├─ BEV 투영 (Homography-based ground plane projection)
    │      └─ UV → 실세계 좌표(m) → BEV 캔버스 픽셀
    │
    ├─ 객체 추적 (BEVObjectTracker, 영상 모드)
    │      └─ 세계 좌표계 greedy nearest-neighbor matching
    │
    └─ [미구현] Unity 3D 렌더링
           └─ Flask 서버를 통해 BEV 결과 전달 예정
```

전체 추론 로직은 `infer.py`에 있으며, `main.py`는 `PipelineConfig`를 구성해 `run_pipeline` (이미지) 또는 `run_video_pipeline` (영상)을 호출하는 CLI 래퍼다.

### 파이프라인 단계 상세

1. **도로 segmentation** — `infer_road_model()` → UV 공간 polygon 목록
2. **횡단보도 segmentation** — `infer_crosswalk_model()` (class index 1만 사용) → UV 공간 polygon 목록
3. **객체 탐지** — `infer_object_model()` → bbox + footpoint (`attention`, `crosswalk` 클래스는 필터링)
4. **카메라 추정** — `estimate_camera_params()`가 PerspectiveFields를 호출해 roll/pitch/vFOV 추출; vFOV 없으면 65° 기본값
5. **BEV 투영** — `project_uv_to_ground()`가 각 UV 점을 `camera_height_m` 높이의 평평한 지면에 ray-cast; `world_to_bev()`로 캔버스 픽셀 변환. **지면이 평면이라는 가정에 의존**
6. **추적 (영상 모드)** — `BEVObjectTracker`가 세계 좌표 기준 greedy 매칭, 클래스 일치 조건, trajectory deque 관리

### 핵심 클래스 / 진입점

| 심볼 | 파일 | 역할 |
|---|---|---|
| `PipelineConfig` | `infer.py:23` | 모델 경로, confidence, BEV 크기, 추적 파라미터 등 모든 설정 |
| `RoadSceneProjector` | `infer.py:598` | 세 모델 + PerspectiveFields 로드; `run()` / `run_video()` 소유 |
| `BEVObjectTracker` | `infer.py:383` | BEV 세계 좌표 기반 greedy 추적기 |
| `render_bev_scene` | `infer.py:474` | BEV 캔버스에 도로/횡단보도/trajectory/객체 렌더링 |
| `render_image_overlay` | `infer.py:548` | 원본 프레임에 bbox와 footpoint 오버레이 |

### 데이터셋 (gitignore됨, 로컬에서만 관리)

| 폴더 | 출처 | 용도 |
|---|---|---|
| `cctv-roadseg-dataset/` | Roboflow | 도로/차선 segmentation 학습 |
| `cctv-crosswalk-dataset/` | AI HUB | 횡단보도 segmentation 학습 |
| `cctv-object-dataset/` | AI HUB | 차량·보행자 탐지 학습 |

### 모델 파일

- `yolo26l-seg.pt` — 도로/횡단보도 학습 시작점 (커밋됨)
- `yolo26s-seg.pt` — 객체 탐지 학습 시작점 (더 경량, 커밋 안 됨)
- 학습 완료 모델은 `runs/` 아래 저장되며 gitignore됨

### 정적 카메라 최적화 (영상 모드)

도로/횡단보도 segmentation과 카메라 파라미터는 **첫 프레임에서만** 계산한다 (CCTV는 고정 카메라 가정). 매 프레임 재계산이 필요하면 `PipelineConfig.video_recompute_camera_each_frame = True`로 설정.

### 외부 의존성: PerspectiveFields

`from perspective2d import PerspectiveFields`로 로드. 패키지 미설치 시 `infer.py`는 레포 루트의 `PerspectiveFields/` 로컬 디렉토리로 fallback한다.

## Planned Components (미구현)

- **Flask 서버**: BEV 좌표 데이터를 Unity로 전달하는 API 서버
- **Unity 3D 렌더링**: WebGL 빌드로 다양한 시점에서 사고 현장 시각화
- **배포**: Docker + AWS EC2 구성
- **속도 추정**: 프레임 간 BEV 좌표 변위로 객체 속도 계산
- **차량 세분류**: 차종·색상 구분 (현재는 단일 class로 탐지)

깃 commit co-author / author에 claude를 남기지 말 것
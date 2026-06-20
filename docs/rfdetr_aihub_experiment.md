# RF-DETR 객체 탐지: AI-Hub 데이터셋 추가 학습 실험

> 작성: 2026-06-20 · 브랜치 `feature/rfdetr-aihub-dataset`
> 목표: 기존 Roboflow CCTV 데이터셋으로만 학습한 RF-DETR 객체 탐지 모델(**baseline**)에
> 대규모 AI-Hub `교통안전(Bbox)` 데이터셋의 일부를 추가(**augmented**)하면 탐지 성능이
> 어떻게 달라지는지 정량 비교한다.

---

## 1. 요약 (TL;DR)

- 신규 데이터셋(AI-Hub `교통문제 해결을 위한 CCTV 교통영상(시내도로)` 교통안전 Bbox)을
  전처리해 기존 Roboflow 데이터셋과 **클래스 체계를 통일**하고 COCO 포맷으로 합쳤다.
- AI-Hub 원본은 약 **13.5만 장 / 79 GB** 로 너무 커서, 기존 train(608장)의 **2배(=1,216장)**
  만 무작위 추출(seed 고정)해 사용했다. (요구 조건: "기존과 동일~최대 2배 비율")
- 동일한 RF-DETR(Nano) 구조·하이퍼파라미터로 **baseline**(Roboflow only)과
  **augmented**(Roboflow + AI-Hub 2×) 두 모델을 학습했다.
- 두 모델을 **동일한 held-out test 셋**에서 평가했다. test 셋은
  Roboflow test(83장) + AI-Hub held-out(250장)으로 구성되며, 도메인별 mAP 를 분리 집계한다.

> 📌 **핵심 결과 표/수치는 §6 에 있다.**

---

## 2. 데이터셋

### 2.1 기존 데이터셋 (Roboflow, `cctv-object-dataset/`)

- 포맷: YOLO segmentation(정규화 polygon), 이미지 640×640.
- 클래스 5종: `bus, car, person, riders, truck` (이미 `attention`/`crosswalk` 제거 완료본).
- 규모: **train 608 / valid 164 / test 83** (총 855장, 5,229 boxes).

| split | images | boxes | bus | car | person | riders | truck |
|---|---|---|---|---|---|---|---|
| train | 608 | 3,782 | 120 | 2,674 | 365 | 509 | 114 |
| valid | 164 | 1,022 | 35 | 718 | 119 | 121 | 29 |
| test  | 83  | 425   | 12 | 288 | 57  | 50  | 18  |

### 2.2 신규 데이터셋 (AI-Hub, `cctv-object-dataset-aihub/`)

- 출처: AI-Hub "교통문제 해결을 위한 CCTV 교통영상(시내도로)" — 교통안전(Bbox).
- 구조: `data/`(원천 이미지)와 `label/`(어노테이션 JSON)이 **분리된 트리**.
  - 이미지: `data/{Training,Validation}/교통안전(Bbox)/[원천]<지점>/<지점>/<카메라ID>/*.jpg`
  - 라벨: `label/01.데이터/{1.Training,2.Validation}/.../1.라벨링데이터/<지점>/<카메라ID>/<지점>_<카메라ID>.json`
- 규모(원본):

| split | 카메라(JSON) | 이미지(JSON 참조) | 디스크 실제 | boxes |
|---|---|---|---|---|
| Training   | 70 | 145,701 | 119,802 | 1,662,454 |
| Validation | 45 | 18,212  | 15,919  | 225,357 |

- 총 용량 약 **79 GB**. JSON 이 참조하는 이미지보다 디스크 실제 파일이 **적다(부분 다운로드)**
  → 전처리 시 **이미지 존재 여부를 반드시 확인**.
- 해상도 **FHD(1920×1080) / HD(1280×720) 혼재** → 이미지마다 실제 크기를 읽어야 함.
- 클래스 8종(원본):

| id | 이름 | 통합 매핑 |
|---|---|---|
| 1 | 승용차 | → `car` |
| 2 | 소형버스 | → `bus` |
| 3 | 대형버스 | → `bus` |
| 4 | 트럭 | → `truck` |
| 5 | 대형 트레일러 | → `truck` |
| 6 | 오토바이(자전거) | → `riders` |
| 7 | 보행자 | → `person` |
| 8 | 분류없음 | **제외** |

### 2.3 AI-Hub 라벨 포맷의 특이점 (표준 COCO 와 다름)

설명서(`설명서.pdf`) 및 실제 JSON 확인 결과:

- 카메라(영상)당 JSON 1개. `images[i]` 와 `annotations[i]` 가 **1:1 대응**.
- 표준 COCO 는 annotation 1개 = 객체 1개지만, 이 포맷은 **annotation 1개에 한 프레임의
  모든 객체**가 들어있다: `bbox`(여러 개)와 `category_id`(여러 개)가 **병렬 리스트**.
- `bbox` 좌표는 **`[x1, y1, x2, y2]` 코너 좌표**(COCO 의 `[x, y, w, h]` 가 아님).
- `iscrowd`, `area`, `segmentation`, `tracking_id` 필드는 채워져 있지 않거나 무의미.

---

## 3. 전처리 (`util/build_aihub_coco.py`)

신규 데이터셋을 기존과 합쳐 RF-DETR 학습용 COCO 로 빌드하는 단일 스크립트.

1. **클래스 통일**: AI-Hub 8종 → 기존 5종(`bus, car, person, riders, truck`)으로 매핑,
   `분류없음` 제외. COCO category id = `통합 idx + 1` (0번은 Roboflow 관례상 placeholder).
2. **좌표 변환**: `[x1,y1,x2,y2]` → COCO `[x,y,w,h]`, 이미지 경계로 clip,
   `w<=1` 또는 `h<=1` 인 degenerate box 제거.
3. **이미지 경로 해석**: 라벨 JSON 위치 → 원천 이미지 경로 유도, **디스크 존재 확인**.
4. **해상도 처리**: 각 이미지의 실제 크기를 PIL 로 읽음(FHD/HD 혼재 대응).
   - ⚠️ OpenCV `cv2.imread` 는 Windows 의 **한글 경로를 못 읽으므로** PIL 사용.
5. **무작위 서브샘플링**: `--seed` 고정. 존재하고 유효 box 가 있는 이미지만 채택.
   - AI-Hub **train** 은 공식 Training 트리, **valid/test** 는 공식 Validation 트리에서 추출
     (train/test 간 영상 분리로 leakage 최소화).
6. **파일명 prefix**: `rf_*`(Roboflow) / `aihub_*`(AI-Hub) → 평가 시 도메인 분리에 사용.

### 3.1 서브샘플 비율 결정

- 요구 조건: "기존 데이터셋과 **동일하거나 최대 2배** 비율".
- 채택: **train 2× (608→1,216), valid 2× (164→328)**, 별도 AI-Hub held-out test 250장.
- 이유: 상한(2×)을 택해 신규 데이터의 효과를 최대한 관찰하되, 원본의 0.8% 수준만 사용해
  학습 시간을 합리적으로 유지. (전체 사용 시 13.5만 장 → 단일 GPU 비현실적)

---

## 4. 빌드 결과 (`datasets-rfdetr/`, ratio=2.0, seed=42)

두 데이터셋의 **test 스플릿은 완전히 동일**하다(공정 비교).

| 데이터셋 | split | images | boxes | 구성 |
|---|---|---|---|---|
| **baseline**  | train | 608  | 3,782  | RF train |
|               | valid | 164  | 1,022  | RF valid |
|               | test  | 333  | 3,581  | RF test(83) + AIHub held-out(250) |
| **augmented** | train | 1,824 | 17,710 | RF train(608) + AIHub(1,216) |
|               | valid | 492  | 5,301  | RF valid(164) + AIHub(328) |
|               | test  | 333  | 3,581  | (baseline 과 동일) |

AI-Hub 서브샘플 클래스 분포(train 1,216장): car 9,384 · person 2,182 · truck 1,411 ·
bus 649 · riders 302.

**augmented/train 합산 클래스 분포**: car 12,058 · person 2,547 · truck 1,525 · bus 769 · riders 811.
→ 모든 클래스에서 box 수가 증가했고, 특히 희소했던 truck(114→1,525), bus(120→769),
person(365→2,547) 의 보강 폭이 크다.

---

## 5. 학습 설정

| 항목 | 값 |
|---|---|
| 모델 | **RF-DETR Nano** (DINOv2 windowed-small backbone, 30.2 M params, 입력 384px) |
| 초기화 | COCO 사전학습 가중치 (rfdetr 기본) |
| epochs | 50 |
| batch | 8 × grad_accum 2 (effective 16) |
| lr | 1e-4 (encoder 1.5e-4), weight decay 1e-4, EMA on |
| 입력 | multi-scale, square resize |
| GPU | NVIDIA RTX 5070 Ti (17 GB), torch 2.10.0+cu128 |
| 공통 | baseline/augmented **모든 설정 동일**, 학습 데이터만 다름 |

- baseline 과 augmented 는 **완전히 동일한 코드/하이퍼파라미터**로 학습하여,
  성능 차이가 오직 "AI-Hub 데이터 추가" 에서 비롯되도록 통제했다.
- 모델 변형(`--rfdetr-size`)은 `train.py` 에 추가. Nano 를 선택한 이유는 2개 모델을
  단일 GPU 에서 현실적 시간 내 학습하기 위함(비교의 상대성은 유지됨).

재현 명령:

```bash
conda activate dl
# 1) 데이터셋 빌드
python util/build_aihub_coco.py --ratio 2.0 --seed 42 --n-test 250
# 2) 학습
python train.py --model rfdetr-object --dataset datasets-rfdetr/baseline  --rfdetr-size nano --epochs 50 --batch 8 --grad-accum 2 --output-dir runs/rfdetr/baseline
python train.py --model rfdetr-object --dataset datasets-rfdetr/augmented --rfdetr-size nano --epochs 50 --batch 8 --grad-accum 2 --output-dir runs/rfdetr/augmented
# 3) 평가 (동일 test 셋, 도메인별 분리)
python util/eval_rfdetr.py --ckpt runs/rfdetr/baseline/checkpoint_best_total.pth  --size nano --test-dir datasets-rfdetr/baseline/test  --tag baseline  --out runs/rfdetr/eval_baseline.json
python util/eval_rfdetr.py --ckpt runs/rfdetr/augmented/checkpoint_best_total.pth --size nano --test-dir datasets-rfdetr/augmented/test --tag augmented --out runs/rfdetr/eval_augmented.json
python util/summarize_results.py
```

---

## 6. 결과

### 6.1 검증(valid) mAP 추이

두 모델의 val set 구성이 다름을 주의: baseline 은 RF valid(164장), augmented 는 RF+AIHub valid(492장).
따라서 **수치 직접 비교는 부적절하며 §6.2 의 공통 test 셋 결과로 비교**해야 한다.

| epoch | baseline val mAP@50:95 (EMA) | baseline val mAP@50 | augmented val mAP@50:95 (EMA) | augmented val mAP@50 |
|---:|---|---|---|---|
| 1  | 0.3092 | 0.5201 | 0.2615 | 0.4443 |
| 5  | 0.3910 | 0.6527 | 0.2930 | 0.4671 |
| 10 | 0.4121 | 0.7057 | 0.3140 | 0.5093 |
| 15 | 0.4389 | 0.7084 | 0.3234 | 0.5243 |
| 20 | 0.4449 | 0.7479 | 0.3287 | 0.5204 |
| 25 | 0.4489 | 0.7219 | 0.3308 | 0.5267 |
| 30 | 0.4678 | 0.7554 | 0.3288 | 0.5214 |
| 35 | **0.4736** ★ | 0.7608 | 0.3310 | 0.5406 |
| 40 | 0.4687 | 0.7587 | 0.3348 | 0.5371 |
| 45 | 0.4688 | 0.7623 | 0.3389 | 0.5364 |
| 50 | 0.4552 | 0.7525 | **0.3395** ★ | 0.5335 |

★ = best EMA mAP@50:95

### 6.2 Test 셋 비교 (동일 held-out, 333장)

공정 비교를 위해 두 모델을 **완전히 동일한 333장**(RF test 83 + AI-Hub held-out 250)으로 평가했다.

| 도메인 (이미지 수) | baseline mAP@50:95 | augmented mAP@50:95 | 변화 | baseline mAP@50 | augmented mAP@50 | 변화 |
|---|---|---|---|---|---|---|
| **전체** (333) | 0.2534 | **0.3667** | **+44.7 %** | 0.4297 | **0.5912** | **+37.6 %** |
| RF 도메인 (83) | 0.5294 | **0.5370** | +1.4 % | 0.8390 | **0.8624** | +2.8 % |
| AI-Hub 도메인 (250) | 0.2112 | **0.3390** | **+60.5 %** | 0.3588 | **0.5402** | **+50.6 %** |

> **결론**: AI-Hub 데이터 추가는 AI-Hub 도메인 성능을 대폭 끌어올리면서(mAP@50:95 +60.5 %),
> 기존 RF 도메인 성능도 소폭 개선(+1.4 %)했다. 도메인 간 성능 격차(catastrophic degradation)는
> 관찰되지 않았다.

### 6.3 클래스별 분석

아래는 AI-Hub 도메인(250장) 기준 클래스별 AP@50:95.

| 클래스 | baseline | augmented | 변화 | 비고 |
|---|---|---|---|---|
| bus     | 0.3161 | **0.5077** | +60.6 % | 학습 데이터 bus 649→769 (+18 %) |
| car     | 0.3602 | **0.4758** | +32.1 % | 가장 많은 클래스 (12,058 box) |
| person  | 0.0466 | **0.1098** | +135.6 % | 절대값은 낮지만 상대 개선 최대 |
| riders  | 0.2222 | **0.3020** | +35.9 % | 오토바이 계열, AI-Hub 302 box |
| truck   | 0.1109 | **0.2994** | +170.0 % | 희소 클래스 최대 수혜 (114→1,525 box) |

- **truck** 이 가장 극적으로 개선됐다. 기존 RF 데이터에 truck 이 114 box 밖에 없었는데
  AI-Hub 에서 1,411 box 가 추가되어 1,525 box 로 늘어난 결과다.
- **person** 은 절대 AP 가 낮다. AI-Hub 의 person 상당수가 소형·원거리 보행자여서
  50 epoch / Nano 모델로는 한계가 있다.
- **car** 가 이미 충분히 많은 데이터를 갖고 있어 상대 개선 폭이 가장 작다.

---

## 7. 결론 및 한계

### 7.1 결론

AI-Hub 교통안전(Bbox) 데이터셋을 전처리·통합한 **augmented** 모델은 동일 test 셋에서
baseline 대비 **전 클래스·전 도메인에서 성능이 향상**됐다.

- 전체 mAP@50:95 **0.253 → 0.367** (+44.7 %): 단 2배 분량의 추가 데이터만으로도 큰 폭의 개선.
- RF 도메인은 거의 유지(+1.4 %): 기존 도메인 성능이 희생되지 않았다.
- AI-Hub 도메인 **0.211 → 0.339** (+60.5 %): 새 도메인 학습의 직접적 효과.
- 데이터가 부족했던 희소 클래스(truck, bus, person)의 개선이 두드러졌다.

데이터 품질과 클래스 분포 균형이 모델 성능에 결정적임을 재확인했고,
AI-Hub 데이터셋의 한국 CCTV 특화 장면이 RF 데이터셋과 상호 보완적으로 작용했다.

### 7.2 한계 / 향후 과제

- AI-Hub 의 0.8 % 만 사용 → 더 큰 비율·전량 학습 시 추가 개선 여지 있음.
- AI-Hub train/test 가 일부 동일 카메라 지점을 공유(촬영 시점은 다름) → 약한 도메인 누수 가능.
- RF-DETR **Nano** 기준. RF-DETR Large 등 상위 모델에서는 절대 성능이 더 높을 것.
- 클래스 통일 시 `소형버스+대형버스→bus`, `트럭+트레일러→truck` 로 묶어 세부 구분은 포기.
- person 클래스 AP 가 여전히 낮음(0.11): 소형·원거리 보행자 hard negative mining, 해상도 업 등 필요.
- 학습 곡선상 augmented 의 val mAP 는 epoch 46 이후 수렴 → 50 epoch 으로 충분히 학습됨.

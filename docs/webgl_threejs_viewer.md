# BEV scene → three.js/WebGL 3D 뷰어

작성자: 22011807 최재현

Python BEV 분석 결과(scene JSON)를 브라우저에서 3D로 재구성하는 웹 앱(`server.py` + `web/`) 설명서.
웹에서 영상/이미지를 업로드하면 Flask 서버가 추론해 scene JSON을 반환하고 three.js 뷰어가 시각화한다.
뷰어는 빌드 스텝이 없고(importmap + CDN three.js), 기존 JSON만 보는 정적 모드로도 동작한다.

---

## 1. 데이터 계약 (Python → 뷰어)

`infer.py`의 `export_scene_json()`이 만들고 `write_scene_json()`이 `output/<name>_scene.json` + `web/data/scene_data.json`에
저장한다(`PipelineConfig.web_data_dir`). 서버 모드에서는 이 JSON이 `POST /api/infer` 응답으로도 반환된다. 모든 좌표는 **월드 미터**.

```jsonc
{
  "camera":   { "height_m": 2.5, "pitch_deg": -20.8, "roll_deg": 1.3, "vfov_deg": 32.2 },
  "road_polygons":      [ { "points": [ { "x": 4.29, "z": 8.43 }, ... ] } ],
  "crosswalk_polygons": [ { "points": [ ... ] } ],
  "objects": [ { "track_id": 235, "class_name": "car", "confidence": 0.88,
                 "x_m": -3.36, "z_m": 8.47,
                 "x_m_smoothed": -3.36, "z_m_smoothed": 8.47,
                 "vx_m": 0, "vz_m": 0, "x_var": 0, "z_var": 0 } ],
  "trajectories": [ { "track_id": 235, "points": [ { "x": .., "z": .. }, ... ] } ],
  "fps": 29.97, "frame_count": 448,
  "tracks": [ { "track_id": 6, "class_name": "car" }, ... ],
  "frames": [ { "frame_index": 0, "objects": [ { "track_id": 6, "x_m": .., "z_m": .., ... } ] }, ... ]
}
```

- **이미지 씬**: `frames`/`fps`가 비어 있음 → 객체를 정적으로 배치 + 궤적 라인.
- **영상 씬**: `frames.length >= 2 && fps > 0` → 타임라인 재생(차량이 프레임을 따라 이동).
  (`web/src/sceneData.js`의 `hasTimeline()` 판별.)

### 좌표계
월드 `(x_m, z_m)` → three.js `(x, 0, z)`, Y-up. X=좌우, Z=전방 깊이, 지면 Y=0(flat-ground 가정).
객체 방향(heading)은 three.js 공간 내 모션 벡터를 `Quaternion.setFromUnitVectors(+Z, dir)`로 적용한다.

---

## 2. 모듈 구성 (`web/`)

| 파일 | 역할 |
|---|---|
| `index.html` | importmap(CDN three.js@0.169 + OrbitControls), UI 레이아웃 |
| `main.js` | 렌더러·씬·조명·그리드·카메라·OrbitControls, scene JSON 로드, `applyScene` 분기, 렌더 루프 |
| `src/sceneData.js` | JSON 헬퍼: `hasTimeline`, `objectWorld`(smoothed→raw fallback), `signedArea`, `describeScene` |
| `src/roadBuilder.js` | 도로/횡단보도 polygon → 삼각분할 메시(`THREE.ShapeUtils.triangulateShape`, min-area 필터) |
| `src/vehicleFactory.js` | car(및 미상 클래스)는 GLB 모델(`assets/coupe.glb`, 첫 사용 전 `preloadVehicleModels()`로 로드 후 clone, 바운딩박스로 실세계 크기·바닥·+Z 전방 정규화), 그 외(truck/bus=박스, person=캡슐, riders)는 절차적 메시. car는 scene JSON `color`(`white`/`black`)에 따라 페인트 재질을 단색으로 교체한 변형 모델을 캐싱해 렌더(GLB가 단색이라 텍스처 곱연산 틴트로는 흰색 표현 불가 → 재질 교체, 휠/유리는 유지); color 미지정·legacy 씬은 원본 GLB |
| `src/objectPlacer.js` | 정적 모드: 객체 배치 + 궤적 기반 방향 |
| `src/trajectoryRenderer.js` | 정적 모드: 트랙별 색상 궤적 라인 |
| `src/playback.js` | 영상 타임라인 컨트롤러: 트랙 타임라인 구성, 프레임 보간(이진탐색), 최소제곱 추세 heading + 슬루 회전, play/pause/seek/loop/speed |
| `src/ui.js` | 미디어 업로드(서버 추론)·JSON 열기/드래그드롭, 재생바, 정보 패널, 뷰 프리셋 |

---

## 3. 서버 (`server.py`, Flask)

`web/` 정적 서빙 + 업로드 추론 API. `RoadSceneProjector`를 하나만 로드해 재사용하고, 추론은 단일 워커 큐로
직렬화한다(YOLO/torch 모델은 동시 실행에 안전하지 않음). `infer`는 워커 안에서 지연 import하므로 ML 의존성이
없는 환경에서도 정적 서빙은 뜬다.

| 엔드포인트 | 설명 |
|---|---|
| `GET /` , `GET /<path>` | `web/` 정적 파일 |
| `POST /api/infer` | multipart `file`(영상/이미지) + 선택 `camera_height`,`ppm` → `{job_id, mode, status}` (202) |
| `GET /api/jobs/<id>` | `{status: queued\|running\|done\|error, elapsed_sec, scene?, error?}` |
| `GET /api/health` | `{ok, projector_loaded}` |

확장자로 모드 판별: `.mp4/.avi/.mov/.mkv/.webm/.m4v` → 영상, `.png/.jpg/.jpeg/.bmp/.webp` → 이미지. 업로드 512 MB 제한.

---

## 4. 실행

### 서버 모드 (업로드 → 추론 → 시각화)
모델·`cv2`·`torch`가 있는 `dl` env에서 실행한다.
```bash
conda activate dl
pip install -r requirements-server.txt   # flask (최초 1회)
python server.py                          # → http://localhost:5000
```
"🎬 영상/이미지 분석"으로 업로드 → 첫 요청 시 모델 로드 → 진행 폴링 후 자동 시각화. 카메라 높이·ppm은 상단 입력으로 조정.

### 시각화 전용 (추론 없이 정적)
```bash
cd web && python -m http.server 8000      # → http://localhost:8000 (file://은 fetch 불가)
```
번들 샘플 자동 로드 + `output/*_scene.json` 드래그&드롭/"JSON 열기"/`?scene=<url>`만 동작("영상/이미지 분석" 비활성).

조작: 마우스 드래그=회전, 휠=줌, 우클릭 드래그=이동. 우상단 `기본/탑다운/카메라` 버튼으로 시점 전환.

---

## 5. 배포

- **서버 모드**: `server.py`를 ML 의존성이 갖춰진 호스트(로컬/VM)에서 실행. 외부 노출 시 프로덕션 WSGI(gunicorn 등) 뒤에 둘 것.
- **정적(시각화 전용)**: `web/`를 GitHub Pages로. `.github/workflows/deploy-pages.yml`이 `web/`를 Pages 아티팩트로 업로드한다.
  저장소 **Settings → Pages → Source: GitHub Actions**(최초 1회) 후 트리거 브랜치로 push → `https://<user>.github.io/<repo>/`.
  이 경우 업로드 추론은 비활성(서버 없음).

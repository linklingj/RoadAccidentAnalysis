# BEV 도로 세그먼트 → Unity 3D 도로 재현 계획서

작성일: 2026-04-29  
작성자: 22011807 최재현

---

## 1. 현재 파이프라인 데이터 구조 파악

### 1-1. Python 쪽에서 생성되는 도로 데이터

`infer.py`의 처리 결과로 나오는 도로 관련 데이터는 아래와 같다.

| 변수명 | 타입 | 좌표계 | 설명 |
|---|---|---|---|
| `road_polygons_uv` | `List[np.ndarray]` | UV 픽셀 공간 | 도로 영역의 polygon 꼭짓점 목록 |
| `crosswalk_polygons_uv` | `List[np.ndarray]` | UV 픽셀 공간 | 횡단보도 영역의 polygon 꼭짓점 목록 |
| `projected_objects` | `List[Dict]` | 실세계 좌표(m) + BEV 픽셀 | 탐지된 객체의 세계 좌표 포함 |

`project_uv_to_ground(u, v, camera, camera_height_m)` 함수가 UV 픽셀 → 실세계 좌표 `(x_m, z_m)`으로 변환한다.

- `x_m`: 카메라 기준 좌우 방향 (미터)  
- `z_m`: 카메라 기준 전방 깊이 (미터)  
- 지면을 y=0 평면으로 가정 (flat ground assumption)

### 1-2. 좌표계 매핑

| 공간 | X | Y | Z |
|---|---|---|---|
| Python (world) | 좌우 (m) | 위/아래 (미사용, 0) | 전방 깊이 (m) |
| Unity | 좌우 | 위/아래 (높이) | 전방 |

**변환 공식:**
```
Python (x_m, z_m) → Unity Vector3(x_m, 0, z_m)
```

### 1-3. Unity 프로젝트 현황

- 씬: `MainScene.unity`, `Reconstruction.unity`
- 보유 애셋: `Low Poly Road Pack` (직선, 곡선, 교차로, S커브 등 .dae 모델 다수)
- 재질: `terrain.mat`

---

## 2. 접근법 비교

### 방법 A: 절차적 메시 생성 (Procedural Mesh)

도로 polygon의 세계 좌표를 그대로 Unity `Mesh`로 변환하는 방식.

**흐름:**
```
BEV polygon (x_m, z_m) → Unity Mesh (XZ평면 삼각분할) → MeshRenderer에 도로 재질 적용
```

**장점:**
- 실제 감지된 도로 형태를 정확히 재현
- 모든 도로 형태(불규칙 형태 포함)에 대응
- 추가 수작업 없이 자동화 가능

**단점:**
- Unity 내 2D polygon 삼각분할 알고리즘 직접 구현 필요 (Ear-Clipping)
- Concave polygon 처리가 까다로움
- 메시 노멀, UV 설정 등 추가 처리 필요

### 방법 B: 사전 제작 타일 배치 (Low Poly Road Pack 활용)

감지된 도로 형태를 분석해 직선/곡선/교차로 타일을 선택·배치하는 방식.

**흐름:**
```
도로 polygon 분석 → 주요 방향 추출 (PCA/스켈레톤화) → 타일 매칭 → 씬에 배치
```

**장점:**
- 시각적 품질이 높음 (사전 제작 에셋)
- 구현이 상대적으로 단순 (타일 배치 로직만)

**단점:**
- 실제 도로 형태와 오차 발생 (타일은 표준화된 형태만 지원)
- 도로 스켈레톤 추출 로직이 복잡 (OpenCV thinning 필요)
- 불규칙한 사거리나 비표준 도로 형태에 대응 불가

### 방법 C: 절차적 메시 + 도로 텍스처 (권장)

방법 A를 기반으로 하되, 도로 텍스처(차선 마킹, 아스팔트)를 UV 매핑으로 적용하는 하이브리드 방식.

**권장 이유:**
- 교통사고 재현이 목적이므로 **정확도가 최우선**
- BEV polygon이 이미 실세계 미터 단위로 존재하므로 변환 비용이 낮음
- Low Poly Road Pack의 텍스처만 분리해 재질로 활용하면 외관도 확보 가능

---

## 3. 전체 시스템 구조 (권장 방법 C 기준)

```
[Python infer.py]
    │  추론 결과 → export_scene_json()
    │
    ▼
[JSON 파일 / Flask API]
    │  GET /scene → scene_data.json
    │
    ▼
[Unity SceneLoader.cs]
    │  JSON 파싱
    │
    ├──▶ [RoadMeshBuilder.cs]
    │        polygon vertices → Mesh 생성 (Ear-Clipping 삼각분할)
    │        → MeshFilter / MeshRenderer → 도로 재질 적용
    │
    ├──▶ [CrosswalkMeshBuilder.cs]
    │        횡단보도 polygon → Mesh 생성 → 횡단보도 재질 적용
    │
    └──▶ [ObjectPlacer.cs]
             detected objects → 차량 모델(Car 1.dae / Car 2.dae) 배치
             trajectory → LineRenderer로 궤적 표시
```

---

## 4. JSON 데이터 포맷 설계

Python 파이프라인 결과를 Unity로 전달하기 위한 JSON 스키마.

```json
{
  "camera": {
    "height_m": 6.5,
    "pitch_deg": -12.3,
    "roll_deg": 0.2,
    "vfov_deg": 65.0
  },
  "road_polygons": [
    [
      {"x": 2.1, "z": 8.5},
      {"x": -1.4, "z": 8.3},
      ...
    ]
  ],
  "crosswalk_polygons": [
    [
      {"x": 1.0, "z": 12.0},
      ...
    ]
  ],
  "objects": [
    {
      "track_id": 1,
      "class_name": "car",
      "confidence": 0.87,
      "x_m": 1.5,
      "z_m": 10.2
    }
  ],
  "trajectories": {
    "1": [
      {"x": 1.5, "z": 10.2},
      {"x": 1.3, "z": 9.8},
      ...
    ]
  },
  "frame_index": 42,
  "timestamp_sec": 1.4
}
```

---

## 5. Python 구현 계획

### 5-1. `infer.py`에 JSON export 함수 추가

```python
import json

def export_scene_json(
    camera: Dict[str, float],
    road_polygons_uv: List[np.ndarray],
    crosswalk_polygons_uv: List[np.ndarray],
    projected_objects: List[Dict[str, Any]],
    trajectories: Optional[Dict[int, List]] = None,
    cfg: PipelineConfig = None,
    frame_index: int = 0,
) -> Dict[str, Any]:
    cfg = cfg or PipelineConfig()

    def poly_to_world(polygons_uv):
        result = []
        for poly in polygons_uv:
            pts = []
            for u, v in poly:
                proj = project_uv_to_ground(float(u), float(v), camera, cfg.camera_height_m)
                if proj:
                    pts.append({"x": round(proj[0], 3), "z": round(proj[1], 3)})
            if len(pts) >= 3:
                result.append(pts)
        return result

    scene = {
        "camera": {k: round(v, 4) for k, v in camera.items()},
        "road_polygons": poly_to_world(road_polygons_uv),
        "crosswalk_polygons": poly_to_world(crosswalk_polygons_uv),
        "objects": [
            {
                "track_id": obj.get("track_id"),
                "class_name": obj["class_name"],
                "confidence": round(obj["confidence"], 3),
                "x_m": round(obj["world_position_m"][0], 3),
                "z_m": round(obj["world_position_m"][1], 3),
            }
            for obj in projected_objects
        ],
        "trajectories": {
            str(tid): [{"x": round(p[0], 3), "z": round(p[1], 3)} for p in pts]
            for tid, pts in (trajectories or {}).items()
        },
        "frame_index": frame_index,
    }
    return scene
```

### 5-2. Flask 서버 (`server.py`) 설계

```python
from flask import Flask, jsonify
from infer import RoadSceneProjector, export_scene_json, PipelineConfig

app = Flask(__name__)
projector = RoadSceneProjector(PipelineConfig())
latest_scene = {}

@app.route("/scene", methods=["GET"])
def get_scene():
    return jsonify(latest_scene)

@app.route("/scene/update", methods=["POST"])
def update_scene():
    # 실시간 처리 후 latest_scene 갱신
    ...
```

정적 분석(이미지 1장)의 경우 `scene_data.json`을 파일로 저장해두고  
Unity에서 `Application.streamingAssetsPath`로 직접 읽는 방식도 가능 (Flask 없이 작동).

---

## 6. Unity C# 구현 계획

### 6-1. 파일 구조

```
Assets/02.Scripts/
├── SceneLoader.cs          # JSON 로드 (HTTP 또는 파일)
├── RoadMeshBuilder.cs      # 도로 polygon → Mesh
├── CrosswalkMeshBuilder.cs # 횡단보도 polygon → Mesh
├── ObjectPlacer.cs         # 차량/보행자 모델 배치
├── TrajectoryRenderer.cs   # 궤적 LineRenderer
└── Triangulator.cs         # Ear-Clipping 삼각분할 유틸
```

### 6-2. `Triangulator.cs` (핵심 알고리즘)

Unity에는 2D polygon 삼각분할이 내장되어 있지 않으므로 Ear-Clipping 알고리즘을 C#으로 구현한다.

```csharp
public static class Triangulator {
    // XZ 평면 상의 polygon을 삼각분할해 인덱스 배열 반환
    public static int[] Triangulate(Vector2[] points) {
        // Ear-Clipping: O(n^2)
        // 1. 볼록 꼭짓점(ear) 판별
        // 2. ear 제거 → 삼각형 추가
        // 3. n-2개 삼각형 생성
    }
}
```

concave polygon 처리를 위해 꼭짓점이 시계 방향(CW)인지 확인한 후  
반시계(CCW) 기준으로 정렬 후 ear-clipping을 수행한다.

### 6-3. `RoadMeshBuilder.cs` 핵심 로직

```csharp
public class RoadMeshBuilder : MonoBehaviour {
    public Material roadMaterial;

    public void BuildFromPolygons(List<List<Vector2>> polygons) {
        foreach (var poly in polygons) {
            var go = new GameObject("RoadPoly");
            var mf = go.AddComponent<MeshFilter>();
            var mr = go.AddComponent<MeshRenderer>();
            mr.material = roadMaterial;

            var verts3D = poly.Select(p => new Vector3(p.x, 0, p.y)).ToArray();
            var tris = Triangulator.Triangulate(poly.ToArray());

            var mesh = new Mesh();
            mesh.vertices = verts3D;
            mesh.triangles = tris;
            mesh.RecalculateNormals();

            // UV: XZ 좌표를 텍스처 좌표로 매핑 (타일링)
            var uvs = poly.Select(p => new Vector2(p.x * 0.1f, p.y * 0.1f)).ToArray();
            mesh.uv = uvs;
            mf.mesh = mesh;
        }
    }
}
```

### 6-4. `SceneLoader.cs` 핵심 로직

```csharp
public class SceneLoader : MonoBehaviour {
    public string sceneJsonUrl = "http://localhost:5000/scene";
    public RoadMeshBuilder roadBuilder;
    public ObjectPlacer objectPlacer;

    IEnumerator Start() {
        using var req = UnityWebRequest.Get(sceneJsonUrl);
        yield return req.SendWebRequest();

        var data = JsonUtility.FromJson<SceneData>(req.downloadHandler.text);
        roadBuilder.BuildFromPolygons(data.road_polygons);
        objectPlacer.PlaceObjects(data.objects);
    }
}
```

정적 분석 모드(Flask 없음)에서는 `sceneJsonUrl` 대신  
`Application.streamingAssetsPath + "/scene_data.json"` 경로를 사용.

---

## 7. 도로 재질(Material) 전략

### 7-1. 절차적 메시용 재질

- **베이스**: Standard Shader (URP) + 아스팔트 텍스처  
- **텍스처 타일링**: UV를 실세계 미터 기준으로 설정 (e.g., 4m × 4m 단위로 타일)  
- **차선 표시**: 별도 Decal 또는 Overlay mesh로 중앙선/차선 표시

### 7-2. Low Poly Road Pack 텍스처 활용

`Low Poly Road Pack/Textures/Road Colorscheme.png`를 도로 메시의 베이스 색상 텍스처로 사용하되,  
UV는 절차적으로 계산하므로 타일 비율을 직접 조정한다.

---

## 8. 구현 우선순위 및 단계

| 단계 | 작업 | 예상 난이도 |
|---|---|---|
| 1 | `export_scene_json()` Python 함수 구현 및 JSON 파일 저장 | 하 |
| 2 | Unity `Triangulator.cs` Ear-Clipping 구현 | 중 |
| 3 | Unity `RoadMeshBuilder.cs` 도로 메시 생성 | 중 |
| 4 | Unity `SceneLoader.cs` JSON 파일 로드 (파일 기반) | 하 |
| 5 | Unity `ObjectPlacer.cs` 차량 모델 배치 | 하 |
| 6 | Unity `TrajectoryRenderer.cs` 궤적 LineRenderer | 하 |
| 7 | Flask 서버 구현 및 Unity HTTP 연동 (실시간) | 중 |
| 8 | 도로 재질 완성 (UV 타일링, 차선 Decal) | 중 |

**1~6단계 완료 시**: 정적 이미지 분석 결과를 Unity 씬에서 재현 가능  
**7~8단계 완료 시**: 실시간 CCTV 영상 → Unity 3D 재현 가능

---

## 9. 알려진 제약 및 고려사항

### 9-1. Flat ground assumption

현재 `project_uv_to_ground()`는 지면이 완전히 평평하다고 가정한다.  
경사로, 과속방지턱, 교량 등은 정확히 투영되지 않는다.  
Unity 씬도 동일하게 y=0 평면을 기준으로 구성하면 일관성이 유지된다.

### 9-2. Polygon 단순화

도로 polygon은 최대 600개 꼭짓점(`max_polygon_points`)을 가질 수 있다.  
Unity에서 Ear-Clipping은 O(n²)이므로, 꼭짓점이 많으면 Ramer-Douglas-Peucker(RDP) 알고리즘으로  
polygon을 단순화한 후 삼각분할하는 것을 권장한다 (허용 오차: 0.05m).

### 9-3. Concave polygon 처리

도로는 항상 convex하지 않다 (ㄱ자 교차로, T자 교차로 등).  
Ear-Clipping은 단순 concave polygon을 지원하지만,  
self-intersecting polygon(도로가 영상 왜곡으로 꼬인 경우)은 전처리 필요.  
검증: `cv2.isContourConvex()` 또는 shoelace formula로 면적 부호 확인.

### 9-4. 카메라-Unity 원점 정렬

Python 좌표계에서 원점(0, 0)은 카메라 직하점.  
Unity 씬에서 카메라 위치 오브젝트를 씬 원점(0, 0, 0)에 배치하고  
`camera.height_m`만큼 y축으로 올려두면 카메라 시점 재현도 가능.

---

## 10. 참고 자료

- Unity Procedural Mesh: `https://docs.unity3d.com/Manual/GeneratingMeshGeometryProcedurally.html`
- Ear-Clipping 알고리즘 참고 구현: `https://github.com/mapbox/earcut` (C# 포트 존재)
- RDP 알고리즘: `https://en.wikipedia.org/wiki/Ramer%E2%80%93Douglas%E2%80%93Peucker_algorithm`
- Unity UnityWebRequest (JSON 수신): `https://docs.unity3d.com/ScriptReference/Networking.UnityWebRequest.html`

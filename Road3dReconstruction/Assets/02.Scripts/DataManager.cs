using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

// ── Primitive point types ──────────────────────────────────────────────────

[Serializable]
public class UVPoint
{
    public float u;
    public float v;
}

[Serializable]
public class IntPoint
{
    public int x;
    public int y;
}

// ── Static scene geometry ──────────────────────────────────────────────────

[Serializable]
public class Polygon
{
    public List<UVPoint> points;
}

[Serializable]
public class WorldPoint
{
    public float x;
    public float z;
}

[Serializable]
public class WorldPolygon
{
    public List<WorldPoint> points;
}

// ── Camera parameters estimated by PerspectiveFields ──────────────────────

[Serializable]
public class CameraParams
{
    public float roll_deg;
    public float pitch_deg;
    public float vfov_deg;
    public float rel_cx;
    public float rel_cy;
    public float rel_focal;
    public float focal_px;
    public float cx;
    public float cy;
}

// ── Per-object data for a single frame ────────────────────────────────────

[Serializable]
public class TrackedObject
{
    public int track_id;        // -1 if untracked
    public string class_name;
    public float confidence;

    // Bounding box in image pixels
    public float bbox_x1;
    public float bbox_y1;
    public float bbox_x2;
    public float bbox_y2;

    // Foot-point in image pixels
    public float foot_u;
    public float foot_v;

    // Ground-plane world coordinates in metres (X forward, Z right)
    public float world_x;
    public float world_z;

    // Position in BEV canvas pixels
    public int bev_x;
    public int bev_y;
}

// ── Trajectory for one tracked object ─────────────────────────────────────

[Serializable]
public class Trajectory
{
    public int track_id;
    public List<IntPoint> points;   // BEV canvas pixel history
}

// ── Per-frame snapshot ─────────────────────────────────────────────────────

[Serializable]
public class FrameData
{
    public int frame_index;
    public List<TrackedObject> tracked_objects;
    public List<Trajectory> trajectories;
}

// ── Root scene data ────────────────────────────────────────────────────────

[Serializable]
public class SceneData
{
    public string video_path;
    public float fps;
    public int frame_width;
    public int frame_height;
    public float camera_height_m;
    public CameraParams camera;
    public List<Polygon> road_polygons_uv;
    public List<Polygon> crosswalk_polygons_uv;
    public List<WorldPolygon> road_polygons_world;
    public List<WorldPolygon> crosswalk_polygons_world;
    public List<FrameData> frames;
}

// ── DataManager ────────────────────────────────────────────────────────────

public class DataManager : MonoBehaviour
{
    [Tooltip("Absolute path to the *_unity.json file produced by infer.py")]
    public string jsonPath = "";

    [Tooltip("camera_height_m が JSON に無い場合のフォールバック値 (m)")]
    public float cameraHeightFallback = 6.5f;

    public SceneData Data { get; private set; }

    public event System.Action OnLoaded;

    void Start()
    {
        if (!string.IsNullOrEmpty(jsonPath))
            Load(jsonPath);
    }

    public void Load(string path)
    {
        if (!File.Exists(path))
        {
            Debug.LogError($"[DataManager] JSON not found: {path}");
            return;
        }

        string raw = File.ReadAllText(path);
        Data = JsonUtility.FromJson<SceneData>(raw);

        // camera_height_m が JSON に無い場合のフォールバック
        if (Data.camera_height_m <= 0f)
            Data.camera_height_m = cameraHeightFallback;

        // world ポリゴンが無い場合は UV から変換
        if ((Data.road_polygons_world == null || Data.road_polygons_world.Count == 0) &&
            Data.road_polygons_uv != null && Data.road_polygons_uv.Count > 0)
        {
            Data.road_polygons_world = ConvertPolygons(Data.road_polygons_uv, Data.camera, Data.camera_height_m);
        }
        if ((Data.crosswalk_polygons_world == null || Data.crosswalk_polygons_world.Count == 0) &&
            Data.crosswalk_polygons_uv != null && Data.crosswalk_polygons_uv.Count > 0)
        {
            Data.crosswalk_polygons_world = ConvertPolygons(Data.crosswalk_polygons_uv, Data.camera, Data.camera_height_m);
        }

        Debug.Log($"[DataManager] Loaded {Data.frames?.Count ?? 0} frames | " +
                  $"{Data.road_polygons_world?.Count ?? 0} road polygon(s) | " +
                  $"{Data.crosswalk_polygons_world?.Count ?? 0} crosswalk polygon(s)");

        OnLoaded?.Invoke();
    }


// ── UV → World 変換ヘルパー ──────────────────────────────────────────────

    static List<WorldPolygon> ConvertPolygons(List<Polygon> polysUV, CameraParams cam, float heightM)
    {
        var result = new List<WorldPolygon>();
        foreach (var poly in polysUV)
        {
            var wPts = new List<WorldPoint>();
            foreach (var uv in poly.points)
            {
                var wp = UVToWorld(uv.u, uv.v, cam, heightM);
                if (wp != null) wPts.Add(wp);
            }
            if (wPts.Count >= 3)
                result.Add(new WorldPolygon { points = wPts });
        }
        return result;
    }

    static WorldPoint UVToWorld(float u, float v, CameraParams cam, float heightM)
    {
        if (cam.focal_px <= 1e-6f) return null;
        float xNorm = (u - cam.cx) / cam.focal_px;
        float yNorm = (v - cam.cy) / cam.focal_px;

        float rollRad  = cam.roll_deg  * Mathf.Deg2Rad;
        float pitchRad = cam.pitch_deg * Mathf.Deg2Rad;
        float cr = Mathf.Cos(rollRad),  sr = Mathf.Sin(rollRad);
        float cp = Mathf.Cos(pitchRad), sp = Mathf.Sin(pitchRad);

        float dirX = xNorm * cr - yNorm * sr;
        float dirY = xNorm * cp * sr + yNorm * cp * cr - sp;
        float dirZ = xNorm * sp * sr + yNorm * sp * cr + cp;

        if (dirY <= 1e-6f) return null;
        float t = heightM / dirY;
        if (t <= 0f) return null;
        float xM = t * dirX;
        float zM = t * dirZ;
        if (zM <= 0f) return null;
        return new WorldPoint { x = xM, z = zM };
    }
}

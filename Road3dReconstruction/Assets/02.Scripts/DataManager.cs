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
    public CameraParams camera;
    public List<Polygon> road_polygons_uv;
    public List<Polygon> crosswalk_polygons_uv;
    public List<FrameData> frames;
}

// ── DataManager ────────────────────────────────────────────────────────────

public class DataManager : MonoBehaviour
{
    [Tooltip("Absolute path to the *_unity.json file produced by infer.py")]
    public string jsonPath = "";

    public SceneData Data { get; private set; }

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
        Debug.Log($"[DataManager] Loaded {Data.frames.Count} frames | " +
                  $"{Data.road_polygons_uv.Count} road polygon(s) | " +
                  $"{Data.crosswalk_polygons_uv.Count} crosswalk polygon(s)");
    }
}

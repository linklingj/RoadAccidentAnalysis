using System;
using System.Collections.Generic;
using UnityEngine;

namespace RoadReconstruction
{
    [Serializable]
    public class SceneCamera
    {
        public float height_m;
        public float pitch_deg;
        public float roll_deg;
        public float vfov_deg;
    }

    [Serializable]
    public class ScenePoint
    {
        public float x;
        public float z;

        public Vector2 ToVector2() { return new Vector2(x, z); }
        public Vector3 ToVector3(float y = 0f) { return new Vector3(x, y, z); }
    }

    [Serializable]
    public class ScenePolygon
    {
        public List<ScenePoint> points = new List<ScenePoint>();
    }

    [Serializable]
    public class SceneObject
    {
        public int track_id;
        public string class_name;
        public float confidence;
        public float x_m;
        public float z_m;

        // Python-side offline RTS smoother output. Missing in legacy JSONs (defaults to 0)
        // — ToWorldSmoothed falls back to raw (x_m, z_m) in that case.
        public float x_m_smoothed;
        public float z_m_smoothed;
        public float vx_m;
        public float vz_m;
        public float x_var;
        public float z_var;

        public Vector3 ToWorld(float y = 0f) { return new Vector3(x_m, y, z_m); }

        public Vector3 ToWorldSmoothed(float y = 0f)
        {
            if (x_m_smoothed == 0f && z_m_smoothed == 0f && (x_m != 0f || z_m != 0f))
            {
                return new Vector3(x_m, y, z_m);
            }
            return new Vector3(x_m_smoothed, y, z_m_smoothed);
        }
    }

    [Serializable]
    public class SceneTrajectory
    {
        public int track_id;
        public List<ScenePoint> points = new List<ScenePoint>();
    }

    [Serializable]
    public class SceneTrack
    {
        public int track_id;
        public string class_name;
    }

    [Serializable]
    public class SceneFrame
    {
        public int frame_index;
        public List<SceneObject> objects = new List<SceneObject>();
    }

    [Serializable]
    public class SceneData
    {
        public SceneCamera camera = new SceneCamera();
        public List<ScenePolygon> road_polygons = new List<ScenePolygon>();
        public List<ScenePolygon> crosswalk_polygons = new List<ScenePolygon>();
        public List<SceneObject> objects = new List<SceneObject>();
        public List<SceneTrajectory> trajectories = new List<SceneTrajectory>();
        public int frame_index;

        // Video timeline (empty for single-image scenes).
        public float fps;
        public int frame_count;
        public List<SceneTrack> tracks = new List<SceneTrack>();
        public List<SceneFrame> frames = new List<SceneFrame>();

        public bool HasTimeline => frames != null && frames.Count >= 2 && fps > 0f;
    }
}

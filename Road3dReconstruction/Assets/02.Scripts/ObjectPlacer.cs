using System;
using System.Collections.Generic;
using UnityEngine;

namespace RoadReconstruction
{
    public class ObjectPlacer : MonoBehaviour
    {
        [Serializable]
        public class ClassPrefab
        {
            public string className = "car";
            public GameObject prefab;
            [Tooltip("Uniform scale applied on top of the prefab.")]
            public float scale = 1f;
            [Tooltip("Local position offset relative to the placement point.")]
            public Vector3 positionOffset = Vector3.zero;
            [Tooltip("Local Euler rotation offset (degrees).")]
            public Vector3 rotationOffset = Vector3.zero;
        }

        [Header("Class -> Prefab")]
        public List<ClassPrefab> classPrefabs = new List<ClassPrefab>();

        [Tooltip("Fallback prefab when no class match is found. Leave empty to skip unknown classes.")]
        public GameObject fallbackPrefab;

        [Header("Layout")]
        [Tooltip("Y offset (m) so the vehicle is placed slightly above the road plane.")]
        public float yOffset = 0.0f;

        [Tooltip("If true, look up trajectory direction from previous frames to face the vehicle along motion.")]
        public bool orientAlongTrajectory = true;

        private readonly List<GameObject> _spawned = new List<GameObject>();

        public void Clear()
        {
            for (int i = 0; i < _spawned.Count; i++)
            {
                if (_spawned[i] != null)
                {
                    if (Application.isPlaying) Destroy(_spawned[i]);
                    else DestroyImmediate(_spawned[i]);
                }
            }
            _spawned.Clear();
        }

        public void PlaceObjects(List<SceneObject> objects, List<SceneTrajectory> trajectories = null)
        {
            Clear();
            if (objects == null) return;

            Dictionary<int, SceneTrajectory> trajLookup = null;
            if (orientAlongTrajectory && trajectories != null)
            {
                trajLookup = new Dictionary<int, SceneTrajectory>(trajectories.Count);
                for (int i = 0; i < trajectories.Count; i++)
                {
                    var t = trajectories[i];
                    if (t == null) continue;
                    trajLookup[t.track_id] = t;
                }
            }

            for (int i = 0; i < objects.Count; i++)
            {
                var obj = objects[i];
                if (obj == null) continue;

                var entry = ResolveEntry(obj.class_name);
                GameObject prefab = entry != null ? entry.prefab : fallbackPrefab;
                if (prefab == null) continue;

                Vector3 worldPos = obj.ToWorld(yOffset);
                Quaternion baseRot = Quaternion.identity;
                if (orientAlongTrajectory && trajLookup != null && trajLookup.TryGetValue(obj.track_id, out var traj) && traj.points != null && traj.points.Count >= 2)
                {
                    Vector3 motion = ComputeForward(traj.points);
                    if (motion.sqrMagnitude > 1e-4f)
                    {
                        baseRot = Quaternion.LookRotation(motion, Vector3.up);
                    }
                }
                if (entry != null)
                {
                    baseRot = baseRot * Quaternion.Euler(entry.rotationOffset);
                    worldPos += baseRot * entry.positionOffset;
                }

                var go = Instantiate(prefab, worldPos, baseRot, transform);
                if (entry != null && entry.scale > 0f && Mathf.Abs(entry.scale - 1f) > 1e-4f)
                {
                    go.transform.localScale = go.transform.localScale * entry.scale;
                }
                string trackTag = obj.track_id >= 0 ? ("T" + obj.track_id) : "det";
                go.name = $"{obj.class_name}_{trackTag}";
                _spawned.Add(go);
            }
        }

        private ClassPrefab ResolveEntry(string className)
        {
            if (string.IsNullOrEmpty(className)) return null;
            for (int i = 0; i < classPrefabs.Count; i++)
            {
                var entry = classPrefabs[i];
                if (entry == null || entry.prefab == null) continue;
                if (string.Equals(entry.className, className, StringComparison.OrdinalIgnoreCase)) return entry;
            }
            return null;
        }

        private static Vector3 ComputeForward(List<ScenePoint> pts)
        {
            // Use the last segment of the trajectory; smoothing across the last few points
            // would average out parking jitter but we keep it simple here.
            int last = pts.Count - 1;
            int prev = Mathf.Max(0, last - 3);
            var p1 = pts[prev].ToVector3();
            var p2 = pts[last].ToVector3();
            return (p2 - p1).normalized;
        }
    }
}

using System.Collections.Generic;
using UnityEngine;

namespace RoadReconstruction
{
    public class TrajectoryRenderer : MonoBehaviour
    {
        [Header("Line Style")]
        public Material lineMaterial;
        public float lineWidth = 0.05f;

        [Tooltip("Y offset (m) so the trajectory renders above road and crosswalk meshes.")]
        public float yOffset = 0.05f;

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

        public void Render(List<SceneTrajectory> trajectories)
        {
            Clear();
            if (trajectories == null) return;

            for (int i = 0; i < trajectories.Count; i++)
            {
                var traj = trajectories[i];
                if (traj == null || traj.points == null || traj.points.Count < 2) continue;

                var go = new GameObject("Trajectory_T" + traj.track_id);
                go.transform.SetParent(transform, false);
                go.transform.localPosition = Vector3.zero;

                var lr = go.AddComponent<LineRenderer>();
                lr.useWorldSpace = false;
                lr.alignment = LineAlignment.View;
                lr.startWidth = lineWidth;
                lr.endWidth = lineWidth;
                lr.numCornerVertices = 2;
                lr.numCapVertices = 2;
                lr.positionCount = traj.points.Count;

                Color color = ColorForTrack(traj.track_id);
                if (lineMaterial != null)
                {
                    lr.sharedMaterial = lineMaterial;
                }
                else
                {
                    // Fallback to a basic unlit material so the line is visible without setup.
                    var fallbackMat = new Material(Shader.Find("Sprites/Default"));
                    lr.sharedMaterial = fallbackMat;
                }
                lr.startColor = color;
                lr.endColor = color;

                for (int p = 0; p < traj.points.Count; p++)
                {
                    var sp = traj.points[p];
                    lr.SetPosition(p, new Vector3(sp.x, yOffset, sp.z));
                }
                _spawned.Add(go);
            }
        }

        public static Color ColorForTrack(int trackId)
        {
            // Match the deterministic palette used in the BEV renderer (HSV-based but consistent).
            float hue = ((trackId * 47) % 360) / 360f;
            return Color.HSVToRGB(hue, 0.7f, 1f);
        }
    }
}

using System.Collections.Generic;
using UnityEngine;

namespace RoadReconstruction
{
    public class RoadMeshBuilder : MonoBehaviour
    {
        [Header("Material")]
        public Material roadMaterial;

        [Header("Layout")]
        [Tooltip("Y offset (m) so the road sits slightly above the ground plane to prevent z-fighting.")]
        public float yOffset = 0.0f;

        [Tooltip("Texture tiling factor (UV units per meter).")]
        public float uvScalePerMeter = 0.1f;

        [Tooltip("Minimum polygon area (m^2) to render. Filters out noisy slivers.")]
        public float minAreaSqM = 0.5f;

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

        public void BuildFromPolygons(List<ScenePolygon> polygons)
        {
            Clear();
            if (polygons == null) return;

            for (int i = 0; i < polygons.Count; i++)
            {
                var poly = polygons[i];
                if (poly == null || poly.points == null || poly.points.Count < 3) continue;

                var verts2D = new Vector2[poly.points.Count];
                for (int v = 0; v < poly.points.Count; v++)
                {
                    verts2D[v] = poly.points[v].ToVector2();
                }

                if (Mathf.Abs(SignedArea(verts2D)) < minAreaSqM) continue;

                var mesh = BuildMesh(verts2D);
                if (mesh == null || mesh.triangles.Length == 0) continue;

                var go = new GameObject("RoadPoly_" + i);
                go.transform.SetParent(transform, false);
                go.transform.localPosition = new Vector3(0f, yOffset, 0f);
                var mf = go.AddComponent<MeshFilter>();
                var mr = go.AddComponent<MeshRenderer>();
                mf.sharedMesh = mesh;
                if (roadMaterial != null) mr.sharedMaterial = roadMaterial;
                _spawned.Add(go);
            }
        }

        private Mesh BuildMesh(Vector2[] poly2D)
        {
            int[] tris = Triangulator.TriangulateXZ(poly2D, reverseWinding: true);
            if (tris.Length == 0) return null;

            var verts3D = new Vector3[poly2D.Length];
            var uvs = new Vector2[poly2D.Length];
            for (int i = 0; i < poly2D.Length; i++)
            {
                verts3D[i] = new Vector3(poly2D[i].x, 0f, poly2D[i].y);
                uvs[i] = new Vector2(poly2D[i].x * uvScalePerMeter, poly2D[i].y * uvScalePerMeter);
            }

            var mesh = new Mesh();
            mesh.name = "RoadMesh";
            if (verts3D.Length > 65535) mesh.indexFormat = UnityEngine.Rendering.IndexFormat.UInt32;
            mesh.vertices = verts3D;
            mesh.triangles = tris;
            mesh.uv = uvs;
            mesh.RecalculateNormals();
            mesh.RecalculateBounds();
            return mesh;
        }

        private static float SignedArea(Vector2[] pts)
        {
            float a = 0f;
            int n = pts.Length;
            for (int i = 0; i < n; i++)
            {
                int j = (i + 1) % n;
                a += pts[i].x * pts[j].y - pts[j].x * pts[i].y;
            }
            return a * 0.5f;
        }
    }
}

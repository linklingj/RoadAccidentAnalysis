using System.Collections.Generic;
using UnityEngine;

namespace RoadReconstruction
{
    public class CrosswalkMeshBuilder : MonoBehaviour
    {
        [Header("Material")]
        public Material crosswalkMaterial;

        [Header("Layout")]
        [Tooltip("Y offset (m) so the crosswalk renders above the road.")]
        public float yOffset = 0.02f;

        [Tooltip("Texture tiling factor (UV units per meter).")]
        public float uvScalePerMeter = 0.4f;

        [Tooltip("Minimum polygon area (m^2) to render.")]
        public float minAreaSqM = 0.2f;

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

                int[] tris = Triangulator.TriangulateXZ(verts2D, reverseWinding: true);
                if (tris.Length == 0) continue;

                var verts3D = new Vector3[verts2D.Length];
                var uvs = new Vector2[verts2D.Length];
                for (int v = 0; v < verts2D.Length; v++)
                {
                    verts3D[v] = new Vector3(verts2D[v].x, 0f, verts2D[v].y);
                    uvs[v] = new Vector2(verts2D[v].x * uvScalePerMeter, verts2D[v].y * uvScalePerMeter);
                }

                var mesh = new Mesh();
                mesh.name = "CrosswalkMesh";
                if (verts3D.Length > 65535) mesh.indexFormat = UnityEngine.Rendering.IndexFormat.UInt32;
                mesh.vertices = verts3D;
                mesh.triangles = tris;
                mesh.uv = uvs;
                mesh.RecalculateNormals();
                mesh.RecalculateBounds();

                var go = new GameObject("CrosswalkPoly_" + i);
                go.transform.SetParent(transform, false);
                go.transform.localPosition = new Vector3(0f, yOffset, 0f);
                var mf = go.AddComponent<MeshFilter>();
                var mr = go.AddComponent<MeshRenderer>();
                mf.sharedMesh = mesh;
                if (crosswalkMaterial != null) mr.sharedMaterial = crosswalkMaterial;
                _spawned.Add(go);
            }
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

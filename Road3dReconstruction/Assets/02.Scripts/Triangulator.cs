using System.Collections.Generic;
using UnityEngine;

namespace RoadReconstruction
{
    public static class Triangulator
    {
        public static int[] Triangulate(Vector2[] points)
        {
            if (points == null || points.Length < 3) return new int[0];

            int n = points.Length;
            var indices = new List<int>(n);
            for (int i = 0; i < n; i++) indices.Add(i);

            float area = SignedArea(points);
            if (Mathf.Abs(area) < 1e-6f) return new int[0];
            if (area < 0f) indices.Reverse();

            var triangles = new List<int>((n - 2) * 3);
            int safety = n * n + 8;

            while (indices.Count >= 3 && safety-- > 0)
            {
                bool earFound = false;
                int cnt = indices.Count;

                for (int i = 0; i < cnt; i++)
                {
                    int prev = indices[(i - 1 + cnt) % cnt];
                    int curr = indices[i];
                    int next = indices[(i + 1) % cnt];

                    if (IsEar(points, indices, prev, curr, next))
                    {
                        triangles.Add(prev);
                        triangles.Add(curr);
                        triangles.Add(next);
                        indices.RemoveAt(i);
                        earFound = true;
                        break;
                    }
                }

                if (!earFound)
                {
                    if (indices.Count == 3)
                    {
                        triangles.Add(indices[0]);
                        triangles.Add(indices[1]);
                        triangles.Add(indices[2]);
                    }
                    break;
                }
            }

            return triangles.ToArray();
        }

        public static int[] TriangulateXZ(Vector2[] pointsXZ, bool reverseWinding = true)
        {
            int[] tris = Triangulate(pointsXZ);
            if (reverseWinding)
            {
                for (int i = 0; i + 2 < tris.Length; i += 3)
                {
                    int tmp = tris[i + 1];
                    tris[i + 1] = tris[i + 2];
                    tris[i + 2] = tmp;
                }
            }
            return tris;
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

        private static bool IsEar(Vector2[] points, List<int> indices, int prev, int curr, int next)
        {
            Vector2 a = points[prev];
            Vector2 b = points[curr];
            Vector2 c = points[next];

            if (Cross(b - a, c - b) <= 0f) return false;

            for (int k = 0; k < indices.Count; k++)
            {
                int idx = indices[k];
                if (idx == prev || idx == curr || idx == next) continue;
                if (PointInTriangle(points[idx], a, b, c)) return false;
            }
            return true;
        }

        private static float Cross(Vector2 a, Vector2 b)
        {
            return a.x * b.y - a.y * b.x;
        }

        private static bool PointInTriangle(Vector2 p, Vector2 a, Vector2 b, Vector2 c)
        {
            float d1 = Sign(p, a, b);
            float d2 = Sign(p, b, c);
            float d3 = Sign(p, c, a);
            bool hasNeg = (d1 < 0f) || (d2 < 0f) || (d3 < 0f);
            bool hasPos = (d1 > 0f) || (d2 > 0f) || (d3 > 0f);
            return !(hasNeg && hasPos);
        }

        private static float Sign(Vector2 p1, Vector2 p2, Vector2 p3)
        {
            return (p1.x - p3.x) * (p2.y - p3.y) - (p2.x - p3.x) * (p1.y - p3.y);
        }
    }
}

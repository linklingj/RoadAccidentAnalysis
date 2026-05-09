using System.Collections.Generic;
using UnityEngine;

public class RoadBuilder : MonoBehaviour
{
    public DataManager dataManager;

    [Header("Materials")]
    public Material roadMaterial;
    public Material crosswalkMaterial;

    [Header("Parents")]
    public Transform roadParent;
    public Transform crosswalkParent;

    void Start()
    {
        if (dataManager == null)
            dataManager = FindObjectOfType<DataManager>();

        if (dataManager != null)
        {
            if (dataManager.Data != null)
                BuildScene(dataManager.Data);
            else
                dataManager.OnLoaded += () => BuildScene(dataManager.Data);
        }
    }

    [ContextMenu("Build")]
    public void Build()
    {
        if (dataManager == null)
            dataManager = FindObjectOfType<DataManager>();

        if (dataManager?.Data == null)
        {
            Debug.LogError("[RoadBuilder] DataManager has no loaded data.");
            return;
        }

        Debug.Log("[RoadBuilder] Building roads and crosswalks...");
        BuildScene(dataManager.Data);
    }

    void BuildScene(SceneData data)
    {
        BuildPolygons(data.road_polygons_world, roadParent, roadMaterial, "Road");
        BuildPolygons(data.crosswalk_polygons_world, crosswalkParent, crosswalkMaterial, "Crosswalk");
    }

    void BuildPolygons(List<WorldPolygon> polys, Transform parent, Material mat, string prefix)
    {
        if (polys == null) return;

        for (int i = 0; i < polys.Count; i++)
        {
            var mesh = BuildMesh(polys[i]);
            if (mesh == null) continue;

            var go = new GameObject($"{prefix}_{i}");
            go.transform.SetParent(parent ?? transform, false);

            var mf = go.AddComponent<MeshFilter>();
            mf.sharedMesh = mesh;

            var mr = go.AddComponent<MeshRenderer>();
            mr.sharedMaterial = mat;
        }
    }

    // Fan triangulation from centroid — works well for roughly-convex polygons.
    static Mesh BuildMesh(WorldPolygon poly)
    {
        var pts = poly.points;
        if (pts == null || pts.Count < 3) return null;

        // Ensure CCW winding in XZ plane so normals point +Y (visible from above).
        // YOLO polygons are CCW in image space (y-down), which maps to CW in world space (y-up).
        float signedArea = 0f;
        int pn = pts.Count;
        for (int i = 0; i < pn; i++)
        {
            int j = (i + 1) % pn;
            signedArea += pts[i].x * pts[j].z - pts[j].x * pts[i].z;
        }
        if (signedArea < 0f)
        {
            var reversed = new List<WorldPoint>(pts);
            reversed.Reverse();
            pts = reversed;
        }

        // Compute centroid
        float cx = 0f, cz = 0f;
        foreach (var p in pts) { cx += p.x; cz += p.z; }
        cx /= pts.Count;
        cz /= pts.Count;

        int n = pts.Count;
        var verts = new Vector3[n + 1];
        verts[0] = new Vector3(cx, 0f, cz); // centroid
        for (int i = 0; i < n; i++)
            verts[i + 1] = new Vector3(pts[i].x, 0f, pts[i].z);

        var tris = new int[n * 3];
        for (int i = 0; i < n; i++)
        {
            tris[i * 3 + 0] = 0;
            tris[i * 3 + 1] = i + 1;
            tris[i * 3 + 2] = (i + 1) % n + 1;
        }

        var mesh = new Mesh { name = "PolyMesh" };
        mesh.vertices = verts;
        mesh.triangles = tris;
        mesh.RecalculateNormals();
        mesh.RecalculateBounds();
        return mesh;
    }
}

using System;
using System.Collections.Generic;
using UnityEngine;

public class CarSpawner : MonoBehaviour
{
    public GameObject carPrefab;
    public Transform carParent;

    public DataManager dataManager;

    [Tooltip("Class names treated as cars (case-insensitive)")]
    public string[] carClassNames = { "car", "vehicle", "bus", "truck", "motorcycle" };

    [ContextMenu("Initiate")]
    public void Generate()
    {
        if (dataManager == null)
            dataManager = FindObjectOfType<DataManager>();

        if (dataManager == null)
        {
            Debug.LogError("[CarSpawner] DataManager not found.");
            return;
        }

        if (dataManager.Data == null)
        {
            Debug.LogError("[CarSpawner] DataManager has no loaded data. Check jsonPath.");
            return;
        }

        SpawnFirstFrameCars();
    }

    void SpawnFirstFrameCars()
    {
        var frames = dataManager.Data.frames;
        if (frames == null || frames.Count == 0)
        {
            Debug.LogWarning("[CarSpawner] No frame data.");
            return;
        }

        var firstFrame = frames[0];
        var carSet = new HashSet<string>(carClassNames, StringComparer.OrdinalIgnoreCase);

        int spawned = 0;
        foreach (var obj in firstFrame.tracked_objects)
        {
            if (!carSet.Contains(obj.class_name))
                continue;

            // world_x/world_z are metres on the ground plane
            Vector3 pos = new Vector3(obj.world_x, 0f, obj.world_z);
            GameObject go = Instantiate(carPrefab, pos, Quaternion.identity);
            go.name = $"Car_T{obj.track_id}_{obj.class_name}";
            go.transform.SetParent(carParent, true);
            spawned++;
        }

        Debug.Log($"[CarSpawner] Spawned {spawned} car(s) from frame 0.");
    }
}

using System.Collections;
using System.IO;
using UnityEngine;
using UnityEngine.Networking;

namespace RoadReconstruction
{
    public class SceneLoader : MonoBehaviour
    {
        public enum SourceMode
        {
            StreamingAssets,
            AbsolutePath,
            HttpUrl,
        }

        [Header("Source")]
        public SourceMode mode = SourceMode.StreamingAssets;

        [Tooltip("File name when reading from StreamingAssets/.")]
        public string streamingAssetsFileName = "scene_data.json";

        [Tooltip("Absolute path on disk (used when mode = AbsolutePath).")]
        public string absoluteFilePath;

        [Tooltip("HTTP URL serving JSON (used when mode = HttpUrl).")]
        public string httpUrl = "http://localhost:5000/scene";

        [Header("Behavior")]
        public bool loadOnStart = true;

        [Tooltip("Scene origin offset applied to all spawned content. Useful for placing the camera at world (0, height, 0).")]
        public Vector3 sceneOriginOffset = Vector3.zero;

        [Header("Targets")]
        public RoadMeshBuilder roadBuilder;
        public CrosswalkMeshBuilder crosswalkBuilder;
        public ObjectPlacer objectPlacer;
        public TrajectoryRenderer trajectoryRenderer;
        [Tooltip("Drives per-frame vehicle animation when the JSON includes a timeline.")]
        public VideoPlaybackController playbackController;

        [Tooltip("Optional camera transform; if assigned, will be placed at scene origin + camera height to mirror the analysis viewpoint.")]
        public Transform sceneCameraAnchor;

        [Header("Debug")]
        public bool verbose = true;

        public SceneData LastLoaded { get; private set; }

        private void Start()
        {
            if (loadOnStart) Load();
        }

        public void Load()
        {
            StartCoroutine(LoadCoroutine());
        }

        // External callers (e.g. AppFlowController after a successful upload) feed
        // the JSON they already have, bypassing the SourceMode switch entirely.
        public void LoadFromJson(string json)
        {
            if (string.IsNullOrEmpty(json))
            {
                LogError("LoadFromJson called with empty payload.");
                return;
            }
            SceneData data;
            try
            {
                data = JsonUtility.FromJson<SceneData>(json);
            }
            catch (System.Exception ex)
            {
                LogError("Failed to parse scene JSON: " + ex.Message);
                return;
            }
            if (data == null)
            {
                LogError("Parsed SceneData is null.");
                return;
            }
            ApplyScene(data);
            LastLoaded = data;
            if (verbose)
            {
                Debug.Log($"[SceneLoader] (in-memory) roads={data.road_polygons.Count}, crosswalks={data.crosswalk_polygons.Count}, objects={data.objects.Count}, trajectories={data.trajectories.Count}");
            }
        }

        public IEnumerator LoadCoroutine()
        {
            string json = null;

            switch (mode)
            {
                case SourceMode.StreamingAssets:
                {
                    string path = Path.Combine(Application.streamingAssetsPath, streamingAssetsFileName);
                    yield return ReadFromUriOrFile(path, result => json = result);
                    break;
                }
                case SourceMode.AbsolutePath:
                {
                    if (string.IsNullOrEmpty(absoluteFilePath))
                    {
                        LogError("AbsolutePath mode requires absoluteFilePath to be set.");
                        yield break;
                    }
                    yield return ReadFromUriOrFile(absoluteFilePath, result => json = result);
                    break;
                }
                case SourceMode.HttpUrl:
                {
                    using (var req = UnityWebRequest.Get(httpUrl))
                    {
                        yield return req.SendWebRequest();
                        if (req.result != UnityWebRequest.Result.Success)
                        {
                            LogError($"HTTP fetch failed: {req.error} ({httpUrl})");
                            yield break;
                        }
                        json = req.downloadHandler.text;
                    }
                    break;
                }
            }

            if (string.IsNullOrEmpty(json))
            {
                LogError("Empty scene JSON.");
                yield break;
            }

            SceneData data;
            try
            {
                data = JsonUtility.FromJson<SceneData>(json);
            }
            catch (System.Exception ex)
            {
                LogError("Failed to parse scene JSON: " + ex.Message);
                yield break;
            }

            if (data == null)
            {
                LogError("Parsed SceneData is null.");
                yield break;
            }

            ApplyScene(data);
            LastLoaded = data;
            if (verbose)
            {
                Debug.Log($"[SceneLoader] roads={data.road_polygons.Count}, crosswalks={data.crosswalk_polygons.Count}, objects={data.objects.Count}, trajectories={data.trajectories.Count}");
            }
        }

        private IEnumerator ReadFromUriOrFile(string path, System.Action<string> onText)
        {
            // StreamingAssets on Android lives inside a JAR, so use UnityWebRequest there.
            // For other platforms, File.ReadAllText is faster.
#if UNITY_ANDROID && !UNITY_EDITOR
            string uri = path.Contains("://") ? path : "file://" + path;
            using (var req = UnityWebRequest.Get(uri))
            {
                yield return req.SendWebRequest();
                if (req.result != UnityWebRequest.Result.Success)
                {
                    LogError($"Failed to read {path}: {req.error}");
                    yield break;
                }
                onText?.Invoke(req.downloadHandler.text);
            }
#else
            if (!File.Exists(path))
            {
                LogError($"Scene JSON not found at: {path}");
                yield break;
            }
            string text = File.ReadAllText(path);
            onText?.Invoke(text);
            yield return null;
#endif
        }

        public void ApplyScene(SceneData data)
        {
            // Translate this transform so child builders place geometry around the desired origin.
            transform.position = sceneOriginOffset;

            if (roadBuilder != null) roadBuilder.BuildFromPolygons(data.road_polygons);
            if (crosswalkBuilder != null) crosswalkBuilder.BuildFromPolygons(data.crosswalk_polygons);

            bool hasTimeline = data.HasTimeline;
            if (hasTimeline && playbackController != null)
            {
                // Timeline drives vehicles directly; skip the static placer and trajectory lines.
                if (objectPlacer != null) objectPlacer.Clear();
                if (trajectoryRenderer != null) trajectoryRenderer.Clear();
                playbackController.Initialize(data);
            }
            else
            {
                if (objectPlacer != null) objectPlacer.PlaceObjects(data.objects, data.trajectories);
                if (trajectoryRenderer != null) trajectoryRenderer.Render(data.trajectories);
                if (playbackController != null) playbackController.ClearVehicles();
            }

            if (sceneCameraAnchor != null && data.camera != null && data.camera.height_m > 0f)
            {
                var pos = sceneOriginOffset;
                pos.y += data.camera.height_m;
                sceneCameraAnchor.position = pos;
                sceneCameraAnchor.rotation = Quaternion.Euler(-data.camera.pitch_deg, 0f, data.camera.roll_deg);
            }
        }

        private void LogError(string msg)
        {
            if (verbose) Debug.LogError("[SceneLoader] " + msg);
        }
    }
}

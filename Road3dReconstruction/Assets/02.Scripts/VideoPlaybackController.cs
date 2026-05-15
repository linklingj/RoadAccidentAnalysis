using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

namespace RoadReconstruction
{
    public class VideoPlaybackController : MonoBehaviour
    {
        private class TrackTimeline
        {
            public int trackId;
            public string className;
            public List<int> frames;
            public List<Vector3> positions;
            public int startFrame;
            public int endFrame;
            public GameObject instance;
            public ObjectPlacer.ClassPrefab prefabEntry;
            public Quaternion lastHeading = Quaternion.identity;
            public bool hasHeading;
        }

        [Header("References")]
        [Tooltip("ObjectPlacer used purely to resolve class -> prefab mappings.")]
        public ObjectPlacer prefabSource;

        [Tooltip("Parent transform for spawned vehicles. Defaults to this transform.")]
        public Transform vehicleParent;

        [Header("Playback")]
        [Tooltip("Auto-play once a timeline is loaded.")]
        public bool autoPlay = true;
        [Range(0.1f, 4f)]
        public float playbackSpeed = 1f;
        [Tooltip("Loop playback when end is reached.")]
        public bool loop = true;
        [Tooltip("Y offset (m) applied to all spawned vehicles.")]
        public float yOffset = 0f;

        [Header("Heading")]
        [Tooltip("Sampling distance (in frames) used to estimate the motion direction. Larger = smoother but laggier.")]
        public int headingWindowFrames = 6;
        [Tooltip("How fast the vehicle rotates toward its motion vector (deg/sec).")]
        public float headingSlewDegPerSec = 540f;
        [Tooltip("Minimum displacement (m) within the heading window to update facing.")]
        public float headingMinDisplacement = 0.05f;

        [Header("UI (optional)")]
        public Slider timeSlider;
        public Button playPauseButton;
        public Text playPauseLabel;
        public Text timeLabel;

        public bool IsPlaying { get; private set; }
        public bool HasTimeline { get; private set; }
        public float CurrentTime { get; private set; }
        public float Duration { get; private set; }

        private SceneData _data;
        private float _fps;
        private int _frameCount;
        private readonly List<TrackTimeline> _tracks = new List<TrackTimeline>();
        private bool _sliderDragging;
        private bool _suppressSliderCallback;

        public void Initialize(SceneData data)
        {
            ClearVehicles();
            _data = data;

            if (data == null || !data.HasTimeline)
            {
                HasTimeline = false;
                IsPlaying = false;
                Duration = 0f;
                CurrentTime = 0f;
                UpdateUiInteractivity();
                UpdateUi();
                return;
            }

            HasTimeline = true;
            _fps = data.fps > 0f ? data.fps : 30f;
            _frameCount = data.frames.Count;
            Duration = Mathf.Max(0f, (_frameCount - 1) / _fps);

            BuildTrackTimelines(data);
            SpawnVehicles();

            CurrentTime = 0f;
            ApplyTime(CurrentTime, force: true);

            HookUi();
            UpdateUiInteractivity();
            UpdateUi();

            IsPlaying = autoPlay;
        }

        private void BuildTrackTimelines(SceneData data)
        {
            var byId = new Dictionary<int, TrackTimeline>();

            // Seed from the registry so the class_name is reliable even if a track is missing
            // from the very first frame it appears in.
            if (data.tracks != null)
            {
                for (int i = 0; i < data.tracks.Count; i++)
                {
                    var t = data.tracks[i];
                    if (t == null) continue;
                    if (byId.ContainsKey(t.track_id)) continue;
                    byId[t.track_id] = new TrackTimeline
                    {
                        trackId = t.track_id,
                        className = t.class_name,
                        frames = new List<int>(),
                        positions = new List<Vector3>(),
                        startFrame = int.MaxValue,
                        endFrame = int.MinValue,
                    };
                }
            }

            for (int f = 0; f < data.frames.Count; f++)
            {
                var frame = data.frames[f];
                if (frame == null || frame.objects == null) continue;
                int fi = frame.frame_index;
                for (int o = 0; o < frame.objects.Count; o++)
                {
                    var obj = frame.objects[o];
                    if (obj == null || obj.track_id < 0) continue;

                    if (!byId.TryGetValue(obj.track_id, out var timeline))
                    {
                        timeline = new TrackTimeline
                        {
                            trackId = obj.track_id,
                            className = obj.class_name,
                            frames = new List<int>(),
                            positions = new List<Vector3>(),
                            startFrame = int.MaxValue,
                            endFrame = int.MinValue,
                        };
                        byId[obj.track_id] = timeline;
                    }
                    if (string.IsNullOrEmpty(timeline.className) && !string.IsNullOrEmpty(obj.class_name))
                    {
                        timeline.className = obj.class_name;
                    }
                    timeline.frames.Add(fi);
                    timeline.positions.Add(new Vector3(obj.x_m, yOffset, obj.z_m));
                    if (fi < timeline.startFrame) timeline.startFrame = fi;
                    if (fi > timeline.endFrame) timeline.endFrame = fi;
                }
            }

            _tracks.Clear();
            foreach (var pair in byId)
            {
                if (pair.Value.frames.Count == 0) continue;
                _tracks.Add(pair.Value);
            }
        }

        private void SpawnVehicles()
        {
            Transform parent = vehicleParent != null ? vehicleParent : transform;
            for (int i = 0; i < _tracks.Count; i++)
            {
                var track = _tracks[i];
                ObjectPlacer.ClassPrefab entry = prefabSource != null ? prefabSource.ResolveEntry(track.className) : null;
                GameObject prefab = entry != null ? entry.prefab : (prefabSource != null ? prefabSource.FallbackPrefab : null);
                if (prefab == null) continue;

                track.prefabEntry = entry;
                track.instance = Instantiate(prefab, parent);
                track.instance.name = $"Vehicle_T{track.trackId}_{track.className}";
                if (entry != null && entry.scale > 0f && Mathf.Abs(entry.scale - 1f) > 1e-4f)
                {
                    track.instance.transform.localScale = track.instance.transform.localScale * entry.scale;
                }
                track.instance.SetActive(false);
            }
        }

        private void Update()
        {
            if (!HasTimeline) return;

            if (IsPlaying)
            {
                CurrentTime += Time.deltaTime * playbackSpeed;
                if (CurrentTime >= Duration)
                {
                    if (loop)
                    {
                        if (Duration > 0f) CurrentTime %= Duration;
                        else CurrentTime = 0f;
                    }
                    else
                    {
                        CurrentTime = Duration;
                        IsPlaying = false;
                    }
                }
                ApplyTime(CurrentTime, force: false);
                UpdateUi();
            }
        }

        public void Play()
        {
            if (!HasTimeline) return;
            if (CurrentTime >= Duration && !loop) CurrentTime = 0f;
            IsPlaying = true;
            UpdateUi();
        }

        public void Pause()
        {
            IsPlaying = false;
            UpdateUi();
        }

        public void TogglePlay()
        {
            if (!HasTimeline) return;
            if (IsPlaying) Pause();
            else Play();
        }

        public void SeekToNormalized(float t01)
        {
            if (!HasTimeline) return;
            t01 = Mathf.Clamp01(t01);
            CurrentTime = Duration * t01;
            ApplyTime(CurrentTime, force: true);
            UpdateUi();
        }

        public void SeekToTime(float seconds)
        {
            if (!HasTimeline) return;
            CurrentTime = Mathf.Clamp(seconds, 0f, Duration);
            ApplyTime(CurrentTime, force: true);
            UpdateUi();
        }

        private void ApplyTime(float t, bool force)
        {
            float frameFloat = t * _fps;
            int frameLow = Mathf.FloorToInt(frameFloat);
            int frameHigh = Mathf.Min(_frameCount - 1, frameLow + 1);
            float alpha = Mathf.Clamp01(frameFloat - frameLow);

            float dt = force ? 1f / Mathf.Max(1f, _fps) : Time.deltaTime;

            for (int i = 0; i < _tracks.Count; i++)
            {
                var track = _tracks[i];
                if (track.instance == null) continue;

                bool active = frameLow >= track.startFrame && frameLow <= track.endFrame + 1;
                if (!active)
                {
                    if (track.instance.activeSelf) track.instance.SetActive(false);
                    track.hasHeading = false;
                    continue;
                }

                if (!SampleFrame(track, frameLow, out var posLow))
                {
                    if (track.instance.activeSelf) track.instance.SetActive(false);
                    continue;
                }
                if (!SampleFrame(track, frameHigh, out var posHigh))
                {
                    posHigh = posLow;
                }

                Vector3 worldPos = Vector3.Lerp(posLow, posHigh, alpha);
                if (!track.instance.activeSelf) track.instance.SetActive(true);

                Quaternion baseHeading = ComputeHeading(track, frameFloat, worldPos);
                Quaternion targetRot = baseHeading;
                if (track.prefabEntry != null)
                {
                    targetRot = targetRot * Quaternion.Euler(track.prefabEntry.rotationOffset);
                }

                Vector3 placementPos = worldPos;
                if (track.prefabEntry != null)
                {
                    placementPos += targetRot * track.prefabEntry.positionOffset;
                }

                if (force)
                {
                    track.instance.transform.SetPositionAndRotation(placementPos, targetRot);
                }
                else
                {
                    track.instance.transform.position = placementPos;
                    track.instance.transform.rotation = Quaternion.RotateTowards(
                        track.instance.transform.rotation,
                        targetRot,
                        headingSlewDegPerSec * dt);
                }
            }
        }

        private Quaternion ComputeHeading(TrackTimeline track, float frameFloat, Vector3 currentPos)
        {
            float pastFrame = frameFloat - headingWindowFrames;
            int pastLow = Mathf.FloorToInt(pastFrame);
            int pastHigh = pastLow + 1;
            float pastAlpha = Mathf.Clamp01(pastFrame - pastLow);

            Vector3 pastPos;
            if (pastLow < track.startFrame)
            {
                if (!SampleFrame(track, track.startFrame, out pastPos))
                {
                    return track.hasHeading ? track.lastHeading : Quaternion.identity;
                }
            }
            else
            {
                if (!SampleFrame(track, pastLow, out var a))
                {
                    return track.hasHeading ? track.lastHeading : Quaternion.identity;
                }
                if (!SampleFrame(track, pastHigh, out var b))
                {
                    b = a;
                }
                pastPos = Vector3.Lerp(a, b, pastAlpha);
            }

            Vector3 motion = currentPos - pastPos;
            motion.y = 0f;
            if (motion.sqrMagnitude < headingMinDisplacement * headingMinDisplacement)
            {
                return track.hasHeading ? track.lastHeading : Quaternion.identity;
            }
            Quaternion h = Quaternion.LookRotation(motion.normalized, Vector3.up);
            track.lastHeading = h;
            track.hasHeading = true;
            return h;
        }

        private static bool SampleFrame(TrackTimeline track, int frameIndex, out Vector3 pos)
        {
            pos = Vector3.zero;
            var fr = track.frames;
            if (fr.Count == 0) return false;
            if (frameIndex <= fr[0])
            {
                pos = track.positions[0];
                return true;
            }
            if (frameIndex >= fr[fr.Count - 1])
            {
                pos = track.positions[fr.Count - 1];
                return true;
            }

            int lo = 0;
            int hi = fr.Count - 1;
            while (lo + 1 < hi)
            {
                int mid = (lo + hi) >> 1;
                if (fr[mid] <= frameIndex) lo = mid;
                else hi = mid;
            }
            int fLo = fr[lo];
            int fHi = fr[hi];
            if (fLo == frameIndex) { pos = track.positions[lo]; return true; }
            if (fHi == frameIndex) { pos = track.positions[hi]; return true; }
            float a = (frameIndex - fLo) / (float)(fHi - fLo);
            pos = Vector3.Lerp(track.positions[lo], track.positions[hi], a);
            return true;
        }

        private void HookUi()
        {
            if (timeSlider != null)
            {
                timeSlider.minValue = 0f;
                timeSlider.maxValue = 1f;
                timeSlider.onValueChanged.RemoveListener(OnSliderChanged);
                timeSlider.onValueChanged.AddListener(OnSliderChanged);

                var trigger = timeSlider.GetComponent<SliderDragRelay>();
                if (trigger == null) trigger = timeSlider.gameObject.AddComponent<SliderDragRelay>();
                trigger.controller = this;
            }
            if (playPauseButton != null)
            {
                playPauseButton.onClick.RemoveListener(TogglePlay);
                playPauseButton.onClick.AddListener(TogglePlay);
            }
        }

        private void OnSliderChanged(float v)
        {
            if (_suppressSliderCallback) return;
            if (!_sliderDragging) return; // ignore programmatic updates when the user isn't dragging.
            SeekToNormalized(v);
        }

        public void OnSliderBeginDrag()
        {
            _sliderDragging = true;
        }

        public void OnSliderEndDrag()
        {
            _sliderDragging = false;
        }

        private void UpdateUiInteractivity()
        {
            if (timeSlider != null) timeSlider.interactable = HasTimeline;
            if (playPauseButton != null) playPauseButton.interactable = HasTimeline;
        }

        private void UpdateUi()
        {
            if (timeSlider != null && !_sliderDragging && Duration > 0f)
            {
                _suppressSliderCallback = true;
                timeSlider.value = Mathf.Clamp01(CurrentTime / Duration);
                _suppressSliderCallback = false;
            }
            if (timeLabel != null)
            {
                timeLabel.text = HasTimeline
                    ? $"{FormatTime(CurrentTime)} / {FormatTime(Duration)}"
                    : "--:-- / --:--";
            }
            if (playPauseLabel != null)
            {
                playPauseLabel.text = IsPlaying ? "Pause" : "Play";
            }
        }

        private static string FormatTime(float sec)
        {
            if (sec < 0f || float.IsNaN(sec)) sec = 0f;
            int m = Mathf.FloorToInt(sec / 60f);
            int s = Mathf.FloorToInt(sec % 60f);
            int ms = Mathf.FloorToInt((sec - Mathf.Floor(sec)) * 100f);
            return $"{m:00}:{s:00}.{ms:00}";
        }

        public void ClearVehicles()
        {
            for (int i = 0; i < _tracks.Count; i++)
            {
                if (_tracks[i].instance != null)
                {
                    if (Application.isPlaying) Destroy(_tracks[i].instance);
                    else DestroyImmediate(_tracks[i].instance);
                }
            }
            _tracks.Clear();
            HasTimeline = false;
            IsPlaying = false;
        }
    }
}

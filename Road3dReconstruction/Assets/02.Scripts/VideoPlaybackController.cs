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
        [Tooltip("Length (in frames) of the recent window whose motion trend determines the heading. Larger = smoother but laggier.")]
        public int headingWindowFrames = 15;
        [Tooltip("Number of evenly-spaced samples taken within the window for the least-squares trend fit. Higher = smoother.")]
        [Range(2, 32)] public int headingTrendSamples = 10;
        [Tooltip("How fast the vehicle rotates toward its trend direction (deg/sec). Lower = stronger temporal smoothing on top of the trend fit.")]
        public float headingSlewDegPerSec = 240f;
        [Tooltip("Minimum displacement (m) accumulated across the window to update facing.")]
        public float headingMinDisplacement = 0.05f;

        [Header("Smoothing")]
        [Tooltip("Replace a sample with the neighbor-interpolated value when it deviates from that interpolation by more than this distance (meters). 0 disables.")]
        public float outlierMaxDeviationM = 2.5f;
        [Tooltip("Number of outlier-replacement passes (handles a couple of consecutive spikes).")]
        public int outlierMaxPasses = 2;
        [Tooltip("Centered moving-average window (samples) applied after outlier removal. 1 disables smoothing.")]
        [Range(1, 11)] public int positionSmoothingWindow = 3;

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
                SmoothTrack(pair.Value);
                _tracks.Add(pair.Value);
            }
        }

        private void SmoothTrack(TrackTimeline track)
        {
            ApplyOutlierFilter(track);
            ApplyMovingAverage(track);
        }

        // Replaces samples whose horizontal distance to the linear interpolation of the previous
        // and next samples exceeds `outlierMaxDeviationM`. This catches single-frame spikes such as
        // (0,0) → (100,100) → (0,1): the middle sample is far from lerp(prev,next), so it is pulled
        // back onto the line between its neighbors. Multiple passes pick up adjacent spikes.
        private void ApplyOutlierFilter(TrackTimeline track)
        {
            if (outlierMaxDeviationM <= 0f) return;
            var frames = track.frames;
            var pos = track.positions;
            int n = pos.Count;
            if (n < 3) return;

            float thrSq = outlierMaxDeviationM * outlierMaxDeviationM;
            int passes = Mathf.Max(1, outlierMaxPasses);
            for (int pass = 0; pass < passes; pass++)
            {
                bool changed = false;
                for (int i = 1; i < n - 1; i++)
                {
                    int fPrev = frames[i - 1];
                    int fCur = frames[i];
                    int fNext = frames[i + 1];
                    float denom = fNext - fPrev;
                    if (denom <= 0f) continue;
                    float t = (fCur - fPrev) / denom;
                    Vector3 expected = Vector3.Lerp(pos[i - 1], pos[i + 1], t);
                    float dx = pos[i].x - expected.x;
                    float dz = pos[i].z - expected.z;
                    if (dx * dx + dz * dz > thrSq)
                    {
                        pos[i] = new Vector3(expected.x, pos[i].y, expected.z);
                        changed = true;
                    }
                }
                if (!changed) break;
            }
        }

        // Centered moving average over `positionSmoothingWindow` samples. Cheap, robust enough for
        // residual jitter once the spike filter has done the heavy lifting.
        private void ApplyMovingAverage(TrackTimeline track)
        {
            int w = positionSmoothingWindow;
            if (w <= 1) return;
            var pos = track.positions;
            int n = pos.Count;
            if (n < 2) return;

            int half = w / 2;
            var buf = new Vector3[n];
            for (int i = 0; i < n; i++)
            {
                int lo = Mathf.Max(0, i - half);
                int hi = Mathf.Min(n - 1, i + half);
                Vector3 sum = Vector3.zero;
                int count = 0;
                for (int k = lo; k <= hi; k++) { sum += pos[k]; count++; }
                buf[i] = sum / count;
            }
            for (int i = 0; i < n; i++) pos[i] = buf[i];
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

        // Fits a straight line through positions sampled across the trailing window
        // [frameFloat - headingWindowFrames, frameFloat] using ordinary least squares.
        // The slope vector is the average velocity in BEV meters per frame — its direction
        // is the smoothed heading. This is stable against single-frame jitter even when
        // the spike filter misses something, and against legitimate small swerves.
        private Quaternion ComputeHeading(TrackTimeline track, float frameFloat, Vector3 currentPos)
        {
            float window = Mathf.Max(1f, headingWindowFrames);
            float startFrame = frameFloat - window;
            int samples = Mathf.Max(2, headingTrendSamples);

            // Clamp the sampling range to the portion of the track that actually exists.
            // Outside [startFrame, endFrame] SampleFrame would clamp to an endpoint, so those
            // pseudo-samples contribute zero motion and bias the fit toward standing still.
            float lo = Mathf.Max(startFrame, track.startFrame);
            float hi = Mathf.Min(frameFloat, track.endFrame);
            if (hi - lo < 1e-3f)
            {
                return track.hasHeading ? track.lastHeading : Quaternion.identity;
            }

            double sumF = 0.0, sumX = 0.0, sumZ = 0.0;
            double sumFF = 0.0, sumFX = 0.0, sumFZ = 0.0;
            int count = 0;
            Vector3 firstPos = currentPos;
            Vector3 latestPos = currentPos;
            for (int s = 0; s < samples; s++)
            {
                float t = s / (float)(samples - 1);
                float f = Mathf.Lerp(lo, hi, t);
                if (!SampleFrameFloat(track, f, out var p)) continue;
                if (count == 0) firstPos = p;
                latestPos = p;
                sumF += f;
                sumX += p.x;
                sumZ += p.z;
                sumFF += (double)f * f;
                sumFX += (double)f * p.x;
                sumFZ += (double)f * p.z;
                count++;
            }
            if (count < 2)
            {
                return track.hasHeading ? track.lastHeading : Quaternion.identity;
            }

            double n = count;
            double denom = n * sumFF - sumF * sumF;
            Vector3 dir;
            if (System.Math.Abs(denom) < 1e-9)
            {
                // Window collapsed to a single frame value — fall back to endpoint delta.
                dir = latestPos - firstPos;
            }
            else
            {
                float slopeX = (float)((n * sumFX - sumF * sumX) / denom);
                float slopeZ = (float)((n * sumFZ - sumF * sumZ) / denom);
                dir = new Vector3(slopeX, 0f, slopeZ);
            }

            // Slope is meters-per-frame; the total displacement implied by the trend over the
            // window is `dir * (hi - lo)`. Compare that against the min-displacement gate.
            float effectiveWindow = hi - lo;
            float displacementSq = dir.sqrMagnitude * effectiveWindow * effectiveWindow;
            if (displacementSq < headingMinDisplacement * headingMinDisplacement)
            {
                return track.hasHeading ? track.lastHeading : Quaternion.identity;
            }

            Quaternion h = Quaternion.LookRotation(dir.normalized, Vector3.up);
            track.lastHeading = h;
            track.hasHeading = true;
            return h;
        }

        private static bool SampleFrameFloat(TrackTimeline track, float frame, out Vector3 pos)
        {
            int low = Mathf.FloorToInt(frame);
            int high = low + 1;
            if (!SampleFrame(track, low, out var a))
            {
                pos = Vector3.zero;
                return false;
            }
            if (!SampleFrame(track, high, out var b))
            {
                b = a;
            }
            float alpha = Mathf.Clamp01(frame - low);
            pos = Vector3.Lerp(a, b, alpha);
            return true;
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

using System.Collections;
using UnityEngine;
using UnityEngine.UI;

namespace RoadReconstruction
{
    // Top-level state machine for the WebGL deployment flow:
    //
    //   UploadPanel (idle, "+" button)
    //      └ user picks a file ──▶ LoadingPanel (upload + server polling)
    //                                     └ done ──▶ ScenePanel (visualization)
    //                                     └ error ──▶ back to UploadPanel with toast
    //
    // The scene must contain (at minimum):
    //   - this AppFlowController on a "AppFlow" GameObject
    //   - a FilePicker MonoBehaviour reachable via this.filePicker
    //   - a SceneLoader configured for the rendering targets
    //   - three UI panels (any RectTransform / CanvasGroup root works)
    //
    // SceneLoader.loadOnStart must be set to false; this controller drives loads.
    [DisallowMultipleComponent]
    public class AppFlowController : MonoBehaviour
    {
        [Header("Server")]
        [Tooltip("Base URL of the Flask inference server. e.g. http://localhost:5000 or https://api.example.com")]
        public string serverBaseUrl = "http://localhost:5000";
        [Tooltip("How often to poll /api/status (seconds).")]
        public float statusPollSeconds = 1.5f;

        [Header("References")]
        public FilePicker filePicker;
        public SceneLoader sceneLoader;

        [Header("UI Panels")]
        [Tooltip("Visible at app start. Contains the '+' upload button.")]
        public GameObject uploadPanel;
        [Tooltip("Visible during upload + server processing.")]
        public GameObject loadingPanel;
        [Tooltip("Visible once a scene has been loaded.")]
        public GameObject scenePanel;

        [Header("Upload UI")]
        public Button plusButton;
        public Text uploadStatusLabel;

        [Header("Loading UI")]
        public Text loadingStatusLabel;
        public Slider loadingProgressBar;
        public Button cancelButton;

        [Header("Scene UI")]
        public Button restartButton;
        public Text errorLabel;

        private InferenceApiClient _api;
        private Coroutine _activeJob;
        private string _activeJobId;

        private void Awake()
        {
            _api = new InferenceApiClient(serverBaseUrl);
            if (sceneLoader != null)
            {
                // Take control of loading away from the inspector setting; if a designer
                // left loadOnStart on we'd race and pull stale StreamingAssets data.
                sceneLoader.loadOnStart = false;
            }
        }

        private void OnEnable()
        {
            if (plusButton != null) plusButton.onClick.AddListener(OnPlusClicked);
            if (cancelButton != null) cancelButton.onClick.AddListener(OnCancelClicked);
            if (restartButton != null) restartButton.onClick.AddListener(GoToUpload);

            if (filePicker != null)
            {
                filePicker.Picked += OnFilePicked;
                filePicker.Cancelled += OnPickCancelled;
                filePicker.Failed += OnPickFailed;
            }
        }

        private void OnDisable()
        {
            if (plusButton != null) plusButton.onClick.RemoveListener(OnPlusClicked);
            if (cancelButton != null) cancelButton.onClick.RemoveListener(OnCancelClicked);
            if (restartButton != null) restartButton.onClick.RemoveListener(GoToUpload);

            if (filePicker != null)
            {
                filePicker.Picked -= OnFilePicked;
                filePicker.Cancelled -= OnPickCancelled;
                filePicker.Failed -= OnPickFailed;
            }
        }

        private void Start()
        {
            GoToUpload();
        }

        private void GoToUpload()
        {
            if (_activeJob != null)
            {
                StopCoroutine(_activeJob);
                _activeJob = null;
            }
            _activeJobId = null;

            ShowPanel(PanelKind.Upload);
            SetUploadStatus(string.Empty);
            SetLoadingProgress(0f, "");
            if (errorLabel != null) errorLabel.text = string.Empty;
        }

        private void OnPlusClicked()
        {
            if (filePicker == null)
            {
                ShowError("FilePicker reference missing");
                return;
            }
            SetUploadStatus("Opening file picker…");
            filePicker.Open();
        }

        private void OnFilePicked(FilePicker.PickResult result)
        {
            if (result.bytes == null || result.bytes.Length == 0)
            {
                ShowError("Selected file is empty");
                return;
            }
            ShowPanel(PanelKind.Loading);
            SetLoadingProgress(0f, $"Uploading {result.filename} ({FormatBytes(result.bytes.Length)})");
            _activeJob = StartCoroutine(RunJob(result));
        }

        private void OnPickCancelled()
        {
            SetUploadStatus("Selection cancelled.");
        }

        private void OnPickFailed(string reason)
        {
            ShowError($"File picker failed: {reason}");
        }

        private void OnCancelClicked()
        {
            // The server keeps running the job (no cancel endpoint yet) but the UI
            // returns to idle so the user can pick something else.
            GoToUpload();
        }

        private IEnumerator RunJob(FilePicker.PickResult media)
        {
            string jobId = null;
            string failure = null;

            yield return _api.UploadAsync(
                media.bytes,
                media.filename,
                media.contentType,
                onUploadProgress: p => SetLoadingProgress(Mathf.Clamp01(p) * 0.5f, $"Uploading… {Mathf.RoundToInt(p * 100f)}%"),
                onSuccess: id => jobId = id,
                onFailure: err => failure = err);

            if (!string.IsNullOrEmpty(failure))
            {
                ShowError(failure);
                yield break;
            }
            if (string.IsNullOrEmpty(jobId))
            {
                ShowError("Server did not return a job id");
                yield break;
            }
            _activeJobId = jobId;
            SetLoadingProgress(0.5f, "Server processing…");

            string serverError = null;
            yield return _api.PollStatusUntilDone(
                jobId,
                statusPollSeconds,
                onProgress: status =>
                {
                    float frac = 0.5f + 0.45f * Mathf.Clamp01(status.progress);
                    string msg = string.IsNullOrEmpty(status.message)
                        ? $"Processing… {Mathf.RoundToInt(status.progress * 100f)}%"
                        : status.message;
                    SetLoadingProgress(frac, msg);
                },
                onDone: _ => SetLoadingProgress(0.95f, "Fetching result…"),
                onFailure: err => serverError = err);

            if (!string.IsNullOrEmpty(serverError))
            {
                ShowError(serverError);
                yield break;
            }

            string json = null;
            yield return _api.FetchResult(
                jobId,
                onSuccess: text => json = text,
                onFailure: err => serverError = err);

            if (!string.IsNullOrEmpty(serverError))
            {
                ShowError(serverError);
                yield break;
            }
            if (string.IsNullOrEmpty(json))
            {
                ShowError("Empty scene JSON returned from server");
                yield break;
            }

            SetLoadingProgress(1f, "Rendering…");

            if (sceneLoader == null)
            {
                ShowError("SceneLoader reference missing");
                yield break;
            }

            sceneLoader.LoadFromJson(json);
            ShowPanel(PanelKind.Scene);
            _activeJob = null;
        }

        private enum PanelKind { Upload, Loading, Scene }

        private void ShowPanel(PanelKind kind)
        {
            if (uploadPanel != null) uploadPanel.SetActive(kind == PanelKind.Upload);
            if (loadingPanel != null) loadingPanel.SetActive(kind == PanelKind.Loading);
            if (scenePanel != null) scenePanel.SetActive(kind == PanelKind.Scene);
        }

        private void SetUploadStatus(string s)
        {
            if (uploadStatusLabel != null) uploadStatusLabel.text = s ?? string.Empty;
        }

        private void SetLoadingProgress(float frac, string msg)
        {
            if (loadingProgressBar != null) loadingProgressBar.value = Mathf.Clamp01(frac);
            if (loadingStatusLabel != null && msg != null) loadingStatusLabel.text = msg;
        }

        private void ShowError(string reason)
        {
            Debug.LogError($"[AppFlow] {reason}");
            if (errorLabel != null) errorLabel.text = reason;
            // Drop back to the upload panel so the user can retry. The error label
            // sits inside the upload panel so it stays visible after the switch.
            ShowPanel(PanelKind.Upload);
            SetUploadStatus("Last attempt failed — try again.");
            _activeJob = null;
        }

        private static string FormatBytes(long bytes)
        {
            string[] units = { "B", "KB", "MB", "GB" };
            double v = bytes;
            int u = 0;
            while (v >= 1024 && u < units.Length - 1)
            {
                v /= 1024.0;
                u++;
            }
            return $"{v:0.##} {units[u]}";
        }
    }
}

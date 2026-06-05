using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Networking;

namespace RoadReconstruction
{
    // Talks to the Flask inference server: uploads a media file, polls status,
    // then fetches the scene JSON. Everything goes through UnityWebRequest so
    // WebGL builds work — File.* would block on the browser sandbox.
    public class InferenceApiClient
    {
        public string BaseUrl { get; }

        public InferenceApiClient(string baseUrl)
        {
            BaseUrl = (baseUrl ?? string.Empty).TrimEnd('/');
        }

        public IEnumerator UploadAsync(
            byte[] bytes,
            string filename,
            string contentType,
            Action<float> onUploadProgress,
            Action<string> onSuccess,
            Action<string> onFailure)
        {
            if (bytes == null || bytes.Length == 0)
            {
                onFailure?.Invoke("empty payload");
                yield break;
            }

            var form = new List<IMultipartFormSection>
            {
                new MultipartFormFileSection("file", bytes, filename, contentType ?? "application/octet-stream"),
            };

            using (var req = UnityWebRequest.Post($"{BaseUrl}/api/upload", form))
            {
                req.downloadHandler = new DownloadHandlerBuffer();
                var op = req.SendWebRequest();
                while (!op.isDone)
                {
                    onUploadProgress?.Invoke(req.uploadProgress);
                    yield return null;
                }
                onUploadProgress?.Invoke(1f);

                if (req.result != UnityWebRequest.Result.Success)
                {
                    onFailure?.Invoke($"upload failed ({req.responseCode}): {req.error} {req.downloadHandler.text}");
                    yield break;
                }

                var parsed = JsonUtility.FromJson<UploadResponse>(req.downloadHandler.text);
                if (parsed == null || string.IsNullOrEmpty(parsed.job_id))
                {
                    onFailure?.Invoke("upload returned no job_id");
                    yield break;
                }
                onSuccess?.Invoke(parsed.job_id);
            }
        }

        public IEnumerator PollStatusUntilDone(
            string jobId,
            float pollIntervalSeconds,
            Action<JobStatus> onProgress,
            Action<JobStatus> onDone,
            Action<string> onFailure)
        {
            string url = $"{BaseUrl}/api/status/{jobId}";
            while (true)
            {
                using (var req = UnityWebRequest.Get(url))
                {
                    yield return req.SendWebRequest();
                    if (req.result != UnityWebRequest.Result.Success)
                    {
                        onFailure?.Invoke($"status failed ({req.responseCode}): {req.error}");
                        yield break;
                    }
                    var status = JsonUtility.FromJson<JobStatus>(req.downloadHandler.text);
                    if (status == null)
                    {
                        onFailure?.Invoke("status returned no payload");
                        yield break;
                    }
                    onProgress?.Invoke(status);
                    if (status.status == "done")
                    {
                        onDone?.Invoke(status);
                        yield break;
                    }
                    if (status.status == "error")
                    {
                        onFailure?.Invoke(string.IsNullOrEmpty(status.error) ? "server error" : status.error);
                        yield break;
                    }
                }
                yield return new WaitForSeconds(Mathf.Max(0.25f, pollIntervalSeconds));
            }
        }

        public IEnumerator FetchResult(
            string jobId,
            Action<string> onSuccess,
            Action<string> onFailure)
        {
            using (var req = UnityWebRequest.Get($"{BaseUrl}/api/result/{jobId}"))
            {
                yield return req.SendWebRequest();
                if (req.result != UnityWebRequest.Result.Success)
                {
                    onFailure?.Invoke($"result failed ({req.responseCode}): {req.error}");
                    yield break;
                }
                onSuccess?.Invoke(req.downloadHandler.text);
            }
        }

        [Serializable]
        public class UploadResponse
        {
            public string job_id;
            public string status;
            public string kind;
        }

        [Serializable]
        public class JobStatus
        {
            public string job_id;
            public string filename;
            public string media_kind;
            public string status;
            public float progress;
            public string message;
            public string error;
            public double created_at;
            public double updated_at;
        }
    }
}

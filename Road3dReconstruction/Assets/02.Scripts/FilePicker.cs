using System;
using System.IO;
using System.Runtime.InteropServices;
using UnityEngine;

#if UNITY_EDITOR
using UnityEditor;
#endif

namespace RoadReconstruction
{
    // Hands the user's chosen file back to managed code regardless of platform.
    //
    // - WebGL: triggers a real browser <input type="file"> via the jslib plugin.
    //          The blob URL is returned synchronously through SendMessage; we
    //          download the bytes with UnityWebRequest which already handles
    //          blob: URLs in WebGL builds.
    // - Editor: uses EditorUtility.OpenFilePanel + File.ReadAllBytes so designers
    //          can iterate without producing a WebGL build every time.
    public class FilePicker : MonoBehaviour
    {
        [Serializable]
        public struct PickResult
        {
            public byte[] bytes;
            public string filename;
            public string contentType;
        }

        public event Action<PickResult> Picked;
        public event Action Cancelled;
        public event Action<string> Failed;

        // The jslib SendMessage handler can only call methods on a named GameObject;
        // we route through a stable singleton so the AppFlowController doesn't
        // have to live on the same GameObject as this picker.
        private static FilePicker _active;

#if UNITY_WEBGL && !UNITY_EDITOR
        [DllImport("__Internal")]
        private static extern void WebGLFilePicker_Open(string gameObject, string method, string accept);

        [DllImport("__Internal")]
        private static extern void WebGLFilePicker_Revoke(string url);
#endif

        public void Open(string accept = "video/*,image/*")
        {
            _active = this;
#if UNITY_WEBGL && !UNITY_EDITOR
            WebGLFilePicker_Open(gameObject.name, nameof(OnWebGLFilePicked), accept ?? string.Empty);
#elif UNITY_EDITOR
            string path = EditorUtility.OpenFilePanel("Select CCTV media", "", "mp4,mov,avi,mkv,webm,png,jpg,jpeg,bmp,webp");
            if (string.IsNullOrEmpty(path))
            {
                Cancelled?.Invoke();
                return;
            }
            try
            {
                var bytes = File.ReadAllBytes(path);
                Picked?.Invoke(new PickResult
                {
                    bytes = bytes,
                    filename = Path.GetFileName(path),
                    contentType = GuessContentType(path),
                });
            }
            catch (Exception ex)
            {
                Failed?.Invoke(ex.Message);
            }
#else
            Failed?.Invoke("File picker is only supported in WebGL builds and the Editor.");
#endif
        }

        // Invoked by SendMessage from WebGLFilePicker.jslib.
        // ReSharper disable once UnusedMember.Global
        public void OnWebGLFilePicked(string payload)
        {
            if (_active == null) return;
#if UNITY_WEBGL && !UNITY_EDITOR
            try
            {
                var msg = JsonUtility.FromJson<WebGLPickMessage>(payload);
                if (msg == null)
                {
                    _active.Failed?.Invoke("Invalid pick payload");
                    return;
                }
                if (msg.cancelled)
                {
                    _active.Cancelled?.Invoke();
                    return;
                }
                _active.StartCoroutine(FetchAndDeliver(msg));
            }
            catch (Exception ex)
            {
                _active.Failed?.Invoke(ex.Message);
            }
#endif
        }

#if UNITY_WEBGL && !UNITY_EDITOR
        private System.Collections.IEnumerator FetchAndDeliver(WebGLPickMessage msg)
        {
            using (var req = UnityEngine.Networking.UnityWebRequest.Get(msg.url))
            {
                yield return req.SendWebRequest();
                if (req.result != UnityEngine.Networking.UnityWebRequest.Result.Success)
                {
                    Failed?.Invoke($"Failed to read blob: {req.error}");
                    WebGLFilePicker_Revoke(msg.url);
                    yield break;
                }

                var bytes = req.downloadHandler.data;
                WebGLFilePicker_Revoke(msg.url);
                Picked?.Invoke(new PickResult
                {
                    bytes = bytes,
                    filename = msg.name,
                    contentType = string.IsNullOrEmpty(msg.type) ? GuessContentType(msg.name) : msg.type,
                });
            }
        }
#endif

        private static string GuessContentType(string filename)
        {
            string ext = Path.GetExtension(filename ?? string.Empty).ToLowerInvariant();
            switch (ext)
            {
                case ".mp4":  return "video/mp4";
                case ".mov":  return "video/quicktime";
                case ".avi":  return "video/x-msvideo";
                case ".mkv":  return "video/x-matroska";
                case ".webm": return "video/webm";
                case ".png":  return "image/png";
                case ".jpg":
                case ".jpeg": return "image/jpeg";
                case ".bmp":  return "image/bmp";
                case ".webp": return "image/webp";
                default:      return "application/octet-stream";
            }
        }

        [Serializable]
        private class WebGLPickMessage
        {
            public string url;
            public string name;
            public long size;
            public string type;
            public bool cancelled;
        }
    }
}

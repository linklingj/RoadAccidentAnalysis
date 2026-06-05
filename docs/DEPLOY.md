# Web Deployment Guide

This branch wires up the end-to-end web flow:

```
[User browser] ───────────────────► Unity WebGL build (static hosting)
       │
       │ POST /api/upload (multipart)
       ▼
[AWS EC2: Flask + gunicorn + GPU] ──► infer.py pipeline ──► scene JSON
       ▲                                                            │
       │                            GET /api/result/<job_id>        │
       └────────────────────────────────────────────────────────────┘
```

The user picks a CCTV video/image via a "+" button, the Unity build uploads it to
the Flask server, the server runs the existing `infer.py` pipeline, and Unity
visualises the returned scene JSON.

## Component checklist

| Piece                       | Status      | Source                                                         |
| --------------------------- | ----------- | -------------------------------------------------------------- |
| Flask inference server      | implemented | `server/`                                                      |
| Dockerfile for the server   | implemented | `server/Dockerfile`                                            |
| Unity upload UI (+ button)  | implemented | `Road3dReconstruction/Assets/01.Scenes/Test.unity` (AppFlow)   |
| Loading panel               | implemented | same scene                                                     |
| WebGL file picker plugin    | implemented | `Road3dReconstruction/Assets/Plugins/WebGL/WebGLFilePicker.jslib` |
| AWS GPU instance + model weights | **manual** | see below                                                 |
| HTTPS termination (ACM/ALB) | **manual**  | see below                                                      |
| Unity WebGL module install  | **manual**  | Unity Hub → Add Modules → WebGL Build Support                  |
| Unity build & static hosting (S3/CloudFront) | **manual** | see below                                      |

---

## What you still need to do

1. **Install the WebGL build module in Unity Hub**
   - Unity Hub → Installs → ⋯ on Unity `6000.3.12f1` → *Add modules* → check
     **WebGL Build Support** → install. (Required because this project's Unity
     install doesn't have it yet — `manage_build platform=webgl` returned
     "Platform 'WebGL' is not installed".)

2. **Point the Unity app at your server URL**
   - In `Test.unity` select the **AppFlow** GameObject.
   - On `AppFlowController`, set **Server Base Url** to your server address
     (e.g. `https://api.your-domain` or `http://<ec2-ip>:5000` for testing).
   - The default `http://localhost:5000` works while you develop locally.

3. **Provision an AWS GPU EC2 instance** (suggested):
   - AMI: *AWS Deep Learning AMI (Ubuntu 22.04)* — pre-installed NVIDIA driver,
     Docker, NVIDIA Container Toolkit.
   - Instance type: `g4dn.xlarge` (cheapest CUDA-capable). For larger videos
     use `g5.xlarge`. CPU-only is supported but ~10–20× slower.
   - Open inbound TCP 5000 (or 443 if you front it with HTTPS).
   - Attach an EBS volume large enough for incoming uploads (`/var/lib/road-accident`).

4. **Get the trained weights onto the instance**
   - `runs/segment/0405-road/weights/best.pt`
   - `runs/segment/0407-crosswalk/weights/best.pt`
   - `runs/segment/0401-object/weights/best.pt`
   - Easiest path: `aws s3 sync runs/ s3://<your-bucket>/runs/` from your dev
     machine, then `aws s3 sync s3://<your-bucket>/runs/ ~/road-accident/runs/`
     on the EC2 host. (They're large; don't bake them into the Docker image.)

5. **Run the inference server in Docker**
   ```bash
   # On the EC2 host, in this repo
   docker build -f server/Dockerfile -t road-accident-server .
   docker run -d --gpus all --restart=always -p 5000:5000 \
     -v ~/road-accident/runs:/app/runs \
     -v ~/road-accident/data:/var/lib/road-accident \
     -e CORS_ORIGINS="https://<your-static-host>" \
     --name road-accident road-accident-server
   ```
   - If you don't have a GPU, drop `--gpus all` and add
     `-e INFER_DEVICE=cpu`.
   - Smoke-test: `curl http://<ec2-ip>:5000/api/health` → `{"status":"ok"}`.

6. **(Recommended) Put HTTPS in front of the server**
   - Browser will block mixed-content uploads (HTTP server + HTTPS static
     site). Easiest fix: an Application Load Balancer with an ACM cert.
   - Point a domain (`api.your-domain.com`) at the ALB.
   - Update `serverBaseUrl` on the Unity `AppFlowController` to
     `https://api.your-domain.com`.

7. **Build the Unity WebGL bundle**
   - File → Build Settings → switch platform to **WebGL** → Build.
   - Or in CI: `unity -batchmode -nographics -projectPath Road3dReconstruction -buildTarget WebGL -executeMethod BuildScript.WebGLBuild` (you'd need to add `BuildScript`; not required for manual builds).
   - **PlayerSettings to verify before build**:
     - Resolution and Presentation → Default Canvas Width/Height: 1920×1080.
     - Publishing Settings → Compression Format: **Brotli** (smaller) or **Gzip**.
     - Publishing Settings → Decompression Fallback: **true** if S3 won't set
       `Content-Encoding`.
     - Other Settings → Color Space: **Linear** (matches URP).
     - Other Settings → Strip Engine Code: enabled (smaller build).

8. **Host the WebGL build statically**
   - `aws s3 sync Build/ s3://<your-bucket>/road3d/`
   - Make the bucket public (or front it with CloudFront).
   - If you used **Brotli** compression and the bucket doesn't auto-set the
     header, run `aws s3 cp --recursive --metadata-directive REPLACE
       --content-encoding br ...` for `*.br` files (same for `*.gz`).
   - For CloudFront, set the cache policy to allow `Content-Encoding`
     pass-through.
   - Test the URL: the upload UI loads, clicking + opens a browser file
     picker, selecting a video uploads to your `serverBaseUrl`, then the BEV
     visualization plays.

9. **CORS sanity check**
   - The server reads `CORS_ORIGINS` (comma-separated). Set it to your static
     host exactly: `https://<your-bucket>.s3.amazonaws.com,https://your-domain.com`.
   - For local dev, leave it as `*`. **Don't ship `*` to production** if you
     plan to add auth later.

---

## Local dev quickstart

```bash
# Terminal 1: inference server (requires the dl conda env)
conda activate dl
pip install -r server/requirements.txt
cp server/.env.example server/.env
export $(grep -v '^#' server/.env | xargs)
python server/app.py

# Terminal 2: Unity Editor
# Open Road3dReconstruction in Unity Hub, hit Play on Test.unity.
# The "+" button opens a native file picker; selecting a video uploads to
# http://localhost:5000 and renders the result.
```

In the Editor, file selection uses `EditorUtility.OpenFilePanel`. In WebGL
builds it uses the browser's native picker via the `.jslib` plugin.

## Files added on this branch

```
server/
  app.py
  jobs.py
  worker.py
  gunicorn_config.py
  requirements.txt
  Dockerfile
  .dockerignore
  .env.example
  README.md

Road3dReconstruction/Assets/Plugins/WebGL/
  WebGLFilePicker.jslib
  WebGLFilePicker.jslib.meta

Road3dReconstruction/Assets/02.Scripts/
  AppFlowController.cs
  FilePicker.cs
  InferenceApiClient.cs
  SpinnerRotator.cs
  SceneLoader.cs       (added LoadFromJson(string) method)

docs/DEPLOY.md          (this file)
```

`Road3dReconstruction/Assets/01.Scenes/Test.unity` was updated:
- `SceneLoader.loadOnStart` flipped to `false` so the AppFlowController
  drives the load.
- `AppFlow` GameObject added (`FilePicker` + `AppFlowController`).
- `AppFlowCanvas` added with `UploadPanel` (with + button, title, hint,
  status, error label) and `LoadingPanel` (with progress bar, status, cancel
  button).
- `Build Settings` now lists `Test.unity` as scene index 0.

## Troubleshooting

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| File picker doesn't open in WebGL | popup blocker tripped because click came from a non-gesture event | the picker is hooked to the button's `onClick`; make sure you only call `FilePicker.Open()` from a UI button handler. |
| CORS errors in browser console | `CORS_ORIGINS` doesn't include the static host | set `CORS_ORIGINS` on the server and restart |
| Upload succeeds but result never returns | model weights missing on the server | check `docker logs road-accident`; expect `FileNotFoundError` for `.pt` paths |
| Server OOMs during video | first run loads all three models | restart and give the instance 8 GB+ RAM; `g4dn.xlarge` has 16 GB |
| "Mixed content" blocked | static site is HTTPS but server is HTTP | put the server behind an ALB with HTTPS, update `serverBaseUrl` |

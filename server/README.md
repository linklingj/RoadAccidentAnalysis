# Inference server

Flask wrapper around `infer.py` that the Unity WebGL client uploads CCTV media
to and polls for BEV scene results.

## Endpoints

| Method | Path                  | Purpose                                                       |
| ------ | --------------------- | ------------------------------------------------------------- |
| POST   | `/api/upload`         | multipart `file` field; returns `{job_id, status}` (202).     |
| GET    | `/api/status/<jobId>` | `{status, progress, message, …}`; status ∈ queued/processing/done/error. |
| GET    | `/api/result/<jobId>` | scene JSON when `done`; 409 otherwise.                        |
| GET    | `/api/health`         | liveness probe.                                               |

The scene JSON schema matches `SceneData` in `Road3dReconstruction/Assets/02.Scripts/SceneSchema.cs`.

## Local development

```bash
conda activate dl
pip install -r server/requirements.txt
cp server/.env.example server/.env
export $(grep -v '^#' server/.env | xargs)
python server/app.py            # dev server on :5000
```

## Docker

```bash
docker build -f server/Dockerfile -t road-accident-server .
docker run --rm -p 5000:5000 \
  -v $PWD/runs:/app/runs \
  -e CORS_ORIGINS="*" \
  road-accident-server
```

The image expects the YOLO/PerspectiveFields weights to be mounted at
`/app/runs` (segment training output). Bake them into the image only if you
control the registry, since they are large.

## AWS EC2 deployment outline

1. Provision a `g4dn.xlarge` (or `g5.xlarge` for newer GPUs) Ubuntu 22.04 AMI.
   CPU-only `c6i.2xlarge` also works but a 30 s video can take ~10 min.
2. Install NVIDIA driver + Docker + NVIDIA Container Toolkit (`g4dn.xlarge`
   AMI from AWS Deep Learning AMI has these pre-installed).
3. Push the Docker image to ECR and pull it on the instance, or build on the
   instance directly.
4. Copy your trained weights to `~/road-accident/runs/` on the host.
5. Run:
   ```bash
   docker run -d --gpus all --restart=always -p 80:5000 \
     -v ~/road-accident/runs:/app/runs \
     -v ~/road-accident/data:/var/lib/road-accident \
     -e CORS_ORIGINS="https://<your-static-host>" \
     --name road-accident road-accident-server
   ```
6. Front the instance with an ALB + ACM cert if you need HTTPS, then point
   the Unity client at `https://api.your-domain/api`.

## Concurrency

A single gunicorn worker with 4 threads serves multiple uploads but processes
one inference at a time (model state is shared in-process). Increase
`GUNICORN_THREADS` only if you have RAM/GPU headroom; bumping `WORKERS` would
duplicate the model in memory.

"""In-memory job registry + background worker queue.

Single-process design: a queue + worker thread share state with the Flask
request handlers. Good enough for a single-EC2-instance deployment behind
gunicorn with ``--workers 1 --threads N``. If you scale out, swap this for
Redis/RQ — the public API (``submit``, ``get``, ``set_progress``…) is small.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from queue import Queue
from typing import Any, Callable, Dict, Optional


JobStatus = str  # "queued" | "processing" | "done" | "error"


@dataclass
class Job:
    job_id: str
    filename: str
    media_kind: str  # "image" | "video"
    input_path: str
    status: JobStatus = "queued"
    progress: float = 0.0  # 0..1
    message: str = ""
    result_path: Optional[str] = None  # absolute path to scene JSON when done
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None

    def to_public(self) -> Dict[str, Any]:
        d = asdict(self)
        d.pop("input_path", None)
        d.pop("result_path", None)
        return d


class JobRegistry:
    def __init__(self) -> None:
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.RLock()
        self._queue: "Queue[str]" = Queue()
        self._worker_started = False
        self._handler: Optional[Callable[[Job], None]] = None
        self._log = logging.getLogger("jobs")

    def configure(self, handler: Callable[[Job], None]) -> None:
        """Register the function that processes a job. Must be set before ``submit``."""
        with self._lock:
            self._handler = handler
            if not self._worker_started:
                t = threading.Thread(target=self._worker_loop, name="job-worker", daemon=True)
                t.start()
                self._worker_started = True

    def submit(self, *, filename: str, media_kind: str, input_path: str) -> Job:
        job = Job(
            job_id=uuid.uuid4().hex,
            filename=filename,
            media_kind=media_kind,
            input_path=input_path,
        )
        with self._lock:
            self._jobs[job.job_id] = job
        self._queue.put(job.job_id)
        return job

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def set_progress(self, job_id: str, progress: float, message: str = "") -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.progress = max(0.0, min(1.0, float(progress)))
            if message:
                job.message = message
            job.updated_at = time.time()

    def _mark_processing(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.status = "processing"
            job.started_at = time.time()
            job.updated_at = job.started_at

    def _mark_done(self, job_id: str, result_path: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.status = "done"
            job.progress = 1.0
            job.result_path = result_path
            job.finished_at = time.time()
            job.updated_at = job.finished_at

    def _mark_error(self, job_id: str, err: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.status = "error"
            job.error = err
            job.finished_at = time.time()
            job.updated_at = job.finished_at

    def _worker_loop(self) -> None:
        while True:
            job_id = self._queue.get()
            if self._handler is None:
                self._mark_error(job_id, "no handler configured")
                continue
            self._mark_processing(job_id)
            job = self.get(job_id)
            if job is None:
                continue
            try:
                self._handler(job)
                if job.status != "done":
                    # Handler should have called set_result; treat absence as error.
                    self._mark_error(job_id, "handler returned without producing a result")
            except Exception as exc:  # noqa: BLE001 — surface any failure to the client
                self._log.exception("Job %s failed", job_id)
                self._mark_error(job_id, f"{type(exc).__name__}: {exc}")

    # Allow the handler to mark completion explicitly (the writer knows the path).
    def set_result(self, job_id: str, result_path: str) -> None:
        self._mark_done(job_id, result_path)


registry = JobRegistry()

"""Background separation jobs.

A job runs one GPU pass and keeps its result on disk. The UI previews that
result, and only commits it to the project stack if it sounds right - so
auditioning a prompt never costs a second inference call.
"""
from __future__ import annotations

import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import audio as A
from . import config
from .engine import Anchor, get_engine, run_chunked
from .store import Project

_pool = ThreadPoolExecutor(max_workers=2)
_jobs: dict[str, "Job"] = {}
_lock = threading.Lock()


@dataclass
class Job:
    id: str
    project_id: str
    description: str
    anchors: list
    status: str = "queued"       # queued | running | done | error
    progress: float = 0.0
    error: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def dir(self) -> Path:
        return config.STORAGE / self.project_id / "jobs" / self.id

    def to_dict(self) -> dict:
        return {
            "id": self.id, "status": self.status, "progress": round(self.progress, 3),
            "error": self.error, "description": self.description, **self.meta,
        }


def get(job_id: str) -> Job | None:
    return _jobs.get(job_id)


def submit(project: Project, description: str, anchors: list) -> Job:
    job = Job(id=uuid.uuid4().hex[:10], project_id=project.id,
              description=description, anchors=anchors)
    with _lock:
        _jobs[job.id] = job
    _pool.submit(_run, job)
    return job


def _run(job: Job) -> None:
    try:
        job.status = "running"
        project = Project.load(job.project_id)

        source = project.render()
        parsed = [Anchor(a[0], float(a[1]), float(a[2])) for a in job.anchors]

        def progress(p: float) -> None:
            job.progress = p

        engine = get_engine(config.ENGINE)
        result = run_chunked(
            engine, source, config.SAMPLE_RATE, job.description, parsed, progress
        )

        out = job.dir()
        out.mkdir(parents=True, exist_ok=True)
        A.write(out / "target.wav", result.target)
        A.write(out / "residual.wav", result.residual)

        peak = float(abs(result.target).max()) if result.target.size else 0.0
        job.meta["target_peak"] = round(peak, 4)
        job.meta["found"] = peak > 1e-3
        job.progress = 1.0
        job.status = "done"

    except Exception as exc:  # surfaced verbatim in the UI - these are usually
        job.error = f"{type(exc).__name__}: {exc}"  # setup problems worth reading
        job.status = "error"
        traceback.print_exc()

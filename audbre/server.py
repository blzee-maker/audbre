"""AudBre HTTP API + static UI."""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException, UploadFile, File, Body
from fastapi.responses import FileResponse, HTMLResponse, Response

from . import audio as A
from . import config, jobs
from .store import Project, list_projects

app = FastAPI(title="AudBre", version="0.1.0")

MAX_UPLOAD = 400 * 1024 * 1024  # 400 MB


def _project(pid: str) -> Project:
    try:
        return Project.load(pid)
    except KeyError:
        raise HTTPException(404, "project not found")


def _wav_response(data: np.ndarray, sample_rate: int = config.SAMPLE_RATE) -> Response:
    """Browsers won't decode 32-bit float WAV, so previews go out as PCM16."""
    buf = io.BytesIO()
    sf.write(buf, np.clip(data, -1.0, 1.0), sample_rate, format="WAV", subtype="PCM_16")
    return Response(buf.getvalue(), media_type="audio/wav",
                    headers={"Cache-Control": "no-store"})


# --------------------------- UI ---------------------------

@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (config.STATIC / "index.html").read_text(encoding="utf-8")


@app.get("/app.js")
def appjs() -> FileResponse:
    return FileResponse(config.STATIC / "app.js", media_type="text/javascript")


@app.get("/style.css")
def appcss() -> FileResponse:
    return FileResponse(config.STATIC / "style.css", media_type="text/css")


# ------------------------ projects ------------------------

@app.get("/api/config")
def get_config() -> dict:
    return {"engine": config.ENGINE, "model": config.MODEL,
            "configured": bool(config.MODAL_URL) or config.ENGINE == "local"}


@app.get("/api/projects")
def api_list() -> list[dict]:
    return list_projects()


@app.post("/api/projects")
async def api_create(file: UploadFile = File(...)) -> dict:
    raw = await file.read()
    if not raw:
        raise HTTPException(400, "empty upload")
    if len(raw) > MAX_UPLOAD:
        raise HTTPException(413, f"file larger than {MAX_UPLOAD // 1024**2} MB")
    try:
        project = Project.create(file.filename or "audio.wav", raw)
    except A.FFmpegError as exc:
        raise HTTPException(400, f"could not decode this file: {exc}")
    return api_get(project.id)


@app.get("/api/projects/{pid}")
def api_get(pid: str) -> dict:
    p = _project(pid)
    data, _ = A.read(p.base_wav)
    return {
        "id": p.id, "name": p.name, "duration": p.duration,
        "sample_rate": p.sample_rate, "has_video": p.has_video,
        "peaks": A.peaks(data),
        "layers": [
            {"id": l.id, "description": l.description, "anchors": l.anchors,
             "enabled": l.enabled}
            for l in p.layers
        ],
    }


@app.get("/api/projects/{pid}/audio/{which}")
def api_audio(pid: str, which: str) -> Response:
    p = _project(pid)
    if which == "base":
        data, _ = A.read(p.base_wav)
    elif which == "current":
        data = p.render()
    else:
        try:
            layer = p.find(which)
        except KeyError:
            raise HTTPException(404, "no such layer")
        path = layer.target_path(p.root)
        if not path.exists():
            raise HTTPException(404, "layer audio missing")
        data, _ = A.read(path)
    return _wav_response(data)


@app.get("/api/projects/{pid}/waveform/current")
def api_waveform(pid: str) -> dict:
    p = _project(pid)
    return {"peaks": A.peaks(p.render())}


# ----------------------- separation -----------------------

@app.post("/api/projects/{pid}/separate")
def api_separate(pid: str, body: dict = Body(...)) -> dict:
    p = _project(pid)
    description = (body.get("description") or "").strip()
    anchors = body.get("anchors") or []

    if not description and not anchors:
        raise HTTPException(400, "describe the sound, mark it on the timeline, or both")
    if config.ENGINE == "modal" and not config.MODAL_URL:
        raise HTTPException(503, "AUDBRE_MODAL_URL is not set - deploy modal_app.py first")

    job = jobs.submit(p, description, anchors)
    return job.to_dict()


@app.get("/api/jobs/{jid}")
def api_job(jid: str) -> dict:
    job = jobs.get(jid)
    if job is None:
        raise HTTPException(404, "unknown job")
    return job.to_dict()


@app.get("/api/jobs/{jid}/audio/{which}")
def api_job_audio(jid: str, which: str) -> Response:
    job = jobs.get(jid)
    if job is None or job.status != "done":
        raise HTTPException(404, "result not ready")
    if which not in ("target", "residual"):
        raise HTTPException(400, "expected 'target' or 'residual'")
    data, _ = A.read(job.dir() / f"{which}.wav")
    return _wav_response(data)


@app.post("/api/projects/{pid}/layers")
def api_commit(pid: str, body: dict = Body(...)) -> dict:
    """Promote a previewed job into a permanent layer on the stack."""
    p = _project(pid)
    job = jobs.get(body.get("job_id", ""))
    if job is None or job.status != "done":
        raise HTTPException(400, "that job has no finished result")

    target, _ = A.read(job.dir() / "target.wav")
    residual, _ = A.read(job.dir() / "residual.wav")
    p.add_layer(job.description, job.anchors, target, residual)
    return api_get(pid)


@app.patch("/api/projects/{pid}/layers/{lid}")
def api_toggle(pid: str, lid: str, body: dict = Body(...)) -> dict:
    p = _project(pid)
    try:
        layer = p.find(lid)
    except KeyError:
        raise HTTPException(404, "no such layer")
    layer.enabled = bool(body.get("enabled", not layer.enabled))
    p.commit()
    return api_get(pid)


@app.delete("/api/projects/{pid}/layers/{lid}")
def api_delete_layer(pid: str, lid: str) -> dict:
    p = _project(pid)
    try:
        p.remove_layer(lid)
    except KeyError:
        raise HTTPException(404, "no such layer")
    return api_get(pid)


# ------------------------- export -------------------------

@app.get("/api/projects/{pid}/export")
def api_export(pid: str, format: str = "wav") -> FileResponse:
    if format not in ("wav", "mp3", "video"):
        raise HTTPException(400, "format must be wav, mp3 or video")

    p = _project(pid)
    p.commit()
    stem = Path(p.name).stem

    if format == "video":
        if not p.has_video:
            raise HTTPException(400, "this project has no video track")
        out = p.root / f"{stem}_audbre{Path(p.name).suffix}"
        A.remux(p.source_path, p.current_wav, out)
    elif format == "mp3":
        out = A.encode_mp3(p.current_wav, p.root / f"{stem}_audbre.mp3")
    else:
        out = p.root / f"{stem}_audbre.wav"
        buf = io.BytesIO()
        sf.write(buf, p.render(), config.SAMPLE_RATE, format="WAV", subtype="PCM_24")
        out.write_bytes(buf.getvalue())

    return FileResponse(out, filename=out.name, media_type="application/octet-stream")


def main() -> None:
    import uvicorn
    print(f"\n  AudBre  ->  http://{config.HOST}:{config.PORT}")
    print(f"  engine: {config.ENGINE}   model: {config.MODEL}\n")
    uvicorn.run(app, host=config.HOST, port=config.PORT, log_level="info")


if __name__ == "__main__":
    main()

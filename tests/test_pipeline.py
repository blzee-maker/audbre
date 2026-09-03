"""End-to-end pipeline test against a stub engine.

The stub stands in for SAM Audio so CI never needs a GPU, weights or network:
it "hears" a 1 kHz tone as the target and returns the rest as the residual,
which is enough to prove the stack, the chunker and the HTTP layer agree.
"""
from __future__ import annotations

import io
import os
import time

import numpy as np
import pytest
import soundfile as sf

SR = 44100
DURATION = 8.0
TONE_HZ = 1000.0
TONE_FROM, TONE_TO = 3.0, 4.0


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    os.environ["AUDBRE_STORAGE"] = str(tmp_path_factory.mktemp("storage"))
    os.environ["AUDBRE_ENGINE"] = "stub"

    from fastapi.testclient import TestClient

    from audbre import jobs
    from audbre.engine.base import SeparationResult

    class StubEngine:
        """Splits out a 1 kHz tone by naive spectral notching."""

        def separate(self, audio, sample_rate, description, anchors):
            spec = np.fft.rfft(audio)
            freqs = np.fft.rfftfreq(audio.size, 1 / sample_rate)
            band = np.abs(freqs - TONE_HZ) < 25
            tone_spec = np.where(band, spec, 0)
            target = np.fft.irfft(tone_spec, n=audio.size).astype("float32")
            return SeparationResult(target=target, residual=(audio - target).astype("float32"))

    jobs.get_engine = lambda name: StubEngine()

    from audbre.server import app

    return TestClient(app)


@pytest.fixture(scope="module")
def wav_bytes():
    t = np.linspace(0, DURATION, int(SR * DURATION), endpoint=False)
    rng = np.random.default_rng(0)
    speech = 0.25 * np.sin(2 * np.pi * 180 * t) + 0.05 * rng.standard_normal(t.size)
    tone = 0.4 * np.sin(2 * np.pi * TONE_HZ * t)
    tone *= ((t >= TONE_FROM) & (t < TONE_TO)).astype(float)
    buf = io.BytesIO()
    sf.write(buf, (speech + tone).astype("float32"), SR, format="WAV", subtype="FLOAT")
    return buf.getvalue()


def band_energy(data, sr, centre=TONE_HZ, width=25.0):
    spec = np.abs(np.fft.rfft(data))
    freqs = np.fft.rfftfreq(data.size, 1 / sr)
    return float(spec[np.abs(freqs - centre) < width].sum())


def fetch(client, url):
    res = client.get(url)
    assert res.status_code == 200, res.text
    data, sr = sf.read(io.BytesIO(res.content), dtype="float32", always_2d=False)
    return (data if data.ndim == 1 else data.mean(axis=1)), sr


def run_job(client, pid, description, anchors):
    res = client.post(f"/api/projects/{pid}/separate",
                      json={"description": description, "anchors": anchors})
    assert res.status_code == 200, res.text
    job_id = res.json()["id"]
    for _ in range(200):
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("done", "error"):
            break
        time.sleep(0.05)
    assert job["status"] == "done", job
    return job


def test_upload_reports_media_info(client, wav_bytes):
    res = client.post("/api/projects", files={"file": ("take.wav", wav_bytes, "audio/wav")})
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["name"] == "take.wav"
    assert body["duration"] == pytest.approx(DURATION, abs=0.1)
    assert body["has_video"] is False
    assert len(body["peaks"]) > 100
    assert body["layers"] == []
    pytest.pid = body["id"]


def test_separation_isolates_the_tone(client):
    job = run_job(client, pytest.pid, "one kilohertz tone", [["+", TONE_FROM, TONE_TO]])
    assert job["found"] is True
    assert job["target_peak"] > 0.05

    target, sr = fetch(client, f"/api/jobs/{job['id']}/audio/target")
    residual, _ = fetch(client, f"/api/jobs/{job['id']}/audio/residual")
    assert band_energy(target, sr) > band_energy(residual, sr) * 10
    pytest.job_id = job["id"]


def test_keeping_a_layer_removes_the_tone(client):
    before, sr = fetch(client, f"/api/projects/{pytest.pid}/audio/current")
    res = client.post(f"/api/projects/{pytest.pid}/layers", json={"job_id": pytest.job_id})
    assert res.status_code == 200, res.text
    project = res.json()
    assert len(project["layers"]) == 1
    assert project["layers"][0]["enabled"] is True

    after, _ = fetch(client, f"/api/projects/{pytest.pid}/audio/current")
    assert band_energy(after, sr) < band_energy(before, sr) * 0.25
    pytest.layer_id = project["layers"][0]["id"]


def test_muting_a_layer_puts_the_sound_back(client):
    muted = client.patch(
        f"/api/projects/{pytest.pid}/layers/{pytest.layer_id}", json={"enabled": False}
    )
    assert muted.status_code == 200
    assert muted.json()["layers"][0]["enabled"] is False

    restored, sr = fetch(client, f"/api/projects/{pytest.pid}/audio/current")
    original, _ = fetch(client, f"/api/projects/{pytest.pid}/audio/base")
    assert band_energy(restored, sr) > band_energy(original, sr) * 0.8

    client.patch(f"/api/projects/{pytest.pid}/layers/{pytest.layer_id}", json={"enabled": True})


def test_separation_without_a_prompt_is_rejected(client):
    res = client.post(f"/api/projects/{pytest.pid}/separate", json={"description": "", "anchors": []})
    assert res.status_code == 400


def test_unanchored_chunks_pass_through_untouched(client, wav_bytes, monkeypatch):
    """A span in one window must not alter audio in another.

    Chunks are shrunk to 3s so an 8s file spans several of them; the anchor
    sits in the first, so the tail must come back bit-for-bit.
    """
    from audbre import config

    monkeypatch.setattr(config, "CHUNK_SECONDS", 3.0)
    monkeypatch.setattr(config, "CHUNK_OVERLAP", 0.25)

    pid = client.post(
        "/api/projects", files={"file": ("chunked.wav", wav_bytes, "audio/wav")}
    ).json()["id"]
    before, sr = fetch(client, f"/api/projects/{pid}/audio/base")
    job = run_job(client, pid, "tone", [["+", 0.2, 0.4]])
    client.post(f"/api/projects/{pid}/layers", json={"job_id": job["id"]})
    after, _ = fetch(client, f"/api/projects/{pid}/audio/current")

    tail = slice(int(6.0 * sr), int(7.5 * sr))
    assert np.allclose(before[tail], after[tail], atol=1e-4)


def test_export_writes_a_wav(client):
    res = client.get(f"/api/projects/{pytest.pid}/export", params={"format": "wav"})
    assert res.status_code == 200
    data, sr = sf.read(io.BytesIO(res.content), dtype="float32", always_2d=False)
    assert sr == SR
    assert data.size == pytest.approx(SR * DURATION, rel=0.02)


def test_export_rejects_video_for_audio_only_project(client):
    res = client.get(f"/api/projects/{pytest.pid}/export", params={"format": "video"})
    assert res.status_code == 400


def test_missing_project_is_a_404(client):
    assert client.get("/api/projects/nope").status_code == 404

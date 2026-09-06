"""Cloud GPU engine - ships each chunk to a Modal function running SAM Audio."""
from __future__ import annotations

import base64
import io

import httpx
import numpy as np
import soundfile as sf

from .. import config
from .base import Anchor, SeparationResult


def _encode(audio: np.ndarray, sample_rate: int) -> str:
    buf = io.BytesIO()
    sf.write(buf, audio.astype("float32"), sample_rate, format="WAV", subtype="FLOAT")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _decode(payload: str) -> np.ndarray:
    data, _ = sf.read(io.BytesIO(base64.b64decode(payload)), dtype="float32", always_2d=False)
    return data if data.ndim == 1 else data.mean(axis=1)


class ModalEngine:
    def __init__(self, url: str | None = None, timeout: float = 600.0):
        self.url = (url or config.MODAL_URL).rstrip("/")
        if not self.url:
            raise RuntimeError(
                "AUDBRE_MODAL_URL is not set. Run `modal deploy modal_app.py` "
                "and copy the printed URL into your .env"
            )
        self.timeout = timeout

    def separate(
        self,
        audio: np.ndarray,
        sample_rate: int,
        description: str,
        anchors: list[Anchor],
    ) -> SeparationResult:
        payload = {
            "audio": _encode(audio, sample_rate),
            "sample_rate": sample_rate,
            "description": description,
            "anchors": [a.to_api() for a in anchors],
            "model": config.MODEL,
        }
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(f"{self.url}/separate", json=payload)
            if resp.status_code != 200:
                raise RuntimeError(f"modal call failed [{resp.status_code}]: {resp.text[:500]}")
            body = resp.json()

        returned = int(body.get("sample_rate", sample_rate))
        if returned != sample_rate:
            raise RuntimeError(
                f"worker returned {returned} Hz for audio sent at {sample_rate} Hz; "
                f"set AUDBRE sample rate to {returned} - chunk lengths will not "
                f"line up otherwise"
            )

        return SeparationResult(
            target=_decode(body["target"]),
            residual=_decode(body["residual"]),
        )

"""ffmpeg-backed audio I/O: decode anything, chunk it, glue it back together."""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from . import config


class FFmpegError(RuntimeError):
    pass


def _run(args: list[str]) -> str:
    proc = subprocess.run(args, capture_output=True, text=True)
    if proc.returncode != 0:
        raise FFmpegError(proc.stderr.strip()[-2000:])
    return proc.stdout


@dataclass
class MediaInfo:
    duration: float
    sample_rate: int
    channels: int
    has_video: bool


def probe(path: Path) -> MediaInfo:
    raw = _run([
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_format", "-show_streams", str(path),
    ])
    data = json.loads(raw)
    streams = data.get("streams", [])
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if audio is None:
        raise FFmpegError("no audio stream found in this file")
    return MediaInfo(
        duration=float(data.get("format", {}).get("duration", 0.0)),
        sample_rate=int(audio.get("sample_rate", config.SAMPLE_RATE)),
        channels=int(audio.get("channels", 1)),
        has_video=any(s.get("codec_type") == "video" for s in streams),
    )


def to_wav(src: Path, dst: Path, sample_rate: int = config.SAMPLE_RATE) -> Path:
    """Decode any container/codec to a float32 WAV at the model's rate."""
    _run([
        "ffmpeg", "-y", "-v", "error", "-i", str(src),
        "-vn", "-ac", "1", "-ar", str(sample_rate),
        "-c:a", "pcm_f32le", str(dst),
    ])
    return dst


def remux(video_src: Path, audio_src: Path, dst: Path) -> Path:
    """Put edited audio back onto the original video without re-encoding it."""
    _run([
        "ffmpeg", "-y", "-v", "error",
        "-i", str(video_src), "-i", str(audio_src),
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        "-shortest", str(dst),
    ])
    return dst


def encode_mp3(src: Path, dst: Path, bitrate: str = "320k") -> Path:
    _run([
        "ffmpeg", "-y", "-v", "error", "-i", str(src),
        "-c:a", "libmp3lame", "-b:a", bitrate, str(dst),
    ])
    return dst


def read(path: Path) -> tuple[np.ndarray, int]:
    data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data, sr


def write(path: Path, data: np.ndarray, sample_rate: int = config.SAMPLE_RATE) -> Path:
    sf.write(str(path), data.astype("float32"), sample_rate, subtype="FLOAT")
    return path


def peaks(data: np.ndarray, buckets: int = 1600) -> list[float]:
    """Downsample to min/max envelope pairs for waveform drawing."""
    if data.size == 0:
        return []
    buckets = max(1, min(buckets, data.size))
    edges = np.linspace(0, data.size, buckets + 1, dtype=int)
    out: list[float] = []
    for i in range(buckets):
        seg = data[edges[i]:edges[i + 1]]
        out.append(float(np.abs(seg).max()) if seg.size else 0.0)
    return out

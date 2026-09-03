"""Shared engine contract plus the chunked driver that wraps any engine."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np

from .. import config


@dataclass
class Anchor:
    """A time span the user pointed at. sign is '+' (this is the sound) or
    '-' (explicitly not this sound)."""
    sign: str
    start: float
    end: float

    def to_api(self) -> list:
        return [self.sign, round(self.start, 3), round(self.end, 3)]


@dataclass
class SeparationResult:
    target: np.ndarray    # the isolated sound
    residual: np.ndarray  # everything else - i.e. the cleaned track


class Engine(Protocol):
    def separate(
        self,
        audio: np.ndarray,
        sample_rate: int,
        description: str,
        anchors: list[Anchor],
    ) -> SeparationResult: ...


def chunk_bounds(n: int, sample_rate: int) -> list[tuple[int, int]]:
    chunk = int(config.CHUNK_SECONDS * sample_rate)
    overlap = int(config.CHUNK_OVERLAP * sample_rate)
    hop = max(1, chunk - overlap)
    if n <= chunk:
        return [(0, n)]
    bounds, start = [], 0
    while start < n:
        end = min(start + chunk, n)
        bounds.append((start, end))
        if end >= n:
            break
        start += hop
    return bounds


def _overlap_add(
    pieces: list[np.ndarray],
    bounds: list[tuple[int, int]],
    n: int,
    sample_rate: int,
) -> np.ndarray:
    """Cross-fade chunks back together so seams are inaudible."""
    out = np.zeros(n, dtype=np.float64)
    weight = np.zeros(n, dtype=np.float64)
    overlap = int(config.CHUNK_OVERLAP * sample_rate)

    for piece, (start, end) in zip(pieces, bounds):
        span = end - start
        piece = np.asarray(piece, dtype=np.float64)[:span]
        if piece.size < span:
            piece = np.pad(piece, (0, span - piece.size))
        w = np.ones(span)
        fade = min(overlap, span // 2)
        if fade > 0:
            if start > 0:
                w[:fade] = np.linspace(0.0, 1.0, fade)
            if end < n:
                w[-fade:] = np.linspace(1.0, 0.0, fade)
        out[start:end] += piece * w
        weight[start:end] += w

    return (out / np.maximum(weight, 1e-8)).astype("float32")


def _anchors_for(
    anchors: list[Anchor], start_s: float, end_s: float
) -> list[Anchor] | None:
    """Re-base anchors into a chunk's local timeline. Returns None if this
    chunk contains none of them."""
    local = [
        Anchor(a.sign, max(0.0, a.start - start_s), min(end_s - start_s, a.end - start_s))
        for a in anchors
        if a.end > start_s and a.start < end_s
    ]
    return local or None


def run_chunked(
    engine: Engine,
    audio: np.ndarray,
    sample_rate: int,
    description: str,
    anchors: list[Anchor],
    on_progress: Callable[[float], None] | None = None,
) -> SeparationResult:
    """Process a recording of any length in overlapping windows.

    When the user has anchored specific moments, chunks that contain no anchor
    are passed through untouched - both faster and truer to the intent, since
    they pointed at one bark, not every bark in the file.
    """
    n = audio.size
    bounds = chunk_bounds(n, sample_rate)
    targets: list[np.ndarray] = []
    residuals: list[np.ndarray] = []

    for i, (start, end) in enumerate(bounds):
        segment = audio[start:end]
        local = _anchors_for(anchors, start / sample_rate, end / sample_rate) if anchors else []

        if anchors and local is None:
            targets.append(np.zeros_like(segment))
            residuals.append(segment.copy())
        else:
            result = engine.separate(segment, sample_rate, description, local or [])
            targets.append(result.target)
            residuals.append(result.residual)

        if on_progress:
            on_progress((i + 1) / len(bounds))

    return SeparationResult(
        target=_overlap_add(targets, bounds, n, sample_rate),
        residual=_overlap_add(residuals, bounds, n, sample_rate),
    )

"""Local CPU engine. Slow but dependency-complete and fully offline.

On a 2 GB-VRAM laptop GPU the large checkpoint will not fit, so this path
deliberately prefers CPU unless there is real headroom.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from .. import config
from .base import Anchor, SeparationResult

_MIN_VRAM_BYTES = 6 * 1024**3


class LocalEngine:
    def __init__(self, model_id: str | None = None):
        self.model_id = model_id or config.MODEL
        self._model = None
        self._processor = None
        self._device = None

    def _load(self):
        if self._model is not None:
            return
        import torch
        from sam_audio import SAMAudio, SAMAudioProcessor

        use_cuda = False
        if torch.cuda.is_available():
            free, _ = torch.cuda.mem_get_info()
            use_cuda = free >= _MIN_VRAM_BYTES

        self._device = torch.device("cuda" if use_cuda else "cpu")
        self._model = SAMAudio.from_pretrained(self.model_id).to(self._device).eval()
        self._processor = SAMAudioProcessor.from_pretrained(self.model_id)

    def separate(
        self,
        audio: np.ndarray,
        sample_rate: int,
        description: str,
        anchors: list[Anchor],
    ) -> SeparationResult:
        import soundfile as sf
        import torch

        self._load()

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "chunk.wav"
            sf.write(str(path), audio.astype("float32"), sample_rate, subtype="FLOAT")

            kwargs = {"audios": [str(path)], "descriptions": [description]}
            if anchors:
                kwargs["anchors"] = [[a.to_api() for a in anchors]]

            inputs = self._processor(**kwargs).to(self._device)
            with torch.inference_mode():
                result = self._model.separate(
                    inputs, predict_spans=not anchors, reranking_candidates=1
                )

        return SeparationResult(
            target=result.target[0].cpu().numpy().astype("float32"),
            residual=result.residual[0].cpu().numpy().astype("float32"),
        )

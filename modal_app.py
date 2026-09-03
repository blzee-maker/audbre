"""AudBre GPU worker.

Deploy with:
    modal deploy modal_app.py

Modal prints a base URL; paste it into .env as AUDBRE_MODAL_URL.
Requires a Modal secret named `huggingface` holding HF_TOKEN, and an approved
access request on the gated facebook/sam-audio-* repos.
"""
from __future__ import annotations

import modal

MODEL_DEFAULT = "facebook/sam-audio-large"
GPU = "A10G"          # 24 GB - comfortable for sam-audio-large
IDLE_SECONDS = 240    # keep warm between edits so iteration feels instant

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "git")
    .pip_install(
        "torch>=2.5",
        "torchaudio>=2.5",
        "numpy>=1.26",
        "soundfile>=0.12.1",
        "fastapi[standard]",
        "huggingface_hub",
    )
    .pip_install("git+https://github.com/facebookresearch/sam-audio.git")
    .env({"HF_HOME": "/cache"})
)

app = modal.App("audbre")
cache = modal.Volume.from_name("audbre-hf-cache", create_if_missing=True)


@app.cls(
    gpu=GPU,
    image=image,
    volumes={"/cache": cache},
    secrets=[modal.Secret.from_name("huggingface")],
    scaledown_window=IDLE_SECONDS,
    timeout=900,
)
class Separator:
    @modal.enter()
    def load(self):
        import torch
        from sam_audio import SAMAudio, SAMAudioProcessor

        self.torch = torch
        self.model_id = MODEL_DEFAULT
        self.model = SAMAudio.from_pretrained(self.model_id).to("cuda").eval()
        self.processor = SAMAudioProcessor.from_pretrained(self.model_id)
        print(f"loaded {self.model_id} @ {self.processor.audio_sampling_rate} Hz")

    @modal.asgi_app()
    def web(self):
        import base64
        import io
        import tempfile
        from pathlib import Path

        import numpy as np
        import soundfile as sf
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel

        api = FastAPI(title="AudBre GPU worker")

        class Request(BaseModel):
            audio: str                 # base64 WAV
            sample_rate: int
            description: str = ""
            anchors: list = []
            model: str = MODEL_DEFAULT

        def encode(arr) -> str:
            buf = io.BytesIO()
            sf.write(buf, np.asarray(arr, dtype="float32"), self.processor.audio_sampling_rate,
                     format="WAV", subtype="FLOAT")
            return base64.b64encode(buf.getvalue()).decode("ascii")

        @api.get("/health")
        def health():
            return {"ok": True, "model": self.model_id,
                    "sample_rate": self.processor.audio_sampling_rate}

        @api.post("/separate")
        def separate(req: Request):
            if not req.description and not req.anchors:
                raise HTTPException(400, "need a description, an anchor, or both")

            raw = base64.b64decode(req.audio)
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "in.wav"
                path.write_bytes(raw)

                kwargs = {"audios": [str(path)], "descriptions": [req.description]}
                if req.anchors:
                    kwargs["anchors"] = [[list(a) for a in req.anchors]]

                inputs = self.processor(**kwargs).to("cuda")
                with self.torch.inference_mode():
                    # Without anchors we let the model propose spans and rerank,
                    # which measurably improves in-the-wild separation.
                    result = self.model.separate(
                        inputs,
                        predict_spans=not req.anchors,
                        reranking_candidates=8 if not req.anchors else 1,
                    )

            return {
                "target": encode(result.target[0].cpu().numpy()),
                "residual": encode(result.residual[0].cpu().numpy()),
                "sample_rate": self.processor.audio_sampling_rate,
            }

        return api

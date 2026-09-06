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
    # Both pins must come last, after sam-audio has pulled its own tree.
    #
    # protobuf: sam-audio drags in one older than Modal's injected client can
    #   use, and the container crash-loops on boot with
    #   "Enum VolumeFsVersion has no value defined for 'ValueType'".
    #   Capped below 7 so wandb stays satisfiable.
    # huggingface_hub / transformers: these two are one decision, not two.
    #   sam-audio's BaseModel._from_pretrained takes proxies and
    #   resume_download as required kwargs, which only hub 0.x passes — under
    #   hub 1.x loading dies with a TypeError. But sam-audio asks for
    #   transformers>=4.54 with no upper bound, so pip takes 5.x, which
    #   imports is_offline_mode and therefore needs hub>=1.0. Holding
    #   transformers in the 4.x line is what makes hub<1.0 satisfiable.
    .pip_install(
        "protobuf>=5.27,<7",
        "huggingface_hub>=0.26,<1.0",
        "transformers>=4.54,<5",
    )
    .env({"HF_HOME": "/cache"})
)

app = modal.App("audbre")
cache = modal.Volume.from_name("audbre-hf-cache", create_if_missing=True)


def disable_visual_ranker() -> None:
    """Build the model without its ImageBind visual ranker.

    SAMAudio.__init__ always calls create_ranker(cfg.visual_ranker), and for
    these checkpoints that config selects ImageBind — which asserts at import
    time and takes the whole container down with it.

    We prompt with text and timeline spans, never with video frames, so that
    ranker has nothing to rank. create_ranker already treats a missing ranker
    as a supported case (`assert config is None; return None`), so we make the
    ImageBind branch produce that same None rather than patching create_ranker
    itself. Because create_ranker resolves the class as a module global, this
    also covers ImageBind nested inside an EnsembleRankerConfig.
    """
    import sam_audio.ranking as ranking

    ranking.ImageBindRanker = lambda config: None


@app.function(
    image=image,
    volumes={"/cache": cache},
    secrets=[modal.Secret.from_name("huggingface")],
    cpu=4,
    memory=16384,
    timeout=1800,
)
def verify() -> str:
    """Load the model on CPU to prove the dependency chain is sound.

    Run with `modal run modal_app.py::verify` before deploying. Construction is
    what keeps breaking, and it breaks identically on CPU — so this answers the
    question for a fraction of the GPU cost, and without a crash-looping web
    container retrying on expensive hardware.
    """
    from sam_audio import SAMAudio, SAMAudioProcessor

    disable_visual_ranker()

    model = SAMAudio.from_pretrained(MODEL_DEFAULT).eval()
    processor = SAMAudioProcessor.from_pretrained(MODEL_DEFAULT)

    print("=" * 52)
    print("LOADED OK      ", MODEL_DEFAULT)
    print("sample rate    ", processor.audio_sampling_rate)
    print("visual ranker  ", model.visual_ranker)
    print("text ranker    ", type(model.text_ranker).__name__)
    print("span predictor ", type(getattr(model, "span_predictor", None)).__name__)
    print("parameters     ", f"{sum(p.numel() for p in model.parameters()) / 1e6:.0f}M")
    print("=" * 52)
    return "ok"


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

        disable_visual_ranker()

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

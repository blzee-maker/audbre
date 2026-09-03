# AudBre

Point at a sound in a recording and take it out.

A dog barking through an interview, a chair creak in a take, a phone buzzing on
the desk — describe it, mark when it happens, and AudBre lifts it off the
recording and leaves everything else alone.

Built on [SAM Audio](https://github.com/facebookresearch/sam-audio), Meta's
promptable audio separation model.

---

## The idea

Removing a sound is striking a line through it. Every removal is a layer on a
stack, struck out and dimmed; un-check one and the text stands back up as the
sound returns to the mix. Nothing is destructive until you export.

```
current = base − Σ (delta for every enabled layer)
```

Each layer stores the *signal that was taken out*, so muting one costs no GPU
round-trip. Layers are still computed in sequence — each sees the output of the
last — so a stack that's been toggled around can be re-rendered exactly.

## How it runs

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="design/architecture-dark.svg">
    <img
      src="design/architecture-light.svg"
      alt="AudBre architecture. Locally, a browser shows the waveform with a grey ghost of the original, lets you drag to mark a span, audition A/B and manage the removal stack; it exchanges the file, prompt and marked span with a FastAPI service that decodes with ffmpeg, cuts 30-second windows with 2 seconds of overlap, cross-fades them back together and stores layer deltas on disk. Across the boundary in the cloud, a Modal A10G container running sam-audio-large receives a 30-second WAV chunk plus anchors and returns the target and residual."
      width="100%">
  </picture>
</p>

Long recordings are cut into 30-second windows with 2 seconds of overlap and
cross-faded back together, so a 40-minute interview never has to fit in GPU
memory. When you've marked a span, windows that don't contain it are passed
through untouched — faster, and truer to the intent: you pointed at *one* bark,
not every bark in the file.

`result.residual` comes straight from the model, so there's no subtract-and-hope
step and no hole where the sound used to be.

## Quick start

Needs Python 3.11+ and `ffmpeg` on PATH.

```bash
git clone https://github.com/blzee-maker/audbre.git
cd audbre
python -m venv .venv && . .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

**1. Get the model weights.** SAM Audio is open source under the SAM License,
but the weights are gated. Accept the licence on
[facebook/sam-audio-large](https://huggingface.co/facebook/sam-audio-large)
(approval usually lands within half an hour), then `hf auth login`.

**2. Deploy the GPU worker.**

```bash
pip install modal && modal setup
modal secret create huggingface HF_TOKEN=hf_...
modal deploy modal_app.py
```

Paste the URL it prints into `.env` as `AUDBRE_MODAL_URL`.

**3. Run it.**

```bash
python -m audbre.server
```

→ http://127.0.0.1:8000

## Running without a GPU

Set `AUDBRE_ENGINE=local` and install the heavier extras:

```bash
pip install -r requirements-local.txt
```

Inference then happens on your CPU. It works and it's fully offline, but expect
minutes rather than seconds per clip — use `sam-audio-small`. The engine refuses
CUDA below 6 GB of free VRAM, so a small laptop GPU won't be picked up and quietly
OOM.

## Configuration

| Variable | Default | Meaning |
| --- | --- | --- |
| `AUDBRE_ENGINE` | `modal` | `modal` (cloud GPU) or `local` (CPU) |
| `AUDBRE_MODAL_URL` | — | Base URL printed by `modal deploy` |
| `AUDBRE_MODEL` | `facebook/sam-audio-large` | `small`, `base` or `large` |
| `AUDBRE_STORAGE` | `./storage` | Where projects are kept |
| `AUDBRE_HOST` / `AUDBRE_PORT` | `127.0.0.1` / `8000` | Server bind |
| `HF_TOKEN` | — | Needs approved access to the gated repo |

## Layout

```
audbre/
  server.py        HTTP API and static UI
  store.py         projects and the non-destructive layer stack
  jobs.py          background separation jobs
  audio.py         ffmpeg decode / encode / remux, waveform peaks
  engine/
    base.py        engine contract, chunking, cross-faded overlap-add
    remote.py      Modal GPU client
    local.py       CPU fallback
  static/          the front end (no build step, no framework)
modal_app.py       the GPU worker
design/            design canvas sources
tests/             end-to-end pipeline tests
```

## Development

```bash
pip install pytest
pytest -q
```

The suite runs the whole pipeline — upload, separate, keep, mute, export —
against a stub engine that notches a 1 kHz tone, so it needs no GPU, no weights
and no network.

## Licence

The code in this repository is MIT (see [LICENSE](LICENSE)).

SAM Audio, its weights, and anything you produce with them are governed by the
[SAM License](https://github.com/facebookresearch/sam-audio/blob/main/LICENSE),
which is not MIT. Read it before using AudBre commercially.

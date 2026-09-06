# AudBre

Point at a sound in a recording and take it out.

A dog barking through an interview, a chair creak in a take, a phone buzzing on
the desk — describe it, mark when it happens, and AudBre lifts it off the
recording and leaves everything else alone.

Built on [SAM Audio](https://github.com/facebookresearch/sam-audio), Meta's
promptable audio separation model.

> [!IMPORTANT]
> **There is no hosted AudBre, and no shared GPU.** You run it on your own
> hardware, against your own Modal account, paying your own costs. The setup
> below deploys a GPU worker into *your* Modal workspace — nothing in this
> repository points at anyone else's, and no endpoint is provided.
>
> Roughly: a few cents per separation, and nothing at all while idle. See
> [What it costs](#what-it-costs).

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

**2. Deploy your own GPU worker.** This creates a Modal workspace billed to
you. Modal gives new accounts a small credit to start with.

```bash
pip install modal && modal setup

# your Hugging Face token, so the container can pull the gated weights
modal secret create huggingface HF_TOKEN=hf_...

# a shared secret so only you can call your worker - see below
modal secret create audbre-worker AUDBRE_WORKER_TOKEN=$(openssl rand -hex 24)

modal deploy modal_app.py
```

Paste the URL it prints into `.env` as `AUDBRE_MODAL_URL`, and the same token
value as `AUDBRE_WORKER_TOKEN`.

> [!WARNING]
> **Modal web endpoints are public.** Anyone who learns your worker URL can
> POST to it and spend your credits. `/separate` therefore requires an
> `X-AudBre-Token` header matching your secret. Do not commit your `.env`, and
> do not paste your worker URL anywhere public.

## What it costs

Only the GPU costs anything, and only while it is running.

| | |
| --- | --- |
| Idle | **\$0** - the container shuts down after 4 minutes |
| Per separation | roughly **\$0.02-0.05** on an A10G |
| Cold start | ~2.5 minutes of GPU time before the first request ([#7](https://github.com/blzee-maker/audbre/issues/7)) |
| The UI, chunking, ffmpeg, exports | free, all local |

The warm window resets on every request, so a long file processed as several
chunks is one continuous billed run rather than one per chunk.

To spend nothing at all, use the CPU path below.

**3. Run it.**

```bash
python -m audbre.server
```

→ http://127.0.0.1:8000

## Choosing a checkpoint, and the GPU it needs

These checkpoints load in **fp32**, and that is what decides the GPU — not
speed. Measured on a 22 GiB A10G:

| Checkpoint | Weights on GPU | Fits a 22 GiB A10G |
| --- | --- | --- |
| `sam-audio-small` | ~6 GB | yes — the default |
| `sam-audio-base` | 20.6 GB | no, OOMs on a 2-second clip |
| `sam-audio-large` | ~7B params | no |

For `base` or `large`, either move `GPU` in `modal_app.py` to `"A100"` (40 GB)
or cast the weights to bfloat16, which halves the requirement but needs the
input tensors cast to match.

Reranking also costs memory: candidates run in parallel, so `separate()` uses
2 candidates for an unanchored prompt and 1 when you have marked a span.

## Deploying from Windows

Two things bite on a `cp1252` console:

```powershell
$env:PYTHONIOENCODING = "utf-8"   # or modal deploy dies mid-build
modal deploy modal_app.py
```

`modal deploy` **exits 0 even when the build fails**, so check `/health`
rather than the exit code. `/health` returns a `build` stamp for exactly this
reason — if it does not match what you just deployed, a warm container is
still serving the old code and needs `modal app stop audbre --yes` first.

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
| `AUDBRE_MODEL` | `facebook/sam-audio-small` | `small`, `base` or `large` — see GPU memory below |
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

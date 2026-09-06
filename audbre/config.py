"""Runtime configuration, read once from the environment."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATIC = Path(__file__).resolve().parent / "static"


def _load_dotenv() -> None:
    """Minimal .env reader so we don't take a dependency for five keys."""
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip())


_load_dotenv()

STORAGE = Path(os.environ.get("AUDBRE_STORAGE", ROOT / "storage"))

ENGINE = os.environ.get("AUDBRE_ENGINE", "modal").lower()
MODAL_URL = os.environ.get("AUDBRE_MODAL_URL", "").rstrip("/")
MODEL = os.environ.get("AUDBRE_MODEL", "facebook/sam-audio-large")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
# Shared secret for your own GPU worker, so a leaked URL cannot be used
# by anyone else to spend your credits.
WORKER_TOKEN = os.environ.get("AUDBRE_WORKER_TOKEN", "")
HOST = os.environ.get("AUDBRE_HOST", "127.0.0.1")
PORT = int(os.environ.get("AUDBRE_PORT", "8000"))

# SAM Audio operates at a fixed 48 kHz; we resample everything on the way in
# and only convert back at export time. This must match the model, or chunks
# come back a different length than they went out and overlap-add silently
# pads or truncates the difference.
SAMPLE_RATE = 48000

# Long recordings are processed in overlapping windows and cross-faded back
# together, so a 40-minute interview doesn't have to fit in GPU memory.
CHUNK_SECONDS = 30.0
CHUNK_OVERLAP = 2.0

STORAGE.mkdir(exist_ok=True)

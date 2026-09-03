"""Engine selection. All engines share one interface, so the UI never cares
whether inference happened on a cloud GPU or the local CPU."""
from __future__ import annotations

from .base import Anchor, Engine, SeparationResult, run_chunked  # noqa: F401


def get_engine(name: str) -> Engine:
    if name == "modal":
        from .remote import ModalEngine
        return ModalEngine()
    if name == "local":
        from .local import LocalEngine
        return LocalEngine()
    raise ValueError(f"unknown engine {name!r} (expected 'modal' or 'local')")

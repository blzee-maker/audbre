"""Project state and the non-destructive removal stack.

Each removal is stored as a *delta*: the signal that was taken out. The audio
you hear is always

    current = base - sum(delta for every enabled layer)

which means toggling a layer off is instant and needs no GPU round-trip. Layers
are still computed sequentially (each one sees the output of the last), so a
stack that has been toggled can be re-rendered for an exact result.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from . import audio as A
from . import config


def _uid(n: int = 8) -> str:
    return uuid.uuid4().hex[:n]


@dataclass
class Layer:
    id: str
    description: str
    anchors: list                      # [["+", start, end], ...]
    enabled: bool = True
    created_at: float = field(default_factory=time.time)

    def delta_path(self, root: Path) -> Path:
        return root / "layers" / f"{self.id}.delta.wav"

    def target_path(self, root: Path) -> Path:
        return root / "layers" / f"{self.id}.target.wav"


@dataclass
class Project:
    id: str
    name: str
    source: str            # original upload (may be video)
    duration: float
    sample_rate: int
    has_video: bool
    layers: list[Layer] = field(default_factory=list)

    @property
    def root(self) -> Path:
        return config.STORAGE / self.id

    @property
    def base_wav(self) -> Path:
        return self.root / "base.wav"

    @property
    def current_wav(self) -> Path:
        return self.root / "current.wav"

    @property
    def source_path(self) -> Path:
        return self.root / self.source

    # ---------- persistence ----------

    def save(self) -> None:
        data = asdict(self)
        (self.root / "project.json").write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, project_id: str) -> "Project":
        path = config.STORAGE / project_id / "project.json"
        if not path.exists():
            raise KeyError(project_id)
        data = json.loads(path.read_text(encoding="utf-8"))
        data["layers"] = [Layer(**l) for l in data.get("layers", [])]
        return cls(**data)

    # ---------- lifecycle ----------

    @classmethod
    def create(cls, upload_name: str, raw: bytes) -> "Project":
        pid = _uid()
        root = config.STORAGE / pid
        (root / "layers").mkdir(parents=True, exist_ok=True)

        safe = Path(upload_name).name
        src = root / safe
        src.write_bytes(raw)

        info = A.probe(src)
        A.to_wav(src, root / "base.wav")
        data, _ = A.read(root / "base.wav")
        A.write(root / "current.wav", data)

        project = cls(
            id=pid,
            name=safe,
            source=safe,
            duration=info.duration,
            sample_rate=info.sample_rate,
            has_video=info.has_video,
        )
        project.save()
        return project

    # ---------- stack ----------

    def render(self) -> np.ndarray:
        """Rebuild current audio from base minus every enabled delta."""
        base, _ = A.read(self.base_wav)
        out = base.astype("float64")
        for layer in self.layers:
            if not layer.enabled:
                continue
            path = layer.delta_path(self.root)
            if not path.exists():
                continue
            delta, _ = A.read(path)
            n = min(out.size, delta.size)
            out[:n] -= delta[:n]
        return np.clip(out, -1.0, 1.0).astype("float32")

    def commit(self) -> Path:
        A.write(self.current_wav, self.render())
        self.save()
        return self.current_wav

    def add_layer(
        self, description: str, anchors: list, target: np.ndarray, residual: np.ndarray
    ) -> Layer:
        layer = Layer(id=_uid(), description=description, anchors=anchors)
        source = self.render()

        n = min(source.size, residual.size)
        delta = np.zeros_like(source)
        delta[:n] = source[:n] - residual[:n]

        A.write(layer.delta_path(self.root), delta)
        A.write(layer.target_path(self.root), target)

        self.layers.append(layer)
        self.commit()
        return layer

    def find(self, layer_id: str) -> Layer:
        for layer in self.layers:
            if layer.id == layer_id:
                return layer
        raise KeyError(layer_id)

    def remove_layer(self, layer_id: str) -> None:
        layer = self.find(layer_id)
        for path in (layer.delta_path(self.root), layer.target_path(self.root)):
            path.unlink(missing_ok=True)
        self.layers.remove(layer)
        self.commit()


def list_projects() -> list[dict]:
    out = []
    for path in sorted(config.STORAGE.glob("*/project.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            out.append({
                "id": data["id"],
                "name": data["name"],
                "duration": data["duration"],
                "layers": len(data.get("layers", [])),
            })
        except (json.JSONDecodeError, KeyError):
            continue
    return out

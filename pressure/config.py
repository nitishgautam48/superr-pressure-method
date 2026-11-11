"""Load config/settings.yaml (every tunable number) with dotted access and a stable hash."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parents[1]
DEFAULT = REPO / "config" / "settings.yaml"


@dataclass
class Config:
    data: dict[str, Any]
    root: Path = REPO
    source: Path = DEFAULT
    overrides: dict[str, Any] = field(default_factory=dict)

    def get(self, dotted: str, default: Any = None) -> Any:
        if dotted in self.overrides:
            return self.overrides[dotted]
        cur: Any = self.data
        for k in dotted.split("."):
            if not isinstance(cur, dict) or k not in cur:
                return default
            cur = cur[k]
        return cur

    def path(self, key: str) -> Path:
        p = Path(self.get(f"paths.{key}"))
        return p if p.is_absolute() else self.root / p

    def with_overrides(self, **kv: Any) -> "Config":
        """Copy with dotted-key overrides, e.g. cfg.with_overrides(**{"pressure.high_pct": 10})."""
        return Config(self.data, self.root, self.source, {**self.overrides, **kv})

    @property
    def hash(self) -> str:
        blob = json.dumps({"d": self.data, "o": self.overrides}, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:12]


def _merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in (b or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path: str | Path | None = None, root: str | Path | None = None) -> Config:
    """settings.yaml, deep-merged with settings_overrides.yaml (written by the dashboard's Weights Lab
    "Save", every change logged with a reason) when that file exists next to it."""
    p = Path(path) if path else DEFAULT
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    ov = p.parent / "settings_overrides.yaml"
    if ov.exists():
        data = _merge(data, (yaml.safe_load(ov.read_text(encoding="utf-8")) or {}).get("settings") or {})
    return Config(data, Path(root) if root else REPO, p)

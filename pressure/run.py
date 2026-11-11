"""One command rebuilds everything:  python -m pressure.run --config config/settings.yaml"""
from __future__ import annotations

import argparse
import datetime as dt
import importlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from . import store
from .config import Config, load_config

log = logging.getLogger("pressure")
# (name, module with run(ctx)) in order; later sessions append their steps here
STEPS: list[tuple[str, str]] = [
    ("data", "pressure.steps_data"),
    ("pressure", "pressure.steps_pressure"),
    ("rating", "pressure.steps_rating"),
    ("reliability", "pressure.steps_reliability"),
]


@dataclass
class RunContext:
    cfg: Config
    tables: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)


def run(cfg: Config, only: list[str] | None = None, progress=None) -> RunContext:
    """``progress(step_name, index, total)`` is called before each step (dashboard upload page)."""
    ctx = RunContext(cfg)
    for i, (name, mod) in enumerate(STEPS):
        if progress:
            progress(name, i, len(STEPS))
        t = time.time()
        importlib.import_module(mod).run(ctx)
        ctx.timings[name] = round(time.time() - t, 2)
        log.info("step %-12s done in %.1fs", name, ctx.timings[name])
        if only and name == only[-1]:
            break
    store.write_json({"source": "real", "run_at": dt.datetime.now().isoformat(timespec="seconds"),
                      "config_hash": cfg.hash, "row_counts": {k: int(len(v)) for k, v in ctx.tables.items()
                                                              if hasattr(v, "__len__") and not isinstance(v, dict)},
                      "metrics": ctx.metrics, "timings_s": ctx.timings, "steps": [s for s, _ in STEPS]},
                     "run_meta.json", cfg.path("processed"))
    return ctx


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="RR Pressure Index pipeline")
    ap.add_argument("--config", default=None)
    ap.add_argument("--until", default=None, help="stop after this step")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ctx = run(load_config(a.config), [a.until] if a.until else None)
    print({k: v for k, v in ctx.metrics.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Write/read data/processed (the dashboard contract) with a schema version and required-column checks."""
from __future__ import annotations

import datetime as dt
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCHEMA_VERSION = "2.0.0"
REQUIRED: dict[str, tuple[str, ...]] = {
    "match_validation.csv": ("match_id", "recon_winner", "actual_winner", "winner_match", "margin_error"),
    "balls_enriched.parquet": ("match_id", "innings", "wp_before", "wp_after", "wpa", "leverage",
                               "pressure_index", "is_pressure", "xruns", "p_out"),
    "batters_rated.parquet": ("player_id", "name", "A", "B", "C", "prs", "rank", "eligible"),
    "bowlers_rated.parquet": ("player_id", "name", "A", "B", "C", "prs", "rank", "eligible"),
    "player_match_pressure.parquet": ("player_id", "match_id", "role", "pressure_balls", "value"),
}


class SchemaError(ValueError):
    pass


def jsonable(o: Any) -> Any:
    if o is None or o is pd.NA or o is pd.NaT:
        return None
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set, np.ndarray)):
        return [jsonable(v) for v in (o.tolist() if isinstance(o, np.ndarray) else o)]
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return None if math.isnan(f) or math.isinf(f) else f
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, (pd.Timestamp, dt.datetime, dt.date)):
        return o.isoformat()
    return o


def _check(name: str, df: pd.DataFrame) -> None:
    need = REQUIRED.get(name, ())
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise SchemaError(f"{name}: missing columns {miss}")


def write_table(df: pd.DataFrame, name: str, base: Path) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    _check(name, df)
    p = base / name
    if name.endswith(".csv"):
        df.to_csv(p, index=False)
    else:
        df.reset_index(drop=True).to_parquet(p, index=False)
    return p


def write_json(obj: dict[str, Any], name: str, base: Path) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    p = base / name
    p.write_text(json.dumps(jsonable({**obj, "schema_version": SCHEMA_VERSION}), indent=2), encoding="utf-8")
    return p


def read_table(name: str, base: Path) -> pd.DataFrame:
    p = base / name
    if not p.exists():
        raise FileNotFoundError(f"{p} not found - run: python -m pressure.run")
    return pd.read_csv(p) if name.endswith(".csv") else pd.read_parquet(p)


def read_json(name: str, base: Path) -> dict[str, Any]:
    p = base / name
    if not p.exists():
        raise FileNotFoundError(f"{p} not found - run: python -m pressure.run")
    return json.loads(p.read_text(encoding="utf-8"))

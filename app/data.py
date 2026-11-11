"""Dashboard data layer: reads ONLY data/processed/ (cached by file mtime) and calls the engine for live
re-rating (Weights Lab, Rankings filters, Pressure Lab threshold). Nothing leaves the machine."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from pressure.config import load_config  # noqa: E402

_override: Path | None = None
_paths: dict[str, str] = {}
RATE_BALL_COLS = ["match_id", "innings", "striker_id", "bowler_id", "is_faced", "is_legal", "reduced_overs", "season",
                  "competition_group", "bat_tier", "bowl_tier", "wpa", "rae_bat", "rae_conc", "is_pressure",
                  "is_pressure_rule", "pressure_index"]


def set_processed_dir(p: str | Path | None) -> None:
    global _override
    _override = Path(p) if p else None
    st.cache_data.clear()


SHOWCASE = REPO / "showcase"


def upload_dir() -> Path | None:
    """Temporary folder of THIS browser session's uploaded-pack run (None if not using one)."""
    try:
        d = st.session_state.get("upload_dir")
    except Exception:  # noqa: BLE001 - outside a Streamlit session
        d = None
    return Path(d) if d and (Path(d) / "processed" / "batters_rated.parquet").exists() else None


def upload_paths(root: Path) -> dict[str, str]:
    return {"paths.raw": str(root / "raw"), "paths.processed": str(root / "processed"),
            "paths.outputs_public": str(root / "public"), "paths.outputs_private": str(root / "private"),
            "paths.decision_log": str(root / "private" / "decision_log.csv"),
            "paths.overrides": str(root / "analyst_overrides.yaml")}


def showcase_mode() -> bool:
    """Read-only hosted mode: no full outputs on disk, only the aggregates-only showcase/ export."""
    if _override or "paths.processed" in _paths or upload_dir():
        return False
    full = load_config().path("processed") / "batters_rated.parquet"
    return not full.exists() and (SHOWCASE / "batters_rated.parquet").exists()


def set_paths(**paths: str) -> None:
    """Redirect any config path (tests: processed, overrides, decision_log, outputs_public, outputs_private)."""
    _paths.clear()
    _paths.update({f"paths.{k}": str(v) for k, v in paths.items()})
    st.cache_data.clear()


def cfg():
    c = load_config()
    ov = dict(_paths)
    up = upload_dir()
    if up:
        ov.update(upload_paths(up))
    elif _override:
        ov["paths.processed"] = str(_override)
    elif showcase_mode():
        ov.update({"paths.processed": str(SHOWCASE), "paths.outputs_public": str(SHOWCASE / "figures"),
                   "paths.outputs_private": str(SHOWCASE)})
    return c.with_overrides(**ov) if ov else c


def processed() -> Path:
    return cfg().path("processed")


def available() -> bool:
    return (processed() / "batters_rated.parquet").exists()


def _mtime(name: str) -> float:
    p = processed() / name
    return p.stat().st_mtime if p.exists() else 0.0


@st.cache_data(show_spinner=False)
def _parquet(base: str, name: str, mtime: float, columns: tuple | None = None) -> pd.DataFrame:
    return pd.read_parquet(Path(base) / name, columns=list(columns) if columns else None)


@st.cache_data(show_spinner=False)
def _json(base: str, name: str, mtime: float) -> dict:
    p = Path(base) / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def table(name: str, columns: list[str] | None = None) -> pd.DataFrame:
    return _parquet(str(processed()), name, _mtime(name), tuple(columns) if columns else None)


def js(name: str) -> dict:
    return _json(str(processed()), name, _mtime(name))


def rated(role: str) -> pd.DataFrame:
    return table(f"{role}s_rated.parquet")


def validation() -> pd.DataFrame:
    p = processed() / "match_validation.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def rate_balls() -> pd.DataFrame:
    return table("balls_enriched.parquet", RATE_BALL_COLS)


@st.cache_data(show_spinner=False, max_entries=64)
def _rerate(base: str, mtime: float, role: str, overrides_json: str, high_pct: float, definition: str,
            competition: str) -> pd.DataFrame:
    from pressure import pressure as PR, rating as RT

    c = cfg()
    s = rate_balls().copy()
    if competition == "IPL":
        s = s[s["competition_group"] == "IPL"]
    elif competition == "T20I":
        s = s[s["competition_group"] == "T20I"]
    col = "is_pressure"
    if definition == "rule":
        col = "is_pressure_rule"
    elif abs(high_pct - float(c.get("pressure.high_pct", 20))) > 1e-9:
        s["_p"] = PR.classify(s[["pressure_index"]], c, high_pct=high_pct)["is_pressure"].to_numpy()
        col = "_p"
    r = RT.rate(s, role, c, json.loads(overrides_json) or None, col)
    r.index.name = "player_id"
    names = rated(role).set_index("player_id")[["name", "display", "team"]]
    return r.join(names, how="left").reset_index()


def rerate(role: str, overrides: dict | None = None, high_pct: float | None = None, definition: str = "leverage",
           competition: str = "both") -> pd.DataFrame:
    hp = float(high_pct if high_pct is not None else cfg().get("pressure.high_pct", 20))
    return _rerate(str(processed()), _mtime("balls_enriched.parquet"), role, json.dumps(overrides or {}, sort_keys=True),
                   hp, definition, competition)


def public_mode() -> bool:
    return bool(load_config().get("app.public_mode", False))


def read_only_note(what: str) -> None:
    st.info(f"{what} needs ball-by-ball data, which is never hosted. It works in the local version "
            "(see README); this online showcase holds aggregates only.", icon=":material/lock:")


def label_tone(label: Any) -> tuple[str, str]:
    return {"Robust": ("gold", "★"), "Likely": ("blue", "◆"), "Sensitive": ("neutral", "○")}.get(label, ("neutral", "–"))

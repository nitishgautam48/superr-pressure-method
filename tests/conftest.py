"""Shared fixtures. Real-data tests skip when data/raw is absent (the pack is never committed)."""
from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import pytest

from pressure.config import load_config

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "data" / "raw"


def make_cfg(root: Path):
    """Config whose outputs go to a temp tree (raw data is read from the repo)."""
    cfg = load_config()
    (root / "config").mkdir(parents=True, exist_ok=True)
    for f in ("analyst_overrides.yaml", "rr_history.yaml"):
        src = REPO / "config" / f
        if src.exists():
            shutil.copy(src, root / "config" / f)
    return cfg.with_overrides(**{"paths.raw": str(RAW), "paths.processed": str(root / "processed"),
                                 "paths.outputs_private": str(root / "private"),
                                 "paths.outputs_public": str(root / "public"),
                                 "paths.decision_log": str(root / "private" / "decision_log.csv"),
                                 "paths.overrides": str(root / "config" / "analyst_overrides.yaml"),
                                 "paths.rr_history": str(root / "config" / "rr_history.yaml")})


needs_data = pytest.mark.skipif(not (RAW / "ball_by_ball.csv").exists(), reason="data pack not present")


@pytest.fixture(scope="session")
def cfg(tmp_path_factory):
    return make_cfg(tmp_path_factory.mktemp("pressure"))


@pytest.fixture(scope="session")
def s1(cfg):
    if not (RAW / "ball_by_ball.csv").exists():
        pytest.skip("data pack not present")
    from pressure.run import RunContext
    from pressure import steps_data

    ctx = RunContext(cfg)
    steps_data.run(ctx)
    return ctx


def ball(**kw):
    base = dict(match_id=1, innings=1, overid=1, over_ball_num=1, innings_ball_num=1, batting_team_id=10,
                bowling_team_id=20, striker_id=1, non_striker_id=2, bowler_id=3, runs=0, is_four=False,
                is_six=False, is_wicket=False, dismissal_type=None, extras_type=None, extras1_runs=6,
                extras_2_runs=0)
    base.update(kw)
    return base

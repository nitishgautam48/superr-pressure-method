"""S3 step: expected outcomes -> metrics -> ratings A/B/C -> eligibility -> model top 10s; writes the contract."""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from . import expected as X, metrics as M, names, rating as RT, store
from .decisions import log_autopilot_default

log = logging.getLogger(__name__)
BAT_COLS = ["balls", "runs", "sr", "dismissals", "balls_per_dismissal", "dot_pct", "boundary_pct", "rae_per100",
            "wpa_per100", "innings"]
BOWL_COLS = ["legal_balls", "runs_conceded", "econ", "dot_pct", "boundary_pct_conceded", "wickets", "balls_per_wicket",
             "rae_conc_per100", "wae_per100", "wpa_per100", "matches"]
BALL_COLS = ["match_id", "innings", "overid", "over_ball_num", "innings_ball_num", "batting_team_id", "bowling_team_id",
             "striker_id", "striker_name", "bowler_id", "bowler_name", "runs", "team_runs", "bowler_runs", "is_legal",
             "is_faced", "is_four", "is_six", "is_team_wicket", "is_bowler_wicket", "striker_out", "dismissal_type",
             "score_before", "wickets_before", "balls_remaining", "phase", "crr", "target", "runs_needed", "rrr",
             "competition_group", "season", "comp_name", "knock_out", "bat_tier", "bowl_tier", "pace_or_spin",
             "batting_type", "reduced_overs", "wp_before", "wp_after", "wpa", "leverage", "pressure_index_raw",
             "ko_multiplier", "pressure_index", "is_pressure", "pressure_band", "is_pressure_rule", "xruns", "p_out",
             "xruns_conc", "p_wkt", "rae_bat", "rae_conc", "wae"]


def rated_table(s: pd.DataFrame, role: str, cfg: Any, directory: pd.DataFrame, overrides=None,
                pressure_col: str = "is_pressure") -> pd.DataFrame:
    r = RT.rate(s, role, cfg, overrides, pressure_col)
    d = s[~s["reduced_overs"]].copy()
    d["is_pressure"] = d[pressure_col].astype(bool)
    if role == "batter":
        t = M.batting_table(d, ["striker_id", "is_pressure"]).rename(columns={"striker_id": "player_id"})
        wide = M.pressure_split(t, "player_id", BAT_COLS)
    else:
        t = M.bowling_table(d, ["bowler_id", "is_pressure"]).rename(columns={"bowler_id": "player_id"})
        wide = M.pressure_split(t, "player_id", BOWL_COLS)
    out = r.join(wide, how="left").join(directory, how="left")
    out.index.name = "player_id"
    out = out.reset_index()
    out["name"] = out["name"].fillna(out["player_id"].astype(str))
    out["role"] = role
    return out


def run(ctx: Any) -> None:
    cfg, t = ctx.cfg, ctx.tables
    s = X.add_expected(t["state"], cfg)
    directory = names.player_directory(t["raw"]["competition_team_player_mapping"], t["comps"])
    base = cfg.path("processed")
    rated = {}
    for role in ("batter", "bowler"):
        rated[role] = rated_table(s, role, cfg, directory)
        store.write_table(rated[role], f"{role}s_rated.parquet", base)
    # per player-match pressure value (consistency charts)
    pm = []
    for role, (pid_col, faced) in RT.ROLE_COLS.items():
        d = s[~s["reduced_overs"]]
        d = d[d[faced]] if faced else d
        pm.append(RT.player_match(d, RT.ball_values(d, role, cfg), RT.ball_weights(d, cfg), d["is_pressure"].astype(bool),
                                  pid_col, role))
    pmt = pd.concat(pm, ignore_index=True)
    store.write_table(pmt, "player_match_pressure.parquet", base)
    sp = M.splits(s[~s["reduced_overs"]])
    store.write_table(sp, "player_splits.parquet", base)
    store.write_table(s[[c for c in BALL_COLS if c in s]], "balls_enriched.parquet", base)

    top_n, next_n = int(cfg.get("rating.top_n", 10)), int(cfg.get("rating.next_n", 10))
    top = {}
    for role, r in rated.items():
        e = r[r["eligible"]].sort_values("rank")
        cols = ["rank", "player_id", "name", "display", "prs", "A", "B", "C", "pressure_balls", "pressure_units",
                "total_balls", "A_se"]
        top[f"{role}s_top"] = e.head(top_n)[cols].to_dict("records")
        top[f"{role}s_next"] = e.iloc[top_n: top_n + next_n][cols].to_dict("records")
        top[f"{role}s_eligible"] = int(len(e))
    store.write_json(top, "model_top10.json", base)
    for k in ("rating.weights", "rating.value_blend", "rating.shrink_k_balls", "rating.associate_weight",
              "rating.emerging_weight", "rating.recency_on", "rating.eligibility"):
        log_autopilot_default(cfg, k, cfg.get(k))
    t.update(state=s, batters=rated["batter"], bowlers=rated["bowler"], player_match=pmt, splits=sp,
             directory=directory)
    ctx.metrics.update(eligible_batters=top["batters_eligible"], eligible_bowlers=top["bowlers_eligible"])
    log.info("S3: %d eligible batters, %d eligible bowlers", top["batters_eligible"], top["bowlers_eligible"])

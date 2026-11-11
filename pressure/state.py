"""Match state per ball, regular innings 1-2 only (super overs excluded).

All "before" values describe the situation as the ball is bowled; "after" values the situation it leaves.
Scores are reconstructed (approximate, see reconstruct.py); wickets and legal balls are exact.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def phase_of(over: pd.Series, cfg: Any) -> pd.Series:
    pp, mid = cfg.get("phases.powerplay"), cfg.get("phases.middle")
    return pd.Series(np.select([over <= pp[1], over <= mid[1]], ["PP", "MID"], "DEATH"), index=over.index)


def build_state(balls: pd.DataFrame, matches: pd.DataFrame, validation: pd.DataFrame, tiers: pd.DataFrame,
                cfg: Any) -> pd.DataFrame:
    b = balls[~balls["is_super_over"]].copy()
    g = b.groupby(["match_id", "innings"], sort=False)
    b["score_after"] = g["team_runs"].cumsum()
    b["score_before"] = b["score_after"] - b["team_runs"]
    b["wickets_after"] = g["is_team_wicket"].cumsum().astype(int)
    b["wickets_before"] = b["wickets_after"] - b["is_team_wicket"].astype(int)
    b["legal_after"] = g["is_legal"].cumsum().astype(int)
    b["legal_before"] = b["legal_after"] - b["is_legal"].astype(int)

    m = matches.set_index("match_id")
    v = validation.set_index("match_id")
    allot1 = m["first_inn_revised_overs"].fillna(120)
    allot2 = m["second_inn_revised_overs"].fillna(allot1)
    b["allotted"] = np.where(b["innings"] == 1, b["match_id"].map(allot1), b["match_id"].map(allot2)).astype(float)
    b["allotted2"] = b["match_id"].map(allot2).astype(float)
    b["reduced_overs"] = b["match_id"].map((allot1 < 120) | (allot2 < 120)).astype(bool)
    b["balls_remaining"] = (b["allotted"] - b["legal_before"]).clip(lower=0)
    b["balls_remaining_after"] = (b["allotted"] - b["legal_after"]).clip(lower=0)
    b["wickets_in_hand"] = 10 - b["wickets_before"]
    b["phase"] = phase_of(b["overid"], cfg)
    b["crr"] = np.where(b["legal_before"] > 0, 6 * b["score_before"] / b["legal_before"].clip(lower=1), np.nan)

    tgt = b["match_id"].map(v["target"])
    inn2 = b["innings"] == 2
    b["target"] = np.where(inn2, tgt, np.nan)
    b["runs_needed"] = np.where(inn2, b["target"] - b["score_before"], np.nan)
    b["runs_needed_after"] = np.where(inn2, b["target"] - b["score_after"], np.nan)
    b["rrr"] = np.where(inn2, 6 * b["runs_needed"] / b["balls_remaining"].clip(lower=1), np.nan)
    b["rrr_minus_crr"] = b["rrr"] - b["crr"]

    for c in ("competition_group", "season_year", "stage", "knock_out", "venue_country", "comp_id", "comp_name",
              "home_team_id", "dls", "tied", "super_over", "no_result", "abandoned", "win_team_id"):
        b[c] = b["match_id"].map(m[c])
    tmap = tiers.set_index(["comp_id", "team_id"])["tier"]
    b["bat_tier"] = [tmap.get((c, t), "ASSOCIATE") for c, t in zip(b["comp_id"], b["batting_team_id"])]
    b["bowl_tier"] = [tmap.get((c, t), "ASSOCIATE") for c, t in zip(b["comp_id"], b["bowling_team_id"])]
    b["is_home"] = b["batting_team_id"] == b["home_team_id"]
    b["season"] = b["season_year"].astype(int)
    b["innings_end"] = g.cumcount(ascending=False) == 0
    return b.reset_index(drop=True)


def outcome_label(b: pd.DataFrame) -> pd.Series:
    """Batting team result: 1 win, 0 loss, 0.5 tie (regulation tie incl. super-over games), NaN no result."""
    tie = b["tied"].astype(bool)
    res = np.where(b["no_result"].astype(bool) | b["abandoned"].astype(bool) | b["win_team_id"].isna(), np.nan,
                   (b["win_team_id"] == b["batting_team_id"]).astype(float))
    return pd.Series(np.where(tie, 0.5, res), index=b.index)

"""Expected-outcome baselines: context-fair expected runs and dismissal / wicket probability.

Binned averages with hierarchical smoothing: each finer bin is shrunk toward its parent bin with ``k``
pseudo-balls, so sparse contexts fall back to broader ones:

  global -> innings x phase -> + wickets -> + RRR bucket -> + pace/spin x batting hand
         -> + competition group x opposition tier

Batting rows are faced balls (target: bat runs, striker dismissed); bowling rows are all deliveries
(target: runs charged to the bowler, bowler wicket). RAE = actual - expected (runs above expected),
WAE = wickets - expected wickets.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _rrr_bucket(s: pd.DataFrame, cfg: Any) -> pd.Series:
    edges = cfg.get("expected.rrr_buckets", [0, 6, 8, 10, 12, 99])
    b = pd.cut(s["rrr"].clip(lower=0, upper=edges[-1] - 0.01), edges, include_lowest=True).astype(str)
    return b.where(s["innings"] == 2, "inn1")


def bowl_kind(bowling_type: pd.Series, pace_or_spin: pd.Series) -> pd.Series:
    t = bowling_type.fillna("").str.lower()
    wrist = t.str.contains("leg-break|legbreak|chinaman|googly")
    finger = t.str.contains("offbreak|off-break|orthodox")
    left = t.str.startswith("left")
    return pd.Series(np.select([wrist, finger, (pace_or_spin == "Pace") & left, pace_or_spin == "Pace"],
                               ["wrist", "finger", "pace_L", "pace_R"], "spin"), index=bowling_type.index)


def opposition_strength(s: pd.DataFrame, team_col: str, cfg: Any) -> pd.Series:
    """Tercile (strong / avg / weak) of a team-season's runs per legal ball in its competition, shrunk.
    For batters pass the bowling team (attack strength: fewer runs conceded = strong); for bowlers the batting
    team (batting strength: more runs scored = strong)."""
    k = float(cfg.get("expected.opposition_shrink_balls", 600))
    key = [team_col, "season", "competition_group"]
    leg = s[s["is_legal"]]
    g = leg.groupby(key)["team_runs"].agg(["sum", "count"])
    comp = leg.groupby(["season", "competition_group"])["team_runs"].mean()
    mu = comp.reindex(pd.MultiIndex.from_arrays([g.index.get_level_values(1), g.index.get_level_values(2)])).to_numpy()
    rate = (g["sum"] + k * mu) / (g["count"] + k) / mu
    good_is_low = team_col == "bowling_team_id"
    score = -rate if good_is_low else rate
    tert = score.groupby(level=2).transform(lambda x: pd.qcut(x.rank(method="first"), 3, labels=["weak", "avg", "strong"]))
    return pd.Series(pd.MultiIndex.from_frame(s[key]).map(tert.astype(str).to_dict()), index=s.index).fillna("avg")


def context_levels(s: pd.DataFrame, cfg: Any, opp_col: str, strength: pd.Series | None = None) -> list[pd.Series]:
    ctx = cfg.get("expected.context", {}) or {}
    wb = pd.cut(s["wickets_before"], [-1, 2, 5, 11], labels=["0-2", "3-5", "6+"]).astype(str)
    l1 = s["innings"].astype(str) + "|" + s["phase"].astype(str)
    if ctx.get("era"):
        l1 = l1 + "|" + s["competition_group"].astype(str) + str("-") + s["season"].astype(str)
    l2 = l1 + "|" + wb
    l3 = l2 + "|" + _rrr_bucket(s, cfg)
    kind = bowl_kind(s["bowling_type"], s["pace_or_spin"]) if ctx.get("bowl_kind") and "bowling_type" in s \
        else s["pace_or_spin"].astype(str)
    l4 = l3 + "|" + kind + "-" + s["batting_type"].astype(str)
    l5 = l4 + "|" + s["competition_group"].astype(str) + "-" + s[opp_col].astype(str)
    levels = [l1, l2, l3, l4, l5]
    if ctx.get("opposition_strength") and strength is not None:
        levels.append(l5 + "|" + strength.astype(str))
    return levels


def smoothed_expectation(y: pd.Series, levels: list[pd.Series], k: float) -> np.ndarray:
    m = np.full(len(y), float(y.mean()))
    yv = y.astype(float)
    for key in levels:
        g = yv.groupby(key.values)
        s_sum = key.map(g.sum()).to_numpy(dtype=float)
        n = key.map(g.count()).to_numpy(dtype=float)
        m = (s_sum + k * m) / (n + k)
    return m


def add_expected(s: pd.DataFrame, cfg: Any) -> pd.DataFrame:
    s = s.copy()
    k = float(cfg.get("expected.smoothing_k", 200))
    fit = ~s["reduced_overs"]
    # batting: faced balls
    bat = fit & s["is_faced"]
    ctx = cfg.get("expected.context", {}) or {}
    att = opposition_strength(s, "bowling_team_id", cfg) if ctx.get("opposition_strength") else None
    bst = opposition_strength(s, "batting_team_id", cfg) if ctx.get("opposition_strength") else None
    s["opp_attack_strength"] = att if att is not None else "n/a"
    s["opp_batting_strength"] = bst if bst is not None else "n/a"
    lv = context_levels(s[bat], cfg, "bowl_tier", att[bat] if att is not None else None)
    s["striker_out"] = s["is_batter_out"] & (s["dismissed_player_id"] == s["striker_id"])
    s["xruns"] = np.nan
    s["p_out"] = np.nan
    s.loc[bat, "xruns"] = smoothed_expectation(s.loc[bat, "runs"], lv, k)
    s.loc[bat, "p_out"] = smoothed_expectation(s.loc[bat, "striker_out"].astype(float), lv, k)
    # bowling: every delivery (wides / no-balls are on the bowler)
    bowl = fit
    lvb = context_levels(s[bowl], cfg, "bat_tier", bst[bowl] if bst is not None else None)
    s["xruns_conc"] = np.nan
    s["p_wkt"] = np.nan
    s.loc[bowl, "xruns_conc"] = smoothed_expectation(s.loc[bowl, "bowler_runs"], lvb, k)
    s.loc[bowl, "p_wkt"] = smoothed_expectation(s.loc[bowl, "is_bowler_wicket"].astype(float), lvb, k)
    s["rae_bat"] = s["runs"] - s["xruns"]
    s["rae_conc"] = s["bowler_runs"] - s["xruns_conc"]
    s["wae"] = s["is_bowler_wicket"].astype(float) - s["p_wkt"]
    return s

"""Pressure index.

leverage = sum over outcomes o in {dot, 1, 2, 4, 6, wicket} of P(o | innings, phase, wickets) *
           |WP(after o) - WP(now)|
i.e. the expected absolute swing in win probability from the state. Normalised so 1.0 = an average ball;
a knockout multiplier (config) is applied on top and kept as a separate column. High-pressure = top X%.
A transparent rule-based definition is built alongside for comparison.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from .wp_model import wp_after_state

log = logging.getLogger(__name__)
OUTCOME_RUNS = {"dot": 0, "1": 1, "2": 2, "4": 4, "6": 6, "wicket": 0}


def wkt_bucket(w: pd.Series) -> pd.Series:
    return pd.cut(w, [-1, 2, 5, 11], labels=["0-2", "3-5", "6+"]).astype(str)


def outcome_of(b: pd.DataFrame) -> pd.Series:
    r = b["runs"].clip(upper=6)
    lab = np.select([b["is_team_wicket"], r == 0, r == 1, r.isin([2, 3]), r.isin([4, 5])],
                    ["wicket", "dot", "1", "2", "4"], "6")
    return pd.Series(lab, index=b.index)


def outcome_probs(s: pd.DataFrame) -> pd.DataFrame:
    """P(outcome | innings, phase, wicket bucket) from legal balls (rows: context, cols: outcomes)."""
    leg = s[s["is_legal"]].copy()
    leg["o"] = outcome_of(leg)
    leg["wb"] = wkt_bucket(leg["wickets_before"])
    tab = pd.crosstab([leg["innings"], leg["phase"], leg["wb"]], leg["o"], normalize="index")
    return tab.reindex(columns=list(OUTCOME_RUNS), fill_value=0.0)


def add_leverage(s: pd.DataFrame, models: dict, cfg: Any) -> tuple[pd.DataFrame, pd.DataFrame]:
    s = s.copy()
    probs = outcome_probs(s)
    key = pd.MultiIndex.from_arrays([s["innings"], s["phase"], wkt_bucket(s["wickets_before"])])
    P = probs.reindex(key).fillna(0.0).to_numpy()
    lev = np.zeros(len(s))
    legal_after = s["legal_before"] + 1
    for j, (o, r) in enumerate(OUTCOME_RUNS.items()):
        wp_o = wp_after_state(s, models, score_after=s["score_before"] + r,
                              wickets_after=s["wickets_before"] + (1 if o == "wicket" else 0), legal_after=legal_after)
        lev += P[:, j] * np.abs(wp_o - s["wp_before"].to_numpy())
    s["leverage"] = lev
    if cfg.get("pressure.normalise_by", "innings") == "innings":
        mean = s.groupby("innings")["leverage"].transform("mean").to_numpy()
    else:
        mean = float(np.nanmean(lev))
    s["pressure_index_raw"] = lev / mean
    mult = float(cfg.get("pressure.knockout_multiplier", 1.0))
    s["ko_multiplier"] = np.where(s["knock_out"].astype(bool), mult, 1.0)
    s["pressure_index"] = s["pressure_index_raw"] * s["ko_multiplier"]
    return s, probs


def classify(s: pd.DataFrame, cfg: Any, high_pct: float | None = None) -> pd.DataFrame:
    """is_pressure (top X% of pressure_index) and a low / medium / high band."""
    s = s.copy()
    hp = float(high_pct if high_pct is not None else cfg.get("pressure.high_pct", 20))
    thr = float(np.nanpercentile(s["pressure_index"], 100 - hp))
    s["is_pressure"] = s["pressure_index"] >= thr
    lo, hi = cfg.get("pressure.bands_pct", [50, 80])
    q_lo, q_hi = np.nanpercentile(s["pressure_index"], [lo, hi])
    s["pressure_band"] = np.select([s["pressure_index"] >= q_hi, s["pressure_index"] >= q_lo], ["high", "medium"], "low")
    s.attrs["pressure_threshold"] = thr
    return s


def rule_based(s: pd.DataFrame, cfg: Any) -> pd.Series:
    r = cfg.get("pressure.rules")
    inn2 = s["innings"] == 2
    chase = inn2 & (s["rrr_minus_crr"] >= r["chase_rrr_minus_crr"]) & (s["balls_remaining"] <= r["chase_balls_remaining_max"])
    collapse = (s["wickets_before"] >= r["collapse_wickets_min"]) & (s["balls_remaining"] >= r["collapse_balls_remaining_min"])
    par = s["balls_remaining"] * s["crr"].fillna(0) / 6
    close = inn2 & (s["phase"] == "DEATH") & ((s["runs_needed"] - par).abs() <= r["close_chase_death_margin_runs"])
    ko = s["knock_out"].astype(bool) & (s["phase"] == "DEATH") & bool(r.get("knockout_death", True))
    return (chase | collapse | close | ko).fillna(False)


def league_stats(s: pd.DataFrame, mask: pd.Series) -> dict[str, float]:
    d = s[mask]
    leg = d[d["is_legal"]]
    faced = d[d["is_faced"]]
    return {"balls": int(len(d)), "share_pct": round(100 * len(d) / max(len(s), 1), 2),
            "dot_pct": round(100 * float((leg["team_runs"] == 0).mean()), 2) if len(leg) else None,
            "boundary_pct": round(100 * float((faced["is_four"] | faced["is_six"]).mean()), 2) if len(faced) else None,
            "wicket_rate_pct": round(100 * float(leg["is_team_wicket"].mean()), 2) if len(leg) else None,
            "run_rate": round(6 * float(d["team_runs"].sum()) / max(int(d["is_legal"].sum()), 1), 2),
            "mean_abs_wpa": round(float(d["wpa"].abs().mean()), 4),
            "mean_pressure_index": round(float(d["pressure_index"].mean()), 3)}


def situation_profile(s: pd.DataFrame, a: pd.Series, b: pd.Series) -> dict[str, dict[str, float]]:
    """How the situations themselves differ: WP closeness, required rate gap, wickets, phase mix."""
    def prof(d: pd.DataFrame) -> dict[str, float]:
        i2 = d[d["innings"] == 2]
        return {"wp_within_0_25_0_75_pct": round(100 * float(d["wp_before"].between(0.25, 0.75).mean()), 1),
                "mean_wickets_before": round(float(d["wickets_before"].mean()), 2),
                "death_overs_pct": round(100 * float((d["phase"] == "DEATH").mean()), 1),
                "innings2_pct": round(100 * float((d["innings"] == 2).mean()), 1),
                "median_rrr_minus_crr_inn2": round(float(i2["rrr_minus_crr"].median()), 2) if len(i2) else None,
                "knockout_pct": round(100 * float(d["knock_out"].astype(bool).mean()), 2)}
    return {"high": prof(s[a]), "other": prof(s[b])}


def agreement(a: pd.Series, b: pd.Series) -> dict[str, float]:
    a, b = a.astype(bool), b.astype(bool)
    both, either = (a & b).sum(), (a | b).sum()
    po = float((a == b).mean())
    pe = float(a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean()))
    return {"leverage_share_pct": round(100 * float(a.mean()), 2), "rule_share_pct": round(100 * float(b.mean()), 2),
            "overlap_jaccard_pct": round(100 * both / max(either, 1), 2),
            "rule_balls_also_leverage_high_pct": round(100 * both / max(b.sum(), 1), 2),
            "leverage_high_also_rule_pct": round(100 * both / max(a.sum(), 1), 2),
            "agreement_pct": round(100 * po, 2), "cohen_kappa": round((po - pe) / (1 - pe), 3) if pe < 1 else None}


def examples(s: pd.DataFrame, n: int) -> list[dict[str, Any]]:
    """Annotated real matches: the highest-pressure close finishes (at least one knockout)."""
    s = s[~s["reduced_overs"]] if "reduced_overs" in s else s
    g = s.groupby("match_id").agg(mean_pi=("pressure_index", "mean"), ko=("knock_out", "first"),
                                  comp=("comp_name", "first"), decided=("win_team_id", lambda x: x.notna().all()))
    g = g[g["decided"]].sort_values("mean_pi", ascending=False)
    picks = list(dict.fromkeys(list(g[g["ko"].astype(bool)].index[:1]) + list(g.index[:n])))[:n]
    out = []
    for mid in picks:
        d = s[s["match_id"] == mid]
        pk = d.loc[d["pressure_index"].idxmax()]
        big = d.loc[d["wpa"].abs().idxmax()]
        out.append({"match_id": int(mid), "competition": str(g.loc[mid, "comp"]), "knock_out": bool(g.loc[mid, "ko"]),
                    "mean_pressure_index": round(float(g.loc[mid, "mean_pi"]), 3),
                    "peak": {"innings": int(pk["innings"]), "over": f"{int(pk['overid']) - 1}.{int(pk['over_ball_num'])}",
                             "pressure_index": round(float(pk["pressure_index"]), 2),
                             "striker": str(pk["striker_name"]), "bowler": str(pk["bowler_name"])},
                    "biggest_swing": {"innings": int(big["innings"]),
                                      "over": f"{int(big['overid']) - 1}.{int(big['over_ball_num'])}",
                                      "wpa": round(float(big["wpa"]), 3), "striker": str(big["striker_name"]),
                                      "bowler": str(big["bowler_name"])}})
    return out

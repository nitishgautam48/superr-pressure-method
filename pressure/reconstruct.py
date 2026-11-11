"""Legal balls, credit rules and score reconstruction.

Run values of extras are missing from the pack. Team runs per ball are reconstructed as
``runs + event_value(extras_type) + extras_2_runs`` (1 per wide / no-ball / leg-bye / bye / other,
5 for a penalty). Totals, targets and run rates built on this are APPROXIMATE (about +/-3 runs);
batter runs, balls faced, dismissals and boundaries are exact.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def prepare_balls(b: pd.DataFrame, cfg: Any) -> pd.DataFrame:
    """Add legal / faced / credit columns and reconstructed team runs. Sorted in delivery order."""
    b = b.sort_values(["match_id", "innings", "innings_ball_num"]).reset_index(drop=True).copy()
    et = b["extras_type"].where(b["extras_type"].notna() & (b["extras_type"].astype(str) != "None"))
    legal_set = set(cfg.get("data.legal_extras", [])) | ({"other"} if cfg.get("data.other_extra_legal") else set())
    b["extras_type"] = et
    b["is_legal"] = et.isna() | et.isin(legal_set)
    b["is_faced"] = ~et.isin(cfg.get("data.not_faced_extras", ["wide"]))
    ev = cfg.get("data.extra_run_events", {})
    b["extra_runs"] = et.map(ev).fillna(0).astype(int) + b["extras_2_runs"].fillna(0).astype(int)
    b["team_runs"] = b["runs"].astype(int) + b["extra_runs"]
    charged = et.isin(cfg.get("data.bowler_charged_extras", []))
    b["bowler_runs"] = b["runs"].astype(int) + np.where(charged, 1 + b["extras_2_runs"].fillna(0).astype(int), 0)
    wk = b["is_wicket"].astype(str).str.lower().eq("true")
    dt = b["dismissal_type"].fillna("")
    b["is_wicket"] = wk
    b["is_bowler_wicket"] = wk & ~dt.isin(cfg.get("data.bowler_wicket_exclude", []))
    b["is_batter_out"] = wk & ~dt.isin(cfg.get("data.batter_dismissal_exclude", []))
    b["is_team_wicket"] = b["is_batter_out"]            # retired-out does not cost a wicket in hand here
    b["is_four"] = b["is_four"].astype(str).str.lower().eq("true")
    b["is_six"] = b["is_six"].astype(str).str.lower().eq("true")
    b["is_super_over"] = b["innings"].isin(cfg.get("data.super_over_innings", [3, 4]))
    return b


def innings_totals(b: pd.DataFrame) -> pd.DataFrame:
    """Per match x innings (1-2): reconstructed total, wickets, legal balls, batting team."""
    reg = b[~b["is_super_over"]]
    g = reg.groupby(["match_id", "innings"])
    return pd.DataFrame({"total": g["team_runs"].sum(), "wickets": g["is_team_wicket"].sum(),
                         "legal_balls": g["is_legal"].sum(), "batting_team_id": g["batting_team_id"].first(),
                         "bat_runs": g["runs"].sum()}).reset_index()


def match_validation(matches: pd.DataFrame, totals: pd.DataFrame, tolerance: float = 10.0) -> pd.DataFrame:
    """Reconstructed vs actual winner and margin, per match with ball-by-ball data."""
    t1 = totals[totals["innings"] == 1].set_index("match_id")
    t2 = totals[totals["innings"] == 2].set_index("match_id")
    m = matches[matches["has_balls"]].set_index("match_id")
    out = pd.DataFrame(index=m.index)
    out["comp_name"], out["competition_group"], out["season"] = m["comp_name"], m["competition_group"], m["season_year"]
    out["result"] = m["result"]
    out["inn1_team"] = t1["batting_team_id"].reindex(out.index)
    out["inn2_team"] = t2["batting_team_id"].reindex(out.index)
    out["inn1_total"] = t1["total"].reindex(out.index)
    out["inn2_total"] = t2["total"].reindex(out.index)
    out["inn2_wickets"] = t2["wickets"].reindex(out.index)
    out["target"] = np.where(m["revised_target"].notna(), m["revised_target"], out["inn1_total"] + 1)
    out["target_source"] = np.where(m["revised_target"].notna(), "revised (DLS)", "reconstructed")
    # Winner from the structure of the chase (legal balls and wickets are exact): a second innings that
    # stops before its allotted balls with wickets in hand can only stop because the target was reached.
    out["inn2_legal_balls"] = t2["legal_balls"].reindex(out.index)
    allotted = np.where(m["second_inn_revised_overs"].notna(), m["second_inn_revised_overs"], 120)
    out["inn2_allotted_balls"] = allotted
    # ... unless the total is far short (an absent / retired batter ends the innings with < 10 wickets down)
    ended_early = ((out["inn2_legal_balls"] < out["inn2_allotted_balls"]) & (out["inn2_wickets"] < 10)
                   & (out["inn2_total"] >= out["target"] - tolerance))
    reached = out["inn2_total"] >= out["target"]
    out["chase_ended_early"] = ended_early
    out["recon_winner"] = np.where(out["inn2_total"].isna(), np.nan,
                                   np.where(ended_early | reached, out["inn2_team"], out["inn1_team"]))
    out["actual_winner"] = m["win_team_id"]
    for c in ("no_result", "abandoned", "dls", "tied", "super_over", "decided", "knock_out"):
        out[c] = m[c]
    out["winner_match"] = np.where(out["decided"] & ~out["super_over"],
                                   out["recon_winner"] == out["actual_winner"], np.nan)
    out["margin_type"], out["actual_margin"] = m["margin_type"], m["margin"]
    runs_margin = out["target"] - 1 - out["inn2_total"]
    out["recon_margin"] = np.where(out["margin_type"] == "runs", runs_margin,
                                   np.where(out["margin_type"] == "wickets", 10 - out["inn2_wickets"], np.nan))
    out["margin_error"] = out["recon_margin"] - out["actual_margin"]
    return out.reset_index()


def validation_summary(v: pd.DataFrame) -> dict[str, Any]:
    d = v[v["decided"] & ~v["super_over"] & v["inn2_total"].notna()]
    runs = d[d["margin_type"] == "runs"]
    wk = d[d["margin_type"] == "wickets"]
    acc = float(d["winner_match"].astype(float).mean()) if len(d) else float("nan")
    return {
        "matches_with_balls": int(len(v)), "decided_compared": int(len(d)),
        "winner_accuracy_pct": round(100 * acc, 2),
        "runs_margin_mae": round(float(runs["margin_error"].abs().mean()), 2) if len(runs) else None,
        "runs_margin_bias": round(float(runs["margin_error"].mean()), 2) if len(runs) else None,
        "runs_margin_within_3_pct": round(100 * float((runs["margin_error"].abs() <= 3).mean()), 1) if len(runs) else None,
        "wickets_margin_exact_pct": round(100 * float((wk["margin_error"] == 0).mean()), 1) if len(wk) else None,
        "no_result": int(v["no_result"].sum()), "abandoned_with_balls": int(v["abandoned"].sum()),
        "dls": int(v["dls"].sum()), "super_over": int(v["super_over"].sum()), "tied": int(v["tied"].sum()),
        "mismatches": d.loc[d["winner_match"] == 0, "match_id"].astype(int).tolist(),
    }

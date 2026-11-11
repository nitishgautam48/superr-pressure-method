"""S1 step: load, tag, reconstruct, validate; writes match_validation.csv + data_quality.json."""
from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from . import data, reconstruct as R, store
from .decisions import log_autopilot_default

log = logging.getLogger(__name__)


def run(ctx: Any) -> None:
    cfg = ctx.cfg
    raw = data.load_raw(cfg)
    comps = data.tag_competitions(raw["competitions"], cfg)
    tiers = data.team_tiers(raw["teams"], comps, cfg)
    balls = R.prepare_balls(raw["ball_by_ball"], cfg)
    matches = data.tag_matches(raw["matches"], comps, tiers, balls, cfg)
    totals = R.innings_totals(balls)
    val = R.match_validation(matches, totals, float(cfg.get("data.chase_early_tolerance_runs", 10)))
    summ = R.validation_summary(val)
    base = cfg.path("processed")
    store.write_table(val, "match_validation.csv", base)

    et = balls["extras_type"].value_counts().to_dict()
    other = balls[balls["extras_type"] == "other"]
    excluded = matches[~matches["has_balls"] | matches["no_result"] | matches["abandoned"]]
    dq = {
        "validation": summ,
        "row_counts": {"balls": int(len(balls)), "regular_balls": int((~balls["is_super_over"]).sum()),
                       "super_over_balls": int(balls["is_super_over"].sum()), "matches_csv": int(len(matches)),
                       "matches_with_balls": int(matches["has_balls"].sum()),
                       "players_in_mapping": int(raw["competition_team_player_mapping"]["player_id"].nunique())},
        "competition_groups": comps["competition_group"].value_counts().to_dict(),
        "team_tiers": tiers["tier"].value_counts().to_dict(),
        "extras_counts": et,
        "other_extra": {"rows": int(len(other)), "with_bat_runs": int((other["runs"] > 0).sum()),
                        "treated_as_legal": bool(cfg.get("data.other_extra_legal")),
                        "evidence": "with 'other' illegal, every completed over containing it has exactly 6 legal "
                                    "balls; with it legal, none do"},
        "excluded_matches": excluded[["match_id", "comp_name", "result", "has_balls", "no_result", "abandoned"]]
        .to_dict("records"),
        "flags": {"dls": int(matches["dls"].sum()), "super_over": int(matches["super_over"].sum()),
                  "tied": int(matches["tied"].sum()), "knock_out": int(matches["knock_out"].sum()),
                  "no_result": int(matches["no_result"].sum()), "abandoned": int(matches["abandoned"].sum())},
        "duplicate_names": _dup_names(raw["competition_team_player_mapping"]),
        "revised_overs_unit": "balls (e.g. 114 = 19 overs); checked against legal balls bowled in DLS chases",
        "limitations": [
            "Run values of extras are missing; team totals, targets and required rates are reconstructed "
            "(about +/-3 runs). Batter runs, balls faced, dismissals and boundaries are exact.",
            "Winners are reconstructed from the chase structure (legal balls and wickets are exact).",
        ],
    }
    store.write_json(dq, "data_quality.json", base)
    for k, v in (("data.other_extra_legal", cfg.get("data.other_extra_legal")),
                 ("data.batter_dismissal_exclude", cfg.get("data.batter_dismissal_exclude"))):
        log_autopilot_default(cfg, k, v)
    ctx.tables.update(balls=balls, matches=matches, comps=comps, tiers=tiers, totals=totals, validation=val,
                      raw=raw)
    ctx.metrics["winner_accuracy_pct"] = summ["winner_accuracy_pct"]
    ctx.metrics["runs_margin_mae"] = summ["runs_margin_mae"]
    log.info("S1: winner accuracy %.2f%% over %d decided matches; runs-margin MAE %.2f",
             summ["winner_accuracy_pct"], summ["decided_compared"], summ["runs_margin_mae"])


def _dup_names(mp: pd.DataFrame) -> list[dict[str, Any]]:
    d = mp.drop_duplicates(["player_id", "player_name"])
    g = d.groupby("player_name")["player_id"].nunique()
    out = []
    for name in g[g > 1].index:
        rows = mp[mp["player_name"] == name].drop_duplicates("player_id")
        out.append({"name": name, "players": rows[["player_id", "team_name", "comp_name"]].to_dict("records")})
    return out

"""S2 step: match state -> WP model -> WPA -> leverage / pressure index -> rule-based -> separation evidence."""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

from . import pressure as PR, state as ST, store, wp_model as W
from .decisions import log_autopilot_default

log = logging.getLogger(__name__)


def run(ctx: Any) -> None:
    cfg, t = ctx.cfg, ctx.tables
    s = ST.build_state(t["balls"], t["matches"], t["validation"], t["tiers"], cfg)
    models, rep = W.fit(s, cfg)
    s = W.add_wp(s, models)
    s, probs = PR.add_leverage(s, models, cfg)
    s = PR.classify(s, cfg)
    s["is_pressure_rule"] = PR.rule_based(s, cfg)
    base = cfg.path("processed")

    terminal = (s["innings"] == 2) & s["innings_end"]
    rep["summary"] = {
        "brier_innings_1": rep["innings_1"]["cv_calibrated"]["brier"],
        "brier_innings_2": rep["innings_2"]["cv_calibrated"]["brier"],
        "brier_weighted": round(float(np.average([rep["innings_1"]["cv_calibrated"]["brier"],
                                                  rep["innings_2"]["cv_calibrated"]["brier"]],
                                                 weights=[rep["innings_1"]["rows"], rep["innings_2"]["rows"]])), 4),
        "validation": "GroupKFold by match_id (no match in two folds) + time split <= 2024 / >= 2025",
        "terminal_balls_take_actual_result": True,
        "large_terminal_swings": int(((s["wpa"].abs() > 0.5) & terminal).sum()),
        "reduced_overs_matches_excluded_from_training": int(s.loc[s["reduced_overs"], "match_id"].nunique()),
        "note": "Scores and targets are reconstructed (about +/-3 runs), so WP inherits that uncertainty.",
    }
    store.write_json(rep, "wp_model_report.json", base)

    full = ~s["reduced_overs"]
    hi, rule = s["is_pressure"] & full, s["is_pressure_rule"] & full
    sep = {
        "threshold": {"high_pct": cfg.get("pressure.high_pct"), "normalise_by": cfg.get("pressure.normalise_by"), "pressure_index_cutoff": round(s.attrs["pressure_threshold"], 4),
                      "knockout_multiplier": cfg.get("pressure.knockout_multiplier")},
        "scope": "full-length matches (reduced-overs matches excluded)",
        "leverage": {"high": PR.league_stats(s, hi), "other": PR.league_stats(s, full & ~hi)},
        "situation_profile": PR.situation_profile(s, hi, full & ~hi),
        "bands": {b: PR.league_stats(s, full & (s["pressure_band"] == b)) for b in ("low", "medium", "high")},
        "rule_based": {"pressure": PR.league_stats(s, rule), "other": PR.league_stats(s, full & ~rule)},
        "by_innings": {f"innings_{i}": {"high": PR.league_stats(s, hi & (s["innings"] == i)),
                                        "other": PR.league_stats(s, full & ~hi & (s["innings"] == i))} for i in (1, 2)},
        "by_competition": {g: {"high_share_pct": round(100 * float(hi[s["competition_group"] == g].mean()), 2)}
                           for g in sorted(s["competition_group"].unique())},
        "by_phase": {p: {"high_share_pct": round(100 * float(hi[s["phase"] == p].mean()), 2),
                         "mean_pressure_index": round(float(s.loc[s["phase"] == p, "pressure_index"].mean()), 3)}
                     for p in ("PP", "MID", "DEATH")},
        "agreement": PR.agreement(hi[full], rule[full]),
        "rank_correlation_of_ratings": None,   # filled in S3/S4 once ratings exist
        "outcome_probabilities": probs.round(4).reset_index().astype({"innings": int}).to_dict("records"),
        "examples": PR.examples(s, int(cfg.get("pressure.example_matches", 4))),
        "leverage_distribution": {"percentiles": {str(q): round(float(np.nanpercentile(s["leverage"], q)), 5)
                                                  for q in (10, 25, 50, 75, 80, 90, 95, 99)},
                                  "mean": round(float(s["leverage"].mean()), 5)},
    }
    store.write_json(sep, "pressure_separation.json", base)
    for k in ("pressure.high_pct", "pressure.knockout_multiplier", "wp_model.tie_handling",
              "wp_model.exclude_reduced_overs", "pressure.normalise_by"):
        log_autopilot_default(cfg, k, cfg.get(k))
    from . import figures as F
    F.fig_definition(cfg, sep)
    F.fig_leverage_distribution(cfg, s, s.attrs["pressure_threshold"])
    F.fig_separation(cfg, sep)
    for ex in sep["examples"][:2]:
        F.fig_example_curve(cfg, s, ex)
    F.fig_calibration(cfg, rep)
    F.export_guard(cfg)
    t["state"], t["wp_models"], t["wp_report"], t["separation"] = s, models, rep, sep
    ctx.metrics.update(wp_brier=rep["summary"]["brier_weighted"], pressure_share_pct=round(100 * float(hi.mean()), 2),
                       rule_share_pct=round(100 * float(rule.mean()), 2),
                       agreement_kappa=sep["agreement"]["cohen_kappa"])
    log.info("S2: Brier %.4f; high-pressure share %.1f%%; rule-based %.1f%%; kappa %s",
             rep["summary"]["brier_weighted"], 100 * hi.mean(), 100 * rule.mean(), sep["agreement"]["cohen_kappa"])

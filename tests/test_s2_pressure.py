"""S2: WP model (grouped CV, calibration, bounds), leverage, pressure index, rule-based, separation."""
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import GroupKFold

from pressure import pressure as PR, state as ST, wp_model as W
from tests.conftest import needs_data


@pytest.fixture(scope="module")
def s2(s1, cfg):
    from pressure import steps_pressure

    steps_pressure.run(s1)
    return s1


@needs_data
def test_no_cross_match_leakage_in_cv(s1, cfg):
    s = ST.build_state(s1.tables["balls"], s1.tables["matches"], s1.tables["validation"], s1.tables["tiers"], cfg)
    X, y, w, g = W.training_rows(s, 2, cfg)
    for tr, te in GroupKFold(n_splits=cfg.get("wp_model.cv_folds")).split(X, y, g):
        assert not set(g[tr]) & set(g[te])
    assert set(np.unique(y)) <= {0, 1} and (w > 0).all()


@needs_data
def test_wp_bounds_and_wpa(s2):
    s = s2.tables["state"]
    for c in ("wp_before", "wp_after"):
        assert s[c].between(0, 1).all(), c
    assert abs(float(s["wpa"].mean())) < 0.01
    # terminal ball of every decided chase ends at the actual result
    term = s[(s["innings"] == 2) & s["innings_end"] & ~s["tied"].astype(bool) & s["win_team_id"].notna()]
    assert set(term["wp_after"].round(6)) <= {0.0, 1.0}


@needs_data
def test_wp_model_beats_no_skill_and_reports(s2):
    rep = s2.tables["wp_report"]
    for inn in (1, 2):
        r = rep[f"innings_{inn}"]
        assert r["cv_calibrated"]["brier"] < r["no_skill_brier"] - 0.02
        assert r["folds_disjoint_by_match"] and "time_split" in r and r["calibration_bins"]
        assert r["calibration"]["chosen"] in ("none", "isotonic", "sigmoid")


@needs_data
def test_leverage_nonnegative_and_index_normalised(s2, cfg):
    s = s2.tables["state"]
    assert (s["leverage"] >= 0).all()
    raw = s.groupby("innings")["pressure_index_raw"].mean()
    assert np.allclose(raw.values, 1.0, atol=1e-6)            # 1.0 = average ball of the innings
    assert abs(100 * s["is_pressure"].mean() - cfg.get("pressure.high_pct")) < 1.0
    assert (s.loc[s["knock_out"].astype(bool), "ko_multiplier"] == cfg.get("pressure.knockout_multiplier")).all()


@needs_data
def test_definition_separates_situations(s2):
    sep = s2.tables["separation"]
    hi, lo = sep["leverage"]["high"], sep["leverage"]["other"]
    assert hi["mean_abs_wpa"] > 2 * lo["mean_abs_wpa"]                # stakes are much higher
    assert hi["wicket_rate_pct"] > lo["wicket_rate_pct"]
    sp = sep["situation_profile"]
    assert sp["high"]["wp_within_0_25_0_75_pct"] > sp["other"]["wp_within_0_25_0_75_pct"] + 20
    assert 0 <= sep["agreement"]["overlap_jaccard_pct"] <= 100 and len(sep["examples"]) >= 2


def test_rule_based_rules(cfg):
    s = pd.DataFrame({"innings": [2, 2, 1, 1, 2], "rrr_minus_crr": [3.0, 1.0, np.nan, np.nan, 0.5],
                      "balls_remaining": [40, 40, 50, 50, 10], "wickets_before": [2, 2, 6, 2, 3],
                      "crr": [8.0, 8.0, 7.0, 7.0, 9.0], "runs_needed": [40, 20, np.nan, np.nan, 15],
                      "phase": ["MID", "MID", "MID", "MID", "DEATH"], "knock_out": [0, 0, 0, 0, 0]})
    assert PR.rule_based(s, cfg).tolist() == [True, False, True, False, True]


def test_classify_threshold(cfg):
    s = pd.DataFrame({"pressure_index": np.linspace(0, 10, 1000)})
    out = PR.classify(s, cfg, high_pct=10)
    assert abs(out["is_pressure"].mean() - 0.10) < 0.005 and set(out["pressure_band"]) == {"low", "medium", "high"}

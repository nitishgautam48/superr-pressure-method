"""S3: expected baselines, shrinkage, eligibility, ratings, names."""
import numpy as np
import pandas as pd
import pytest

from pressure import expected as X, rating as RT
from tests.conftest import needs_data


@pytest.fixture(scope="module")
def s3(s1, cfg):
    from pressure import steps_pressure, steps_rating

    if "state" not in s1.tables or "wp_report" not in s1.tables:
        steps_pressure.run(s1)
    steps_rating.run(s1)
    return s1


def test_smoothing_falls_back_to_parent():
    y = pd.Series([1.0] * 1000 + [5.0])                       # one ball in a rare bin
    lv = [pd.Series(["all"] * 1001), pd.Series(["common"] * 1000 + ["rare"])]
    m = X.smoothed_expectation(y, lv, k=200)
    assert abs(m[-1] - 1.0) < 0.05 and abs(m[0] - 1.0) < 0.01  # rare bin shrunk to its parent


def test_shrinkage_moves_small_samples_toward_mean():
    v = pd.Series([1.0] * 5 + [0.0] * 2000)
    key = pd.Series(["small"] * 5 + ["big"] * 2000)
    w = pd.Series(1.0, index=v.index)
    shr, n, mu = RT._shrunk_mean(v, w, key, k=100)
    assert mu < shr["small"] < 1.0 and abs(shr["small"] - mu) < abs(1.0 - mu) * 0.1
    assert abs(shr["big"] - 0.0) < 0.01


@needs_data
def test_expected_baselines_are_unbiased(s3):
    s = s3.tables["state"]
    f = s[s["is_faced"] & ~s["reduced_overs"]]
    assert abs(float(f["rae_bat"].mean())) < 0.01                 # runs above expected average ~ 0
    assert f["p_out"].between(0, 1).all() and f["xruns"].between(0, 6).all()
    assert abs(float(s.loc[~s["reduced_overs"], "wae"].mean())) < 0.002


@needs_data
def test_eligibility_filters(s3, cfg):
    el = cfg.get("rating.eligibility")
    b, w = s3.tables["batters"], s3.tables["bowlers"]
    eb = b[b["eligible"]]
    assert (eb["pressure_balls"] >= el["batter_min_pressure_balls"]).all()
    assert (eb["pressure_units"] >= el["batter_min_pressure_innings"]).all()
    nb = b[~b["eligible"]]
    assert ((nb["pressure_balls"] < el["batter_min_pressure_balls"]) |
            (nb["pressure_units"] < el["batter_min_pressure_innings"])).all()
    ew = w[w["eligible"]]
    assert (ew["pressure_balls"] >= el["bowler_min_pressure_balls"]).all()
    assert (ew["pressure_units"] >= el["bowler_min_pressure_matches"]).all()
    assert b["rank"].notna().sum() == len(eb) and b.loc[~b["eligible"], "rank"].isna().all()


@needs_data
def test_ratings_shape_and_ranks(s3, cfg):
    for t in (s3.tables["batters"], s3.tables["bowlers"]):
        e = t[t["eligible"]].sort_values("rank")
        assert len(e) >= 20 and e["player_id"].is_unique
        assert list(e["rank"].head(10)) == list(range(1, 11))
        assert e["prs_raw"].is_monotonic_decreasing and e["prs"].between(0, 100).all()
        w = cfg.get("rating.weights")
        recon = sum(w[c] * e[f"z_{c}"] for c in "ABC")
        assert np.allclose(recon, e["prs_raw"])
        assert e["name"].notna().all() and e["A_se"].gt(0).all()


@needs_data
def test_associate_weight_toggle_changes_weights(s3, cfg):
    s = s3.tables["state"].head(5000)
    w1 = RT.ball_weights(s, cfg)
    w0 = RT.ball_weights(s, cfg, {"associate_exclude": True})
    assoc = (s["bat_tier"] == "ASSOCIATE") | (s["bowl_tier"] == "ASSOCIATE")
    assert (w0[assoc] == 0).all() and (w0[~assoc] == w1[~assoc]).all()


def test_bowl_kind_mapping():
    t = pd.Series(["Right-arm Leg-Break", "Slow left-arm orthodox", "Left-arm Chinaman", "Left-arm Fast",
                   "Right-arm medium", "Right-arm offbreak"])
    k = X.bowl_kind(t, pd.Series(["Spin", "Spin", "Spin", "Pace", "Pace", "Spin"]))
    assert k.tolist() == ["wrist", "finger", "wrist", "pace_L", "pace_R", "finger"]


def test_era_and_context_switches(cfg):
    s = pd.DataFrame({"innings": [1, 1], "phase": ["PP", "PP"], "wickets_before": [0, 0], "rrr": [float("nan")] * 2,
                      "pace_or_spin": ["Pace", "Pace"], "batting_type": ["RHB", "RHB"], "bowling_type": ["Left-arm Fast"] * 2,
                      "competition_group": ["IPL", "IPL"], "season": [2022, 2026], "bowl_tier": ["IPL", "IPL"]})
    on = X.context_levels(s, cfg.with_overrides(**{"expected.context": {"era": True, "bowl_kind": True}}), "bowl_tier")
    off = X.context_levels(s, cfg.with_overrides(**{"expected.context": {}}), "bowl_tier")
    assert on[0].iloc[0] != on[0].iloc[1] and off[0].iloc[0] == off[0].iloc[1]     # seasons split only with era
    assert "pace_L" in on[3].iloc[0] and "Pace" in off[3].iloc[0]


@needs_data
def test_opposition_strength_terciles(s3, cfg):
    s = s3.tables["state"]
    st_ = X.opposition_strength(s, "bowling_team_id", cfg)
    assert set(st_.unique()) <= {"weak", "avg", "strong"}
    share = st_.value_counts(normalize=True)
    assert all(0.15 < share.get(k, 0) < 0.55 for k in ("weak", "avg", "strong"))

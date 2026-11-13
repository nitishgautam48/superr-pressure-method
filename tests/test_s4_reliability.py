"""S4: fast bootstrap = rating, labels, analyst layer, final lists, export guard, RR lens."""
import numpy as np
import pandas as pd
import pytest

from pressure import analyst, decisions, figures as F, rating as RT, reliability as RL
from tests.conftest import needs_data


@pytest.fixture(scope="module")
def s4(s1, cfg):
    from pressure import steps_pressure, steps_rating, steps_reliability

    if "wp_report" not in s1.tables:
        steps_pressure.run(s1)
    if "batters" not in s1.tables or "label" not in s1.tables["batters"]:
        steps_rating.run(s1)
        steps_reliability.run(s1)
    return s1


@needs_data
def test_fast_bootstrap_reproduces_rating(s4, cfg):
    s = s4.tables["state"]
    for role in ("batter", "bowler"):
        r = RT.rate(s, role, cfg)
        agg = RL.match_aggregates(s, role, cfg)
        el = np.isin(agg["players"], r.index[r["eligible"]])
        prs = RL.fast_scores(agg, np.ones(len(agg["matches"])), el, cfg)
        assert np.allclose(prs[el], r["prs_raw"].reindex(agg["players"]).to_numpy()[el])


def test_labels(cfg):
    assert RL.label(90, 85, cfg) == "Robust" and RL.label(90, 60, cfg) == "Likely" and RL.label(40, 95, cfg) == "Sensitive"


@needs_data
def test_reliability_columns_and_files(s4, cfg):
    b = s4.tables["batters"]
    e = b[b["eligible"]]
    for c in ("boot_rank_lo", "boot_rank_hi", "p_top10", "variant_top10_pct", "label", "rank_ipl", "rank_t20i"):
        assert c in b
    assert (e["boot_rank_lo"] <= e["boot_rank_hi"]).all() and e["p_top10"].between(0, 100).all()
    assert set(e["label"]) <= {"Robust", "Likely", "Sensitive"}
    base = cfg.path("processed")
    for f in ("sensitivity.json", "persistence.json", "final_lists.json", "model_vs_final.json", "rr_lens.json"):
        assert (base / f).exists(), f


@needs_data
def test_final_lists_unique_and_spelled_as_mapping(s4):
    mp = s4.tables["raw"]["competition_team_player_mapping"]
    names = set(mp["player_name"])
    for role in ("batter", "bowler"):
        fl = s4.tables["finals"][f"{role}s_final"]
        ids = [x["player_id"] for x in fl]
        assert len(fl) == 10 and len(set(ids)) == 10
        assert all(x["name"] in names for x in fl)
        assert [x["rank"] for x in fl] == list(range(1, 11))


@needs_data
def test_analyst_overrides(s4, cfg):
    rated = {"batter": s4.tables["batters"], "bowler": s4.tables["bowlers"]}
    b = rated["batter"][rated["batter"]["eligible"]].sort_values("rank")
    top, outsider = int(b.iloc[0]["player_id"]), int(b.iloc[15]["player_id"])
    ov = [{"role": "batter", "action": "drop", "player_id": top, "reason": "Too few pressure innings to trust"},
          {"role": "batter", "action": "pin", "player_id": outsider, "rank": 3, "reason": "Best chase record in the data"},
          {"role": "batter", "action": "swap_rank", "player_id": int(b.iloc[1]["player_id"]), "rank": 1, "reason": "x"}]
    out = analyst.apply(rated, cfg, ov, log=False)
    fl = out["batters_final"]
    ids = [x["player_id"] for x in fl]
    assert top not in ids and ids[2] == outsider and len(set(ids)) == 10
    assert next(x for x in fl if x["player_id"] == outsider)["provenance"] == "analyst-add"
    assert len(out["refused"]) == 1 and "reason" in out["refused"][0]["error"]   # short reason refused
    assert out["batters_model"][0]["player_id"] == top                           # model list kept


def test_decision_log_requires_reason(cfg):
    with pytest.raises(decisions.DecisionRefused):
        decisions.log_decision(cfg, "rating.weights", "a", "b", "short")
    row = decisions.log_decision(cfg, "rating.weights", "a", "b", "Testing the decision log works")
    assert row["reason"].startswith("Testing")


@needs_data
def test_export_guard_blocks_row_level(s4, cfg):
    F.export_guard(cfg)                                  # current exports are clean
    p = cfg.path("outputs_public") / "zz_bad.csv"
    pd.DataFrame({"innings_ball_num": [1, 2]}).to_csv(p, index=False)
    with pytest.raises(F.ExportGuardError):
        F.export_guard(cfg)
    p.unlink()


@needs_data
def test_rr_targets_funnel_alignment_and_final(s4, cfg):
    import json

    from pressure import analyst, rr_target

    res = rr_target.build(cfg)
    rr_target.apply_to_rated(cfg, res)
    finals = analyst.refresh_final(cfg)
    mp = s4.tables["raw"]["competition_team_player_mapping"]
    for role, side in (("batter", "batting"), ("bowler", "bowling")):
        steps = res[role]["steps"]
        left = [s["remaining"] for s in steps]
        assert left == sorted(left, reverse=True)                        # funnel only ever removes players
        t = pd.DataFrame(res[role]["table"])
        assert (t["aligned_phases"].str.len() > 0).all()               # every target helps a phase RR needs
        assert t["fit"].between(0, 100).all() and t["target_score"].between(0, 100).all()
        assert not t["auction_status"].str.startswith("in RR").any()   # current RR squad excluded
        fl = finals[f"{role}s_final"]
        assert finals["list_basis"] == "RR auction targets"
        assert [x["player_id"] for x in fl] == t.sort_values("target_rank")["player_id"].head(10).tolist()
        assert all(x["name"] in set(mp["player_name"]) for x in fl) and len({x["player_id"] for x in fl}) == 10
    assert set(res["needs"]["batting"]) == {"PP", "MID", "DEATH"}


def test_availability_file_requires_reasons(cfg, tmp_path):
    from pressure import rr_target

    p = cfg.path("overrides").parent / "auction_pool.yaml"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("exclude:\n  - {player_id: 1, reason: short}\n  - {player_id: 2, reason: Retained by his franchise for 2027}\n")
    try:
        excl, incl, errors = rr_target.load_availability(cfg)
        assert excl == {2: "Retained by his franchise for 2027"} and len(errors) == 1
    finally:
        p.unlink()

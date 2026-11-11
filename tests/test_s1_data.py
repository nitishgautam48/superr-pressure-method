"""S1: legal balls, credit rules, score reconstruction, match validation."""
import pandas as pd
import pytest

from pressure import data, reconstruct as R
from tests.conftest import ball, needs_data


def _prep(rows, cfg):
    return R.prepare_balls(pd.DataFrame(rows), cfg)


def test_legal_ball_logic(cfg):
    rows = [ball(innings_ball_num=i + 1, extras_type=e, runs=r) for i, (e, r) in enumerate(
        [(None, 1), ("wide", 0), ("Overstep", 4), ("legbyes", 0), ("byes", 0), ("other", 2), ("Penalty", 0)])]
    b = _prep(rows, cfg)
    assert b["is_legal"].tolist() == [True, False, False, True, True, False, False]
    assert b["is_faced"].tolist() == [True, False, True, True, True, True, True]  # every delivery except wides


def test_score_reconstruction_and_credit(cfg):
    rows = [ball(innings_ball_num=1, runs=4, is_four=True),
            ball(innings_ball_num=2, extras_type="wide"),
            ball(innings_ball_num=3, extras_type="legbyes"),
            ball(innings_ball_num=4, extras_type="Overstep", runs=6, is_six=True),
            ball(innings_ball_num=5, extras_type="byes", extras_2_runs=4),
            ball(innings_ball_num=6, is_wicket=True, dismissal_type="RunOut"),
            ball(innings_ball_num=7, is_wicket=True, dismissal_type="Caught"),
            ball(innings_ball_num=8, is_wicket=True, dismissal_type="Retire Out")]
    b = _prep(rows, cfg)
    assert b["team_runs"].tolist() == [4, 1, 1, 7, 5, 0, 0, 0]
    assert b["bowler_runs"].tolist() == [4, 1, 0, 7, 0, 0, 0, 0]       # leg-byes / byes not charged
    assert b["is_bowler_wicket"].tolist() == [False] * 5 + [False, True, False]   # run out not credited
    assert b["is_batter_out"].tolist() == [False] * 5 + [True, True, False]       # retired out not a dismissal
    t = R.innings_totals(b)
    assert int(t["total"].iloc[0]) == 18 and int(t["wickets"].iloc[0]) == 2 and int(t["legal_balls"].iloc[0]) == 6


def test_parse_result_formats():
    assert data.parse_result("West Indies Won by 9 Wickets") == ("wickets", 9)
    assert data.parse_result("Royal Challengers Bengaluru Won 3 Wkts") == ("wickets", 3)
    assert data.parse_result("Punjab Kings Won by 7 Runs(D/L Method)") == ("runs", 7)
    assert data.parse_result("No Result") == (None, None)


def test_competition_and_tier_tags(cfg):
    comps = pd.DataFrame({"comp_id": [1, 2, 3], "comp_name": ["IPL 2024", "Asia Cup Rising Stars 2025",
                                                                "India in Japan T20I Match 2026"],
                          "match_format": ["T20", "T20", "T20I"], "season": ["2024", "2025", "2026"]})
    c = data.tag_competitions(comps, cfg)
    assert c["competition_group"].tolist() == ["IPL", "EMERGING", "T20I"]
    teams = pd.DataFrame({"comp_id": [1, 2, 3, 3], "team_id": [5, 6, 7, 8],
                          "team_name": ["Rajasthan Royals", "India A", "India", "Japan"]})
    t = data.team_tiers(teams, c, cfg)
    assert t["tier"].tolist() == ["IPL", "A_TEAM", "FULL_MEMBER", "ASSOCIATE"]


@needs_data
def test_validation_meets_done_criterion(s1, cfg):
    s = R.validation_summary(s1.tables["validation"])
    assert s["winner_accuracy_pct"] >= cfg.get("data.validation_target_pct")
    assert s["runs_margin_mae"] is not None and s["runs_margin_mae"] < 4      # about +/-3 runs, no big bias
    assert abs(s["runs_margin_bias"]) < 1
    assert s["wickets_margin_exact_pct"] >= 95


@needs_data
def test_known_matches(s1):
    v = s1.tables["validation"].set_index("match_id")
    # 13511: "West Indies Won by 9 Wickets" - chase ended early with 1 wicket down
    r = v.loc[13511]
    assert r["recon_winner"] == r["actual_winner"] and r["recon_margin"] == 9
    # every DLS chase uses the revised target (in balls / runs from matches.csv)
    d = v[v["target_source"] == "revised (DLS)"]
    assert len(d) >= 20 and (d["inn2_allotted_balls"] <= 120).all()


@needs_data
def test_data_shape(s1):
    b = s1.tables["balls"]
    assert len(b) == 200999 and b["match_id"].nunique() == 867
    assert int(b["is_super_over"].sum()) == 38
    assert int((b["extras_type"] == "other").sum()) == 264 and not b.loc[b["extras_type"] == "other", "is_legal"].any()

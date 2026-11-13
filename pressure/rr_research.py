"""RR research from the data pack only: season record, close-match record, pressure profile ranked against the other
IPL teams, squad continuity, and a structured inclusion / exclusion funnel for auction targets.

    python -m pressure.rr_research          # writes outputs/private/rr_research.json and prints a summary
"""
from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from .config import load_config

RR = "Rajasthan Royals"
PHASES = ("PP", "MID", "DEATH")


def _ipl(cfg: Any) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw, proc = cfg.path("raw"), cfg.path("processed")
    comps = pd.read_csv(raw / "competitions.csv")
    teams = pd.read_csv(raw / "teams.csv")
    m = pd.read_csv(raw / "matches.csv").merge(comps[["comp_id", "comp_name", "season"]], on="comp_id")
    m = m[m["comp_name"].str.match(r"^IPL\b")].copy()
    m["season"] = m["season"].astype(str).str[:4].astype(int)
    v = pd.read_csv(proc / "match_validation.csv")[["match_id", "inn1_team", "inn2_team", "margin_type", "actual_margin"]]
    m = m.merge(v, on="match_id", how="left")
    return m, teams, comps


def season_record(m: pd.DataFrame, rr_ids: set[int]) -> list[dict[str, Any]]:
    out = []
    for season, g in m.groupby("season"):
        league = g[g["stage"].astype(str).str.lower().eq("league")]
        pts: dict[int, float] = {}
        for _, r in league.iterrows():
            for t in (r["home_team_id"], r["away_team_id"]):
                pts.setdefault(t, 0)
            if pd.notna(r["win_team_id"]):
                pts[r["win_team_id"]] += 2
            else:
                pts[r["home_team_id"]] += 1
                pts[r["away_team_id"]] += 1
        table = sorted(pts.items(), key=lambda kv: -kv[1])
        rr = next(t for t in pts if t in rr_ids) if any(t in rr_ids for t in pts) else None
        mine = g[(g["home_team_id"].isin(rr_ids)) | (g["away_team_id"].isin(rr_ids))]
        won = int(mine["win_team_id"].isin(rr_ids).sum())
        nr = int(mine["win_team_id"].isna().sum())
        playoff = mine[~mine["stage"].astype(str).str.lower().eq("league")]
        out.append({"season": int(season), "matches": int(len(mine)), "won": won, "lost": int(len(mine) - won - nr),
                    "no_result": nr, "win_pct": round(100 * won / max(len(mine) - nr, 1), 1),
                    "league_points": pts.get(rr), "league_position": [t for t, _ in table].index(rr) + 1 if rr in pts else None,
                    "playoff_matches": playoff["stage"].astype(str).tolist(),
                    "playoff_wins": int(playoff["win_team_id"].isin(rr_ids).sum())})
    return out


def close_and_situation(m: pd.DataFrame, rr_ids: set[int]) -> dict[str, Any]:
    mine = m[(m["home_team_id"].isin(rr_ids) | m["away_team_id"].isin(rr_ids)) & m["win_team_id"].notna()].copy()
    close = ((mine["margin_type"] == "runs") & (mine["actual_margin"] <= 10)) | \
            ((mine["margin_type"] == "wickets") & (mine["actual_margin"] <= 3))
    mine["rr_won"] = mine["win_team_id"].isin(rr_ids)
    mine["rr_chased"] = mine["inn2_team"].isin(rr_ids)
    def rec(d):
        return {"matches": int(len(d)), "won": int(d["rr_won"].sum()),
                "win_pct": round(100 * d["rr_won"].mean(), 1) if len(d) else None}
    by = {}
    for s, g in mine.groupby("season"):
        c = g[close.loc[g.index]]
        by[int(s)] = {"close": rec(c), "chasing": rec(g[g["rr_chased"]]), "defending": rec(g[~g["rr_chased"]])}
    return {"all_seasons": {"close": rec(mine[close]), "chasing": rec(mine[mine["rr_chased"]]),
                            "defending": rec(mine[~mine["rr_chased"]]), "not_close": rec(mine[~close])},
            "by_season": by, "close_definition": "won by <= 10 runs or <= 3 wickets"}


def pressure_rank(balls: pd.DataFrame, rr_ids: set[int], seasons: list[int]) -> dict[str, Any]:
    """Win probability added per 100 high-pressure balls, by phase, for every IPL team; RR's rank (1 = best of 10)."""
    b = balls[(balls["competition_group"] == "IPL") & balls["is_pressure"] & ~balls["reduced_overs"] &
              balls["season"].isin(seasons)]
    out: dict[str, Any] = {}
    for side, col, sign in (("batting", "batting_team_id", 1), ("bowling", "bowling_team_id", -1)):
        rows = []
        for ph in PHASES + ("ALL",):
            d = b if ph == "ALL" else b[b["phase"] == ph]
            if ph != "ALL":
                d = d[d["phase"] == ph]
            team = d.groupby(col).agg(balls=("wpa", "size"), wpa=("wpa", "sum"), runs=("team_runs", "sum"),
                                      legal=("is_legal", "sum"), wk=("is_team_wicket", "sum"))
            team["wpa_per100"] = sign * 100 * team["wpa"] / team["balls"]
            team["run_rate"] = 6 * team["runs"] / team["legal"].clip(lower=1)
            team["wicket_pct"] = 100 * team["wk"] / team["legal"].clip(lower=1)
            team["rank"] = team["wpa_per100"].rank(ascending=False).astype(int)
            rr = team[team.index.isin(rr_ids)]
            if len(rr):
                r = rr.iloc[0]
                rows.append({"phase": ph, "rr_wpa_per100": round(float(r["wpa_per100"]), 3),
                             "league_avg_wpa_per100": round(float(team["wpa_per100"].mean()), 3),
                             "rr_rank_of_teams": int(r["rank"]), "teams": int(len(team)),
                             "rr_run_rate": round(float(r["run_rate"]), 2), "league_run_rate": round(float(team["run_rate"].mean()), 2),
                             "rr_wicket_pct": round(float(r["wicket_pct"]), 2),
                             "league_wicket_pct": round(float(team["wicket_pct"].mean()), 2), "rr_balls": int(r["balls"])})
        out[side] = rows
    return out


def squad(mapping: pd.DataFrame, comps: pd.DataFrame, rated: dict[str, pd.DataFrame]) -> dict[str, Any]:
    mp = mapping.merge(comps[["comp_id", "season"]], on="comp_id")
    mp["season"] = mp["season"].astype(str).str[:4].astype(int)
    rr = mp[mp["team_name"] == RR]
    by_season = {int(s): sorted(g["player_name"].unique().tolist()) for s, g in rr.groupby("season")}
    latest = max(by_season)
    ids = set(rr.loc[rr["season"] == latest, "player_id"])
    contrib = []
    for role, r in rated.items():
        x = r[r["player_id"].isin(ids) & r["eligible"]]
        for _, p in x.iterrows():
            contrib.append({"name": p["name"], "role": role, "rank": int(p["rank"]), "prs": round(float(p["prs"]), 1),
                            "label": p.get("label")})
    seasons = sorted(by_season)
    cont = {f"{a}->{b}": len(set(by_season[a]) & set(by_season[b])) for a, b in zip(seasons, seasons[1:])}
    return {"latest_season": latest, "squad_size_by_season": {s: len(v) for s, v in by_season.items()},
            "continuity_players_kept": cont, "latest_squad": by_season[latest],
            "latest_squad_rated": sorted(contrib, key=lambda c: (c["role"], c["rank"]))}


def funnel(balls: pd.DataFrame, rated: dict[str, pd.DataFrame], mapping: pd.DataFrame, comps: pd.DataFrame,
           rules: dict[str, Any]) -> dict[str, Any]:
    """Structured inclusion / exclusion, step by step, with counts and who drops out of the current model top 10."""
    mp = mapping.merge(comps[["comp_id", "season"]], on="comp_id")
    mp["season"] = mp["season"].astype(str).str[:4].astype(int)
    latest_ipl = mp[mp["comp_name"].str.match(r"^IPL\b")]["season"].max()
    rr_now = set(mp[(mp["team_name"] == RR) & (mp["season"] == latest_ipl)]["player_id"])
    b = balls[~balls["reduced_overs"]]
    out = {}
    for role, r in rated.items():
        idcol, unit = ("striker_id", "is_faced") if role == "batter" else ("bowler_id", "is_legal")
        d = b[b[unit]]
        last_season = d.groupby(idcol)["season"].max()
        recent = d[d["season"] >= rules["recent_from"]]
        rec_p = recent[recent["is_pressure"]].groupby(idcol).size()
        ipl_ids = set(d.loc[d["competition_group"] == "IPL", idcol])
        full_ids = set(d.loc[(d["competition_group"] == "T20I") &
                             ((d["bowl_tier"] if role == "batter" else d["bat_tier"]) == "FULL_MEMBER"), idcol])
        steps, pool = [], r.copy()
        model_top = set(r[r["rank"] <= 10]["player_id"])

        def step(name, mask, why):
            nonlocal pool
            dropped = pool[~mask.reindex(pool.index, fill_value=False)]
            pool = pool[mask.reindex(pool.index, fill_value=False)]
            steps.append({"step": name, "rule": why, "remaining": int(len(pool)),
                          "model_top10_dropped": dropped[dropped["player_id"].isin(model_top)]["name"].tolist()})
        steps.append({"step": "Start", "rule": "every rated player", "remaining": int(len(pool)), "model_top10_dropped": []})
        step("Reliable under pressure", (pool["pressure_balls"] >= rules["min_pressure_balls"]) &
             (pool["pressure_units"] >= rules["min_pressure_units"]),
             f">= {rules['min_pressure_balls']} pressure balls and >= {rules['min_pressure_units']} innings/matches with pressure balls")
        step("Currently playing", pool["player_id"].map(last_season) >= rules["active_from"],
             f"played in {rules['active_from']} or later (data runs to {int(b['season'].max())})")
        step("Current form sample", pool["player_id"].map(rec_p).fillna(0) >= rules["recent_pressure_balls"],
             f">= {rules['recent_pressure_balls']} pressure balls since {rules['recent_from']}")
        step("Proven at the right level", pool["player_id"].isin(ipl_ids | full_ids),
             "has IPL or full-member T20I experience (not associate / A-team only)")
        if rules.get("exclude_current_rr"):
            step("Not already at RR", ~pool["player_id"].isin(rr_now), f"not in RR's {int(latest_ipl)} squad")
        pool = pool.copy()
        pool["re_rank"] = pool["prs_raw"].rank(ascending=False, method="first").astype(int)
        out[role] = {"pool_ids": [int(x) for x in pool["player_id"]], "steps": steps, "top10_after_funnel": pool.sort_values("re_rank").head(10)[
            ["re_rank", "name", "team", "prs", "label", "pressure_balls"]].to_dict("records"),
            "current_rr_in_pool_before_exclusion": sorted(r[r["player_id"].isin(rr_now) & r["eligible"]]["name"].tolist())}
    return out


def build(cfg: Any | None = None, rules: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = cfg or load_config()
    proc, raw = cfg.path("processed"), cfg.path("raw")
    m, teams, comps = _ipl(cfg)
    rr_ids = set(teams.loc[teams["team_name"] == RR, "team_id"].astype(int))
    balls = pd.read_parquet(proc / "balls_enriched.parquet",
                            columns=["batting_team_id", "bowling_team_id", "competition_group", "is_pressure", "reduced_overs",
                                     "season", "phase", "wpa", "team_runs", "is_legal", "is_team_wicket", "is_faced",
                                     "striker_id", "bowler_id", "bat_tier", "bowl_tier"])
    rated = {r: pd.read_parquet(proc / f"{r}s_rated.parquet") for r in ("batter", "bowler")}
    mapping = pd.read_csv(raw / "competition_team_player_mapping.csv")
    seasons = sorted(balls.loc[balls["competition_group"] == "IPL", "season"].unique().tolist())
    rules = rules or {"min_pressure_balls": 180, "min_pressure_units": 15, "active_from": 2025, "recent_from": 2025,
                      "recent_pressure_balls": 40, "exclude_current_rr": True}
    res_f = funnel(balls, rated, mapping, comps, rules)
    sp = pd.read_parquet(proc / "player_splits.parquet")
    need = {}
    for role, phase, min_balls in (("batter", "DEATH", 40), ("bowler", "PP", 40)):
        x = sp[(sp["role"] == role) & (sp["split"] == "phase") & (sp["split_value"] == phase) & sp["is_pressure"] &
               sp["player_id"].isin(res_f[role]["pool_ids"]) & (sp["balls"] >= min_balls)].copy()
        names = rated[role].set_index("player_id")
        x["name"], x["overall_rank_in_pool"] = x["player_id"].map(names["name"]), x["player_id"].map(names["rank"])
        cols = ["name", "balls", "sr", "dismissals", "wpa_per100"] if role == "batter" else \
               ["name", "balls", "econ", "wickets", "wpa_per100"]
        need[f"{role}_{phase}"] = x.sort_values("wpa_per100", ascending=False)[cols + ["overall_rank_in_pool"]].head(10).round(2).to_dict("records")
    return {"note": "Data pack only. 2027 retentions / releases are not in the data and are not used.",
            "season_record": season_record(m, rr_ids), "close_and_situation": close_and_situation(m, rr_ids),
            "pressure_rank_all": pressure_rank(balls, rr_ids, seasons),
            "pressure_rank_recent": pressure_rank(balls, rr_ids, [s for s in seasons if s >= 2025]),
            "squad": squad(mapping, comps, rated), "funnel_rules": rules,
            "funnel": res_f, "need_specialists": need}


if __name__ == "__main__":
    c = load_config()
    res = build(c)
    p = c.path("outputs_private") / "rr_research.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    print(f"written {p}")

"""RR auction targets: the brief asks for reliable players under pressure whom RR should back at the IPL 2027 auction.

  1. pressure rating (PRS) from rating.py - the data-only base
  2. structured funnel (rr_research.funnel) + the analyst's availability file (config/auction_pool.yaml)
  3. RR needs: RR's rank among the IPL teams per phase under pressure (batting and bowling), recent seasons
  4. RR fit: the player's shrunk pressure WPA per 100 balls in each phase, weighted by RR's need for that phase
  5. target score = (1 - w) x PRS percentile + w x fit percentile, both within the funnel pool
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import yaml

from . import rr_research as RRR

PHASES = ("PP", "MID", "DEATH")


def load_availability(cfg: Any) -> tuple[dict[int, str], dict[int, str], list[str]]:
    p = cfg.path("overrides").parent / "auction_pool.yaml"          # next to analyst_overrides.yaml
    doc = (yaml.safe_load(p.read_text(encoding="utf-8")) if p.exists() else None) or {}
    excl, incl, errors = {}, {}, []
    n = int(cfg.get("analyst.min_reason_chars", 10))
    for key, bucket in (("exclude", excl), ("include", incl)):
        for e in doc.get(key) or []:
            if not isinstance(e, dict) or "player_id" not in e:
                errors.append(f"{key}: entry without player_id: {e}")
            elif len(str(e.get("reason", "")).strip()) < n:
                errors.append(f"{key}: player {e['player_id']} refused - reason needs {n}+ characters")
            else:
                bucket[int(e["player_id"])] = str(e["reason"]).strip() + (f" ({e['source']})" if e.get("source") else "")
    return excl, incl, errors


def needs(rank_rows: dict[str, list[dict]]) -> dict[str, dict[str, float]]:
    """Need weight per side and phase: 0 = RR best of the teams, 1 = RR worst."""
    out = {}
    for side, rows in rank_rows.items():
        out[side] = {r["phase"]: round((r["rr_rank_of_teams"] - 1) / max(r["teams"] - 1, 1), 3)
                     for r in rows if r["phase"] in PHASES}
    return out


def fit_scores(splits: pd.DataFrame, ids: list[int], role: str, need_w: dict[str, float], k: float) -> pd.DataFrame:
    sp = splits[(splits["role"] == role) & (splits["split"] == "phase") & splits["is_pressure"] &
                splits["player_id"].isin(ids)].copy()
    sp["wpa_sum"] = sp["wpa_per100"].fillna(0) * sp["balls"] / 100
    sp["shrunk"] = 100 * sp["wpa_sum"] / (sp["balls"] + k)
    wide = sp.pivot_table(index="player_id", columns="split_value", values="shrunk", aggfunc="first").reindex(ids)
    balls = sp.pivot_table(index="player_id", columns="split_value", values="balls", aggfunc="first").reindex(ids)
    wsum = sum(need_w.get(ph, 0) for ph in PHASES) or 1.0
    fit = sum(need_w.get(ph, 0) * wide.get(ph, pd.Series(0.0, index=wide.index)).fillna(0.0) for ph in PHASES) / wsum
    best = pd.Series([max(PHASES, key=lambda ph: need_w.get(ph, 0) * (wide.loc[i, ph] if ph in wide and pd.notna(wide.loc[i, ph]) else -9))
                      for i in wide.index], index=wide.index)
    out = pd.DataFrame({"fit_raw": fit, "best_need_phase": best})
    for ph in PHASES:
        out[f"pressure_wpa100_{ph}"] = wide.get(ph)
        out[f"pressure_balls_{ph}"] = balls.get(ph)
    return out


def _pct(x: pd.Series) -> pd.Series:
    return 100 * x.rank(pct=True, method="average")


def status_labels(mapping: pd.DataFrame, comps: pd.DataFrame) -> pd.DataFrame:
    mp = mapping.merge(comps[["comp_id", "season"]], on="comp_id")
    mp["season"] = mp["season"].astype(str).str[:4].astype(int)
    ipl = mp[mp["comp_name"].str.match(r"^IPL\b")]
    latest = int(ipl["season"].max())
    now = ipl[ipl["season"] == latest].groupby("player_id")["team_name"].first()
    intl = mp[~mp["comp_name"].str.match(r"^IPL\b")]
    india = set(intl.loc[intl["team_name"].isin(["India", "India A"]), "player_id"])
    other = set(intl.loc[~intl["team_name"].isin(["India", "India A"]), "player_id"])
    ids = mp["player_id"].unique()
    st = pd.DataFrame(index=pd.Index(ids, name="player_id"))
    st["ipl_team_latest"] = now.reindex(st.index)
    st["auction_status"] = np.where(st["ipl_team_latest"] == RRR.RR, f"in RR {latest} squad (retain candidate)",
                                    np.where(st["ipl_team_latest"].notna(), f"with another IPL team in {latest} (check retention)",
                                             f"not in IPL {latest} (likely in the pool)"))
    st["overseas"] = np.where(st.index.isin(india), "Indian", np.where(st.index.isin(other), "Overseas",
                                                                    "unknown (no T20I in data)"))
    return st


def build(cfg: Any) -> dict[str, Any]:
    proc, raw = cfg.path("processed"), cfg.path("raw")
    m, teams, comps = RRR._ipl(cfg)
    rr_ids = set(teams.loc[teams["team_name"] == RRR.RR, "team_id"].astype(int))
    balls = pd.read_parquet(proc / "balls_enriched.parquet",
                            columns=["batting_team_id", "bowling_team_id", "competition_group", "is_pressure", "reduced_overs",
                                     "season", "phase", "wpa", "team_runs", "is_legal", "is_team_wicket", "is_faced",
                                     "striker_id", "bowler_id", "bat_tier", "bowl_tier"])
    rated = {r: pd.read_parquet(proc / f"{r}s_rated.parquet") for r in ("batter", "bowler")}
    mapping = pd.read_csv(raw / "competition_team_player_mapping.csv")
    splits = pd.read_parquet(proc / "player_splits.parquet")
    f = cfg.get("rr_target.funnel")
    rules = {"min_pressure_balls": f["min_pressure_balls"], "min_pressure_units": f["min_pressure_units"],
             "active_from": f["active_from"], "recent_from": f["recent_from"],
             "recent_pressure_balls": f["recent_pressure_balls"], "exclude_current_rr": f["exclude_current_rr"]}
    fun = RRR.funnel(balls, rated, mapping, comps, rules)
    seasons = sorted(balls.loc[balls["competition_group"] == "IPL", "season"].unique().tolist())
    ranks = RRR.pressure_rank(balls, rr_ids, [s for s in seasons if s >= int(cfg.get("rr_target.need_seasons_from", 2025))])
    nw = needs(ranks)
    excl, incl, errors = load_availability(cfg)
    st = status_labels(mapping, comps)
    w = float(cfg.get("rr_target.fit_weight", 0.3))
    k = float(cfg.get("rr_target.fit_shrink_balls", 60))
    out: dict[str, Any] = {"status_all": st.reset_index().to_dict("records"), "needs": nw, "rank_rows": ranks, "fit_weight": w, "availability_errors": errors,
                           "season_record": RRR.season_record(m, rr_ids), "close_and_situation": RRR.close_and_situation(m, rr_ids)}
    for role, side in (("batter", "batting"), ("bowler", "bowling")):
        r = rated[role].set_index("player_id")
        pool = list(fun[role]["pool_ids"])
        steps = list(fun[role]["steps"])
        gone = [p for p in pool if p in excl]
        pool = [p for p in pool if p not in excl]
        steps.append({"step": "Available at auction (your research)", "rule": "not on the exclude list in auction_pool.yaml",
                      "remaining": len(pool), "model_top10_dropped": [r.loc[p, "name"] for p in gone if p in r.index]})
        added = [p for p in incl if p in r.index and bool(r.loc[p, "eligible"]) and p not in pool]
        pool += added
        if added:
            steps.append({"step": "Added back (your research)", "rule": "include list in auction_pool.yaml",
                          "remaining": len(pool), "model_top10_dropped": []})
        fs = fit_scores(splits, pool, role, nw[side], k)
        al = cfg.get("rr_target.alignment") or {}
        need_ph = [ph for ph in PHASES if nw[side].get(ph, 0) >= float(al.get("min_need_weight", 0.4))]
        fs["aligned_phases"] = [", ".join(ph for ph in need_ph
                                          if pd.notna(fs.loc[i, f"pressure_balls_{ph}"]) and fs.loc[i, f"pressure_balls_{ph}"] >= al.get("min_phase_balls", 40)
                                          and pd.notna(fs.loc[i, f"pressure_wpa100_{ph}"]) and fs.loc[i, f"pressure_wpa100_{ph}"] > 0)
                                for i in fs.index]
        not_aligned = [p for p in pool if not fs.loc[p, "aligned_phases"]]
        steps.append({"step": "Aligned with RR's needs", "rule": f"positive under pressure (>= {al.get('min_phase_balls', 40)} balls) in a phase "
                      f"RR needs: {', '.join(need_ph)}", "remaining": len(pool) - len(not_aligned),
                      "model_top10_dropped": [r.loc[p, "name"] for p in not_aligned if p in r.index and r.loc[p, "rank"] <= 10]})
        out.setdefault("not_aligned", {})[role] = [{"name": r.loc[p, "name"], "reason": "no positive pressure record in "
                                                   + ", ".join(need_ph)} for p in not_aligned]
        pool = [p for p in pool if p not in not_aligned]
        fs = fs.loc[pool]
        t = r.loc[pool, ["name", "team", "prs", "prs_raw", "label", "p_top10", "pressure_balls"]].join(fs)
        t["prs_pct_pool"] = _pct(t["prs_raw"])
        t["fit"] = _pct(t["fit_raw"])
        t["target_score"] = (1 - w) * t["prs_pct_pool"] + w * t["fit"]
        t["target_rank"] = t["target_score"].rank(ascending=False, method="first").astype(int)
        t = t.join(st, how="left").sort_values("target_rank")
        t["availability_note"] = [incl.get(p, "") for p in t.index]
        out[role] = {"steps": steps, "need_weights": nw[side], "table": t.reset_index().round(3).to_dict("records")}
    return out


def apply_to_rated(cfg: Any, res: dict[str, Any]) -> None:
    """Write target_rank / target_score / fit / status into the rated parquets (dashboard + analyst layer)."""
    from . import store

    base = cfg.path("processed")
    for role in ("batter", "bowler"):
        r = store.read_table(f"{role}s_rated.parquet", base)
        t = pd.DataFrame(res[role]["table"]).set_index("player_id")
        for c in ("target_rank", "target_score", "fit", "prs_pct_pool", "best_need_phase"):
            r[c] = r["player_id"].map(t[c])
        st = pd.DataFrame(res["status_all"]).set_index("player_id")
        for c in ("auction_status", "overseas", "ipl_team_latest"):
            r[c] = r["player_id"].map(st[c])
        store.write_table(r, f"{role}s_rated.parquet", base)


def log_defaults(cfg: Any) -> None:
    from .decisions import log_autopilot_default

    for k in ("rr_target.use_for_final", "rr_target.funnel", "rr_target.fit_weight", "rr_target.need_seasons_from"):
        log_autopilot_default(cfg, k, cfg.get(k), "RR auction-target layer added on request")

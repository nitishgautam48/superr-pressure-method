"""RR lens: evidence numbers only, for the analyst's own "how each pick strengthens RR".

* RR's own pressure profile from the data (batting and bowling, by phase, chase vs defend, vs league).
* Per pick: pressure numbers by phase (death / middle), vs pace / spin, chase vs defend, already at RR or not.
"""
from __future__ import annotations

from typing import Any

import pandas as pd
import yaml

RR = "Rajasthan Royals"


def rr_profile(s: pd.DataFrame, rr_team_ids: set[int]) -> dict[str, Any]:
    ipl = s[(s["competition_group"] == "IPL") & ~s["reduced_overs"] & s["is_pressure"]]
    out = {}
    for side, col in (("batting", "batting_team_id"), ("bowling", "bowling_team_id")):
        rr = ipl[ipl[col].isin(rr_team_ids)]
        rows = []
        for (ph,), grp in [((p,), ipl[ipl["phase"] == p]) for p in ("PP", "MID", "DEATH")]:
            mine = rr[rr["phase"] == ph]
            rows.append({"phase": ph, **_rates(mine, "rr"), **_rates(grp, "league")})
        for sit, m in (("chase", ipl["innings"] == 2), ("set/defend", ipl["innings"] == 1)):
            rows.append({"phase": sit, **_rates(rr[m.reindex(rr.index, fill_value=False)], "rr"), **_rates(ipl[m], "league")})
        out[side] = rows
    return out


def _rates(d: pd.DataFrame, p: str) -> dict[str, float]:
    legal = max(int(d["is_legal"].sum()), 1)
    return {f"{p}_balls": int(len(d)), f"{p}_run_rate": round(6 * d["team_runs"].sum() / legal, 2),
            f"{p}_wicket_pct": round(100 * d["is_team_wicket"].sum() / legal, 2),
            f"{p}_wpa_per100": round(100 * d["wpa"].sum() / max(len(d), 1), 3)}


def pick_evidence(splits: pd.DataFrame, rated: pd.DataFrame, ids: list[int], role: str) -> list[dict[str, Any]]:
    sp = splits[(splits["role"] == role) & splits["is_pressure"]]
    r = rated.set_index("player_id")
    rate_col = "sr" if role == "batter" else "econ"
    out = []
    for p in ids:
        x = sp[sp["player_id"] == p]
        ev = {"player_id": int(p), "name": str(r.loc[p, "name"]), "plays_for_rr": bool(r.loc[p].get("plays_for_rr", False)),
              "rr_seasons": r.loc[p].get("rr_seasons"), "ipl_team": r.loc[p].get("ipl_team")}
        for _, y in x.iterrows():
            key = f"{y['split']}={y['split_value']}"
            ev[key] = {"balls": int(y["balls"]), rate_col: None if pd.isna(y[rate_col]) else round(float(y[rate_col]), 1),
                       "wpa_per100": None if pd.isna(y["wpa_per100"]) else round(float(y["wpa_per100"]), 3)}
        out.append(ev)
    return out


def build(s: pd.DataFrame, splits: pd.DataFrame, rated: dict[str, pd.DataFrame], finals: dict[str, Any],
          teams: pd.DataFrame, cfg: Any) -> dict[str, Any]:
    rr_ids = set(teams.loc[teams["team_name"] == RR, "team_id"].astype(int))
    p = cfg.path("rr_history")
    hist = yaml.safe_load(p.read_text(encoding="utf-8")) if p.exists() else {}
    return {"rr_history": hist or {}, "rr_pressure_profile": rr_profile(s, rr_ids),
            "batters": pick_evidence(splits, rated["batter"], [x["player_id"] for x in finals["batters_final"]], "batter"),
            "bowlers": pick_evidence(splits, rated["bowler"], [x["player_id"] for x in finals["bowlers_final"]], "bowler"),
            "note": "Evidence numbers only (pressure balls). The 'why RR' text is the analyst's."}

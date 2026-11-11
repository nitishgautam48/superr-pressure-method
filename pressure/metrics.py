"""Player metrics, computed separately for high-pressure and other balls (raw, unweighted
counts so every number can be checked by hand), plus splits by phase, chase vs defend and pace vs spin."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _div(a, b, k=1.0):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return np.where(b > 0, k * a / np.where(b > 0, b, 1), np.nan)


def batting_table(s: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Batting metrics grouped by ``by`` (must include striker_id). Dismissals use the dismissed player."""
    f = s[s["is_faced"]]
    g = f.groupby(by)
    t = pd.DataFrame({"balls": g.size(), "runs": g["runs"].sum(), "dots": g["runs"].apply(lambda r: (r == 0).sum()),
                      "boundaries": g.apply(lambda d: (d["is_four"] | d["is_six"]).sum(), include_groups=False),
                      "rae": g["rae_bat"].sum(), "wpa": g["wpa"].sum(),
                      "innings": g.apply(lambda d: d[["match_id", "innings"]].drop_duplicates().shape[0],
                                         include_groups=False)})
    outs = s[s["is_batter_out"]].copy()
    outs["striker_id"] = outs["dismissed_player_id"]
    t["dismissals"] = outs.groupby(by).size().reindex(t.index).fillna(0).astype(int)
    t["sr"] = _div(t["runs"], t["balls"], 100)
    t["balls_per_dismissal"] = _div(t["balls"], t["dismissals"])
    t["dot_pct"] = _div(t["dots"], t["balls"], 100)
    t["boundary_pct"] = _div(t["boundaries"], t["balls"], 100)
    t["rae_per100"] = _div(t["rae"], t["balls"], 100)
    t["wpa_per100"] = _div(t["wpa"], t["balls"], 100)
    return t.reset_index()


def bowling_table(s: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    g = s.groupby(by)
    t = pd.DataFrame({"deliveries": g.size(), "legal_balls": g["is_legal"].sum(), "runs_conceded": g["bowler_runs"].sum(),
                      "dots": g.apply(lambda d: ((d["team_runs"] == 0) & d["is_legal"]).sum(), include_groups=False),
                      "boundaries": g.apply(lambda d: (d["is_four"] | d["is_six"]).sum(), include_groups=False),
                      "wickets": g["is_bowler_wicket"].sum(), "rae_conc": g["rae_conc"].sum(), "wae": g["wae"].sum(),
                      "wpa": -g["wpa"].sum(),
                      "matches": g["match_id"].nunique()})
    t["econ"] = _div(t["runs_conceded"], t["legal_balls"], 6)
    t["dot_pct"] = _div(t["dots"], t["legal_balls"], 100)
    t["boundary_pct_conceded"] = _div(t["boundaries"], t["deliveries"], 100)
    t["balls_per_wicket"] = _div(t["legal_balls"], t["wickets"])
    t["rae_conc_per100"] = _div(t["rae_conc"], t["legal_balls"], 100)
    t["wae_per100"] = _div(t["wae"], t["legal_balls"], 100)
    t["wpa_per100"] = _div(t["wpa"], t["legal_balls"], 100)
    return t.reset_index()


def pressure_split(table: pd.DataFrame, id_col: str, cols: list[str]) -> pd.DataFrame:
    """Wide frame: <col>_p (pressure) and <col>_np (other balls) per player."""
    p = table[table["is_pressure"]].set_index(id_col)[cols].add_suffix("_p")
    n = table[~table["is_pressure"]].set_index(id_col)[cols].add_suffix("_np")
    return p.join(n, how="outer")


def splits(s: pd.DataFrame) -> pd.DataFrame:
    """Long table for profiles / RR lens: role, split (phase, chase/defend, pace/spin), pressure flag."""
    s = s.copy()
    s["situation"] = np.where(s["innings"] == 2, "chase", "set/defend")
    out = []
    for split in ("phase", "situation", "pace_or_spin"):
        bt = batting_table(s, ["striker_id", split, "is_pressure"]).rename(columns={"striker_id": "player_id", split: "split_value"})
        bt["role"], bt["split"] = "batter", split
        out.append(bt[["player_id", "role", "split", "split_value", "is_pressure", "balls", "runs", "sr", "dismissals",
                       "rae_per100", "wpa_per100"]])
        bw = bowling_table(s, ["bowler_id", split, "is_pressure"]).rename(columns={"bowler_id": "player_id", split: "split_value"})
        bw["role"], bw["split"] = "bowler", split
        bw = bw.rename(columns={"legal_balls": "balls"})
        out.append(bw[["player_id", "role", "split", "split_value", "is_pressure", "balls", "runs_conceded", "econ", "wickets",
                       "rae_conc_per100", "wpa_per100"]])
    return pd.concat(out, ignore_index=True)

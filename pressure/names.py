"""Player identity: exact mapping spelling (for the Unstop dropdowns) + "name (team, competition)" display."""
from __future__ import annotations

import pandas as pd


def player_directory(mapping: pd.DataFrame, comps: pd.DataFrame) -> pd.DataFrame:
    m = mapping.merge(comps[["comp_id", "season_year", "competition_group"]], on="comp_id", how="left")
    m = m.sort_values(["player_id", "season_year", "comp_id"])
    last = m.groupby("player_id").tail(1).set_index("player_id")
    ipl = m[m["competition_group"] == "IPL"].groupby("player_id").tail(1).set_index("player_id")
    intl = m[m["competition_group"] != "IPL"].groupby("player_id").tail(1).set_index("player_id")
    d = pd.DataFrame({"name": last["player_name"], "team": last["team_name"], "competition": last["comp_name"],
                      "ipl_team": ipl["team_name"].reindex(last.index), "intl_team": intl["team_name"].reindex(last.index)})
    rr = m[m["team_name"] == "Rajasthan Royals"]
    d["plays_for_rr"] = d.index.isin(rr["player_id"])
    d["rr_seasons"] = rr.groupby("player_id")["season_year"].apply(lambda x: ",".join(map(str, sorted(set(x))))) \
        .reindex(d.index)
    dup = d["name"].duplicated(keep=False)
    d["display"] = d["name"] + " (" + d["team"].astype(str) + ", " + d["competition"].astype(str) + ")"
    d["display_short"] = d["name"].where(~dup, d["display"])
    d.index.name = "player_id"
    return d

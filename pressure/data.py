"""Raw loaders + competition / team-tier tagging + match flags."""
from __future__ import annotations

import logging
import re
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)
FILES = ("ball_by_ball", "matches", "competitions", "teams", "playing_eleven", "competition_team_player_mapping")


def load_raw(cfg: Any) -> dict[str, pd.DataFrame]:
    raw = cfg.path("raw")
    missing = [f for f in FILES if not (raw / f"{f}.csv").exists()]
    if missing:
        raise FileNotFoundError(f"missing data files in {raw}: {missing}")
    out = {f: pd.read_csv(raw / f"{f}.csv", low_memory=False) for f in FILES}
    log.info("loaded raw: %s", {k: len(v) for k, v in out.items()})
    return out


def season_of(s: Any) -> int:
    m = re.match(r"(\d{4})", str(s))
    return int(m.group(1)) if m else -1


def tag_competitions(comps: pd.DataFrame, cfg: Any) -> pd.DataFrame:
    """competition_group = IPL | EMERGING | T20I; season as the first calendar year."""
    c = comps.copy()
    kw = [k.lower() for k in cfg.get("tiers.emerging_comp_keywords", [])]
    name = c["comp_name"].astype(str)
    is_ipl = name.str.match(r"^IPL\b")
    is_em = (c["match_format"] == "T20") & ~is_ipl & name.str.lower().map(lambda n: any(k in n for k in kw))
    c["competition_group"] = np.select([is_ipl, is_em], ["IPL", "EMERGING"], "T20I")
    c.loc[(c["match_format"] == "T20") & ~is_ipl & ~is_em, "competition_group"] = "EMERGING"
    c["season_year"] = c["season"].map(season_of)
    return c


def team_tiers(teams: pd.DataFrame, comps: pd.DataFrame, cfg: Any) -> pd.DataFrame:
    """Per (comp_id, team_id): tier = IPL | FULL_MEMBER | ASSOCIATE | A_TEAM."""
    full = set(cfg.get("tiers.full_members", []))
    suffix = cfg.get("tiers.a_team_suffix", " A")
    t = teams.merge(comps[["comp_id", "competition_group"]], on="comp_id", how="left")
    name = t["team_name"].astype(str)
    t["tier"] = np.select([t["competition_group"] == "IPL", name.str.endswith(suffix), name.isin(full)],
                          ["IPL", "A_TEAM", "FULL_MEMBER"], "ASSOCIATE")
    return t[["comp_id", "team_id", "team_name", "tier"]]


_MARGIN = re.compile(r"won\s+(?:by\s+)?(\d+)\s*(run|wkt|wicket)", re.I)


def parse_result(text: Any) -> tuple[str | None, int | None]:
    """'X Won by 7 wickets' -> ('wickets', 7); 'won by 20 runs (D/L)' -> ('runs', 20)."""
    m = _MARGIN.search(str(text))
    if not m:
        return None, None
    return ("runs" if m.group(2).lower() == "run" else "wickets"), int(m.group(1))


def tag_matches(matches: pd.DataFrame, comps: pd.DataFrame, tiers: pd.DataFrame, balls: pd.DataFrame,
                cfg: Any) -> pd.DataFrame:
    """Match table with competition group, season, team tiers and status flags."""
    m = matches.merge(comps[["comp_id", "comp_name", "match_format", "competition_group", "season_year"]],
                      on="comp_id", how="left")
    tmap = tiers.set_index(["comp_id", "team_id"])["tier"]
    for side in ("home", "away"):
        m[f"{side}_tier"] = [tmap.get((c, t), "ASSOCIATE") for c, t in zip(m["comp_id"], m[f"{side}_team_id"])]
    res = m["result"].astype(str)
    low = res.str.lower()
    m["no_result"] = low.str.contains("no result")
    m["abandoned"] = low.str.contains("abandon")
    m["dls"] = m["revised_target"].notna() | low.str.contains(r"d/l|dls", regex=True)
    so = set(balls.loc[balls["innings"].isin(cfg.get("data.super_over_innings", [3, 4])), "match_id"])
    m["super_over"] = m["match_id"].isin(so) | low.str.contains("super over")
    m["tied"] = low.str.contains(r"\btie[d]?\b", regex=True) | m["super_over"]
    parsed = res.map(parse_result)
    m["margin_type"] = [p[0] for p in parsed]
    m["margin"] = [p[1] for p in parsed]
    m["has_balls"] = m["match_id"].isin(set(balls["match_id"]))
    m["decided"] = m["win_team_id"].notna() & ~m["no_result"] & ~m["abandoned"]
    return m

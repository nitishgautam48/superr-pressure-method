"""Ratings: Pressure Reliability Score from three shrunk components.

Per-ball value (common unit for pressure and other balls):
  batter  v = w_wpa * WPA / sd(WPA) + w_rae * (runs - xruns) / sd(RAE)
  bowler  v = w_wpa * (-WPA) / sd(WPA) + w_rae * (xruns_conc - runs_conc) / sd(RAE_conc)
Ball weights = season recency x associate-opposition weight x emerging-competition weight (all config).

A) PRESSURE QUALITY   empirical-Bayes shrunk mean v on pressure balls (k pseudo-balls toward the role mean)
B) PRESSURE DELTA     shrunk mean v on pressure balls minus the player's OWN shrunk mean v on other balls
C) CONSISTENCY        over matches with >= N pressure balls: share of matches with positive pressure value
                      (shrunk toward the role mean by match count) and a low spread (MAD) of per-match value
PRS = wA*z(A) + wB*z(B) + wC*z(C), z-scored within the eligible pool of the role, shown as a 0-100 percentile.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

ROLE_COLS = {"batter": ("striker_id", "is_faced"), "bowler": ("bowler_id", None)}


def ball_weights(s: pd.DataFrame, cfg: Any, overrides: dict[str, Any] | None = None) -> pd.Series:
    o = overrides or {}
    g = lambda k, d=None: o.get(k, cfg.get(f"rating.{k}", d))  # noqa: E731
    w = pd.Series(1.0, index=s.index)
    if g("recency_on", True):
        sw = {int(k): float(v) for k, v in (g("season_weights", {}) or {}).items()}
        w *= s["season"].map(sw).fillna(min(sw.values()) if sw else 1.0)
    return w * _opp_weight(s, g) * np.where(s["competition_group"] == "EMERGING", float(g("emerging_weight", 0.7)), 1.0)


def _opp_weight(s: pd.DataFrame, g) -> np.ndarray:
    assoc = (s["bat_tier"] == "ASSOCIATE") | (s["bowl_tier"] == "ASSOCIATE")
    aw = 0.0 if g("associate_exclude", False) else float(g("associate_weight", 0.5))
    return np.where(assoc, aw, 1.0)


def ball_values(s: pd.DataFrame, role: str, cfg: Any, blend: dict[str, float] | None = None) -> pd.Series:
    b = blend or cfg.get("rating.value_blend", {"wpa": 0.5, "rae": 0.5})
    if role == "batter":
        f = s["is_faced"]
        wpa, rae = s.loc[f, "wpa"], s.loc[f, "rae_bat"]
        v = b["wpa"] * wpa / wpa.std() + b["rae"] * rae / rae.std()
        return v.reindex(s.index)
    wpa, rae = -s["wpa"], -s["rae_conc"]
    return b["wpa"] * wpa / wpa.std() + b["rae"] * rae / rae.std()


def _shrunk_mean(v: pd.Series, w: pd.Series, key: pd.Series, k: float) -> tuple[pd.Series, pd.Series, float]:
    """Weighted mean per key shrunk toward the weighted population mean with k pseudo-balls."""
    ok = v.notna() & (w > 0)
    vw = (v * w)[ok]
    num, den = vw.groupby(key[ok]).sum(), w[ok].groupby(key[ok]).sum()
    mu = float(num.sum() / den.sum()) if den.sum() else 0.0
    return (num + k * mu) / (den + k), den, mu


def rate(s: pd.DataFrame, role: str, cfg: Any, overrides: dict[str, Any] | None = None,
         pressure_col: str = "is_pressure") -> pd.DataFrame:
    """Rating table for one role (one row per player). ``overrides`` replace rating.* config keys."""
    o = overrides or {}
    g = lambda k, d=None: o.get(k, cfg.get(f"rating.{k}", d))  # noqa: E731
    pid_col, faced = ROLE_COLS[role]
    d = s[~s["reduced_overs"]]
    if faced:
        d = d[d[faced]]
    w = ball_weights(d, cfg, o)
    v = ball_values(d, role, cfg, g("value_blend"))
    k = float(g("shrink_k_balls", 100))
    pid = d[pid_col]
    pres = d[pressure_col].astype(bool)

    A, n_p, mu_p = _shrunk_mean(v[pres], w[pres], pid[pres], k)
    base, n_np, mu_np = _shrunk_mean(v[~pres], w[~pres], pid[~pres], k)
    out = pd.DataFrame({"A": A, "baseline_value": base, "w_pressure_balls": n_p, "w_other_balls": n_np})
    out["A"] = out["A"].fillna(mu_p)
    out["baseline_value"] = out["baseline_value"].fillna(mu_np)
    out["B"] = out["A"] - out["baseline_value"]
    # raw counts for eligibility (legal balls for bowlers, faced balls for batters)
    cnt = d[pres & (d["is_legal"] if role == "bowler" else True)]
    out["pressure_balls"] = cnt.groupby(cnt[pid_col]).size().reindex(out.index).fillna(0).astype(int)
    unit = ["match_id", "innings"] if role == "batter" else ["match_id"]
    out["pressure_units"] = d[pres].groupby(pid[pres])[unit].apply(lambda x: x.drop_duplicates().shape[0]) \
        .reindex(out.index).fillna(0).astype(int)
    allb = d[d["is_legal"]] if role == "bowler" else d
    out["total_balls"] = allb.groupby(allb[pid_col]).size().reindex(out.index).fillna(0).astype(int)
    # sample-size uncertainty of A (normal approximation; bootstrap in S4)
    vv = (v[pres] - v[pres].groupby(pid[pres]).transform("mean")) ** 2
    sd = np.sqrt(vv.groupby(pid[pres]).mean()).reindex(out.index)
    out["A_se"] = sd.fillna(v[pres].std()) / np.sqrt(out["w_pressure_balls"].fillna(0) + k)

    # C) consistency over matches with >= N pressure balls
    pm = player_match(d, v, w, pres, pid_col, role)
    pm = pm[pm["pressure_balls"] >= int(g("consistency_min_pressure_balls", 3))]
    km = float(g("consistency_shrink_matches", 5))
    grp = pm.groupby("player_id")
    share = grp["value"].apply(lambda x: (x > 0).mean())
    nm = grp.size()
    mu_share = float((pm["value"] > 0).mean()) if len(pm) else 0.5
    out["consistency_matches"] = nm.reindex(out.index).fillna(0).astype(int)
    out["share_positive"] = ((share * nm + km * mu_share) / (nm + km)).reindex(out.index).fillna(mu_share)
    codes, uniq = pd.factorize(pm["player_id"])
    mad = pd.Series(wmad(pm["value"].to_numpy(float), np.ones(len(pm)), codes, len(uniq)), index=uniq)
    out["mad"] = mad.reindex(out.index)
    out["mad"] = out["mad"].fillna(out["mad"].median())

    el = g("eligibility", {}) or {}
    if role == "batter":
        out["eligible"] = (out["pressure_balls"] >= el.get("batter_min_pressure_balls", 120)) & \
                          (out["pressure_units"] >= el.get("batter_min_pressure_innings", 12))
    else:
        out["eligible"] = (out["pressure_balls"] >= el.get("bowler_min_pressure_balls", 120)) & \
                          (out["pressure_units"] >= el.get("bowler_min_pressure_matches", 12))
    return score(out, cfg, o)


def wmedian(values: np.ndarray, weights: np.ndarray, groups: np.ndarray, n_groups: int) -> np.ndarray:
    """Weighted median per group (vectorised). With unit or integer weights this equals numpy's median of the
    expanded sample (mid-point of the two middle values when the cumulative weight hits exactly half)."""
    out = np.full(n_groups, np.nan)
    keep = weights > 0
    v, w, g = values[keep], weights[keep], groups[keep]
    if not len(v):
        return out
    o = np.lexsort((v, g))
    v, w, g = v[o], w[o], g[o]
    start = np.r_[0, np.flatnonzero(np.diff(g)) + 1]
    sizes = np.diff(np.r_[start, len(v)])
    cw = np.cumsum(w)
    base = np.r_[0.0, cw[start[1:] - 1]]
    rel = cw - np.repeat(base, sizes)
    half = np.repeat(np.add.reduceat(w, start) / 2, sizes)
    cand = np.flatnonzero(rel >= half - 1e-12)
    cg = np.searchsorted(start, cand, side="right") - 1
    uniq, idx = np.unique(cg, return_index=True)
    first = np.full(len(start), 0)
    first[uniq] = cand[idx]
    end = np.r_[start[1:], len(v)]
    exact = np.isclose(rel[first], half[first]) & (first + 1 < end)
    nxt = np.minimum(first + 1, len(v) - 1)
    out[g[start]] = np.where(exact, (v[first] + v[nxt]) / 2, v[first])
    return out


def wmad(values: np.ndarray, weights: np.ndarray, groups: np.ndarray, n_groups: int) -> np.ndarray:
    med = wmedian(values, weights, groups, n_groups)
    return wmedian(np.abs(values - med[groups]), weights, groups, n_groups)


def player_match(d: pd.DataFrame, v: pd.Series, w: pd.Series, pres: pd.Series, pid_col: str, role: str) -> pd.DataFrame:
    x = pd.DataFrame({"player_id": d[pid_col], "match_id": d["match_id"], "v": v, "w": w, "p": pres,
                      "season": d["season"], "competition_group": d["competition_group"]})
    x = x[x["p"]]
    g = x.groupby(["player_id", "match_id"])
    return pd.DataFrame({"pressure_balls": g.size(), "value": g["v"].mean(), "value_sum": g["v"].sum(),
                         "season": g["season"].first(), "competition_group": g["competition_group"].first()}) \
        .reset_index().assign(role=role)


def _z(x: pd.Series, ref: pd.Series) -> pd.Series:
    sd = float(ref.std())
    return (x - float(ref.mean())) / (sd if sd > 0 else 1.0)


def score(out: pd.DataFrame, cfg: Any, overrides: dict[str, Any] | None = None) -> pd.DataFrame:
    """C from its parts, PRS from z-scores within the eligible pool, 0-100 percentile, rank."""
    o = overrides or {}
    g = lambda k, d=None: o.get(k, cfg.get(f"rating.{k}", d))  # noqa: E731
    out = out.copy()
    el = out["eligible"] if out["eligible"].any() else pd.Series(True, index=out.index)
    cb = g("consistency_blend", {"share_positive": 0.5, "low_spread": 0.5})
    out["C"] = cb["share_positive"] * _z(out["share_positive"], out.loc[el, "share_positive"]) + \
        cb["low_spread"] * _z(-out["mad"], -out.loc[el, "mad"])
    W = g("weights", {"A": 0.5, "B": 0.25, "C": 0.25})
    for c in ("A", "B", "C"):
        out[f"z_{c}"] = _z(out[c], out.loc[el, c])
    out["prs_raw"] = sum(float(W[c]) * out[f"z_{c}"] for c in ("A", "B", "C"))
    ref = np.sort(out.loc[el, "prs_raw"].to_numpy())
    out["prs"] = (100 * np.searchsorted(ref, out["prs_raw"].to_numpy(), side="right") / max(len(ref), 1)).clip(0, 100).round(1)
    out["rank"] = out.loc[out["eligible"], "prs_raw"].rank(ascending=False, method="first").reindex(out.index)
    out["rank"] = out["rank"].astype("Int64")
    return out.sort_values(["rank", "prs_raw"], ascending=[True, False], na_position="last")

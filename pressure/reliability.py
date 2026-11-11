"""Reliability: bootstrap by match, split-half persistence, sensitivity, IPL vs T20I, labels.

The bootstrap resamples MATCHES with replacement. To make 1,000 draws fast, each role is pre-aggregated to
player x match sums (sum of w*v and w on pressure / other balls, per-match pressure value); a draw is a vector
of match counts and every component is a weighted sum of those rows. A draw with every match counted once
reproduces rating.rate() exactly (tested). Eligibility is fixed to the full-sample eligible set.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from . import pressure as PR, rating as RT

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------------------------------
def match_aggregates(s: pd.DataFrame, role: str, cfg: Any, pressure_col: str = "is_pressure",
                     overrides: dict | None = None) -> dict[str, Any]:
    pid_col, faced = RT.ROLE_COLS[role]
    d = s[~s["reduced_overs"]]
    if faced:
        d = d[d[faced]]
    o = overrides or {}
    w = RT.ball_weights(d, cfg, o).to_numpy(float)
    v = RT.ball_values(d, role, cfg, o.get("value_blend", cfg.get("rating.value_blend"))).to_numpy(float)
    pres = d[pressure_col].astype(bool).to_numpy()
    x = pd.DataFrame({"p": d[pid_col].to_numpy(), "m": d["match_id"].to_numpy(), "pres": pres,
                      "wv": w * v, "w": w, "v": v})
    xp, xo = x[x["pres"]], x[~x["pres"]]
    gp = xp.groupby(["p", "m"]).agg(Sp=("wv", "sum"), Wp=("w", "sum"), pv=("v", "mean"), pb=("v", "size"))
    go = xo.groupby(["p", "m"]).agg(So=("wv", "sum"), Wo=("w", "sum"))
    agg = gp.join(go, how="outer").fillna({"Sp": 0, "Wp": 0, "So": 0, "Wo": 0, "pb": 0}).reset_index()
    players, pc = np.unique(agg["p"].to_numpy(), return_inverse=True)
    matches, mc = np.unique(agg["m"].to_numpy(), return_inverse=True)
    return {"role": role, "players": players, "pc": pc, "matches": matches, "mc": mc,
            "Sp": agg["Sp"].to_numpy(float), "Wp": agg["Wp"].to_numpy(float), "So": agg["So"].to_numpy(float),
            "Wo": agg["Wo"].to_numpy(float), "pv": agg["pv"].to_numpy(float), "pb": agg["pb"].to_numpy(float)}


def fast_scores(agg: dict, counts: np.ndarray, eligible: np.ndarray, cfg: Any, weights: dict | None = None) -> np.ndarray:
    """PRS raw per player (np.nan for ineligible) for one vector of match counts."""
    g = lambda k, dflt=None: cfg.get(f"rating.{k}", dflt)  # noqa: E731
    k, km = float(g("shrink_k_balls", 100)), float(g("consistency_shrink_matches", 5))
    n = len(agg["players"])
    c = counts[agg["mc"]]
    Sp, Wp = np.bincount(agg["pc"], c * agg["Sp"], n), np.bincount(agg["pc"], c * agg["Wp"], n)
    So, Wo = np.bincount(agg["pc"], c * agg["So"], n), np.bincount(agg["pc"], c * agg["Wo"], n)
    mu_p = Sp.sum() / max(Wp.sum(), 1e-12)
    mu_o = So.sum() / max(Wo.sum(), 1e-12)
    A = np.where(Wp > 0, (Sp + k * mu_p) / (Wp + k), mu_p)
    base = np.where(Wo > 0, (So + k * mu_o) / (Wo + k), mu_o)
    B = A - base
    q = (agg["pb"] >= int(g("consistency_min_pressure_balls", 3))) & ~np.isnan(agg["pv"])
    cq = c * q
    nm = np.bincount(agg["pc"], cq, n)
    pos = np.bincount(agg["pc"], cq * (np.nan_to_num(agg["pv"]) > 0), n)
    mu_s = pos.sum() / max(nm.sum(), 1e-12)
    share = (pos + km * mu_s) / (nm + km)
    mad = RT.wmad(np.nan_to_num(agg["pv"]), cq.astype(float), agg["pc"], n)
    mad = np.where(np.isnan(mad), np.nanmedian(mad), mad)
    el = eligible

    def z(a):
        sd = a[el].std(ddof=1)
        return (a - a[el].mean()) / (sd if sd > 0 else 1.0)
    cb = g("consistency_blend", {"share_positive": 0.5, "low_spread": 0.5})
    C = cb["share_positive"] * z(share) + cb["low_spread"] * z(-mad)
    W = weights or g("weights")
    prs = W["A"] * z(A) + W["B"] * z(B) + W["C"] * z(C)
    return np.where(el, prs, np.nan)


def bootstrap(agg: dict, eligible_ids: set, cfg: Any) -> pd.DataFrame:
    rng = np.random.default_rng(int(cfg.get("seed", 42)))
    n_draws = int(cfg.get("reliability.bootstrap_draws", 1000))
    top_n = int(cfg.get("rating.top_n", 10))
    el = np.isin(agg["players"], list(eligible_ids))
    nmat = len(agg["matches"])
    ranks = np.full((n_draws, len(agg["players"])), np.nan)
    for d in range(n_draws):
        counts = rng.multinomial(nmat, np.full(nmat, 1 / nmat)).astype(float)
        prs = fast_scores(agg, counts, el, cfg)
        r = pd.Series(prs).rank(ascending=False, method="first").to_numpy()
        ranks[d] = r
    q = (1 - float(cfg.get("reliability.interval", 0.8))) / 2
    R = ranks[:, el]
    out = pd.DataFrame({"player_id": agg["players"][el], "boot_rank_median": np.median(R, axis=0),
                        "boot_rank_lo": np.quantile(R, q, axis=0), "boot_rank_hi": np.quantile(R, 1 - q, axis=0),
                        "p_top10": 100 * np.mean(R <= top_n, axis=0)})
    hist = {int(p): np.bincount(np.clip(R[:, i], 1, 40).astype(int), minlength=41)[1:].tolist()
            for i, p in enumerate(agg["players"][el])}
    out.attrs["rank_hist"] = hist
    return out


# ---------------------------------------------------------------------------------------------------
def variants(s: pd.DataFrame, role: str, cfg: Any, base: pd.DataFrame) -> dict[str, pd.Series]:
    """Rank of each player under each sensitivity variant (eligible players only)."""
    out: dict[str, pd.Series] = {}
    for pct in cfg.get("reliability.thresholds_pct", [10, 20, 30]):
        col = f"_p{pct}"
        s[col] = PR.classify(s[["pressure_index"]], cfg, high_pct=pct)["is_pressure"].to_numpy()
        out[f"threshold_{pct}%"] = RT.rate(s, role, cfg, None, col)["rank"]
        s.drop(columns=[col], inplace=True)
    out["rule_based"] = RT.rate(s, role, cfg, None, "is_pressure_rule")["rank"]
    for name, w in (cfg.get("reliability.weight_sets") or {}).items():
        out[f"weights_{name}"] = RT.score(base, cfg, {"weights": w})["rank"]
    out["associates_excluded"] = RT.rate(s, role, cfg, {"associate_exclude": True})["rank"]
    out["no_recency"] = RT.rate(s, role, cfg, {"recency_on": False})["rank"]
    return out


def label(p_boot: float, p_var: float, cfg: Any) -> str:
    """Robust if top 10 in >= 80% of bootstraps AND of variants; Likely >= 50% on both; else Sensitive."""
    m = min(p_boot, p_var)
    for thr, lab in cfg.get("reliability.labels", [[80, "Robust"], [50, "Likely"], [0, "Sensitive"]]):
        if m >= thr:
            return lab
    return "Sensitive"


def _relaxed(cfg: Any) -> dict:
    el = dict(cfg.get("rating.eligibility"))
    f = float(cfg.get("reliability.subset_eligibility_factor", 0.5))
    return {"eligibility": {k: int(round(v * f)) for k, v in el.items()}}


def _corr(a: pd.Series, b: pd.Series) -> dict[str, Any]:
    j = pd.concat([a, b], axis=1).dropna()
    if len(j) < 5:
        return {"n": int(len(j)), "pearson": None, "spearman": None}
    return {"n": int(len(j)), "pearson": round(float(pearsonr(j.iloc[:, 0], j.iloc[:, 1])[0]), 3),
            "spearman": round(float(spearmanr(j.iloc[:, 0], j.iloc[:, 1])[0]), 3)}


def split_half(s: pd.DataFrame, role: str, cfg: Any) -> dict[str, Any]:
    sh = cfg.get("reliability.split_half")
    ov = _relaxed(cfg)
    out = {}
    for name, (ma, mb) in {"seasons": (s["season"].isin(sh["first"]), s["season"].isin(sh["second"])),
                           "odd_even_matches": (s["match_id"] % 2 == 1, s["match_id"] % 2 == 0)}.items():
        ra, rb = RT.rate(s[ma], role, cfg, ov), RT.rate(s[mb], role, cfg, ov)
        both = ra.index[ra["eligible"]].intersection(rb.index[rb["eligible"]])
        out[name] = {c: _corr(ra.loc[both, c], rb.loc[both, c]) for c in ("A", "B", "C", "prs_raw")}
        out[name]["points"] = [{"player_id": int(p), "A_first": round(float(ra.loc[p, "A"]), 4),
                                "A_second": round(float(rb.loc[p, "A"]), 4), "B_first": round(float(ra.loc[p, "B"]), 4),
                                "B_second": round(float(rb.loc[p, "B"]), 4)} for p in both]
    return out


def competition_split(s: pd.DataFrame, role: str, cfg: Any) -> pd.DataFrame:
    ov = _relaxed(cfg)
    ipl = s["competition_group"] == "IPL"
    t20i_full = (s["competition_group"] == "T20I") & (s["bat_tier"] == "FULL_MEMBER") & (s["bowl_tier"] == "FULL_MEMBER")
    ri, rt = RT.rate(s[ipl], role, cfg, ov), RT.rate(s[t20i_full], role, cfg, ov)
    out = pd.DataFrame({"rank_ipl": ri["rank"], "prs_ipl": ri["prs"], "rank_t20i": rt["rank"], "prs_t20i": rt["prs"]})
    out["rank_gap"] = (out["rank_ipl"] - out["rank_t20i"]).abs()
    gap = int(cfg.get("reliability.dependence_rank_gap", 15))
    out["competition_dependent"] = out["rank_gap"] > gap
    return out


def blend_diagnostic(s: pd.DataFrame, role: str, cfg: Any) -> list[dict[str, Any]]:
    """Split-half persistence of A, B, C, PRS for each WPA weight in the value blend (evidence for the choice)."""
    sh = cfg.get("reliability.split_half")
    halves = {"seasons": (s["season"].isin(sh["first"]), s["season"].isin(sh["second"])),
              "odd_even_matches": (s["match_id"] % 2 == 1, s["match_id"] % 2 == 0)}
    rows = []
    for wpa in cfg.get("reliability.blend_grid_wpa", [0.5, 0.25, 0.0]):
        ov = {**_relaxed(cfg), "value_blend": {"wpa": float(wpa), "rae": 1 - float(wpa)}}
        for name, (ma, mb) in halves.items():
            ra, rb = RT.rate(s[ma], role, cfg, ov), RT.rate(s[mb], role, cfg, ov)
            both = ra.index[ra["eligible"]].intersection(rb.index[rb["eligible"]])
            rows.append({"wpa_weight": float(wpa), "split": name, "n": int(len(both)),
                         **{c: _corr(ra.loc[both, c], rb.loc[both, c])["pearson"] for c in ("A", "B", "C", "prs_raw")}})
    return rows

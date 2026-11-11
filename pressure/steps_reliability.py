"""S4 step: bootstrap, sensitivity, split-half, IPL vs T20I, labels, analyst layer, RR lens, figures 6-12."""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from . import analyst, figures as F, rating as RT, reliability as RL, rr_lens, store

log = logging.getLogger(__name__)
LIMITATIONS = [
    "Team totals, targets and required rates are reconstructed because extras' run values are missing "
    "(about +/-3 runs); batter runs, balls, dismissals and boundaries are exact.",
    "Pressure samples are small (eligibility starts at 120 pressure balls); shrinkage pulls small samples toward "
    "the role average, and the bootstrap intervals show the remaining uncertainty.",
    "T20I includes associate nations; balls involving associates are down-weighted (0.5).",
    "IPL 2023+ is the impact-player era; batting depth and bowling usage changed mid-sample.",
    "No pitch, weather, injury or match-up intent data; context models use phase, wickets, required rate, "
    "pace/spin, batting hand, competition and opposition tier only.",
    "Rain-reduced matches are excluded from WP training and ratings.",
]


def run(ctx: Any) -> None:
    cfg, t = ctx.cfg, ctx.tables
    s = t["state"]
    base = cfg.path("processed")
    rated = {"batter": t["batters"].set_index("player_id"), "bowler": t["bowlers"].set_index("player_id")}
    sens, pers, comp_corr = {}, {}, {}
    for role in ("batter", "bowler"):
        r = rated[role]
        elig = set(r.index[r["eligible"]])
        agg = RL.match_aggregates(s, role, cfg)
        boot = RL.bootstrap(agg, elig, cfg).set_index("player_id")
        rated_base = RT.rate(s, role, cfg)
        var = RL.variants(s, role, cfg, rated_base)
        vt = pd.DataFrame(var)
        top_n = int(cfg.get("rating.top_n", 10))
        r = r.join(boot)
        r["variant_top10_pct"] = (100 * (vt <= top_n).sum(axis=1) / vt.shape[1]).reindex(r.index)
        r["label"] = [RL.label(pb, pv, cfg) if e else None
                      for pb, pv, e in zip(r["p_top10"].fillna(0), r["variant_top10_pct"].fillna(0), r["eligible"])]
        cs = RL.competition_split(s, role, cfg)
        r = r.join(cs)
        rule_rank = var["rule_based"]
        both = r.index[r["eligible"]].intersection(rule_rank.dropna().index)
        comp_corr[role] = round(float(spearmanr(r.loc[both, "rank"], rule_rank.loc[both])[0]), 3) if len(both) > 5 else None
        show = r[r["eligible"]].sort_values("rank").head(25).index
        sens[role] = {"variants": list(var), "matrix": [{"player_id": int(p), "name": str(r.loc[p, "name"]),
                                                          "model_rank": int(r.loc[p, "rank"]),
                                                          **{k: (None if pd.isna(v.get(p)) else int(v.get(p)))
                                                             for k, v in var.items()}} for p in show],
                      "top10_frequency": {int(p): round(float(r.loc[p, "variant_top10_pct"]), 1) for p in show},
                      "rank_hist": {int(k): v for k, v in boot.attrs.get("rank_hist", {}).items() if k in set(show)},
                      "competition_corr_spearman": _comp_corr(cs)}
        pers[role] = RL.split_half(s, role, cfg)
        pers[role]["blend_diagnostic"] = RL.blend_diagnostic(s, role, cfg)
        rated[role] = r
    # labels count + honest persistence verdict
    for role in pers:
        a = pers[role]["seasons"]["A"]["pearson"]
        b = pers[role]["seasons"]["B"]["pearson"]
        pers[role]["verdict"] = {"quality_persistence_r": a, "delta_persistence_r": b,
                                 "delta_is_weak": b is None or b < 0.2,
                                 "note": "If delta persistence is weak, lean on quality (A) and consistency (C)."}
    pers["limitations"] = LIMITATIONS
    store.write_json(sens, "sensitivity.json", base)
    store.write_json(pers, "persistence.json", base)
    sep = store.read_json("pressure_separation.json", base)
    sep["rank_correlation_of_ratings"] = {"leverage_vs_rule_spearman": comp_corr}
    store.write_json({k: v for k, v in sep.items() if k != "schema_version"}, "pressure_separation.json", base)

    out = {role: rated[role].reset_index() for role in rated}
    finals = analyst.apply(out, cfg)
    for role in ("batter", "bowler"):
        prov = {x["player_id"]: x["provenance"] for x in finals[f"{role}s_final"]}
        out[role]["provenance"] = out[role]["player_id"].map(prov)
        out[role]["in_final"] = out[role]["player_id"].isin(prov)
        store.write_table(out[role], f"{role}s_rated.parquet", base)
    store.write_json(finals, "final_lists.json", base)
    store.write_json({"model": {r: finals[f"{r}s_model"] for r in ("batter", "bowler")},
                      "final": {r: finals[f"{r}s_final"] for r in ("batter", "bowler")},
                      "changes": {r: finals[f"{r}s_changes"] for r in ("batter", "bowler")},
                      "refused_overrides": finals["refused"]}, "model_vs_final.json", base)
    rr = rr_lens.build(s, t["splits"], out, finals, t["raw"]["teams"], cfg)
    store.write_json(rr, "rr_lens.json", base)

    dq = store.read_json("data_quality.json", base)
    for role, tag in (("batter", "a"), ("bowler", "b")):
        F.fig_top10_table(cfg, out[role], role, tag)
        F.fig_pressure_vs_baseline(cfg, out[role], role, tag)
        F.fig_consistency(cfg, out[role], t["player_match"], role, tag)
        F.fig_rank_stability(cfg, out[role], role, tag)
    F.fig_persistence(cfg, pers)
    F.fig_ipl_vs_t20i(cfg, out)
    F.fig_validation(cfg, dq, t["validation"])
    F.export_guard(cfg)
    t["batters"], t["bowlers"], t["finals"] = out["batter"], out["bowler"], finals
    lab = {role: out[role].loc[out[role]["rank"] <= 10, "label"].value_counts().to_dict() for role in out}
    ctx.metrics.update(top10_labels=lab, persistence_A_r={r: pers[r]["seasons"]["A"]["pearson"] for r in ("batter", "bowler")},
                       persistence_B_r={r: pers[r]["seasons"]["B"]["pearson"] for r in ("batter", "bowler")},
                       leverage_vs_rule_rank_rho=comp_corr, refused_overrides=len(finals["refused"]))
    log.info("S4: labels %s; persistence A %s, B %s", lab, ctx.metrics["persistence_A_r"], ctx.metrics["persistence_B_r"])


def _comp_corr(cs: pd.DataFrame) -> dict[str, Any]:
    j = cs.dropna(subset=["rank_ipl", "rank_t20i"])
    if len(j) < 5:
        return {"n": int(len(j)), "spearman": None}
    return {"n": int(len(j)), "spearman": round(float(spearmanr(j["rank_ipl"], j["rank_t20i"])[0]), 3),
            "dependent_players": int(j["competition_dependent"].sum())}

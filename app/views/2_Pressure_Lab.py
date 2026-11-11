"""Pressure lab: any match ball by ball; threshold and definition with live league separation stats."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from app import charts, components as C, data as D, state
from pressure import pressure as PR

MODE = state.theme_mode()
C.page_header("Pressure lab", "Win probability and pressure for any match; change the definition and watch the league stats.")
if D.showcase_mode():
    sep = D.js("pressure_separation.json")
    hi, lo, rb = sep["leverage"]["high"], sep["leverage"]["other"], sep["rule_based"]["pressure"]
    C.section_header("League separation", f"Leverage definition · top {sep['threshold']['high_pct']}%")
    st.dataframe(pd.DataFrame([{"Metric": n, "High pressure": hi[k], "Other balls": lo[k], "Rule-based": rb[k]}
                               for n, k in (("Share of balls %", "share_pct"), ("Dot %", "dot_pct"), ("Boundary %", "boundary_pct"),
                                            ("Wicket %", "wicket_rate_pct"), ("Run rate", "run_rate"),
                                            ("Mean |WP swing|", "mean_abs_wpa"))]), hide_index=True, width="stretch")
    D.read_only_note("The ball-by-ball match explorer and the live threshold slider")
    for f in sorted(D.cfg().path("outputs_public").glob("04_example_match_*.png")):
        st.image(str(f), width="stretch")
    st.stop()
cols = ["match_id", "innings", "overid", "over_ball_num", "innings_ball_num", "batting_team_id", "striker_name",
        "bowler_name", "wp_after", "wpa", "pressure_index", "is_team_wicket", "is_pressure_rule", "comp_name", "season",
        "reduced_overs", "is_legal", "is_faced", "team_runs", "is_four", "is_six", "knock_out"]
b = D.table("balls_enriched.parquet", cols)
val = D.validation().set_index("match_id")
c1, c2 = st.columns([1, 1])
thr_pct = c1.slider("High pressure = top X% of balls", 5, 40, int(D.cfg().get("pressure.high_pct", 20)), key="pl_thr")
definition = c2.segmented_control("Definition", ["Leverage", "Rule-based"], default="Leverage", key="pl_def")
full = b[~b["reduced_overs"]].copy()
thr = float(full["pressure_index"].quantile(1 - thr_pct / 100))
full["is_pressure"] = full["pressure_index"] >= thr if definition != "Rule-based" else full["is_pressure_rule"].astype(bool)
full["wpa"] = full["wpa"]
hi, lo = PR.league_stats(full, full["is_pressure"]), PR.league_stats(full, ~full["is_pressure"])
rows = [("Share of balls", "share_pct", "%"), ("Dot %", "dot_pct", ""), ("Boundary %", "boundary_pct", ""),
        ("Wicket %", "wicket_rate_pct", ""), ("Run rate", "run_rate", ""), ("Mean |WP swing| ×100", "mean_abs_wpa", "x100")]
C.section_header("League separation", f"{definition} definition · {'top ' + str(thr_pct) + '%' if definition != 'Rule-based' else 'rules from settings.yaml'}")
C.kpi_row([C.kpi_tile_html(lab, f"{(hi[k] * (100 if u == 'x100' else 1)):.2f}",
                           f"other balls {(lo[k] * (100 if u == 'x100' else 1)):.2f}") for lab, k, u in rows])

C.section_header("One match, ball by ball")
ms = (b.groupby("match_id").agg(comp=("comp_name", "first"), season=("season", "first"), mean=("pressure_index", "mean"))
      .join(val[["result"]], how="left").sort_values("mean", ascending=False))
ex = [e["match_id"] for e in D.js("pressure_separation.json").get("examples", [])]
opts = ex + [m for m in ms.index if m not in ex]
mid = st.selectbox("Match (highest-pressure first; examples from the report at the top)", opts, key="pl_match",
                   format_func=lambda m: f"{ms.loc[m, 'comp']} · {ms.loc[m, 'result']} · mean pressure {ms.loc[m, 'mean']:.2f}")
d = b[b["match_id"] == mid]
if d.empty:
    C.empty_state("No balls for this match.")
else:
    if d["reduced_overs"].any():
        st.warning("Rain-reduced match: excluded from model training and ratings; curves shown for reference.",
                   icon=":material/warning:")
    charts.show(charts.match_curves(d, thr, MODE), key="pl_curve")
    top = d.nlargest(5, "pressure_index")[["innings", "overid", "over_ball_num", "striker_name", "bowler_name",
                                           "pressure_index", "wpa"]]
    top["ball"] = (top["overid"] - 1).astype(str) + "." + top["over_ball_num"].astype(str)
    st.dataframe(top[["innings", "ball", "striker_name", "bowler_name", "pressure_index", "wpa"]], hide_index=True,
                 width="stretch", column_config={"pressure_index": st.column_config.NumberColumn("Pressure", format="%.2f"),
                                                 "wpa": st.column_config.NumberColumn("WPA (batting side)", format="%+.3f"),
                                                 "striker_name": "Batter", "bowler_name": "Bowler"})

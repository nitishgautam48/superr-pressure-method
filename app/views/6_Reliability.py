"""Reliability: sensitivity heatmap, split-half persistence, IPL vs T20I, limitations."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from app import charts, components as C, data as D, state

MODE = state.theme_mode()
C.page_header("Reliability", "How stable are the rankings, and where is the model less certain?")
sens, pers = D.js("sensitivity.json"), D.js("persistence.json")
role = st.segmented_control("Role", ["batter", "bowler"], default="batter", key="rl_role",
                            format_func=lambda r: r.title() + "s") or "batter"
s = sens.get(role, {})
C.section_header("Sensitivity: rank under each variant", "Top 25 model players · thresholds, rule-based pressure, weight sets, toggles")
if s.get("matrix"):
    m = pd.DataFrame(s["matrix"]).set_index("name")
    charts.show(charts.heatmap(m[["model_rank"] + s["variants"]], MODE), key="rl_heat")
p = pers.get(role, {})
with C.pair_grid("rl_mid"):
    with st.container():
        C.section_header("Split-half persistence (2022-23 vs 2024-26)")
        sh = p.get("seasons", {})
        pts = pd.DataFrame(sh.get("points", []))
        if len(pts):
            charts.show(charts.scatter(pts.assign(name=pts["player_id"].astype(str)), "A_first", "A_second", "name",
                                       "Quality A, 2022-23", "Quality A, 2024-26", mode=MODE), key="rl_sh")
        st.caption(" · ".join(f"{k}: r = {sh.get(k, {}).get('pearson')}" for k in ("A", "B", "C", "prs_raw")) +
                   f" (n = {sh.get('A', {}).get('n')})")
        v = p.get("verdict", {})
        if v.get("delta_is_weak"):
            st.warning("Pressure delta (B) does not persist between periods; the plan says to lean on quality (A) and "
                       "consistency (C). See the blend diagnostic.", icon=":material/warning:")
    with st.container():
        C.section_header("Blend diagnostic", "Persistence of each component by WPA weight in the value blend")
        bd = pd.DataFrame(p.get("blend_diagnostic", []))
        if len(bd):
            st.dataframe(bd, hide_index=True, width="stretch")
r = D.rated(role)
both = r.dropna(subset=["rank_ipl", "rank_t20i"])
C.section_header("IPL-only vs T20I-only (full members)", f"{len(both)} players rated in both · {int(both['competition_dependent'].sum())} flagged competition-dependent")
if len(both):
    charts.show(charts.scatter(both, "rank_ipl", "rank_t20i", "name", "Rank, IPL only", "Rank, T20I only",
                               highlight=both.loc[both["eligible"] & (both["rank"] <= 10), "name"].tolist(), mode=MODE), key="rl_comp")
C.section_header("Limitations")
C._show('<div class="wr-card"><ul style="margin:0 0 0 18px;line-height:1.6">' +
        "".join(f"<li>{C._e(x)}</li>" for x in pers.get("limitations", [])) + "</ul></div>")

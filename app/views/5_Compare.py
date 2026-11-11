"""Compare 2-4 players: pressure vs normal metrics and match consistency."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from app import charts, components as C, data as D, state

MODE = state.theme_mode()
C.page_header("Compare", "Two to four players of the same role side by side.")
role = st.segmented_control("Role", ["batter", "bowler"], default=st.session_state.get("selected_role", "batter"),
                            key="cp_role", format_func=lambda r: r.title() + "s") or "batter"
r = D.rated(role)
e = r[r["eligible"]].sort_values("rank")
nm = dict(zip(e["player_id"].astype(int), e["display"].fillna(e["name"])))
pre = [p for p in state.compare_list() if p in nm][:4] or list(nm)[:2]
chosen = st.multiselect("Players (2-4)", list(nm), default=pre, max_selections=4, format_func=nm.get, key=f"cp_pick_{role}")
if len(chosen) < 2:
    C.empty_state("Pick at least two eligible players.")
    st.stop()
x = r.set_index("player_id").loc[chosen]
rate, lab = ("sr", "Strike rate") if role == "batter" else ("econ", "Economy")
with C.pair_grid("cp_top"):
    with st.container():
        C.section_header(f"{lab}: pressure vs other balls")
        charts.show(charts.compare_bars(pd.DataFrame({"label": x["name"], "pressure": x[f"{rate}_p"], "normal": x[f"{rate}_np"]}),
                                        lab, MODE), key="cp_rate")
    with st.container():
        C.section_header("Components and reliability")
        t = x[["name", "rank", "prs", "z_A", "z_B", "z_C", "p_top10", "label", "pressure_balls"]].reset_index(drop=True)
        st.dataframe(t, hide_index=True, width="stretch",
                     column_config={"name": st.column_config.TextColumn("Player", pinned=True), "rank": "#",
                                    "prs": st.column_config.NumberColumn("PRS", format="%.0f"),
                                    "z_A": st.column_config.NumberColumn("A (z)", format="%+.2f"),
                                    "z_B": st.column_config.NumberColumn("B (z)", format="%+.2f"),
                                    "z_C": st.column_config.NumberColumn("C (z)", format="%+.2f"),
                                    "p_top10": st.column_config.NumberColumn("P(top 10)", format="%.0f%%"),
                                    "pressure_balls": "Pressure balls"})
pm = D.table("player_match_pressure.parquet")
pm = pm[(pm["role"] == role) & pm["player_id"].isin(chosen) & (pm["pressure_balls"] >= 3)]
C.section_header("Match-by-match consistency")
charts.show(charts.strip(pm, {p: str(x.loc[p, "name"]) for p in chosen}, MODE), key="cp_strip")

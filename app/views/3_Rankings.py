"""Rankings: sortable tables with live filters (re-rated by the engine, cached)."""
from __future__ import annotations

import streamlit as st

from app import components as C, data as D, state

C.page_header("Rankings", "All rated players. Filters re-run the rating live (same method, different inputs).")
c = D.cfg()
f1, f2, f3, f4 = st.columns([1.2, 1, 1, 1])
comp = f1.segmented_control("Competition", ["Both", "IPL", "T20I"], default="Both", key="rk_comp")
minb = f2.number_input("Min pressure balls", 30, 400, int(c.get("rating.eligibility.batter_min_pressure_balls", 120)), 10,
                       key="rk_minb")
assoc = f3.toggle("Include associates", value=not c.get("rating.associate_exclude", False), key="rk_assoc")
rec = f4.toggle("Recency weights", value=bool(c.get("rating.recency_on", True)), key="rk_rec")
el = dict(c.get("rating.eligibility"))
scale = minb / el["batter_min_pressure_balls"]
units = 12 if comp == "Both" else 6
ov = {"eligibility": {"batter_min_pressure_balls": minb, "bowler_min_pressure_balls": minb,
                      "batter_min_pressure_innings": units, "bowler_min_pressure_matches": units},
      "associate_exclude": not assoc, "recency_on": rec}
default = comp == "Both" and minb == el["batter_min_pressure_balls"] and assoc and rec
if D.showcase_mode() and not default:
    D.read_only_note("Live re-rating with filters")
    default = True
t_bat, t_bowl = st.tabs(["Batters", "Bowlers"])
for tab, role in ((t_bat, "batter"), (t_bowl, "bowler")):
    with tab:
        if default:
            r = D.rated(role)
        else:
            with st.spinner("Re-rating…"):
                r = D.rerate(role, ov, competition={"Both": "both"}.get(comp, comp))
        e = r[r["eligible"]].sort_values("rank").copy()
        if e.empty:
            C.empty_state("No eligible players with these filters. Lower the minimum pressure balls.")
            continue
        cols = ["rank", "name", "team", "prs", "z_A", "z_B", "z_C", "pressure_balls", "pressure_units"]
        cols += [x for x in ("label", "p_top10", "boot_rank_lo", "boot_rank_hi") if x in e]
        st.caption(f"{len(e)} eligible {role}s" + ("" if default else " · live re-rating (bootstrap columns need a full run)"))
        sel = st.dataframe(e[cols], hide_index=True, width="stretch", height=min(40 + 35 * len(e), 640),
                           on_select="rerun", selection_mode="single-row", key=f"rk_tbl_{role}",
                           column_config={"rank": st.column_config.NumberColumn("#", format="%d"),
                                          "name": st.column_config.TextColumn("Player", pinned=True), "team": "Team",
                                          "prs": st.column_config.ProgressColumn("PRS", min_value=0, max_value=100, format="%.0f"),
                                          "z_A": st.column_config.ProgressColumn("A quality (z)", min_value=-3, max_value=3, format="%.2f"),
                                          "z_B": st.column_config.ProgressColumn("B vs own baseline (z)", min_value=-3, max_value=3, format="%.2f"),
                                          "z_C": st.column_config.ProgressColumn("C consistency (z)", min_value=-3, max_value=3, format="%.2f"),
                                          "pressure_balls": "Pressure balls",
                                          "pressure_units": "Innings" if role == "batter" else "Matches",
                                          "label": "Label", "p_top10": st.column_config.NumberColumn("P(top 10)", format="%.0f%%"),
                                          "boot_rank_lo": st.column_config.NumberColumn("Rank lo", format="%.0f"),
                                          "boot_rank_hi": st.column_config.NumberColumn("Rank hi", format="%.0f")})
        rows = sel.selection.rows if sel is not None else []
        if rows and st.button("Open profile", key=f"rk_open_{role}", type="primary"):
            state.select_player(e.iloc[rows[0]]["player_id"], role)
            st.switch_page("views/4_Player_Profile.py")

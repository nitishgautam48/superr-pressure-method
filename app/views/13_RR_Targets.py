"""RR targets: who RR should back at the IPL 2027 auction - season record, needs, funnel, alignment gate, target lists."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app import charts, components as C, data as D, state

MODE = state.theme_mode()
res = D.js("rr_targets.json")
C.page_header("RR auction targets", "Reliable players under pressure whom RR should back at the IPL 2027 auction - "
              "data first, then RR's needs, then your availability research.")
if not res:
    C.empty_state("No RR target results yet - run the pipeline.")
    st.stop()
st.caption("Retentions and releases for 2027 are not in the data pack: add your own research in config/auction_pool.yaml "
           "(exclude / include with a reason).")
for e in res.get("availability_errors", []):
    st.warning(e, icon=":material/warning:")

C.section_header("RR's last seasons (from the data)")
st.dataframe(pd.DataFrame(res.get("season_record", [])), hide_index=True, width="stretch")
cs = res.get("close_and_situation", {}).get("all_seasons", {})
st.caption(f"Close matches ({res.get('close_and_situation', {}).get('close_definition', '')}): {cs.get('close', {}).get('won')}/"
           f"{cs.get('close', {}).get('matches')} won · chasing {cs.get('chasing', {}).get('won')}/{cs.get('chasing', {}).get('matches')}"
           f" · defending {cs.get('defending', {}).get('won')}/{cs.get('defending', {}).get('matches')}")

C.section_header("Where RR needs help under pressure", "RR's rank among the IPL teams per phase (10 = worst)")
fig = go.Figure()
for side, col in (("batting", charts.PINK), ("bowling", charts.blue(MODE))):
    rows = [r for r in res["rank_rows"][side] if r["phase"] != "ALL"]
    fig.add_trace(go.Bar(name=f"RR {side}", x=[r["phase"] for r in rows], y=[r["rr_rank_of_teams"] for r in rows],
                         marker_color=col, text=[f"{r['rr_rank_of_teams']}/{r['teams']}" for r in rows], textposition="outside",
                         hovertemplate="%{x}: rank %{y}<extra>" + side + "</extra>"))
fig.update_layout(template=f"warroom_{MODE}", barmode="group", yaxis={"range": [0, 11.5], "title": "RR rank (higher = bigger need)"},
                  height=340)
charts.show(fig, key="rt_needs")

for role in ("batter", "bowler"):
    r = res[role]
    C.section_header(f"{role.title()}s: funnel and targets")
    with C.pair_grid(f"rt_{role}"):
        with st.container():
            st.dataframe(pd.DataFrame([{"Step": s["step"], "Rule": s["rule"], "Left": s["remaining"],
                                        "Model top 10 removed": ", ".join(s["model_top10_dropped"])} for s in r["steps"]]),
                         hide_index=True, width="stretch")
        with st.container():
            t = pd.DataFrame(r["table"])
            cols = ["target_rank", "name", "team", "target_score", "prs_pct_pool", "fit", "aligned_phases", "label",
                    "auction_status", "overseas"]
            st.dataframe(t[cols].head(20), hide_index=True, width="stretch",
                         column_config={"target_rank": st.column_config.NumberColumn("#", format="%d"),
                                        "name": st.column_config.TextColumn("Player", pinned=True),
                                        "target_score": st.column_config.ProgressColumn("Target score", min_value=0, max_value=100, format="%.0f"),
                                        "prs_pct_pool": st.column_config.NumberColumn("Pressure (pct)", format="%.0f"),
                                        "fit": st.column_config.NumberColumn("RR fit (pct)", format="%.0f"),
                                        "aligned_phases": "Helps RR in", "label": "Reliability",
                                        "auction_status": "Status", "overseas": "Nationality"})
            ov = (t.head(10)["overseas"] == "Overseas").sum()
            st.caption(f"Top 10: {ov} overseas (RR can hold at most 8 overseas players in the squad, 4 in an XI).")
    na = res.get("not_aligned", {}).get(role, [])
    if na:
        with st.expander(f"{len(na)} {role}s removed by the RR alignment gate"):
            st.dataframe(pd.DataFrame(na), hide_index=True, width="stretch")
st.caption(f"Target score = {1 - res.get('fit_weight', 0.3):.0%} pressure rating + {res.get('fit_weight', 0.3):.0%} RR fit "
           "(percentiles within the pool). The submission list in Overview starts from this order; your decisions apply on top.")

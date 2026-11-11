"""Data quality: reconstruction validation, excluded matches, extras limitation."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app import charts, components as C, data as D, state

MODE = state.theme_mode()
dq = D.js("data_quality.json")
v = dq.get("validation", {})
C.page_header("Data quality", "How far the reconstructed scores can be trusted, and what was left out.")
C.kpi_row([C.kpi_tile_html("Winner reconstructed correctly", f"{v.get('winner_accuracy_pct', 0):.1f}%",
                           f"{v.get('decided_compared', 0)} decided matches"),
           C.kpi_tile_html("Run-margin error (MAE)", f"{v.get('runs_margin_mae', 0):.2f} runs", f"bias {v.get('runs_margin_bias', 0):+.2f}"),
           C.kpi_tile_html("Within 3 runs", f"{v.get('runs_margin_within_3_pct', 0):.0f}%"),
           C.kpi_tile_html("Wicket margins exact", f"{v.get('wickets_margin_exact_pct', 0):.1f}%"),
           C.kpi_tile_html("Excluded matches", f"{dq.get('excluded_match_count', len(dq.get('excluded_matches', [])))}")])
st.info("Run values of extras are missing from the pack, so team totals, targets and required rates are reconstructed "
        "(about ±3 runs). Batter runs, balls faced, dismissals and boundaries are exact. Winners are reconstructed "
        "from the chase structure (legal balls and wickets are exact).", icon=":material/info:")
val = D.validation()
if D.showcase_mode():
    o = dq.get("other_extra", {})
    st.markdown(f"- 'other' extras: {o.get('rows')} rows, treated as a no-ball variant - {o.get('evidence', '')}\n"
                f"- Revised overs are in {dq.get('revised_overs_unit', 'balls')}\n- Flags: {dq.get('flags', {})}")
    f = D.cfg().path("outputs_public") / "12_data_validation.png"
    if f.exists():
        st.image(str(f), width="stretch")
    st.stop()
with C.pair_grid("dq_mid"):
    with st.container():
        C.section_header("Run-margin error", "Reconstructed minus actual, matches won by runs")
        e = val.loc[val["margin_type"] == "runs", "margin_error"].dropna().clip(-12, 12).round()
        cnt = e.value_counts().sort_index()
        fig = go.Figure(go.Bar(x=cnt.index, y=cnt.values, marker_color=charts.blue(MODE),
                               hovertemplate="error %{x} runs: %{y} matches<extra></extra>"))
        fig.update_layout(template=f"warroom_{MODE}", height=320, xaxis_title="Runs (capped ±12)", yaxis_title="Matches")
        charts.show(fig, key="dq_err")
    with st.container():
        C.section_header("Decisions on the raw data")
        o = dq.get("other_extra", {})
        C._show('<div class="wr-card"><ul style="margin:0 0 0 18px;line-height:1.6">'
                f'<li>"other" extras ({o.get("rows")} rows, {o.get("with_bat_runs")} with bat runs): treated as a no-ball '
                f'variant (not legal) - {C._e(o.get("evidence", ""))}.</li>'
                f'<li>Revised overs in matches.csv are in {C._e(dq.get("revised_overs_unit", "balls"))}.</li>'
                f'<li>Flags: {C._e(str(dq.get("flags", {})))}.</li>'
                f'<li>Same name, different players: {", ".join(C._e(x["name"]) for x in dq.get("duplicate_names", []))}.</li>'
                "</ul></div>")
mism = val[val["winner_match"] == 0]
C.section_header("Matches where the reconstructed winner differs", f"{len(mism)} matches")
st.dataframe(mism[["match_id", "comp_name", "result", "inn1_total", "inn2_total", "target"]], hide_index=True, width="stretch")
C.section_header("Excluded matches", "No result, abandoned or no ball-by-ball data")
st.dataframe(pd.DataFrame(dq.get("excluded_matches", [])), hide_index=True, width="stretch")

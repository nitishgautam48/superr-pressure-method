"""Overview: KPIs and the two top-10 boards (model vs final)."""
from __future__ import annotations

import streamlit as st

from app import boards, components as C, data as D

dq, rep, sep, meta = D.js("data_quality.json"), D.js("wp_model_report.json"), D.js("pressure_separation.json"), D.js("run_meta.json")
fl = D.js("final_lists.json")
bat, bowl = D.rated("batter"), D.rated("bowler")
C.page_header("RR Pressure Index", "Who performs best when the game is on the line - IPL and T20I, 2022-2026",
              [("Your uploaded pack" if D.upload_dir() else "Real data pack", "success"),
               ("Model proposes · you decide", "neutral")])
rc = dq.get("row_counts", {})
C.kpi_row([C.kpi_tile_html("Matches", f"{rc.get('matches_with_balls', 0):,}"),
           C.kpi_tile_html("Balls", f"{rc.get('regular_balls', 0):,}"),
           C.kpi_tile_html("High-pressure balls", f"{sep.get('leverage', {}).get('high', {}).get('share_pct', 0):.1f}%",
                           help="Top 20% of the pressure index"),
           C.kpi_tile_html("Eligible batters", f"{int(bat['eligible'].sum())}"),
           C.kpi_tile_html("Eligible bowlers", f"{int(bowl['eligible'].sum())}"),
           C.kpi_tile_html("WP model Brier", f"{rep.get('summary', {}).get('brier_weighted', 0):.3f}",
                           help="Lower is better; no-skill = 0.25")])
view = st.segmented_control("List", ["Model", "Final"], default="Final", key="ov_view")
key = "final" if view == "Final" else "model"
changes = sum(len(fl.get(f"{r}s_changes", [])) for r in ("batter", "bowler"))
st.caption("Final = model plus your logged decisions (Player profile → Analyst decision). "
           f"{changes} model pick(s) changed so far.")
with C.pair_grid("ov_boards"):
    with st.container():
        C._show(boards.board_html("Top 10 batters under pressure", fl.get(f"batters_{key}", []), bat, key == "final"))
    with st.container():
        C._show(boards.board_html("Top 10 bowlers under pressure", fl.get(f"bowlers_{key}", []), bowl, key == "final"))
st.caption("Labels: ★ Robust = top 10 in ≥ 80% of 1,000 match resamples and of sensitivity variants · ◆ Likely ≥ 50% · "
           "○ Sensitive < 50%. Rank 80% = bootstrap interval.")

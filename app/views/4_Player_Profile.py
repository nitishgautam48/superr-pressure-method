"""Player profile: pressure vs normal, splits, consistency, rank distribution, analyst decision card."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from app import charts, components as C, data as D, state
from pressure import analyst

MODE = state.theme_mode()
role = st.segmented_control("Role", ["batter", "bowler"], key="pp_role",
                            default=st.session_state.get("selected_role", "batter"),
                            format_func=lambda r: r.title() + "s") or "batter"
r = D.rated(role)
e = r.sort_values(["rank", "prs_raw"], ascending=[True, False], na_position="last")
ids = e["player_id"].astype(int).tolist()
nm = dict(zip(e["player_id"].astype(int), e["display"].fillna(e["name"])))
cur = state.selected_player()
cur = cur if cur in nm else ids[0]
pid = st.selectbox("Player", ids, index=ids.index(cur), key=f"pp_pick_{role}_{cur}",
                   format_func=lambda p: nm[p] + (f"  ·  #{int(e.set_index('player_id').loc[p, 'rank'])}"
                                                   if pd.notna(e.set_index('player_id').loc[p, 'rank']) else "  ·  not eligible"))
state.select_player(pid, role)
p = r.set_index("player_id").loc[pid]
tone, icon = D.label_tone(p.get("label"))
badges = [(str(p.get("label") or "not eligible"), tone)] + ([("Plays for RR", "pink")] if p.get("plays_for_rr") else [])
C.page_header(str(p["name"]), f"{p.get('team') or ''} · {p.get('competition') or ''}", badges)
C.kpi_row([C.kpi_tile_html("Model rank", f"#{int(p['rank'])}" if pd.notna(p["rank"]) else "–"),
           C.kpi_tile_html("PRS", f"{p['prs']:.0f}"),
           C.kpi_tile_html("Rank 80% interval", f"{p['boot_rank_lo']:.0f}-{p['boot_rank_hi']:.0f}" if pd.notna(p.get("boot_rank_lo")) else "–"),
           C.kpi_tile_html("P(top 10)", f"{p['p_top10']:.0f}%" if pd.notna(p.get("p_top10")) else "–"),
           C.kpi_tile_html("Pressure balls", f"{int(p['pressure_balls'])}", f"{int(p['pressure_units'])} {'innings' if role == 'batter' else 'matches'}"),
           C.kpi_tile_html("A quality (z)", f"{p['z_A']:+.2f}"),
           C.kpi_tile_html("B vs own baseline (z)", f"{p['z_B']:+.2f}"),
           C.kpi_tile_html("C consistency (z)", f"{p['z_C']:+.2f}")])
if st.button("Add to compare", key="pp_cmp"):
    st.toast("Added to compare." if state.add_to_compare(pid) else "Compare already has 4 players.")

if role == "batter":
    mets = [("SR", "sr"), ("Balls per dismissal", "balls_per_dismissal"), ("Dot %", "dot_pct"), ("Boundary %", "boundary_pct"),
            ("RAE / 100", "rae_per100"), ("WPA / 100", "wpa_per100")]
    rate, rate_lab = "sr", "Strike rate"
else:
    mets = [("Economy", "econ"), ("Balls per wicket", "balls_per_wicket"), ("Dot %", "dot_pct"),
            ("Boundary % conceded", "boundary_pct_conceded"), ("RAE conceded / 100", "rae_conc_per100"), ("WPA / 100", "wpa_per100")]
    rate, rate_lab = "econ", "Economy"
cmp_df = pd.DataFrame([{"Metric": lab, "Pressure": p.get(f"{c}_p"), "Other balls": p.get(f"{c}_np")} for lab, c in mets])
sp = D.table("player_splits.parquet")
sp = sp[(sp["player_id"] == pid) & (sp["role"] == role)]
with C.pair_grid("pp_top"):
    with st.container():
        C.section_header("Pressure vs normal")
        st.dataframe(cmp_df, hide_index=True, width="stretch",
                     column_config={"Pressure": st.column_config.NumberColumn(format="%.2f"),
                                    "Other balls": st.column_config.NumberColumn(format="%.2f")})
    with st.container():
        C.section_header("Splits", f"{rate_lab}: pressure vs other balls")
        rows = []
        for split, vals in (("phase", ["PP", "MID", "DEATH"]), ("situation", ["chase", "set/defend"]), ("pace_or_spin", ["Pace", "Spin"])):
            for v in vals:
                a = sp[(sp["split"] == split) & (sp["split_value"] == v)]
                pr, nr = a[a["is_pressure"]], a[~a["is_pressure"]]
                rows.append({"label": v, "pressure": float(pr[rate].iloc[0]) if len(pr) else None,
                             "normal": float(nr[rate].iloc[0]) if len(nr) else None})
        charts.show(charts.compare_bars(pd.DataFrame(rows), rate_lab, MODE), key="pp_splits")

pm = D.table("player_match_pressure.parquet")
pm = pm[(pm["player_id"] == pid) & (pm["role"] == role) & (pm["pressure_balls"] >= int(D.cfg().get("rating.consistency_min_pressure_balls", 3)))]
with C.pair_grid("pp_mid"):
    with st.container():
        C.section_header("Match-by-match consistency", f"{len(pm)} matches with ≥ 3 pressure balls")
        if len(pm):
            charts.show(charts.strip(pm, {pid: str(p["name"])}, MODE), key="pp_strip")
        else:
            C.empty_state("No matches with enough pressure balls.")
    with st.container():
        C.section_header("Bootstrap rank distribution")
        hist = D.js("sensitivity.json").get(role, {}).get("rank_hist", {}).get(str(pid))
        if hist:
            charts.show(charts.rank_hist(hist, int(p["rank"]) if pd.notna(p["rank"]) else None, MODE), key="pp_hist")
        elif pd.notna(p.get("boot_rank_lo")):
            st.info(f"80% of 1,000 resamples put him between rank {p['boot_rank_lo']:.0f} and {p['boot_rank_hi']:.0f} "
                    f"(full histogram stored for the top 25).", icon=":material/info:")
        else:
            C.empty_state("Not eligible, so not bootstrapped.")

C.section_header("Analyst decision", "Your call, logged with your reason. The model list is always kept.")
if D.showcase_mode():
    st.info("Decisions are made in the local version; this showcase is read-only.", icon=":material/lock:")
    st.stop()
with st.form("pp_decide", border=True):
    a1, a2 = st.columns([1, 1])
    action = a1.selectbox("Action", ["pin", "drop", "swap_rank"], key="pp_action",
                          format_func={"pin": "Pin (keep / add)", "drop": "Drop from the list", "swap_rank": "Move to rank"}.get)
    rank = a2.number_input("Rank (pin / move)", 1, 10, int(p["rank"]) if pd.notna(p["rank"]) and p["rank"] <= 10 else 10, key="pp_rank")
    reason = st.text_area("Reason (required, at least 10 characters)", key="pp_reason")
    if st.form_submit_button("Save decision", type="primary"):
        o = {"role": role, "action": action, "player_id": int(pid), "reason": reason}
        if action == "swap_rank" or (action == "pin" and (pd.isna(p["rank"]) or int(rank) != int(p["rank"]))):
            o["rank"] = int(rank)
        c = D.cfg()
        err = analyst.add_override(c, o, {"batter": D.rated("batter"), "bowler": D.rated("bowler")})
        if err:
            st.error(f"Not saved: {err}", icon=":material/error:")
        else:
            analyst.refresh_final(c)
            st.cache_data.clear()
            st.success("Saved and logged. Final lists updated.", icon=":material/check:")

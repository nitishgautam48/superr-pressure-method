"""Weights lab: what-if re-rank (Apply), movement vs the model; Save requires a reason (decision log)."""
from __future__ import annotations

import pandas as pd
import streamlit as st
import yaml

from app import components as C, data as D
from pressure.decisions import DecisionRefused, log_decision

c = D.cfg()
C.page_header("Weights lab", "Try other weights and thresholds. Nothing changes until you Save with a reason.")
W = c.get("rating.weights")
if D.showcase_mode():
    D.read_only_note("The Weights lab (live re-rating)")
    st.write("Current weights:", W, "· value blend:", c.get("rating.value_blend"))
    st.stop()
with st.form("wl_form", border=True):
    a, b, cc = st.columns(3)
    wA = a.slider("wA quality", 0.0, 1.0, float(W["A"]), 0.05, key="wl_A")
    wB = b.slider("wB vs own baseline", 0.0, 1.0, float(W["B"]), 0.05, key="wl_B")
    wC = cc.slider("wC consistency", 0.0, 1.0, float(W["C"]), 0.05, key="wl_C")
    d1, d2, d3, d4, d5 = st.columns(5)
    thr = d1.slider("Pressure threshold (top %)", 10, 30, int(c.get("pressure.high_pct", 20)), 5, key="wl_thr")
    k = d2.slider("Shrinkage k (balls)", 25, 300, int(c.get("rating.shrink_k_balls", 100)), 25, key="wl_k")
    aw = d3.slider("Associate weight", 0.0, 1.0, float(c.get("rating.associate_weight", 0.5)), 0.1, key="wl_aw")
    wpa = d4.slider("WPA share of value", 0.0, 1.0, float(c.get("rating.value_blend.wpa", 0.5)), 0.25, key="wl_wpa")
    rec = d5.toggle("Recency weights", value=bool(c.get("rating.recency_on", True)), key="wl_rec")
    applied = st.form_submit_button("Apply", type="primary")
tot = wA + wB + wC
if tot <= 0:
    st.error("Give at least one component a weight.", icon=":material/error:")
    st.stop()
ov = {"weights": {"A": wA / tot, "B": wB / tot, "C": wC / tot}, "shrink_k_balls": k, "associate_weight": aw,
      "value_blend": {"wpa": wpa, "rae": 1 - wpa}, "recency_on": rec}
st.caption(f"Normalised weights: A {wA / tot:.2f} · B {wB / tot:.2f} · C {wC / tot:.2f}")
with C.pair_grid("wl_res"):
    for role in ("batter", "bowler"):
        with st.container():
            with st.spinner("Re-rating…"):
                new = D.rerate(role, ov, high_pct=thr)
            base = D.rated(role).set_index("player_id")["rank"]
            top = new[new["eligible"]].sort_values("rank").head(10)
            rows = []
            for _, x in top.iterrows():
                o = base.get(x["player_id"])
                mv = ("new" if o is None or pd.isna(o) else
                      f"▲ {int(o - x['rank'])}" if o > x["rank"] else f"▼ {int(x['rank'] - o)}" if o < x["rank"] else "=")
                rows.append({"#": int(x["rank"]), "Player": x["name"], "Model rank": None if o is None or pd.isna(o) else int(o),
                             "Move": mv, "PRS": x["prs"]})
            C.section_header(f"New top 10 {role}s")
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                         column_config={"PRS": st.column_config.NumberColumn(format="%.0f")})
if D.public_mode() or D.upload_dir():
    st.caption("Saving settings is switched off here (it would change the app's settings for everyone). "
               "Try weights above; save them in your own local copy.")
    st.stop()
C.section_header("Save these settings", "Writes config/settings_overrides.yaml (used by the next pipeline run) and logs each change.")
with st.form("wl_save"):
    reason = st.text_area("Reason (required, at least 10 characters)", key="wl_reason")
    if st.form_submit_button("Save"):
        new_settings = {"rating": {"weights": ov["weights"], "shrink_k_balls": k, "associate_weight": aw,
                                   "value_blend": ov["value_blend"], "recency_on": rec}, "pressure": {"high_pct": thr}}
        old = {"rating.weights": W, "rating.shrink_k_balls": c.get("rating.shrink_k_balls"),
               "rating.associate_weight": c.get("rating.associate_weight"), "rating.value_blend": c.get("rating.value_blend"),
               "rating.recency_on": c.get("rating.recency_on"), "pressure.high_pct": c.get("pressure.high_pct")}
        flat = {"rating.weights": ov["weights"], "rating.shrink_k_balls": k, "rating.associate_weight": aw,
                "rating.value_blend": ov["value_blend"], "rating.recency_on": rec, "pressure.high_pct": thr}
        changed = {kk: v for kk, v in flat.items() if str(v) != str(old[kk])}
        try:
            if not changed:
                st.info("Nothing changed from the current settings.", icon=":material/info:")
            else:
                for kk, v in changed.items():
                    log_decision(c, kk, old[kk], v, reason)
                p = c.source.parent / "settings_overrides.yaml"
                p.write_text("# Saved from the Weights lab; every change is in outputs/private/decision_log.csv\n"
                             + yaml.safe_dump({"settings": new_settings}, sort_keys=False), encoding="utf-8")
                st.success(f"Saved {len(changed)} change(s). Run `python -m pressure.run` to rebuild bootstraps, "
                           "figures and final lists with them.", icon=":material/check:")
        except DecisionRefused as exc:
            st.error(f"Not saved: {exc}", icon=":material/error:")

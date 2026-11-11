"""My decisions: decision log, model vs final with reasons, refused overrides, undo."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from app import components as C, data as D
from pressure import analyst
from pressure.decisions import read_log

c = D.cfg()
C.page_header("My decisions", "Everything you changed, with your reasons. The model's lists are never overwritten.")
mvf = D.js("model_vs_final.json")
with C.pair_grid("md_lists"):
    for role in ("batter", "bowler"):
        with st.container():
            C.section_header(f"{role.title()}s: model vs final")
            m = {x["rank"]: x["name"] for x in mvf.get("model", {}).get(role, [])}
            f = mvf.get("final", {}).get(role, [])
            st.dataframe(pd.DataFrame([{"#": x["rank"], "Model": m.get(x["rank"]), "Final": x["name"],
                                        "Provenance": x["provenance"], "Reason": x["reason"]} for x in f]),
                         hide_index=True, width="stretch")
            for ch in mvf.get("changes", {}).get(role, []):
                st.caption(f"Out of the list: {ch['name']} ({ch['change']}) - {ch['reason']}")
ref = mvf.get("refused_overrides", [])
if ref:
    st.error(f"{len(ref)} override(s) refused: " + "; ".join(f"{o.get('player_id')}: {o.get('error')}" for o in ref),
             icon=":material/error:")
if D.showcase_mode():
    C.section_header("Decision log")
    st.dataframe(pd.DataFrame(D.js("decision_log.json") if isinstance(D.js("decision_log.json"), list) else []),
                 hide_index=True, width="stretch")
    st.stop()
C.section_header("Current overrides", "From config/analyst_overrides.yaml")
ovs = analyst.load_overrides(c)
if not ovs:
    C.empty_state("No overrides yet. Use the Analyst decision card on a player's profile.")
else:
    st.dataframe(pd.DataFrame(ovs), hide_index=True, width="stretch")
    with st.form("md_undo"):
        i = st.number_input("Remove override number (1 = first row)", 1, len(ovs), 1, key="md_idx")
        why = st.text_input("Reason (required)", key="md_why")
        if st.form_submit_button("Remove"):
            err = analyst.remove_override(c, int(i) - 1, why)
            if err:
                st.error(err, icon=":material/error:")
            else:
                analyst.refresh_final(c)
                st.cache_data.clear()
                st.success("Removed and logged.", icon=":material/check:")
C.section_header("Decision log", "outputs/private/decision_log.csv")
log = pd.DataFrame(read_log(c))
if log.empty:
    C.empty_state("The log is empty.")
else:
    st.dataframe(log.iloc[::-1], hide_index=True, width="stretch")
    st.caption(f"{int((log['source'] == 'autopilot').sum())} autopilot defaults to review "
               "(reason 'autopilot default - review').")

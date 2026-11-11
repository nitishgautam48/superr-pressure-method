"""Methodology: technical documentation of the system - formulas, settings, decisions and evidence, all read live
from config and the processed outputs. Documentation of what the code does, not report text."""
from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from app import components as C, data as D

c = D.cfg()
dq, rep, sep = D.js("data_quality.json"), D.js("wp_model_report.json"), D.js("pressure_separation.json")
pers, sens, meta = D.js("persistence.json"), D.js("sensitivity.json"), D.js("run_meta.json")
C.page_header("Methodology", "How the RR Pressure Index is computed - formulas, settings and evidence, read live from "
              "the current run.", [("System documentation", "neutral")])
st.info("This page documents what the code does, for transparency and reproducibility. It is not report text: the "
        "methodology report must be written in your own words.", icon=":material/info:")


def g(path, d=None):
    return c.get(path, d)


def facts(rows: list[tuple[str, object]]) -> None:
    st.table(pd.DataFrame(rows, columns=["Item", "Value"]).astype(str).set_index("Item"))


steps = st.tabs(["1 Data", "2 Win probability", "3 Pressure", "4 Baselines", "5 Ratings", "6 Fair comparison",
                 "7 Reliability", "8 Reproducibility", "Figures"])

with steps[0]:
    v = dq.get("validation", {})
    C.section_header("Data and reconstruction")
    facts([("Deliveries / matches with ball-by-ball", f"{dq.get('row_counts', {}).get('balls', 0):,} / "
                                                     f"{dq.get('row_counts', {}).get('matches_with_balls', 0):,}"),
           ("Super-over balls (excluded from baselines)", dq.get("row_counts", {}).get("super_over_balls")),
           ("Legal delivery", "extras_type empty, leg-bye or bye (wide, no-ball and 'other' are not legal)"),
           ("Balls faced", "every delivery except wides"),
           ("'other' extras", f"{dq.get('other_extra', {}).get('rows')} rows, treated as a no-ball variant - "
                              f"{dq.get('other_extra', {}).get('evidence', '')}"),
           ("Bowler wickets exclude", ", ".join(g("data.bowler_wicket_exclude", []))),
           ("Batter dismissals exclude", ", ".join(g("data.batter_dismissal_exclude", []))),
           ("Excluded matches (no result / abandoned / no balls)", len(dq.get("excluded_matches", [])))])
    st.markdown("**Team runs per ball (reconstructed - extras' run values are missing from the pack)**")
    st.latex(r"\text{team\_runs} = \text{bat\_runs} + \mathbb{1}[\text{wide, no-ball, leg-bye, bye, other}] + "
             r"5\cdot\mathbb{1}[\text{penalty}] + \text{extras\_2\_runs}")
    st.markdown("**Winner reconstruction:** a chase that ends before its allotted balls with wickets in hand and "
                f"within {g('data.chase_early_tolerance_runs')} runs of the target = target reached; otherwise totals decide. "
                "DLS games use the revised target and balls.")
    facts([("Winner correct", f"{v.get('winner_accuracy_pct')}% of {v.get('decided_compared')} decided matches"),
           ("Run-margin error", f"MAE {v.get('runs_margin_mae')} runs, bias {v.get('runs_margin_bias')}, "
                                f"{v.get('runs_margin_within_3_pct')}% within 3 runs"),
           ("Wicket margins exact", f"{v.get('wickets_margin_exact_pct')}%")])

with steps[1]:
    C.section_header("Win-probability model", "Batting side's chance of winning, before and after every ball")
    rows = []
    for inn in (1, 2):
        r = rep.get(f"innings_{inn}", {})
        for name, m in r.get("candidates", {}).items():
            rows.append({"Innings": inn, "Model": name, "CV log-loss": m["log_loss"], "CV Brier": m["brier"],
                         "Chosen": "✓" if name == r.get("chosen") else ""})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    facts([("Features, innings 1", "score, wickets, balls remaining, phase, competition, team tiers"),
           ("Features, innings 2", "runs needed, balls remaining, wickets in hand, required rate, competition, team tiers"),
           ("Validation", rep.get("summary", {}).get("validation", "")),
           ("Calibration (chosen by cross-fitted log-loss)",
            f"innings 1: {rep.get('innings_1', {}).get('calibration', {}).get('chosen')}, "
            f"innings 2: {rep.get('innings_2', {}).get('calibration', {}).get('chosen')}"),
           ("Time split (train ≤ 2024, test ≥ 2025) Brier",
            f"{rep.get('innings_1', {}).get('time_split', {}).get('brier')} / {rep.get('innings_2', {}).get('time_split', {}).get('brier')} "
            f"(base rate {rep.get('innings_1', {}).get('time_split', {}).get('test_base_rate_brier')} / "
            f"{rep.get('innings_2', {}).get('time_split', {}).get('test_base_rate_brier')})"),
           ("Ties", f"{g('wp_model.tie_handling')} (0.5); no-results excluded; rain-reduced matches excluded from training")])
    st.latex(r"\text{WPA}_{\text{ball}} = WP_{\text{after}} - WP_{\text{before}}\quad(\text{batter } +,\ \text{bowler } -)")

with steps[2]:
    C.section_header("Pressure definition")
    st.latex(r"\text{leverage} = \sum_{o\in\{0,1,2,4,6,W\}} P(o \mid \text{innings, phase, wickets})\,"
             r"\bigl|WP(\text{after } o) - WP(\text{now})\bigr|")
    st.latex(r"\text{pressure index} = \frac{\text{leverage}}{\overline{\text{leverage}}_{\text{same innings}}}"
             r"\times(1.15 \text{ if knockout})")
    th = sep.get("threshold", {})
    facts([("High-pressure ball", f"top {th.get('high_pct')}% of the pressure index (cut-off {th.get('pressure_index_cutoff')})"),
           ("Normalised by", th.get("normalise_by")),
           ("Bands", f"low < {g('pressure.bands_pct')[0]}th pct ≤ medium < {g('pressure.bands_pct')[1]}th pct ≤ high")])
    r = g("pressure.rules", {})
    st.markdown("**Rule-based alternative (comparison):** pressure if any of")
    facts([("Steep chase", f"innings 2, RRR − CRR ≥ {r.get('chase_rrr_minus_crr')} and ≤ {r.get('chase_balls_remaining_max')} balls left"),
           ("Collapse", f"≥ {r.get('collapse_wickets_min')} wickets down and ≥ {r.get('collapse_balls_remaining_min')} balls left"),
           ("Close death-overs chase", f"|runs needed − par| ≤ {r.get('close_chase_death_margin_runs')}"),
           ("Knockout", "death overs of a knockout match")])
    hi, lo = sep.get("leverage", {}).get("high", {}), sep.get("leverage", {}).get("other", {})
    rb = sep.get("rule_based", {}).get("pressure", {})
    met = [("Dot %", "dot_pct"), ("Boundary %", "boundary_pct"), ("Wicket %", "wicket_rate_pct"), ("Run rate", "run_rate"),
           ("Mean |WP swing|", "mean_abs_wpa")]
    st.markdown("**Separation evidence (league-wide)**")
    st.dataframe(pd.DataFrame([{"Metric": n, "High pressure": hi.get(k), "Other balls": lo.get(k), "Rule-based": rb.get(k)}
                               for n, k in met]), hide_index=True, width="stretch")
    a = sep.get("agreement", {})
    st.caption(f"Leverage vs rule-based: kappa {a.get('cohen_kappa')}, Jaccard {a.get('overlap_jaccard_pct')}%; "
               f"rating-rank Spearman {sep.get('rank_correlation_of_ratings', {}).get('leverage_vs_rule_spearman')}.")

with steps[3]:
    C.section_header("Context-fair baselines", "Expected runs, dismissal and wicket probability for each ball")
    ctx = g("expected.context", {}) or {}
    levels = ["innings × phase" + (" × scoring era (season × competition)" if ctx.get("era") else ""), "+ wickets fallen",
              "+ required-rate bucket", "+ " + ("bowler type (left/right-arm pace, wrist/finger spin)" if ctx.get("bowl_kind")
                                                 else "pace/spin") + " × batting hand",
              "+ competition × opposition tier"] + (["+ opposition team-season strength"] if ctx.get("opposition_strength") else [])
    facts([(f"Level {i + 1}", lv) for i, lv in enumerate(levels)])
    st.latex(r"\hat{m}_{\text{bin}} = \frac{\sum y + k\,\hat{m}_{\text{parent}}}{n + k},\quad k = "
             + str(g("expected.smoothing_k")))
    st.latex(r"\text{RAE} = \text{runs} - \hat{m}_{\text{runs}},\qquad \text{WAE} = \text{wickets} - \hat{m}_{\text{wicket}}")

with steps[4]:
    C.section_header("Ratings")
    b, W = g("rating.value_blend"), g("rating.weights")
    st.latex(rf"v_{{\text{{ball}}}} = {b['wpa']}\,\frac{{\text{{WPA}}}}{{\sigma_{{WPA}}}} + {b['rae']}\,\frac{{\text{{RAE}}}}{{\sigma_{{RAE}}}}"
             r"\quad(\text{bowlers: } -\text{WPA}, -\text{RAE conceded})")
    st.latex(r"\text{shrunk mean} = \frac{\sum w_i v_i + k\,\mu_{\text{role}}}{\sum w_i + k},\quad k = "
             + str(g("rating.shrink_k_balls")) + r"\text{ weighted balls}")
    facts([("A - pressure quality", "shrunk mean value on high-pressure balls"),
           ("B - vs own baseline", "A minus the player's own shrunk mean on other balls"),
           ("C - consistency", f"share of matches (≥ {g('rating.consistency_min_pressure_balls')} pressure balls) with positive "
                               f"value, shrunk by {g('rating.consistency_shrink_matches')} matches, + low MAD of per-match value"),
           ("Eligibility, batters", f"≥ {g('rating.eligibility')['batter_min_pressure_balls']} pressure balls and ≥ "
                                    f"{g('rating.eligibility')['batter_min_pressure_innings']} innings with pressure balls"),
           ("Eligibility, bowlers", f"≥ {g('rating.eligibility')['bowler_min_pressure_balls']} pressure legal balls and ≥ "
                                    f"{g('rating.eligibility')['bowler_min_pressure_matches']} matches with pressure spells")])
    st.latex(rf"\text{{PRS}} = {W['A']:.2f}\,z(A) + {W['B']:.2f}\,z(B) + {W['C']:.2f}\,z(C)\quad"
             r"(z \text{ within the eligible pool; shown as a 0-100 percentile})")

with steps[5]:
    C.section_header("Fair comparisons")
    sw = g("rating.season_weights", {})
    facts([("Associate opposition", "excluded" if g("rating.associate_exclude") else f"weight {g('rating.associate_weight')}"),
           ("Emerging (A-team) cups", f"weight {g('rating.emerging_weight')}"),
           ("Recency", ("on: " + ", ".join(f"{k} {v}" for k, v in sw.items())) if g("rating.recency_on") else "off"),
           ("Team tiers", ", ".join(f"{k} {v}" for k, v in dq.get("team_tiers", {}).items())),
           ("IPL-only vs T20I-only rank correlation",
            f"batters {sens.get('batter', {}).get('competition_corr_spearman', {}).get('spearman')}, "
            f"bowlers {sens.get('bowler', {}).get('competition_corr_spearman', {}).get('spearman')} "
            f"(rank gap > {g('reliability.dependence_rank_gap')} = flagged)")])

with steps[6]:
    C.section_header("Reliability")
    facts([("Bootstrap", f"{g('reliability.bootstrap_draws'):,} resamples of matches; {int(g('reliability.interval') * 100)}% rank interval and P(top 10)"),
           ("Sensitivity variants", ", ".join(sens.get("batter", {}).get("variants", []))),
           ("Labels", "Robust ≥ 80% in top 10 across bootstraps AND variants; Likely ≥ 50%; otherwise Sensitive")])
    rows = [{"Role": r, "Split": sp, "A": pers.get(r, {}).get(sp, {}).get("A", {}).get("pearson"),
             "B": pers.get(r, {}).get(sp, {}).get("B", {}).get("pearson"), "C": pers.get(r, {}).get(sp, {}).get("C", {}).get("pearson"),
             "n": pers.get(r, {}).get(sp, {}).get("A", {}).get("n")}
            for r in ("batter", "bowler") for sp in ("seasons", "odd_even_matches")]
    st.markdown("**Split-half persistence (Pearson r between halves)**")
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    pc = c.path("outputs_private") / "parameter_comparison.json"
    if pc.exists():
        comp = json.loads(pc.read_text())
        st.markdown("**Settings compared by persistence of A (season split)**")
        st.dataframe(pd.DataFrame([{"Setting": k, "Batters": v["batter"]["persist_seasons_A"],
                                    "Bowlers": v["bowler"]["persist_seasons_A"]} for k, v in comp.items()]),
                     hide_index=True, width="stretch")
    st.markdown("**Limitations**")
    C._show('<div class="wr-card"><ul style="margin:0 0 0 18px;line-height:1.6">' +
            "".join(f"<li>{C._e(x)}</li>" for x in pers.get("limitations", [])) + "</ul></div>")

with steps[7]:
    C.section_header("Reproducibility")
    facts([("Rebuild everything", "python -m pressure.run --config config/settings.yaml"),
           ("Run at", meta.get("run_at")), ("Config hash", meta.get("config_hash")),
           ("Step timings (s)", ", ".join(f"{k} {v}" for k, v in (meta.get("timings_s") or {}).items())),
           ("Settings", "config/settings.yaml, merged with config/settings_overrides.yaml (saved, reasoned changes)"),
           ("Seed", g("seed")), ("Decisions", "outputs/private/decision_log.csv (every change with a reason)")])
    ov = c.source.parent / "settings_overrides.yaml"
    if ov.exists():
        st.code(ov.read_text(encoding="utf-8"), language="yaml")

with steps[8]:
    C.section_header("Report figures", "outputs/public - aggregates only, watermarked, with a space for your reasoning")
    pngs = sorted(c.path("outputs_public").glob("*.png")) if c.path("outputs_public").exists() else []
    if not pngs:
        C.empty_state("No figures yet - run the pipeline.")
    for p in pngs:
        with st.expander(p.stem.replace("_", " ")):
            st.image(str(p), width="stretch")

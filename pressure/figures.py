"""Report figure exports: PNG 1920x1080 + CSV of the plotted aggregates, to outputs/public/.

Every figure carries the watermark and a "[Your reasoning here]" panel for the analyst's own words. No report prose
is written here. export_guard() refuses row-level raw data in outputs/public/.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

log = logging.getLogger(__name__)
PINK, BLUE, GOLD, SLATE, INK, MUTED = "#E5307A", "#254AA5", "#D4A017", "#7A869A", "#151923", "#5B6474"
BANNED_COLUMNS = {"innings_ball_num", "over_ball_num", "delivery_ist_time", "bowl_speed", "stroke", "delivery_style",
                  "striker_id", "non_striker_id", "dismissed_player_id"}
MAX_PUBLIC_ROWS = 400


class ExportGuardError(RuntimeError):
    pass


def _canvas(cfg: Any, title: str, subtitle: str = "", left: float = 0.06) -> tuple[plt.Figure, plt.Axes, plt.Axes]:
    w, h = cfg.get("figures.size_px", [1920, 1080])
    dpi = int(cfg.get("figures.dpi", 100))
    fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi, facecolor="white")
    fig.text(0.04, 0.94, title, fontsize=26, fontweight="bold", color=INK, ha="left")
    if subtitle:
        fig.text(0.04, 0.905, subtitle, fontsize=15, color=MUTED, ha="left")
    ax = fig.add_axes([left, 0.12, 0.66 - left, 0.74])
    note = fig.add_axes([0.70, 0.12, 0.26, 0.74])
    note.set_xticks([]), note.set_yticks([])
    for sp in note.spines.values():
        sp.set_edgecolor("#C9CED8"), sp.set_linestyle((0, (6, 4)))
    note.text(0.5, 0.5, cfg.get("figures.placeholder", "[Your reasoning here]"), ha="center", va="center",
              fontsize=18, color=MUTED, style="italic", transform=note.transAxes)
    fig.text(0.5, 0.03, cfg.get("figures.watermark", "Confidential"), ha="center", fontsize=13, color="#9AA3B5")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(labelsize=13)
    ax.grid(axis="y", color="#E3E5EC", linewidth=1)
    ax.set_axisbelow(True)
    return fig, ax, note


def _save(fig: plt.Figure, data: pd.DataFrame, name: str, cfg: Any) -> Path:
    out = cfg.path("outputs_public")
    out.mkdir(parents=True, exist_ok=True)
    p = out / f"{name}.png"
    fig.savefig(p, dpi=fig.dpi, facecolor="white")
    plt.close(fig)
    data.to_csv(out / f"{name}.csv", index=False)
    return p


def export_guard(cfg: Any) -> list[str]:
    """Raise if any CSV in outputs/public looks row-level (ball identifiers or too many rows)."""
    bad = []
    for f in sorted(cfg.path("outputs_public").glob("*.csv")):
        d = pd.read_csv(f, nrows=MAX_PUBLIC_ROWS + 1)
        cols = set(map(str, d.columns))
        if cols & BANNED_COLUMNS:
            bad.append(f"{f.name}: row-level columns {sorted(cols & BANNED_COLUMNS)}")
        if len(d) > MAX_PUBLIC_ROWS:
            bad.append(f"{f.name}: more than {MAX_PUBLIC_ROWS} rows")
    if bad:
        raise ExportGuardError("; ".join(bad))
    return [f.name for f in sorted(cfg.path("outputs_public").glob("*"))]


# --- S2 figures ------------------------------------------------------------------------------------
def fig_definition(cfg: Any, sep: dict) -> Path:
    fig, ax, _ = _canvas(cfg, "1. How pressure is defined", "Leverage = expected swing in win probability from the ball")
    ax.axis("off")
    steps = [("Match state", "score, wickets, balls left,\ntarget, required rate"),
             ("Win probability", "calibrated model per innings\n(grouped-by-match CV)"),
             ("Leverage", "Σ P(outcome) × |WP after − WP now|\noutcomes: dot,1,2,4,6,wicket"),
             ("Pressure index", f"1.0 = average ball of the innings\nhigh = top {sep['threshold']['high_pct']}%"
                                f" · knockout ×{sep['threshold']['knockout_multiplier']}")]
    for i, (h, b) in enumerate(steps):
        y = 0.85 - i * 0.24
        ax.add_patch(plt.Rectangle((0.05, y - 0.08), 0.9, 0.16, color=[PINK, BLUE, GOLD, SLATE][i], alpha=0.12,
                                   transform=ax.transAxes))
        ax.text(0.09, y + 0.02, h, fontsize=20, fontweight="bold", color=INK, transform=ax.transAxes, va="center")
        ax.text(0.45, y, b, fontsize=15, color=INK, transform=ax.transAxes, va="center")
        if i < 3:
            ax.annotate("", xy=(0.5, y - 0.1), xytext=(0.5, y - 0.08), xycoords="axes fraction",
                        arrowprops=dict(arrowstyle="-|>", color=MUTED))
    a = sep["agreement"]
    data = pd.DataFrame([{"step": h, "detail": b.replace("\n", " ")} for h, b in steps] +
                        [{"step": "rule-based agreement", "detail": f"kappa {a['cohen_kappa']}, Jaccard {a['overlap_jaccard_pct']}%"}])
    return _save(fig, data, "01_pressure_definition", cfg)


def fig_leverage_distribution(cfg: Any, state: pd.DataFrame, thr: float) -> Path:
    fig, ax, _ = _canvas(cfg, "2. Pressure index distribution", f"Threshold for high pressure marked (top {cfg.get('pressure.high_pct')}%)")
    v = state.loc[~state["reduced_overs"], "pressure_index"].clip(upper=8)
    counts, edges = np.histogram(v, bins=80)
    centers = (edges[:-1] + edges[1:]) / 2
    ax.bar(centers, counts, width=edges[1] - edges[0], color=np.where(centers >= thr, PINK, SLATE))
    ax.axvline(thr, color=INK, linestyle="--", linewidth=2)
    ax.text(thr, counts.max() * 0.95, f"  high pressure ≥ {thr:.2f}", fontsize=15, color=INK)
    ax.set_xlabel("Pressure index (1.0 = average ball of the innings; capped at 8 for display)", fontsize=14)
    ax.set_ylabel("Balls", fontsize=14)
    return _save(fig, pd.DataFrame({"pressure_index_bin": centers.round(3), "balls": counts}), "02_leverage_distribution", cfg)


def fig_separation(cfg: Any, sep: dict) -> Path:
    fig, ax, _ = _canvas(cfg, "3. High- vs low-pressure balls (league-wide)",
                         "Same players, different situations: what changes when the stakes are high")
    rows = []
    for metric, label in (("dot_pct", "Dot %"), ("boundary_pct", "Boundary %"), ("wicket_rate_pct", "Wicket %"),
                          ("run_rate", "Run rate"), ("mean_abs_wpa", "Mean |WP swing| ×100")):
        hi, lo = sep["leverage"]["high"][metric], sep["leverage"]["other"][metric]
        rb, ro = sep["rule_based"]["pressure"][metric], sep["rule_based"]["other"][metric]
        k = 100 if metric == "mean_abs_wpa" else 1
        rows.append({"metric": label, "leverage_high": hi * k, "leverage_other": lo * k,
                     "rule_pressure": rb * k, "rule_other": ro * k})
    d = pd.DataFrame(rows)
    ax.axis("off")
    box = ax.get_position()
    series = (("leverage_high", PINK, "High"), ("leverage_other", SLATE, "Other"), ("rule_pressure", BLUE, "Rule"))
    n = len(d)
    for i, r in d.iterrows():
        sub = fig.add_axes([box.x0 + i * box.width / n + 0.008, box.y0, box.width / n - 0.022, box.height * 0.86])
        vals = [r[c] for c, _, _ in series]
        bars = sub.bar(range(3), vals, color=[c for _, c, _ in series], width=0.7)
        sub.bar_label(bars, fmt="%.1f", fontsize=12, padding=2)
        lo = min(vals)
        sub.set_ylim(0 if r["metric"].startswith("Mean") else lo * 0.85, max(vals) * 1.12)
        sub.set_title(r["metric"], fontsize=15, color=INK)
        sub.set_xticks(range(3), [lab for _, _, lab in series], fontsize=12)
        for sp in ("top", "right"):
            sub.spines[sp].set_visible(False)
        sub.tick_params(axis="y", labelsize=11)
    fig.text(box.x0, box.y0 + box.height * 0.95, "High = top 20% leverage · Other = remaining balls · Rule = "
             "rule-based pressure. Each panel has its own scale.", fontsize=12, color=MUTED)
    return _save(fig, d, "03_pressure_separation", cfg)


def fig_example_curve(cfg: Any, state: pd.DataFrame, example: dict) -> Path:
    mid = example["match_id"]
    d = state[state["match_id"] == mid]
    fig, ax, _ = _canvas(cfg, "4. Pressure over a real match", f"{example['competition']} · match {mid}"
                         + (" · knockout" if example["knock_out"] else ""))
    rows = []
    for inn, c in ((1, BLUE), (2, PINK)):
        x = d[d["innings"] == inn].groupby("overid").agg(pressure=("pressure_index", "mean"),
                                                         wp=("wp_before", "first")).reset_index()
        off = 0 if inn == 1 else 20
        ax.plot(x["overid"] + off, x["pressure"], color=c, linewidth=3, label=f"Innings {inn}: mean pressure per over")
        rows += [{"innings": inn, "over": int(o), "mean_pressure_index": round(p, 3), "wp_batting_start_of_over": round(w, 3)}
                 for o, p, w in zip(x["overid"], x["pressure"], x["wp"])]
    pk = example["peak"]
    ax.axvline((0 if pk["innings"] == 1 else 20) + int(pk["over"].split(".")[0]) + 1, color=GOLD, linestyle=":", linewidth=2)
    ax.text(0.02, 0.92, f"Peak: innings {pk['innings']}, over {pk['over']} - {pk['striker']} v {pk['bowler']} "
            f"(index {pk['pressure_index']})", transform=ax.transAxes, fontsize=14, color=INK)
    ax.set_xlabel("Over (1-20 first innings, 21-40 second innings)", fontsize=14)
    ax.set_ylabel("Pressure index", fontsize=14)
    ax.legend(fontsize=13, frameon=False, loc="upper left", bbox_to_anchor=(0, 0.88))
    return _save(fig, pd.DataFrame(rows), f"04_example_match_{mid}", cfg)


def fig_calibration(cfg: Any, rep: dict) -> Path:
    fig, ax, _ = _canvas(cfg, "5. Win-probability model calibration",
                         f"Grouped-by-match CV · Brier innings 1 {rep['innings_1']['cv_calibrated']['brier']} · "
                         f"innings 2 {rep['innings_2']['cv_calibrated']['brier']} (no-skill 0.25)")
    ax.plot([0, 1], [0, 1], color=SLATE, linestyle="--", linewidth=2, label="Perfect calibration")
    rows = []
    for inn, c in ((1, BLUE), (2, PINK)):
        bins = rep[f"innings_{inn}"]["calibration_bins"]
        ax.plot([b["p_mean"] for b in bins], [b["observed"] for b in bins], marker="o", markersize=9, color=c,
                linewidth=3, label=f"Innings {inn}")
        rows += [{"innings": inn, **b} for b in bins]
    ax.set_xlabel("Predicted win probability", fontsize=14)
    ax.set_ylabel("Observed win rate", fontsize=14)
    ax.set_xlim(0, 1), ax.set_ylim(0, 1)
    ax.legend(fontsize=13, frameon=False, loc="upper left")
    return _save(fig, pd.DataFrame(rows), "05_wp_calibration", cfg)


S2_FIGURES: list[Callable] = [fig_definition, fig_leverage_distribution, fig_separation, fig_example_curve, fig_calibration]


# --- S4 figures ------------------------------------------------------------------------------------
LABEL_COLOR = {"Robust": GOLD, "Likely": BLUE, "Sensitive": SLATE}


def fig_top10_table(cfg: Any, rated: pd.DataFrame, role: str, tag: str) -> Path:
    e = rated[rated["eligible"]].sort_values("rank").head(int(cfg.get("rating.top_n", 10)))
    fig, ax, _ = _canvas(cfg, f"6{tag}. Model top 10 {role}s under pressure",
                         "PRS 0-100 · 80% bootstrap rank interval · P(top 10) over 1,000 match resamples · label")
    ax.axis("off")
    rate = "sr_p" if role == "batter" else "econ_p"
    cols = ["#", "Player", "PRS", "Rank 80%", "P(top 10)", "Label", "Pressure balls", "SR (P)" if role == "batter" else "Econ (P)"]
    rows = [[int(r["rank"]), r["name"], f"{r['prs']:.0f}", f"{r['boot_rank_lo']:.0f}-{r['boot_rank_hi']:.0f}",
             f"{r['p_top10']:.0f}%", r["label"], int(r["pressure_balls"]), f"{r[rate]:.1f}"] for _, r in e.iterrows()]
    tb = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="left", colLoc="left")
    tb.auto_set_font_size(False), tb.set_fontsize(14), tb.scale(1, 2.6)
    for (i, j), c in tb.get_celld().items():
        c.set_edgecolor("#E3E5EC")
        if i == 0:
            c.set_text_props(fontweight="bold", color=INK)
        elif j == 5:
            c.set_text_props(color=LABEL_COLOR.get(rows[i - 1][5], INK), fontweight="bold")
    data = pd.DataFrame(rows, columns=cols)
    return _save(fig, data, f"06{tag}_top10_{role}s", cfg)


def fig_pressure_vs_baseline(cfg: Any, rated: pd.DataFrame, role: str, tag: str) -> Path:
    e = rated[rated["eligible"]]
    top = e.sort_values("rank").head(int(cfg.get("rating.top_n", 10)))
    fig, ax, _ = _canvas(cfg, f"7{tag}. Pressure value vs own baseline - {role}s",
                         "x = value on other balls · y = value on pressure balls (A) · above the line = rises under pressure")
    ax.scatter(e["baseline_value"], e["A"], s=50, color=SLATE, alpha=0.6, label="Eligible")
    ax.scatter(top["baseline_value"], top["A"], s=120, color=PINK, label="Model top 10", zorder=3)
    lo, hi = float(min(e["baseline_value"].min(), e["A"].min())), float(max(e["baseline_value"].max(), e["A"].max()))
    ax.plot([lo, hi], [lo, hi], color=MUTED, linestyle="--", linewidth=1.5, label="Same as baseline")
    for _, r in top.iterrows():
        ax.annotate(r["name"], (r["baseline_value"], r["A"]), xytext=(6, 4), textcoords="offset points", fontsize=12)
    ax.set_xlabel("Shrunk value per ball, other balls (baseline)", fontsize=14)
    ax.set_ylabel("Shrunk value per ball, pressure balls (A)", fontsize=14)
    ax.legend(fontsize=13, frameon=False, loc="upper left")
    data = e[["name", "baseline_value", "A", "B"]].round(4).sort_values("A", ascending=False)
    return _save(fig, data, f"07{tag}_pressure_vs_baseline_{role}s", cfg)


def fig_consistency(cfg: Any, rated: pd.DataFrame, pm: pd.DataFrame, role: str, tag: str) -> Path:
    top = rated[rated["eligible"]].sort_values("rank").head(int(cfg.get("rating.top_n", 10)))
    fig, ax, _ = _canvas(cfg, f"8{tag}. Match-by-match pressure value - top 10 {role}s",
                         f"One dot per match with ≥ {cfg.get('rating.consistency_min_pressure_balls')} pressure balls · "
                         "right of 0 = positive pressure value", left=0.17)
    rows = []
    rng = np.random.default_rng(0)
    for i, (_, r) in enumerate(top.iterrows()):
        x = pm[(pm["player_id"] == r["player_id"]) & (pm["role"] == role) &
               (pm["pressure_balls"] >= int(cfg.get("rating.consistency_min_pressure_balls", 3)))]["value"]
        y = np.full(len(x), i) + rng.uniform(-0.25, 0.25, len(x))
        ax.scatter(x.clip(-3, 3), y, s=30, color=np.where(x > 0, PINK, SLATE), alpha=0.75)
        ax.plot([x.median()] * 2, [i - 0.35, i + 0.35], color=INK, linewidth=3)
        rows.append({"rank": int(r["rank"]), "name": r["name"], "matches": int(len(x)),
                     "share_positive_pct": round(100 * float((x > 0).mean()), 1) if len(x) else None,
                     "median_value": round(float(x.median()), 3) if len(x) else None,
                     "mad": round(float((x - x.median()).abs().median()), 3) if len(x) else None})
    ax.axvline(0, color=MUTED, linewidth=1)
    ax.set_yticks(range(len(top)), [f"{int(r['rank'])}. {r['name']}" for _, r in top.iterrows()], fontsize=12)
    ax.invert_yaxis()
    ax.set_xlabel("Mean value per pressure ball in the match (capped ±3) · black bar = median", fontsize=13)
    return _save(fig, pd.DataFrame(rows), f"08{tag}_consistency_{role}s", cfg)


def fig_rank_stability(cfg: Any, rated: pd.DataFrame, role: str, tag: str) -> Path:
    e = rated[rated["eligible"]].sort_values("rank").head(int(cfg.get("rating.top_n", 10)) + 5)
    fig, ax, _ = _canvas(cfg, f"9{tag}. Rank stability - {role}s",
                         "Bar = 80% bootstrap rank interval · dot = model rank · % = top 10 across sensitivity variants",
                         left=0.15)
    for i, (_, r) in enumerate(e.iterrows()):
        c = LABEL_COLOR.get(r["label"], SLATE)
        ax.plot([r["boot_rank_lo"], r["boot_rank_hi"]], [i, i], color=c, linewidth=8, alpha=0.6, solid_capstyle="round")
        ax.scatter([r["rank"]], [i], color=INK, s=60, zorder=3)
        ax.text(max(r["boot_rank_hi"], r["rank"]) + 0.8, i, f"{r['variant_top10_pct']:.0f}% · {r['label']}",
                va="center", fontsize=11, color=c)
    ax.axvline(10.5, color=MUTED, linestyle=":", linewidth=1.5)
    ax.set_yticks(range(len(e)), [r["name"] for _, r in e.iterrows()], fontsize=12)
    ax.invert_yaxis()
    ax.set_xlabel("Rank (lower is better) · dotted line = top-10 cut", fontsize=13)
    data = e[["rank", "name", "boot_rank_lo", "boot_rank_median", "boot_rank_hi", "p_top10", "variant_top10_pct", "label"]].round(1)
    return _save(fig, data, f"09{tag}_rank_stability_{role}s", cfg)


def fig_persistence(cfg: Any, persistence: dict) -> Path:
    fig, ax, _ = _canvas(cfg, "10. Split-half persistence (2022-23 vs 2024-26)",
                         "Does pressure performance in one period predict the next? Eligible players in both halves")
    ax.axis("off")
    box = ax.get_position()
    rows = []
    for i, role in enumerate(("batter", "bowler")):
        sh = persistence[role]["seasons"]
        for j, comp in enumerate(("A", "B")):
            sub = fig.add_axes([box.x0 + j * box.width / 2 + 0.02, box.y0 + (1 - i) * box.height / 2 + 0.04,
                                box.width / 2 - 0.05, box.height / 2 - 0.09])
            pts = pd.DataFrame(sh["points"])
            if len(pts):
                sub.scatter(pts[f"{comp}_first"], pts[f"{comp}_second"], s=28, color=PINK if comp == "A" else BLUE, alpha=0.7)
            cc = sh[comp]
            sub.set_title(f"{role}s - {'quality A' if comp == 'A' else 'delta B'}: r = {cc['pearson']}, "
                          f"rho = {cc['spearman']} (n = {cc['n']})", fontsize=13)
            sub.tick_params(labelsize=10)
            rows.append({"role": role, "component": comp, **{k: cc[k] for k in ("n", "pearson", "spearman")}})
    return _save(fig, pd.DataFrame(rows), "10_split_half_persistence", cfg)


def fig_ipl_vs_t20i(cfg: Any, rated: dict[str, pd.DataFrame]) -> Path:
    fig, ax, _ = _canvas(cfg, "11. IPL-only vs T20I-only (full members) ranks",
                         "Players rated in both (half eligibility minimums) · far from the diagonal = depends on one competition")
    rows = []
    for role, c in (("batter", PINK), ("bowler", BLUE)):
        r = rated[role].dropna(subset=["rank_ipl", "rank_t20i"])
        ax.scatter(r["rank_ipl"], r["rank_t20i"], s=45, color=c, alpha=0.7, label=f"{role}s (n={len(r)})")
        for _, x in r[r["competition_dependent"].astype(bool) & r["eligible"]].sort_values("rank").head(6).iterrows():
            ax.annotate(x["name"], (x["rank_ipl"], x["rank_t20i"]), xytext=(5, 3), textcoords="offset points", fontsize=11)
        rows += r[["name", "rank_ipl", "rank_t20i", "competition_dependent"]].assign(role=role).to_dict("records")
    m = max([x["rank_ipl"] for x in rows] + [x["rank_t20i"] for x in rows] + [10])
    ax.plot([1, m], [1, m], color=MUTED, linestyle="--")
    ax.set_xlabel("Rank, IPL only", fontsize=14), ax.set_ylabel("Rank, T20I full members only", fontsize=14)
    ax.legend(fontsize=13, frameon=False)
    data = pd.DataFrame(rows).sort_values(["role", "rank_ipl"]).head(MAX_PUBLIC_ROWS)
    return _save(fig, data, "11_ipl_vs_t20i", cfg)


def fig_validation(cfg: Any, dq: dict, val: pd.DataFrame) -> Path:
    s = dq["validation"]
    fig, ax, _ = _canvas(cfg, "12. Data validation: reconstructed vs actual results",
                         f"Winner correct in {s['winner_accuracy_pct']}% of {s['decided_compared']} decided matches · "
                         f"run-margin MAE {s['runs_margin_mae']} runs (bias {s['runs_margin_bias']})")
    e = val.loc[val["margin_type"] == "runs", "margin_error"].dropna().clip(-12, 12)
    counts = e.round().value_counts().sort_index()
    ax.bar(counts.index, counts.values, color=BLUE)
    ax.set_xlabel("Reconstructed minus actual run margin (runs; capped ±12)", fontsize=14)
    ax.set_ylabel("Matches won by runs", fontsize=14)
    data = pd.DataFrame({"margin_error_runs": counts.index.astype(int), "matches": counts.values})
    return _save(fig, data, "12_data_validation", cfg)

"""Reusable UI components (badges, KPI tiles, headers, grids, empty states, page guard).

Each component has a pure ``*_html`` builder (unit-tested) and a thin render function.
Status never relies on colour alone: every tone is paired with an icon and a text label.
"""
from __future__ import annotations

import html
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd
import streamlit as st

from app import state

ROBUST_BADGE = {"Robust pick": ("gold", "★"), "Likely pick": ("blue", "◆"), "Weight-sensitive": ("neutral", "○")}
FLAG_TEXT = {"small_sample": "Small sample", "one_tournament_wonder": "One-tournament", "high_volatility": "Volatile",
             "matchup_weakness": "Match-up weakness", "declining": "Declining", "data_quality": "Data quality",
             "age_uncertain": "Age uncertain"}
FLAG_TONE = {"age_uncertain": "danger", "data_quality": "danger"}
CARD_STATS = {
    "bat": [("death_sr", "Death SR", "{:.0f}"), ("boundary_pct", "Bdry %", "{:.1f}"), ("sr_vs_spin", "SR v spin", "{:.0f}")],
    "bowl": [("death_econ", "Death eco", "{:.2f}"), ("dot_pct_bowl", "Dot %", "{:.1f}"), ("balls_per_wkt", "Balls/wkt", "{:.1f}")],
    "ar": [("death_sr", "Death SR", "{:.0f}"), ("boundary_pct", "Bdry %", "{:.1f}"), ("death_econ", "Death eco", "{:.2f}")],
}
_e = html.escape


def _fmt(v: Any, f: str = "{:.1f}") -> str:
    if v is None or v is pd.NA or (isinstance(v, float) and math.isnan(v)):
        return "–"
    try:
        return f.format(float(v))
    except (TypeError, ValueError):
        return _e(str(v))


def _show(markup: str) -> None:
    """Render component HTML. st.markdown keeps inline SVG (st.html sanitises it away); builders emit
    single-line HTML so the Markdown parser never splits a block."""
    st.markdown(markup.replace("\n", " "), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
def badge_html(text: str, tone: str = "neutral", icon: str | None = None, title: str | None = None) -> str:
    t = f' title="{_e(title)}"' if title else ""
    i = f'<span aria-hidden="true">{_e(icon)}</span>' if icon else ""
    return f'<span class="wr-badge tone-{_e(tone)}"{t}>{i}{_e(text)}</span>'


def badge(text: str, tone: str = "neutral", icon: str | None = None) -> None:
    _show(badge_html(text, tone, icon))


def robustness_badge_html(label: Any, pct: Any = None) -> str:
    if not isinstance(label, str) or label not in ROBUST_BADGE:
        return badge_html("Robustness n/a", "neutral", "○")
    tone, icon = ROBUST_BADGE[label]
    extra = f" {float(pct):.0f}%" if pct is not None and not (isinstance(pct, float) and math.isnan(pct)) else ""
    return badge_html(f"{label}{extra}", tone, icon, title="Share of 1,000 weight draws with the player in the top 10")


def page_header_html(title: str, subtitle: str = "", badges: Sequence[tuple[str, str]] | None = None,
                     eyebrow: bool = False) -> str:
    """Page title (H1, capped at 28px per s4/s4b-D). ``eyebrow=True`` puts the badges above the title (Home)."""
    b = "".join(badge_html(t, tone, "◆" if tone == "warning" else None) for t, tone in (badges or []))
    sub = f'<div class="wr-sub">{_e(subtitle)}</div>' if subtitle else ""
    if eyebrow:
        return (f'<div class="wr-header wr-hero" style="display:block"><div class="wr-eyebrow">{b}</div>'
                f'<h1 class="wr-h1">{_e(title)}</h1>{sub}</div>')
    return (f'<div class="wr-header"><div><h1 class="wr-h1">{_e(title)}</h1>{sub}</div>'
            f'<div class="wr-badges">{b}</div></div>')


def page_header(title: str, subtitle: str = "", badges: Sequence[tuple[str, str]] | None = None,
                eyebrow: bool = False) -> None:
    _show(page_header_html(title, subtitle, badges, eyebrow))


def section_header_html(title: str, subtitle: str | None = None) -> str:
    sub = f'<div class="wr-sub">{_e(subtitle)}</div>' if subtitle else ""
    return f'<div class="wr-section"><h2 class="wr-h2" style="margin:0">{_e(title)}</h2>{sub}</div>'


def section_header(title: str, subtitle: str | None = None) -> None:
    _show(section_header_html(title, subtitle))


def kpi_tile_html(label: str, value: Any, delta: str | None = None, help: str | None = None,
                  delta_good: bool | None = None) -> str:
    t = f' title="{_e(help)}"' if help else ""
    d = ""
    if delta:
        if delta_good is None:
            d = f'<div class="wr-kpi-delta wr-caption">{_e(delta)}</div>'
        else:
            tone, icon = ("success", "▲") if delta_good else ("danger", "▼")
            d = f'<div class="wr-kpi-delta">{badge_html(delta, tone, icon)}</div>'
    return (f'<div class="wr-card wr-kpi"{t}><div class="wr-kpi-label">{_e(label)}</div>'
            f'<div class="wr-kpi-value wr-num">{_e(str(value))}</div>{d}</div>')


def kpi_row(tiles: Iterable[str]) -> None:
    """Fluid KPI grid: repeat(auto-fit, minmax(180px, 1fr))."""
    _show('<div class="wr-kpi-grid">' + "".join(tiles) + "</div>")


def kpi_tile(label: str, value: Any, delta: str | None = None, help: str | None = None) -> None:
    _show(kpi_tile_html(label, value, delta, help))


def rsi_ring_html(score: Any, size: int = 56) -> str:
    """SVG donut 0-100 with the number as plain text inside <text> (s4b-F)."""
    try:
        s = max(0.0, min(100.0, float(score)))
        label = str(int(s))  # floor: 99.8 shows 99, never a misleading 100
    except (TypeError, ValueError):
        s, label = 0.0, "–"
    r = 15.9155  # circumference = 100 units
    stroke = 3.2
    return (f'<svg class="wr-ring" width="{size}" height="{size}" viewBox="0 0 36 36" role="img" '
            f'aria-label="RSI {label}">'
            f'<circle cx="18" cy="18" r="{r}" fill="none" stroke="var(--wr-ring-track)" stroke-width="{stroke}"/>'
            f'<circle cx="18" cy="18" r="{r}" fill="none" stroke="#E5307A" stroke-width="{stroke}" '
            f'stroke-linecap="round" stroke-dasharray="{s:.1f} {100 - s:.1f}" transform="rotate(-90 18 18)"/>'
            f'<text x="50%" y="50%" text-anchor="middle" dominant-baseline="central" font-size="11">{label}</text>'
            f'</svg>')


def rsi_ring(score: Any, size: int = 56) -> None:
    _show(rsi_ring_html(score, size))


def initials(name: str) -> str:
    parts = [p for p in str(name).replace(".", " ").split() if p]
    return (parts[0][0] + (parts[-1][0] if len(parts) > 1 else "")).upper() if parts else "?"


def initials_avatar_html(name: str) -> str:
    return f'<span class="wr-avatar" aria-hidden="true">{_e(initials(name))}</span>'


def _side(row: Any) -> str:
    role = row.get("listed_role")
    if role in ("Pacer", "Spinner"):
        return "bowl"
    if role == "All-rounder":
        return "ar"
    return "bat"


def flag_badges_html(flags: Any, reasons: Any = None) -> str:
    flags = list(flags) if isinstance(flags, (list, tuple)) or hasattr(flags, "__iter__") and not isinstance(flags, str) else []
    reasons = list(reasons) if reasons is not None and not isinstance(reasons, str) else []
    out = []
    for i, f in enumerate(flags):
        why = reasons[i] if i < len(reasons) else ""
        out.append(badge_html(FLAG_TEXT.get(f, str(f).replace("_", " ")), FLAG_TONE.get(f, "warning"), "⚠", title=why))
    return "".join(out)


RUPEE = "\u20b9"


def money(v: Any, nd: int = 2) -> str:
    """Rupee crore amount, e.g. ₹2.25 cr ("Rs" in the plan text is the same currency)."""
    return f"{RUPEE}{_fmt(v, '{:.' + str(nd) + 'f}')} cr"


def chip_html(text: str, tone: str | None = None, icon: str | None = None) -> str:
    """Count / status chip (Home dataset card). Neutral chips use the surface tint; tones reuse badge colours."""
    if tone:
        return badge_html(text, tone, icon).replace("wr-badge", "wr-chip wr-badge", 1)
    i = f'<span aria-hidden="true">{_e(icon)}</span>' if icon else ""
    return f'<span class="wr-chip">{i}{_e(text)}</span>'


def player_card_html(row: Any, compact: bool = False, fills: str | None = None) -> str:
    """Mockup layout + s5 contents: #rank and role; avatar, name, age chip, state, RSI ring; archetype and
    robustness badges; 3 role stats; expected price band; risk flags."""
    name = str(row.get("name", ""))
    rank = row.get("rank")
    rank_txt = f"#{int(rank)}" if rank is not None and not pd.isna(rank) else "–"
    head = (f'<div class="wr-pcard-head"><span class="wr-pcard-rank">{_e(rank_txt)}</span>'
            f'<span class="wr-pcard-role">{_e(str(row.get("listed_role") or ""))}</span></div>')
    meta = (f'<span class="wr-agechip wr-num">{_fmt(row.get("age"))} yrs</span> · '
            f'{_e(str(row.get("state_team") or ""))}')
    top = (f'<div class="wr-pcard-top">{initials_avatar_html(name)}'
           f'<div class="wr-pcard-id"><div class="wr-pcard-name" title="{_e(name)}">{_e(name)}</div>'
           f'<div class="wr-pcard-meta">{meta}</div></div>'
           f'{rsi_ring_html(row.get("rsi"), 48 if compact else 56)}</div>')
    tags = (badge_html(str(row.get("archetype") or "Unclassified"), "neutral")
            + robustness_badge_html(row.get("robustness_label"), row.get("robustness_pct")))
    if fills:
        tags += badge_html(f"fills: {fills}", "teal")
    body = [head, top, f'<div class="wr-row">{tags}</div>']
    if not compact:
        stats = "".join(f'<div><div class="k">{_e(lab)}</div><div class="v wr-num">{_fmt(row.get(col), f)}</div></div>'
                        for col, lab, f in CARD_STATS[_side(row)])
        body.append(f'<div class="wr-stats-line">{stats}</div>')
        body.append(f'<div><div class="wr-price" title="Expected auction price">{money(row.get("expected_price_cr"))}</div>'
                    f'<div class="wr-price-band">{RUPEE}{_fmt(row.get("price_lo_cr"), "{:.2f}")}'
                    f'–{_fmt(row.get("price_hi_cr"), "{:.2f}")} cr band</div></div>')
    else:
        body.append(f'<div class="wr-price-band">{money(row.get("expected_price_cr"))} expected</div>')
    flags = flag_badges_html(row.get("risk_flags"), row.get("risk_reasons"))
    if flags:
        body.append(f'<div class="wr-row">{flags}</div>')
    return f'<div class="wr-pcard{" compact" if compact else ""}">{"".join(body)}</div>'


def player_card(row: Any, compact: bool = False, on_click_key: str | None = None, fills: str | None = None) -> None:
    """Card inside a keyed container (styled as a card); optional "Open profile" button."""
    pid = str(row.get("player_id"))
    with st.container(key=f"wr_card_{on_click_key or pid}"):
        _show(player_card_html(row, compact, fills))
        if on_click_key:
            if st.button("Open profile", key=on_click_key, width="stretch",
                         help=f"Open {row.get('name')}'s profile"):
                open_profile(pid)


def open_profile(pid: str) -> None:
    state.select_player(pid)
    page = Path(__file__).parent / "views" / "4_Player_Profile.py"
    if page.exists():
        st.switch_page(str(page.relative_to(Path(__file__).parent)))
    else:
        st.toast("The Player Profile page arrives in phase F2.", icon="ℹ️")


def card_grid(key: str):
    """A container laid out as repeat(auto-fill, minmax(240px, 1fr)) (see theme.css)."""
    return st.container(key=f"wr_grid_{key}")


def tile_grid(key: str):
    """Equal-width tiles that always span the row: repeat(auto-fit, minmax(260px, 1fr))."""
    return st.container(key=f"wr_tiles_{key}")


def fmt_run_at(value: Any) -> str:
    """'2026-11-07T06:34:43' -> '07 Nov 2026, 06:34' (falls back to the raw text)."""
    try:
        return pd.Timestamp(str(value)).strftime("%d %b %Y, %H:%M")
    except (ValueError, TypeError):
        return str(value or "–")


def mini_grid(key: str):
    return st.container(key=f"wr_minigrid_{key}")


def flag_list_html(flags: Any, reasons: Any, penalties_pts: dict[str, float] | None = None) -> str:
    flags = list(flags or [])
    reasons = list(reasons or [])
    if not flags:
        return f'<div class="wr-flags">{badge_html("No risk flags", "success", "✓")}</div>'
    items = []
    for i, f in enumerate(flags):
        pts = (penalties_pts or {}).get(f)
        pts_txt = f' <span class="wr-num">({pts:+.1f} RSI)</span>' if pts is not None else ""
        items.append(f'<div class="wr-flag">{badge_html(FLAG_TEXT.get(f, f), FLAG_TONE.get(f, "warning"), "⚠")}'
                     f'<span class="why">{_e(reasons[i] if i < len(reasons) else "")}{pts_txt}</span></div>')
    return '<div class="wr-flags">' + "".join(items) + "</div>"


def flag_list(flags: Any, reasons: Any, penalties_pts: dict[str, float] | None = None) -> None:
    _show(flag_list_html(flags, reasons, penalties_pts))


def stat_compare_table_html(players: pd.DataFrame, metrics: Sequence[tuple[str, str, bool, str]]) -> str:
    """Rows = metrics, columns = players; best value per row gets '▲ best' + bold (not colour alone)."""
    names = players["name"].astype(str).tolist()
    head = "<tr><th>Metric</th>" + "".join(f"<th>{_e(n)}</th>" for n in names) + "</tr>"
    rows = []
    for col, label, higher, f in metrics:
        if col not in players:
            continue
        vals = pd.to_numeric(players[col], errors="coerce")
        best = (vals.idxmax() if higher else vals.idxmin()) if vals.notna().sum() > 1 else None
        cells = []
        for idx, v in vals.items():
            is_best = idx == best
            mark = f' {badge_html("best", "success", "▲")}' if is_best else ""
            cells.append(f'<td class="wr-num{" best" if is_best else ""}">{_fmt(v, f)}{mark}</td>')
        rows.append(f"<tr><td>{_e(label)}</td>{''.join(cells)}</tr>")
    return f'<div class="wr-table-wrap"><table class="wr-table">{head}{"".join(rows)}</table></div>'


def stat_compare_table(players: pd.DataFrame, metrics: Sequence[tuple[str, str, bool, str]]) -> None:
    _show(stat_compare_table_html(players, metrics))


def empty_state_html(message: str) -> str:
    return f'<div class="wr-card wr-empty"><div class="icon" aria-hidden="true">○</div><div>{_e(message)}</div></div>'


def empty_state(message: str, action_label: str | None = None, key: str | None = None) -> bool:
    _show(empty_state_html(message))
    return bool(action_label) and st.button(action_label, key=key or f"empty_{hash(message)}")


def privacy_banner_html(fields_sent: Sequence[str]) -> str:
    if not fields_sent:
        body = "Nothing has left this machine. Answers so far came from local templates."
    else:
        chips = "".join(badge_html(f, "blue") for f in fields_sent)
        body = f'Computed fields sent to the model (never raw pack rows):<div class="wr-chips" style="margin-top:6px">{chips}</div>'
    return (f'<div class="wr-card wr-privacy" role="note"><div class="wr-row">{badge_html("Privacy", "blue", "🔒")}</div>'
            f'<div class="wr-caption" style="margin-top:6px;font-size:13px">{body}</div></div>')


def privacy_banner(fields_sent: Sequence[str]) -> None:
    _show(privacy_banner_html(fields_sent))


def pair_grid(key: str):
    """Two-chart row: side by side at >= 1280px viewport, stacked below (s4b E)."""
    return st.container(key=f"wr_pair_{key}")


def profile_header_html(row: Any) -> str:
    """Player profile header: avatar, name, age / state / role, archetype + robustness badges, price band, RSI ring."""
    name = str(row.get("name", ""))
    rank = row.get("rank")
    rank_txt = f"Rank #{int(rank)}" if rank is not None and not pd.isna(rank) else "Not ranked (ineligible)"
    meta = " · ".join(_e(str(x)) for x in (f"{_fmt(row.get('age'))} yrs", row.get("state_team"), row.get("listed_role"),
                                          (f"{row.get('batting_hand')}HB" if row.get("batting_hand") in ("L", "R") else row.get("batting_hand")),
                                          row.get("bowling_style")) if x is not None and not (not isinstance(x, str) and pd.isna(x))
                      and str(x) not in ("", "nan", "None", "<NA>"))
    tags = (badge_html(str(row.get("archetype") or "Unclassified"), "neutral")
            + robustness_badge_html(row.get("robustness_label"), row.get("robustness_pct"))
            + badge_html(rank_txt, "pink" if row.get("in_top10") else "neutral", "#"))
    price = (f'<div class="wr-price">{money(row.get("expected_price_cr"))} expected</div>'
             f'<div class="wr-price-band">{RUPEE}{_fmt(row.get("price_lo_cr"), "{:.2f}")}–'
             f'{_fmt(row.get("price_hi_cr"), "{:.2f}")} cr band</div>')
    return (f'<div class="wr-card wr-profile-head">{initials_avatar_html(name)}'
            f'<div class="wr-profile-id"><h1 class="wr-h1">{_e(name)}</h1><div class="wr-caption">{meta}</div>'
            f'<div class="wr-row" style="margin-top:8px">{tags}</div></div>'
            f'<div class="wr-profile-price">{price}</div>{rsi_ring_html(row.get("rsi"), 72)}</div>')


def comp_card_html(row: Any, horizon: int | None = None) -> str:
    """Comparable historical player: similarity %, age then, later IPL outcome. ``became_regular`` is
    measured within ``horizon`` seasons of the snapshot, so the badge says so."""
    reg = row.get("became_regular")
    within = f" within {horizon} seasons" if horizon else ""
    outcome = (badge_html(f"IPL regular{within}", "success", "✓") if reg is True or reg == 1
               else badge_html(f"Not a regular{within}", "neutral", "○") if reg is not None and not pd.isna(reg) else "")
    sim = row.get("similarity")
    sim_v = float(sim) * (100 if sim is not None and not pd.isna(sim) and float(sim) <= 1 else 1) if sim is not None else None
    return (f'<div class="wr-card wr-comp"><div class="wr-pcard-head"><span class="wr-pcard-rank">'
            f'{_fmt(sim_v, "{:.0f}")}% similar</span><span class="wr-pcard-role">age then {_fmt(row.get("age_then"), "{:.0f}")}'
            f'{" · " + _e(str(int(row["season_then"]))) if row.get("season_then") is not None and not pd.isna(row.get("season_then")) else ""}'
            f'</span></div><div class="wr-pcard-name" style="margin:6px 0">{_e(str(row.get("comp_name", "")))}</div>'
            f'<div class="wr-caption" style="font-size:13px">{_e(str(row.get("ipl_outcome_summary") or "No IPL outcome recorded"))}</div>'
            f'<div class="wr-row" style="margin-top:8px">{outcome}</div></div>')


def callout_html(title: str, body: str) -> str:
    return f'<div class="wr-card wr-callout"><h3 class="wr-h3">{_e(title)}</h3><p class="wr-why">{_e(body)}</p></div>'


def safe_run(page: Any) -> bool:
    """Run a navigation page; an unexpected error shows a friendly message instead of a traceback, so the
    sidebar and other pages keep working. st.stop / st.rerun / st.switch_page are BaseException and pass."""
    try:
        page.run()
        return True
    except Exception as exc:  # noqa: BLE001
        import logging

        logging.getLogger("app").exception("page failed")
        st.error(f"This page hit a problem and stopped ({type(exc).__name__}). The details are in the app log; "
                 "other pages still work. If the processed outputs look stale, re-run the pipeline.",
                 icon=":material/error:")
        return False

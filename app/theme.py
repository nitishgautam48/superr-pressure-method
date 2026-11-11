"""War Room design system: tokens, Plotly templates and CSS injection.

* Tokens for dark (default) and light, plus brand accents. Badge/text tones have
  mode-specific steps chosen to pass WCAG AA (checked in tests).
* Chart palette: the brand colourway was run through the data-viz validator. As a
  categorical set only three hues pass all-pairs in both modes, so overlays are capped
  at three series (a fourth facets into small multiples) and slate is used only for
  de-emphasis, never as a category.
* ``css(mode)`` implements the responsive shell: fluid sidebar, fluid padding/type,
  auto-fit grids, visible keyboard focus, hidden footer / deploy button.
"""
from __future__ import annotations

from typing import Literal

import plotly.graph_objects as go
import plotly.io as pio

Mode = Literal["dark", "light"]

ACCENTS = {
    "pink": "#E5307A", "royal_blue": "#254AA5", "gold": "#D4A017", "teal": "#1FA2A6",
    "slate": "#7A869A", "success": "#2BB673", "warning": "#F2A93B", "danger": "#E5484D",
}

TOKENS: dict[str, dict[str, str]] = {
    "dark": {"bg": "#0E1117", "surface": "#161B26", "surface_2": "#1E2533", "border": "#2A3242",
             "text": "#E8EAF0", "text_muted": "#9AA3B5"},
    "light": {"bg": "#F7F7FA", "surface": "#FFFFFF", "surface_2": "#F0F1F5", "border": "#E3E5EC",
              "text": "#151923", "text_muted": "#5B6474"},
}

# Text colour for each tone, per mode (AA on the tinted badge background - see tests).
TONE_TEXT = {
    "dark": {"neutral": "#C3C9D6", "pink": "#F48BB3", "blue": "#9DB4F2", "gold": "#E6BE4F",
             "success": "#5FD39B", "warning": "#F6BE66", "danger": "#F38A8D", "teal": "#5CCFD2"},
    "light": {"neutral": "#4A5262", "pink": "#A8144F", "blue": "#254AA5", "gold": "#7A5900",
              "success": "#126B3F", "warning": "#7E4F00", "danger": "#A61F24", "teal": "#0D6C6F"},
}
TONE_BASE = {"neutral": ACCENTS["slate"], "pink": ACCENTS["pink"], "blue": ACCENTS["royal_blue"],
             "gold": ACCENTS["gold"], "success": ACCENTS["success"], "warning": ACCENTS["warning"],
             "danger": ACCENTS["danger"], "teal": ACCENTS["teal"]}
TONE_ALPHA = 0.16

# Validated categorical series (all-pairs, both modes): pink, blue, gold.
SERIES = {"dark": ["#E5307A", "#6A8EE6", "#B8860B"], "light": ["#E5307A", "#254AA5", "#D4A017"]}
POSITIVE = {"dark": "#6A8EE6", "light": "#254AA5"}      # diverging poles (signed values)
NEGATIVE = {"dark": "#F06A6E", "light": "#C8363B"}
MUTED_MARK = {"dark": "#5E6A80", "light": "#A3ABBA"}     # de-emphasised marks ("everyone else")

GOOGLE_FONTS = ("https://fonts.googleapis.com/css2?family=Inter:wght@400;600"
                "&family=JetBrains+Mono:wght@400;600&display=swap")


# ---------------------------------------------------------------------------
def _rgb(hex_: str) -> tuple[float, float, float]:
    h = hex_.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))  # type: ignore[return-value]


def blend(fg: str, bg: str, alpha: float) -> str:
    f, b = _rgb(fg), _rgb(bg)
    return "#" + "".join(f"{round((a * alpha + c * (1 - alpha)) * 255):02X}" for a, c in zip(f, b))


def contrast(fg: str, bg: str) -> float:
    """WCAG 2.x contrast ratio."""
    def lum(hex_: str) -> float:
        out = []
        for c in _rgb(hex_):
            out.append(c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4)
        return 0.2126 * out[0] + 0.7152 * out[1] + 0.0722 * out[2]
    a, b = sorted((lum(fg), lum(bg)), reverse=True)
    return (a + 0.05) / (b + 0.05)


def tone_colors(tone: str, mode: Mode) -> tuple[str, str]:
    """(background, text) for a badge tone on the surface colour."""
    tone = tone if tone in TONE_BASE else "neutral"
    return blend(TONE_BASE[tone], TOKENS[mode]["surface"], TONE_ALPHA), TONE_TEXT[mode][tone]


# ---------------------------------------------------------------------------
def plotly_template(mode: Mode) -> go.layout.Template:
    t = TOKENS[mode]
    tpl = go.layout.Template()
    tpl.layout = go.Layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        colorway=SERIES[mode] + [ACCENTS["teal"], MUTED_MARK[mode]],
        font={"family": "Inter, system-ui, sans-serif", "size": 12.5, "color": t["text_muted"]},
        title={"font": {"size": 14, "color": t["text"]}, "x": 0, "xanchor": "left"},
        height=360, margin={"l": 40, "r": 16, "t": 32, "b": 40},
        hoverlabel={"bgcolor": t["surface_2"], "bordercolor": t["border"],
                    "font": {"family": "Inter, system-ui, sans-serif", "size": 12.5, "color": t["text"]}},
        xaxis={"gridcolor": t["border"], "linecolor": t["border"], "zerolinecolor": t["border"],
               "tickcolor": t["border"], "showline": False, "automargin": True},
        yaxis={"gridcolor": t["border"], "linecolor": t["border"], "zerolinecolor": t["border"],
               "tickcolor": t["border"], "showline": False, "automargin": True},
        polar={"bgcolor": "rgba(0,0,0,0)", "radialaxis": {"gridcolor": t["border"], "linecolor": t["border"]},
               "angularaxis": {"gridcolor": t["border"], "linecolor": t["border"]}},
        legend={"bgcolor": "rgba(0,0,0,0)", "font": {"color": t["text_muted"]}, "orientation": "h",
                "y": -0.18, "x": 0},
        barcornerradius=4, bargap=0.35, hovermode="closest",
    )
    return tpl


def register_templates() -> None:
    for m in ("dark", "light"):
        pio.templates[f"warroom_{m}"] = plotly_template(m)  # type: ignore[arg-type]


register_templates()


# ---------------------------------------------------------------------------
def css(mode: Mode) -> str:
    """CSS injected on every page: tokens, responsive shell (s4b) and component styles."""
    t = TOKENS[mode]
    tones = "\n".join(
        f".wr-badge.tone-{k}{{background:{tone_colors(k, mode)[0]};color:{tone_colors(k, mode)[1]};"
        f"border-color:{blend(TONE_BASE[k], t['surface'], 0.35)}}}" for k in TONE_BASE)
    ring_track = t["border"]
    return f"""<style>
@import url('{GOOGLE_FONTS}');
:root {{
  --wr-bg:{t['bg']}; --wr-surface:{t['surface']}; --wr-surface-2:{t['surface_2']}; --wr-border:{t['border']};
  --wr-text:{t['text']}; --wr-muted:{t['text_muted']}; --wr-pink:{ACCENTS['pink']}; --wr-blue:{ACCENTS['royal_blue']};
  --wr-gold:{ACCENTS['gold']}; --wr-ring-track:{ring_track};
  --wr-gap: clamp(16px, 2vw, 28px);
}}
/* ---------- responsive shell (s4b A, C, D) ---------- */
html, body {{ overflow-x: hidden; }}
[data-testid="stAppViewContainer"] {{ min-width: 0; }}
[data-testid="stMain"] {{ min-width: 0; }}
[data-testid="stMainBlockContainer"], .block-container {{
  max-width: 100% !important;
  padding: calc(3.75rem + 8px) clamp(16px, 2.5vw, 48px) 48px !important;  /* clear the fixed top bar */
}}
section[data-testid="stSidebar"] {{
  width: clamp(220px, 18vw, 280px) !important; min-width: clamp(220px, 18vw, 280px) !important;
  max-width: 280px !important;
}}
section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {{ overflow-y: auto; }}
.st-key-wr_sidebar_footer {{
  background: {t['surface']}; border-top: 1px solid {t['border']}; padding: 10px 0 6px; margin-top: 8px;
}}
/* sticky must sit on the wrapper (the footer's own parent is exactly its height, so it could never stick) */
section[data-testid="stSidebar"] [data-testid="stLayoutWrapper"]:has(> .st-key-wr_sidebar_footer) {{
  position: sticky; bottom: 0; z-index: 2; background: {t['surface']};
}}
h1, .wr-h1 {{ font-size: clamp(22px, calc(1.6vw + 8px), 28px) !important; font-weight: 600 !important;
             letter-spacing: -0.01em; line-height: 1.2 !important; color: {t['text']}; margin: 0; }}
h2, .wr-h2 {{ font-size: 20px !important; font-weight: 600 !important; }}
h3, .wr-h3 {{ font-size: 16px !important; font-weight: 600 !important; }}
.wr-caption {{ font-size: 12px; color: {t['text_muted']}; }}
/* hide footer + deploy button; keep the menu for Settings -> Theme */
footer, [data-testid="stAppDeployButton"], .stAppDeployButton, [data-testid="stDecoration"] {{ display: none !important; }}
/* respect reduced-motion preferences */
@media (prefers-reduced-motion: reduce) {{ *, *::before, *::after {{ animation-duration: 0.01ms !important;
  transition-duration: 0.01ms !important; scroll-behavior: auto !important; }} }}
/* visible keyboard focus (s4 accessibility) */
*:focus-visible {{ outline: 2px solid {ACCENTS['royal_blue'] if mode == 'light' else '#8FA8EE'} !important;
                  outline-offset: 2px !important; }}
.wr-num {{ font-family: 'JetBrains Mono', ui-monospace, monospace; font-variant-numeric: tabular-nums; }}

/* ---------- components ---------- */
.wr-header {{ display:flex; flex-wrap:wrap; align-items:flex-end; justify-content:space-between;
             gap: 8px 16px; margin-bottom: var(--wr-gap); }}
.wr-header .wr-sub {{ color:{t['text_muted']}; font-size:14px; margin-top:4px; }}
.wr-badges {{ display:flex; flex-wrap:wrap; gap:6px; }}
.wr-section {{ margin: var(--wr-gap) 0 12px; }}
.wr-section .wr-sub {{ color:{t['text_muted']}; font-size:13px; margin-top:2px; }}
.wr-card {{ background:{t['surface']}; border:1px solid {t['border']}; border-radius:12px;
           padding:16px 18px; min-width:0; }}
.wr-kpi-grid {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
               gap:16px; margin-bottom: var(--wr-gap); }}
.wr-kpi .wr-kpi-label {{ font-size:12px; color:{t['text_muted']}; }}
.wr-kpi .wr-kpi-value {{ font-size: clamp(22px, 1.8vw, 32px); font-weight:600; color:{t['text']};
                        line-height:1.15; margin-top:6px; }}
.wr-kpi .wr-kpi-delta {{ font-size:12px; margin-top:4px; }}
.wr-badge {{ display:inline-flex; align-items:center; gap:4px; font-size:12px; font-weight:600;
            padding:2px 8px; border-radius:999px; border:1px solid transparent; white-space:nowrap; }}
{tones}
.wr-avatar {{ width:40px; height:40px; border-radius:50%; display:inline-flex; align-items:center;
             justify-content:center; font-weight:600; font-size:14px; flex:none;
             background:{t['surface_2']}; color:{t['text']}; border:1px solid {t['border']}; }}
.wr-ring {{ flex:none; }}
.wr-ring text {{ fill:{t['text']}; font-family:'JetBrains Mono', ui-monospace, monospace; font-weight:600; }}
.wr-pcard {{ display:flex; flex-direction:column; gap:10px; height:100%; }}
.wr-pcard-top {{ display:flex; align-items:center; gap:10px; min-width:0; }}
.wr-pcard-name {{ font-weight:600; color:{t['text']}; font-size:15px; white-space:nowrap; overflow:hidden;
                 text-overflow:ellipsis; min-width:0; }}
.wr-pcard-meta {{ font-size:12px; color:{t['text_muted']}; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
.wr-pcard-id {{ flex:1; min-width:0; }}
.wr-stats {{ display:grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap:6px; }}
.wr-stat {{ background:{t['surface_2']}; border-radius:8px; padding:6px 8px; min-width:0; }}
.wr-stat .k {{ font-size:11px; color:{t['text_muted']}; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
.wr-stat .v {{ font-size:14px; font-weight:600; color:{t['text']}; }}
.wr-row {{ display:flex; flex-wrap:wrap; align-items:center; gap:6px; }}
.wr-flags {{ display:flex; flex-direction:column; gap:6px; }}
.wr-flag {{ display:flex; gap:8px; align-items:flex-start; font-size:13px; color:{t['text']}; }}
.wr-flag .why {{ color:{t['text_muted']}; }}
.wr-empty {{ text-align:center; padding:28px 16px; color:{t['text_muted']}; }}
.wr-empty .icon {{ font-size:22px; }}
.wr-privacy {{ border-left:3px solid {ACCENTS['royal_blue']}; }}
.wr-table-wrap {{ overflow-x:auto; border:1px solid {t['border']}; border-radius:12px; }}
table.wr-table {{ border-collapse:collapse; width:100%; font-size:13px; color:{t['text']}; }}
table.wr-table th, table.wr-table td {{ padding:8px 12px; border-bottom:1px solid {t['border']}; text-align:left;
                                        white-space:nowrap; }}
table.wr-table th {{ color:{t['text_muted']}; font-weight:600; font-size:12px; background:{t['surface']}; }}
table.wr-table td:first-child, table.wr-table th:first-child {{ position:sticky; left:0; background:{t['surface']}; }}
table.wr-table td.best {{ font-weight:600; }}
.wr-tiles {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap:16px; }}
.wr-tile h3 {{ margin:0 0 6px; }}
.wr-tile p {{ margin:0; color:{t['text_muted']}; font-size:13px; }}
.wr-chips {{ display:flex; flex-wrap:wrap; gap:8px; }}
/* fluid card grid: keyed containers become CSS grids (s4b E) */
/* 260px min (spec says 240) so the 5 top-10 cards fill one row at 1920 and 3 fit at 1366 */
[class*="st-key-wr_grid_"] {{ display:grid !important; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
                              gap:16px !important; align-items:stretch; }}
[class*="st-key-wr_grid_"] > div {{ width:auto !important; min-width:0 !important; }}
[class*="st-key-wr_tiles_"] {{ display:grid !important; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
                               gap:16px !important; align-items:stretch; }}
[class*="st-key-wr_tiles_"] > div {{ width:auto !important; min-width:0 !important; }}
[class*="st-key-wr_minigrid_"] {{ display:grid !important; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
                                  gap:12px !important; }}
[class*="st-key-wr_minigrid_"] > div {{ width:auto !important; min-width:0 !important; }}
/* two-chart rows: side by side >= 1280px viewport, stacked below (s4b E) */
[class*="st-key-wr_pair_"] {{ display:grid !important; grid-template-columns: minmax(0, 1fr); gap:16px !important; }}
@media (min-width: 1280px) {{ [class*="st-key-wr_pair_"] {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} }}
[class*="st-key-wr_pair_"] > div {{ width:auto !important; min-width:0 !important; }}
.wr-profile-head {{ display:flex; flex-wrap:wrap; align-items:center; gap:16px; }}
.wr-profile-head .wr-avatar {{ width:56px; height:56px; font-size:18px; }}
.wr-profile-id {{ flex:1 1 280px; min-width:0; }}
.wr-profile-id h1 {{ margin:0 0 4px !important; }}
.wr-profile-price {{ text-align:right; }}
.wr-comp {{ height:100%; }}
.wr-callout {{ border-left:3px solid {ACCENTS['pink']}; }}
.wr-why {{ font-size:14px; color:{t['text']}; margin:4px 0 0; }}
/* auction planner: constraints beside results >= 1366px, full width above them below (s4b E) */
.st-key-wr_planner {{ display:grid !important; grid-template-columns: minmax(0, 1fr); gap:20px !important; align-items:start; }}
@media (min-width: 1366px) {{ .st-key-wr_planner {{ grid-template-columns: minmax(300px, 360px) minmax(0, 1fr); }} }}
.st-key-wr_planner > div {{ width:auto !important; min-width:0 !important; }}
/* upload stepper: numbered steps, current one bold with a text label (not colour alone) */
.wr-stepper {{ display:flex; flex-wrap:wrap; gap:8px; margin:0 0 16px; }}
.wr-step {{ display:inline-flex; align-items:center; gap:8px; padding:6px 12px; border-radius:8px; font-size:14px;
           background:{t['surface']}; border:1px solid {t['border']}; color:{t['text_muted']}; }}
.wr-step .n {{ display:inline-flex; width:22px; height:22px; border-radius:50%; align-items:center; justify-content:center;
              font-size:12px; font-weight:600; background:{t['surface_2']}; color:{t['text']}; }}
.wr-step.now {{ color:{t['text']}; font-weight:600; border-color:{ACCENTS['pink']}; }}
.wr-step.now .n {{ background:{ACCENTS['pink']}; color:#FFFFFF; }}
.wr-step.done {{ color:{t['text']}; }}
/* compact cards (squad grids): no avatar, so names get the room */
.wr-pcard.compact .wr-avatar {{ display:none; }}
/* player cards: keyed containers styled as cards */
[class*="st-key-wr_card_"] {{ background:{t['surface']}; border:1px solid {t['border']}; border-radius:12px;
                              padding:14px 16px; min-width:0; height:100%; }}
[class*="st-key-wr_card_"] [data-testid="stElementContainer"]:has(button) {{ margin-top:auto; }}
/* ---------- mockup alignment (sidebar brand + flat nav, Home tiles, dataset card, cards) ---------- */
.wr-brand {{ display:flex; align-items:center; gap:10px; padding:4px 4px 14px; }}
.wr-mark {{ width:24px; height:24px; border-radius:7px; background:{ACCENTS['pink']}; position:relative; flex:none; overflow:hidden; }}
.wr-mark::after {{ content:""; position:absolute; right:0; bottom:0; width:12px; height:12px; background:{ACCENTS['royal_blue']}; }}
.wr-brand-name {{ font-weight:600; font-size:14px; color:{t['text']}; line-height:1.2; }}
.wr-brand-sub {{ font-size:12px; color:{t['text_muted']}; }}
section[data-testid="stSidebar"] [data-testid="stPageLink"] a {{ border-radius:8px; padding:6px 12px; }}
section[data-testid="stSidebar"] [data-testid="stPageLink"] a span {{ color:{t['text_muted']}; font-size:14px; }}
section[data-testid="stSidebar"] [data-testid="stPageLink"] a[aria-current="page"],
section[data-testid="stSidebar"] [data-testid="stPageLink"] a:hover {{ background:{t['surface_2']}; }}
section[data-testid="stSidebar"] [data-testid="stPageLink"] a[aria-current="page"] span {{ color:{t['text']}; font-weight:600; }}
.wr-side-row {{ display:flex; align-items:center; justify-content:space-between; gap:8px; font-size:14px;
               color:{t['text_muted']}; margin:4px 0; }}
.wr-eyebrow {{ display:flex; flex-wrap:wrap; gap:8px; margin-bottom:12px; }}
.wr-hero .wr-sub {{ font-size:16px; }}
.wr-tile-num {{ font-family:'JetBrains Mono', ui-monospace, monospace; font-size:12px; color:{t['text_muted']}; }}
.wr-tile h2 {{ margin:8px 0 6px !important; }}
.wr-tile p {{ margin:0; color:{t['text_muted']}; font-size:14px; line-height:1.5; }}
[class*="st-key-wr_card_tile_"] [data-testid="stPageLink"] a {{ padding:0; background:none !important; }}
[class*="st-key-wr_card_tile_"] [data-testid="stPageLink"] a span {{ color:{'#8FA8EE' if mode == 'dark' else ACCENTS['royal_blue']};
                                                                     font-weight:600; }}
.wr-coming {{ font-size:14px; color:{t['text_muted']}; font-weight:600; }}
.wr-ds-head {{ display:flex; flex-wrap:wrap; justify-content:space-between; align-items:baseline; gap:6px 16px;
              margin-bottom:12px; }}
.wr-ds-head h3 {{ margin:0 !important; }}
.wr-chip {{ display:inline-flex; align-items:center; gap:4px; font-size:13px; font-weight:600; padding:4px 10px;
           border-radius:6px; background:{t['surface_2']}; color:{t['text']}; white-space:nowrap; }}
.wr-pcard-head {{ display:flex; justify-content:space-between; align-items:baseline; font-size:12px; }}
.wr-pcard-rank {{ font-family:'JetBrains Mono', ui-monospace, monospace; font-weight:600;
                 color:{'#F48BB3' if mode == 'dark' else '#A8144F'}; font-size:13px; }}
.wr-pcard-role {{ color:{t['text_muted']}; }}
.wr-agechip {{ display:inline-block; padding:0 6px; border-radius:4px; background:{t['surface_2']}; color:{t['text']}; }}
.wr-stats-line {{ display:grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap:6px; padding:10px 0;
                 border-top:1px solid {t['border']}; border-bottom:1px solid {t['border']}; }}
.wr-stats-line .k {{ font-size:11px; color:{t['text_muted']}; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
.wr-stats-line .v {{ font-size:15px; font-weight:600; color:{t['text']}; }}
.wr-price {{ font-family:'JetBrains Mono', ui-monospace, monospace; font-weight:600; font-size:14px; color:{t['text']}; }}
.wr-price-band {{ font-size:11px; color:{t['text_muted']}; }}
/* login card */
.st-key-wr_login {{ max-width: 420px; margin: 8vh auto 0; background:{t['surface']}; border:1px solid {t['border']};
                   border-radius:12px; padding:24px; }}
</style>
"""

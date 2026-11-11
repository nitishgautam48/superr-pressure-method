"""Shared top-10 board markup (Overview, Weights Lab, Checklist)."""
from __future__ import annotations

import pandas as pd

from app import components as C
from app import data as D

PROV_TONE = {"model": ("neutral", "◇"), "analyst-confirmed": ("success", "✓"), "analyst-moved": ("blue", "↕"),
             "analyst-add": ("pink", "+")}


def board_html(title: str, rows: list[dict], rated: pd.DataFrame, show_prov: bool) -> str:
    r = rated.set_index("player_id")
    body = []
    for x in rows:
        p = r.loc[x["player_id"]] if x["player_id"] in r.index else None
        lab = p["label"] if p is not None else None
        tone, icon = D.label_tone(lab)
        interval = (f'{p["boot_rank_lo"]:.0f}-{p["boot_rank_hi"]:.0f}' if p is not None and pd.notna(p.get("boot_rank_lo"))
                    else "–")
        prov = ""
        if show_prov:
            pt, pi = PROV_TONE.get(x.get("provenance", "model"), ("neutral", "◇"))
            prov = C.badge_html(x.get("provenance", "model"), pt, pi, title=x.get("reason") or None)
        team = C._e(str(p["team"])) if p is not None and pd.notna(p.get("team")) else ""
        body.append(f'<tr><td class="wr-num">{x["rank"]}</td><td><b>{C._e(x["name"])}</b>'
                    f'<div class="wr-caption">{team}</div></td><td class="wr-num">{x["prs"]:.0f}</td>'
                    f'<td>{C.badge_html(str(lab or "n/a"), tone, icon)}</td><td class="wr-num">{interval}</td>'
                    f'<td class="wr-num">{(p["p_top10"] if p is not None else 0):.0f}%</td><td>{prov}</td></tr>')
    head = ("<tr><th>#</th><th>Player</th><th>PRS</th><th>Label</th><th>Rank 80%</th><th>P(top 10)</th>"
            f"<th>{'Provenance' if show_prov else ''}</th></tr>")
    return (f'<div class="wr-card"><h3 class="wr-h3">{C._e(title)}</h3><div class="wr-table-wrap">'
            f'<table class="wr-table">{head}{"".join(body)}</table></div></div>')

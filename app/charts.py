"""Plotly charts for the dashboard (warroom templates from theme.py): hover values everywhere, no pies / 3D,
at most three overlaid series, colour always paired with a label."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from app import theme

PINK, GOLD = theme.ACCENTS["pink"], theme.ACCENTS["gold"]


def _t(mode: str) -> str:
    return f"warroom_{mode}"


def blue(mode: str) -> str:
    return theme.SERIES[mode][1]


def muted(mode: str) -> str:
    return theme.MUTED_MARK[mode]


def match_curves(d: pd.DataFrame, thr: float, mode: str = "dark") -> go.Figure:
    """Win probability (batting side of innings 1 kept as one line) and pressure index, ball by ball."""
    d = d.sort_values(["innings", "innings_ball_num"]).reset_index(drop=True)
    x = np.arange(1, len(d) + 1)
    team1 = d.loc[d["innings"] == 1, "batting_team_id"].iloc[0] if (d["innings"] == 1).any() else None
    wp1 = np.where(d["batting_team_id"] == team1, d["wp_after"], 1 - d["wp_after"])
    lab = [f"Inn {i} · {o - 1}.{b} · {s} v {w}" for i, o, b, s, w in
           zip(d["innings"], d["overid"], d["over_ball_num"], d["striker_name"], d["bowler_name"])]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.55, 0.45],
                        subplot_titles=["Win probability of the team batting first", "Pressure index (1.0 = average ball)"])
    fig.add_trace(go.Scatter(x=x, y=wp1, mode="lines", line={"color": blue(mode), "width": 2}, name="Win probability",
                             customdata=lab, hovertemplate="%{customdata}<br>WP %{y:.2f}<extra></extra>"), row=1, col=1)
    hi = d["pressure_index"] >= thr
    fig.add_trace(go.Bar(x=x, y=d["pressure_index"], marker_color=np.where(hi, PINK, muted(mode)), name="Pressure index",
                         customdata=lab, hovertemplate="%{customdata}<br>pressure %{y:.2f}<extra></extra>"), row=2, col=1)
    fig.add_hline(y=thr, line={"dash": "dot", "color": theme.TOKENS[mode]["text_muted"]}, row=2, col=1,
                  annotation_text=f"high pressure ≥ {thr:.2f}", annotation_position="top left")
    split = int((d["innings"] == 1).sum())
    if 0 < split < len(d):
        fig.add_vline(x=split + 0.5, line={"color": theme.TOKENS[mode]["text_muted"], "width": 1})
    for i, w in enumerate(d["is_team_wicket"]):
        if w:
            fig.add_trace(go.Scatter(x=[i + 1], y=[wp1[i]], mode="markers", marker={"color": GOLD, "size": 9,
                                     "symbol": "x"}, showlegend=False, hovertemplate=lab[i] + "<br>wicket<extra></extra>"),
                          row=1, col=1)
    fig.update_yaxes(range=[0, 1], row=1, col=1)
    top = float(max(thr * 2.5, d["pressure_index"].quantile(0.98) * 1.3))
    fig.update_yaxes(range=[0, top], row=2, col=1, title_text=f"capped at {top:.0f}; hover = exact")
    fig.update_layout(template=_t(mode), height=520, showlegend=False, bargap=0)
    fig.update_xaxes(title_text="Ball of the match (gold × = wicket)", row=2, col=1)
    return fig


def compare_bars(df: pd.DataFrame, metric_label: str, mode: str = "dark") -> go.Figure:
    """df columns: label, pressure, normal (one row per player or split)."""
    fig = go.Figure([go.Bar(name="Pressure", x=df["label"], y=df["pressure"], marker_color=PINK,
                            text=[f"{v:.1f}" if pd.notna(v) else "–" for v in df["pressure"]], textposition="outside",
                            hovertemplate="%{x}: %{y:.2f}<extra>Pressure</extra>"),
                     go.Bar(name="Other balls", x=df["label"], y=df["normal"], marker_color=muted(mode),
                            text=[f"{v:.1f}" if pd.notna(v) else "–" for v in df["normal"]], textposition="outside",
                            hovertemplate="%{x}: %{y:.2f}<extra>Other balls</extra>")])
    fig.update_layout(template=_t(mode), barmode="group", yaxis_title=metric_label, height=340)
    return fig


def strip(pm: pd.DataFrame, names: dict, mode: str = "dark") -> go.Figure:
    """One row per player, one dot per match (pressure value), median tick."""
    fig = go.Figure()
    rng = np.random.default_rng(0)
    order = list(names)
    for i, pid in enumerate(order):
        v = pm.loc[pm["player_id"] == pid, "value"].clip(-3, 3)
        y = i + rng.uniform(-0.22, 0.22, len(v))
        fig.add_trace(go.Scatter(x=v, y=y, mode="markers", name=names[pid], showlegend=False,
                                 marker={"size": 8, "color": np.where(v > 0, PINK, muted(mode)), "opacity": 0.8},
                                 hovertemplate=names[pid] + ": %{x:.2f}<extra></extra>"))
        if len(v):
            fig.add_trace(go.Scatter(x=[v.median()] * 2, y=[i - 0.35, i + 0.35], mode="lines", showlegend=False,
                                     line={"color": theme.TOKENS[mode]["text"], "width": 3},
                                     hovertemplate=f"median {v.median():.2f}<extra></extra>"))
    fig.add_vline(x=0, line={"color": theme.TOKENS[mode]["text_muted"], "width": 1})
    fig.update_yaxes(tickvals=list(range(len(order))), ticktext=[names[p] for p in order], autorange="reversed")
    fig.update_layout(template=_t(mode), height=max(220, 70 * len(order) + 80),
                      xaxis_title="Mean value per pressure ball in a match (capped ±3) · pink = positive")
    return fig


def rank_hist(hist: Sequence[int], model_rank: int | None, mode: str = "dark") -> go.Figure:
    x = np.arange(1, len(hist) + 1)
    fig = go.Figure(go.Bar(x=x, y=hist, marker_color=np.where(x <= 10, PINK, muted(mode)),
                           hovertemplate="rank %{x}: %{y} of 1,000 resamples<extra></extra>"))
    if model_rank:
        fig.add_vline(x=model_rank, line={"color": GOLD, "width": 2}, annotation_text=f"model rank {model_rank}")
    fig.update_layout(template=_t(mode), height=280, xaxis_title="Rank in bootstrap resample (40 = 40 or worse)",
                      yaxis_title="Resamples")
    return fig


def heatmap(m: pd.DataFrame, mode: str = "dark") -> go.Figure:
    z = m.to_numpy(dtype=float)
    fig = go.Figure(go.Heatmap(z=np.clip(z, 1, 40), x=list(m.columns), y=list(m.index), colorscale=[[0, PINK], [0.25, blue(mode)], [1, muted(mode)]],
                               zmin=1, zmax=40, text=np.where(np.isnan(z), "", np.nan_to_num(z).astype(int).astype(str)),
                               texttemplate="%{text}", hovertemplate="%{y}<br>%{x}: rank %{z:.0f}<extra></extra>",
                               colorbar={"title": "Rank"}))
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(template=_t(mode), height=max(360, 26 * len(m) + 120), xaxis_tickangle=-30)
    return fig


def scatter(df: pd.DataFrame, x: str, y: str, label: str, xt: str, yt: str, highlight: Sequence = (),
            diagonal: bool = True, mode: str = "dark") -> go.Figure:
    hi = df[label].isin(highlight)
    fig = go.Figure()
    for flag, color, size, name in ((False, muted(mode), 8, "Others"), (True, PINK, 12, "Highlighted")):
        d = df[hi == flag]
        fig.add_trace(go.Scatter(x=d[x], y=d[y], mode="markers+text" if flag else "markers", name=name,
                                 text=d[label] if flag else None, textposition="top center",
                                 marker={"size": size, "color": color}, customdata=d[label],
                                 hovertemplate="%{customdata}<br>" + xt + " %{x:.3f}<br>" + yt + " %{y:.3f}<extra></extra>"))
    if diagonal and len(df):
        lo, hi_ = float(np.nanmin(df[[x, y]].to_numpy())), float(np.nanmax(df[[x, y]].to_numpy()))
        fig.add_trace(go.Scatter(x=[lo, hi_], y=[lo, hi_], mode="lines", line={"dash": "dash", "color": muted(mode)},
                                 name="Equal", hoverinfo="skip"))
    fig.update_layout(template=_t(mode), xaxis_title=xt, yaxis_title=yt, height=420)
    return fig


def show(fig: go.Figure, key: str | None = None) -> None:
    import streamlit as st

    st.plotly_chart(fig, width="stretch", theme=None, key=key, config={"responsive": True, "displaylogo": False})

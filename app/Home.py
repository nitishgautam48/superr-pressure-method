"""RR Pressure Index dashboard - entry point and router.

    APP_PASSWORD=... streamlit run app/Home.py

Before login the only registered page is the login card, so no URL can show data.
"""
from __future__ import annotations

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR.parent) not in sys.path:
    sys.path.insert(0, str(APP_DIR.parent))

import streamlit as st  # noqa: E402

from app import components as C  # noqa: E402
from app import data as D  # noqa: E402
from app import state, theme  # noqa: E402

st.set_page_config(page_title="RR Pressure Index", page_icon="✦", layout="wide", initial_sidebar_state="expanded")
MODE = state.theme_mode()
st.html(theme.css(MODE))

PAGE_FILES = [("1_Overview.py", "Overview"), ("2_Pressure_Lab.py", "Pressure lab"), ("3_Rankings.py", "Rankings"),
              ("4_Player_Profile.py", "Player profile"), ("5_Compare.py", "Compare"), ("6_Reliability.py", "Reliability"), ("11_Methodology.py", "Methodology"),
              ("7_Weights_Lab.py", "Weights lab"), ("8_My_Decisions.py", "My decisions"),
              ("9_Data_Quality.py", "Data quality"), ("10_Submission_Checklist.py", "Submission checklist"),
              ("12_Run_On_Data.py", "Run on a data pack")]


def login_page() -> None:
    with st.container(key="wr_login"):
        st.html('<h1 class="wr-h1">RR Pressure Index</h1>'
                '<div class="wr-caption" style="margin:6px 0 16px">Private analysis tool. Log in to continue.</div>')
        if not state.app_password():
            st.error("No password is configured. Set APP_PASSWORD in .streamlit/secrets.toml or the environment.",
                     icon=":material/lock:")
            return
        with st.form("login", border=False):
            pw = st.text_input("Password", type="password", key="login_pw")
            if st.form_submit_button("Log in", type="primary", width="stretch"):
                if state.login(pw):
                    st.rerun()
                st.error("Wrong password. Try again.", icon=":material/error:")


def sidebar(nav_pages) -> None:
    meta = D.js("run_meta.json")
    with st.sidebar:
        st.html('<div class="wr-brand"><span class="wr-mark" aria-hidden="true"></span><div>'
                '<div class="wr-brand-name">RR Pressure Index</div><div class="wr-brand-sub">'
                + ("Method · upload your data" if D.public_mode() else "Private · SupeRR Selector") + '</div>'
                '</div></div>')
        for p in nav_pages:
            st.page_link(p)
        st.divider()
        src = ("Your upload" if D.upload_dir() else "Showcase" if D.showcase_mode()
               else ("Real pack" if D.available() else "No outputs"))
        C._show(f'<div class="wr-side-row"><span>Theme</span>{C.badge_html(MODE.title(), "neutral", "◐")}</div>'
                f'<div class="wr-caption" style="margin:-2px 0 8px">Switch in ⋮ menu → Settings</div>'
                f'<div class="wr-side-row"><span>Data</span>{C.badge_html(src, "blue" if src == "Showcase" else ("success" if D.available() else "warning"), "◆")}</div>'
                + ('<div class="wr-caption" style="margin:4px 0 8px">Read-only · aggregates only</div>' if src == "Showcase" else ""))
        with st.container(key="wr_sidebar_footer"):
            st.html(f'<div class="wr-caption">Last run {C._e(C.fmt_run_at(meta.get("run_at")))} · config '
                    f'{C._e(str(meta.get("config_hash", "–")))}</div>')
            if st.button("Log out", key="logout"):
                state.logout()
                st.rerun()


if not state.is_authed():
    st.navigation([st.Page(login_page, title="Log in", url_path="login", default=True)], position="hidden").run()
    st.stop()
if not D.available():   # no outputs yet: upload page (+ methodology, readable before any data)
    first = [st.Page("views/12_Run_On_Data.py", title="Run on a data pack", default=True),
             st.Page("views/11_Methodology.py", title="Methodology")]
    nav = st.navigation(first, position="hidden")
    sidebar(first)
    C.safe_run(nav)
    st.stop()

HIDDEN_IN_SHOWCASE = {"10_Submission_Checklist.py"}   # private prep (answers, raw mapping) never hosted
PAGES = [st.Page(f"views/{f}", title=t, default=(i == 0)) for i, (f, t) in enumerate(PAGE_FILES)
         if (APP_DIR / "views" / f).exists() and not ((D.showcase_mode() or D.public_mode()) and f in HIDDEN_IN_SHOWCASE)]
nav = st.navigation(PAGES, position="hidden")
sidebar(PAGES)
C.safe_run(nav)

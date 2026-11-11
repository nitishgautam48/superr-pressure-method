"""Session helpers: password gate (st.secrets / APP_PASSWORD), selected player, compare list, theme."""
from __future__ import annotations

import hmac
import os
from typing import Any

import streamlit as st


def app_password() -> str | None:
    try:
        pw = st.secrets.get("APP_PASSWORD")
    except Exception:  # noqa: BLE001
        pw = None
    return pw or os.environ.get("APP_PASSWORD") or None


def check_password(candidate: str) -> bool:
    pw = app_password()
    return bool(pw) and hmac.compare_digest(candidate.encode(), pw.encode())


def is_authed() -> bool:
    return bool(st.session_state.get("authed"))


def login(candidate: str) -> bool:
    st.session_state["authed"] = ok = check_password(candidate)
    return ok


def logout() -> None:
    for k in list(st.session_state.keys()):
        del st.session_state[k]


def select_player(pid: Any, role: str | None = None) -> None:
    st.session_state["selected_player"] = int(pid)
    if role:
        st.session_state["selected_role"] = role


def selected_player() -> int | None:
    return st.session_state.get("selected_player")


def compare_list() -> list[int]:
    return st.session_state.setdefault("compare", [])


def add_to_compare(pid: int, limit: int = 4) -> bool:
    lst = compare_list()
    if pid in lst:
        return True
    if len(lst) >= limit:
        return False
    lst.append(int(pid))
    return True


def theme_mode() -> str:
    try:
        t = st.context.theme.type
    except Exception:  # noqa: BLE001
        t = None
    return t if t in ("dark", "light") else "dark"

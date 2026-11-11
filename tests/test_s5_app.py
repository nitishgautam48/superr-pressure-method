"""S5: every dashboard page renders on real outputs; gate; analyst card; reason rules."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from tests.conftest import needs_data

HOME = str(Path(__file__).resolve().parents[1] / "app" / "Home.py")
PAGES = ["1_Overview.py", "2_Pressure_Lab.py", "3_Rankings.py", "4_Player_Profile.py", "5_Compare.py", "6_Reliability.py",
         "7_Weights_Lab.py", "8_My_Decisions.py", "9_Data_Quality.py", "10_Submission_Checklist.py",
         "11_Methodology.py"]


@pytest.fixture(scope="module")
def app_data(s1, cfg):
    from app import data as D
    from pressure import steps_pressure, steps_rating, steps_reliability

    if "finals" not in s1.tables:
        if "wp_report" not in s1.tables:
            steps_pressure.run(s1)
        steps_rating.run(s1)
        steps_reliability.run(s1)
    D.set_paths(processed=cfg.path("processed"), overrides=cfg.path("overrides"), decision_log=cfg.path("decision_log"),
                outputs_public=cfg.path("outputs_public"), outputs_private=cfg.path("outputs_private"),
                raw=cfg.path("raw"))
    yield D
    D.set_paths()


def _at(monkeypatch):
    monkeypatch.setenv("APP_PASSWORD", "pw")
    at = AppTest.from_file(HOME, default_timeout=240)
    at.session_state["authed"] = True
    return at.run()


def test_gate_shows_nothing_before_login(monkeypatch):
    monkeypatch.setenv("APP_PASSWORD", "pw")
    at = AppTest.from_file(HOME, default_timeout=60).run()
    txt = " ".join(h.proto.body for h in at.get("html"))
    assert "Log in to continue" in txt and "Top 10" not in txt
    at.text_input(key="login_pw").set_value("nope")
    at.get("form_submit_button")[0].click()
    at.run()
    assert any("Wrong password" in e.value for e in at.error)


@needs_data
def test_every_page_renders(monkeypatch, app_data):
    at = _at(monkeypatch)
    for p in PAGES:
        at.switch_page(f"views/{p}").run()
        assert not at.exception, (p, [e.value for e in at.exception])


@needs_data
def test_overview_boards_and_toggle(monkeypatch, app_data):
    at = _at(monkeypatch)
    at.switch_page("views/1_Overview.py").run()
    txt = " ".join(m.value for m in at.markdown)
    assert "Top 10 batters under pressure" in txt and "Top 10 bowlers under pressure" in txt
    fl = app_data.js("final_lists.json")
    assert fl["batters_final"][0]["name"] in txt


@needs_data
def test_analyst_card_requires_reason_then_updates_final(monkeypatch, app_data):
    at = _at(monkeypatch)
    at.session_state["selected_role"] = "batter"
    at.switch_page("views/4_Player_Profile.py").run()
    at.text_area(key="pp_reason").set_value("short")
    at.get("form_submit_button")[0].click()
    at.run()
    assert any("reason" in e.value for e in at.error)
    pid = at.session_state["selected_player"]
    at.selectbox(key="pp_action").set_value("pin")
    at.text_area(key="pp_reason").set_value("Highest pressure value across both competitions")
    at.get("form_submit_button")[0].click()
    at.run()
    assert not at.exception and any("Saved" in s.value for s in at.success)
    fl = app_data.js("final_lists.json")
    pick = next(x for x in fl["batters_final"] if x["player_id"] == pid)
    assert pick["provenance"] == "analyst-confirmed" and pick["reason"].startswith("Highest")


@needs_data
def test_weights_lab_save_refused_without_reason(monkeypatch, app_data):
    at = _at(monkeypatch)
    at.switch_page("views/7_Weights_Lab.py").run()
    at.slider(key="wl_A").set_value(0.7)
    at.get("form_submit_button")[0].click()        # Apply
    at.run()
    assert not at.exception
    at.text_area(key="wl_reason").set_value("no")
    at.get("form_submit_button")[1].click()        # Save
    at.run()
    assert any("reason" in e.value for e in at.error)


@needs_data
def test_methodology_documents_live_settings(monkeypatch, app_data):
    at = _at(monkeypatch)
    at.switch_page("views/11_Methodology.py").run()
    assert not at.exception
    assert len(at.tabs) == 9 and len(at.get("latex")) >= 8
    assert any("not report text" in i.value for i in at.info)


def test_no_page_reachable_without_login(monkeypatch):
    """Regression: a folder named pages/ made every page its own URL, skipping the login gate."""
    assert not (Path(HOME).parent / "pages").exists()
    monkeypatch.setenv("APP_PASSWORD", "pw")
    at = AppTest.from_file(HOME, default_timeout=60).run()
    for p in PAGES:
        try:
            at.switch_page(f"views/{p}").run()
        except Exception:  # noqa: BLE001 - an unregistered page may be refused outright
            continue
        txt = " ".join(h.proto.body for h in at.get("html")) + " ".join(m.value for m in at.markdown)
        assert "Log in to continue" in txt and "Top 10" not in txt and not at.dataframe, p

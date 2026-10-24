"""End-to-end: run the real Streamlit script headlessly on synthetic data."""
from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).parent.parent / "app.py")


@pytest.fixture()
def at(synthetic_dir, monkeypatch):
    monkeypatch.setenv("GLOBALLENS_DATA_DIR", str(synthetic_dir))
    st.cache_data.clear()
    st.cache_resource.clear()
    app = AppTest.from_file(APP, default_timeout=120)
    app.run()
    return app


def no_errors(app):
    assert not app.exception, [e.value for e in app.exception]
    assert not app.error, [e.value for e in app.error]


def test_app_boots_with_default_scenario(at):
    no_errors(at)
    assert at.title[0].value == "GlobalLens"
    assert len(at.tabs) >= 5
    # Pakistan is the default home country
    assert at.sidebar.selectbox[0].value == "Pakistan"
    assert len(at.metric) >= 8


def test_every_scenario_runs(at):
    sel = next(s for s in at.selectbox if s.label == "Scenario")
    for name in sel.options:
        sel.select(name).run()
        no_errors(at)
        sel = next(s for s in at.selectbox if s.label == "Scenario")


def test_every_industry_preset_runs(at):
    sel = next(s for s in at.selectbox if s.label == "What are you taking abroad?")
    for name in sel.options:
        sel.select(name).run()
        no_errors(at)
        sel = next(s for s in at.selectbox if s.label == "What are you taking abroad?")


def test_changing_home_country_updates_the_overview(at):
    at.sidebar.selectbox[0].select("Brazil").run()
    no_errors(at)
    assert any("Brazil at a glance" in m.value for m in at.markdown)


def test_custom_scenario_with_nothing_picked_degrades_gracefully(at):
    next(s for s in at.selectbox if s.label == "Scenario").select("Custom").run()
    no_errors(at)
    assert any("at least one country" in i.value for i in at.info)


def test_tight_filters_show_a_warning_not_a_crash(at):
    # only two synthetic countries are "Low income" -> fewer than the 5 needed
    inc = next(m for m in at.multiselect if m.label == "Income groups")
    inc.set_value(["Low income"]).run()
    no_errors(at)
    assert any("Too few countries" in w.value for w in at.warning)
    # the other tabs must still render (combined view explains itself instead of crashing)
    assert any("Market recommender" in i.value for i in at.info)


def test_combined_view_renders_brief_when_both_engines_have_input(at):
    no_errors(at)
    md = " ".join(m.value for m in at.markdown)
    assert "Decision brief" in md and "Best risk-adjusted bets" in md


def test_weights_slider_changes_the_recommendation(at):
    def top_text(app):
        return [m.value for m in app.markdown if "Why the top three" in m.value or "strongest on" in m.value]

    before = top_text(at)
    # crank "Market size" to 10 and everything else to 0
    for s in at.slider:
        if s.label in ("Market size",):
            s.set_value(10)
        elif s.label in ("Growth", "Purchasing power", "Digital & financial access",
                         "Business environment", "Stability", "Logistics & trade"):
            s.set_value(0)
    at.run()
    no_errors(at)
    assert top_text(at) != before

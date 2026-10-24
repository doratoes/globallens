import numpy as np
import pytest

import shocks


@pytest.fixture()
def model(df):
    return shocks.build_trade_model(df)


def test_haversine_known_distances():
    # London - Paris ~ 344 km; identical points = 0; symmetric
    lat, lon = np.array([51.5, 48.9, 51.5]), np.array([-0.1, 2.3, -0.1])
    D = shocks.haversine_matrix(lat, lon)
    assert D[0, 1] == pytest.approx(344, rel=0.03)
    assert D[0, 2] == pytest.approx(0, abs=1e-6)
    assert np.allclose(D, D.T)


def test_trade_matrix_matches_reported_totals(df, model):
    d = df.loc[model.iso3]
    X = d["exports_usd"].to_numpy()
    M = d["imports_usd"].to_numpy()
    M = M * X.sum() / M.sum()
    assert model.max_row_error < 1e-6 and model.max_col_error < 1e-6
    assert np.allclose(model.T.sum(axis=1), X, rtol=1e-6)
    assert np.allclose(model.T.sum(axis=0), M, rtol=1e-6)
    assert np.allclose(np.diag(model.T), 0)
    assert (model.T >= 0).all()


def test_nearby_big_economies_trade_more(df, model):
    i = {c: k for k, c in enumerate(model.iso3)}
    # gravity logic: Germany trades more with neighbouring France than with far-away New Zealand
    assert model.T[i["DEU"], i["FRA"]] > model.T[i["DEU"], i["NZL"]]


def test_countries_missing_inputs_are_left_out(df):
    broken = df.copy()
    broken.loc["PAK", "lat"] = np.nan
    broken.loc["KEN", "exports_usd"] = np.nan
    m = shocks.build_trade_model(broken)
    assert "PAK" not in m.iso3 and "KEN" not in m.iso3
    assert len(m.iso3) == len(df) - 2


def test_no_shock_means_no_risk(df, model):
    r = shocks.simulate_shock(df, model, [], 30)
    assert (r["trade_at_risk_pct_gdp"] == 0).all()
    assert not r["shocked"].any()


def test_risk_grows_with_the_size_of_the_shock(df, model):
    small = shocks.simulate_shock(df, model, ["CHN"], 10)
    big = shocks.simulate_shock(df, model, ["CHN"], 40)
    assert (big["trade_at_risk_pct_gdp"] >= small["trade_at_risk_pct_gdp"] - 1e-12).all()
    assert big["trade_at_risk_pct_gdp"].sum() > 3.5 * small["trade_at_risk_pct_gdp"].sum()


def test_own_loss_is_exact_for_the_shocked_country(df, model):
    r = shocks.simulate_shock(df, model, ["CHN"], 20)
    assert r.loc["CHN", "own_usd"] == pytest.approx(0.2 * df.loc["CHN", "exports_usd"])
    assert r.loc["CHN", "own_pct_gdp"] == pytest.approx(0.2 * df.loc["CHN", "exports_gdp"])
    assert r.drop("CHN")["own_usd"].eq(0).all()


def test_channels_add_up_and_toggles_work(df, model):
    r = shocks.simulate_shock(df, model, ["DEU", "FRA"], 25)
    parts = r[["supply_pct_gdp", "demand_pct_gdp", "own_pct_gdp"]].sum(axis=1)
    assert np.allclose(parts, r["trade_at_risk_pct_gdp"])
    no_dem = shocks.simulate_shock(df, model, ["DEU", "FRA"], 25, demand_spillover=False)
    assert (no_dem["demand_pct_gdp"] == 0).all()
    assert r["demand_pct_gdp"].sum() > 0


def test_substitution_reduces_supply_losses(df, model):
    none = shocks.simulate_shock(df, model, ["CHN"], 30, substitution=0.0)
    full = shocks.simulate_shock(df, model, ["CHN"], 30, substitution=1.0)
    assert (full["supply_pct_gdp"] <= none["supply_pct_gdp"] + 1e-12).all()
    assert full["supply_pct_gdp"].sum() < none["supply_pct_gdp"].sum()
    # with substitution off, supply loss equals the raw lost import value
    raw = 0.3 * model.T[model.iso3.index("CHN"), :] / df.loc[model.iso3, "gdp"].to_numpy() * 100
    assert np.allclose(none["supply_pct_gdp"].to_numpy(), raw)


def test_energy_scope_only_bites_fuel_exporters(df, model):
    gulf = shocks.SCENARIOS["Gulf energy shock"]["countries"]
    allx = shocks.simulate_shock(df, model, gulf, 30, scope="all")
    en = shocks.simulate_shock(df, model, gulf, 30, scope="energy")
    assert en["own_usd"].sum() < allx["own_usd"].sum()
    # a hit country with no fuel exports loses nothing in energy scope
    d2 = df.copy()
    d2.loc["SAU", "fuel_exports"] = 0.0
    en2 = shocks.simulate_shock(d2, model, ["SAU"], 50, scope="energy")
    assert en2["own_usd"].sum() == 0 and en2["trade_at_risk_pct_gdp"].sum() == 0


def test_unknown_shocked_country_is_ignored_not_crashing(df, model):
    r = shocks.simulate_shock(df, model, ["XXX"], 30)
    assert (r["trade_at_risk_pct_gdp"] == 0).all()


def test_tiers_vulnerability_and_driver(df, model):
    r = shocks.simulate_shock(df, model, ["CHN"], 40)
    assert set(r["tier"]) <= {"Low", "Moderate", "High", "Severe"}
    assert r["vulnerability"].between(0, 100).all()
    worst = r.drop("CHN")["trade_at_risk_pct_gdp"].idxmax()
    assert r.loc[worst, "main_driver"] == df.loc["CHN", "name"]
    # more at risk -> never a lower tier
    order = {"Low": 0, "Moderate": 1, "High": 2, "Severe": 3}
    t = r.sort_values("trade_at_risk_pct_gdp")["tier"].map(order)
    assert t.is_monotonic_increasing


def test_scenarios_are_well_formed(df, model):
    for name, sc in shocks.SCENARIOS.items():
        assert sc["scope"] in ("all", "energy") and 5 <= sc["drop"] <= 100, name
        if name != "Custom":
            assert set(sc["countries"]) & set(model.iso3), name

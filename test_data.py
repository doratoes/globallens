"""Data pipeline: parsing of World Bank responses (mocked) and the load path."""
import numpy as np
import pytest

import data


class FakeResp:
    def __init__(self, payload):
        self._p = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._p


class FakeSession:
    """Serves canned World Bank-shaped JSON, two pages for the indicator call."""

    def __init__(self):
        self.calls = []

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append((url, dict(params)))
        if url.endswith("/country"):
            rows = [
                {"id": "PAK", "name": "Pakistan", "region": {"value": "South Asia"},
                 "incomeLevel": {"value": "Lower middle income"}, "latitude": "33.6", "longitude": "73.0"},
                {"id": "WLD", "name": "World", "region": {"value": "Aggregates"},
                 "incomeLevel": {"value": "Aggregates"}, "latitude": "", "longitude": ""},
                {"id": "ABW", "name": "Aruba", "region": {"value": "Latin America & Caribbean"},
                 "incomeLevel": {"value": "High income"}, "latitude": "", "longitude": ""},
            ]
            return FakeResp([{"page": 1, "pages": 1}, rows])
        page = params["page"]
        rows = {
            1: [{"countryiso3code": "PAK", "date": "2023", "value": 12.5},
                {"countryiso3code": "WLD", "date": "2023", "value": 3.0}],
            2: [{"countryiso3code": "ABW", "date": "2022", "value": None},
                {"countryiso3code": "ZZZ", "date": "2022", "value": 1.0}],
        }[page]
        return FakeResp([{"page": page, "pages": 2}, rows])


def test_countries_exclude_aggregates_and_parse_blank_coordinates():
    c = data.fetch_countries(FakeSession())
    assert list(c["iso3"]) == ["PAK", "ABW"]
    assert c.loc[0, "lat"] == pytest.approx(33.6)
    assert np.isnan(c.loc[1, "lat"])


def test_indicator_pagination_and_null_values_dropped():
    s = FakeSession()
    out = data.fetch_indicator(s, "gdp_growth", "NY.GDP.MKTP.KD.ZG")
    # both pages were requested, null value dropped, every real row kept
    assert [c[1]["page"] for c in s.calls] == [1, 2]
    assert s.calls[0][1]["mrnev"] == 1
    assert set(out["iso3"]) == {"PAK", "WLD", "ZZZ"}
    assert out.loc[out["iso3"] == "PAK", "year"].iloc[0] == 2023


def test_load_dataset_shape_and_derived_columns(df):
    assert len(df) >= 50
    assert set(data.INDICATORS) <= set(df.columns)
    assert df.index.is_unique
    assert np.allclose(df["trade_open"], df["exports_gdp"] + df["imports_gdp"])
    assert (df["n_indicators"] >= data.MIN_INDICATORS).all()
    # sparse indicators stay NaN rather than being invented
    assert df["lpi"].isna().sum() > 0


def test_vintage_reports_every_cached_indicator(df):
    v = data.vintage()
    assert list(v.index) == list(data.INDICATORS)
    assert v["countries"].min() > 0
    assert v["median_year"].between(2014, 2025).all()

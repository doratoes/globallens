"""Data layer for GlobalLens.

Pulls country-level indicators from the World Bank Open Data API, caches them as
tidy CSVs in ./data, and serves one clean wide DataFrame (one row per country)
to the rest of the app.

    python data.py --refresh     # re-download everything and rewrite the cache
    python data.py --info        # show what is currently cached

The cache is committed to the repo on purpose: the deployed app starts instantly
and does not depend on the World Bank API being up. If the cache is missing the
app will try to build it on first load.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

API = "https://api.worldbank.org/v2"
# "most recent non-empty value" within this window, so every country is
# represented by its latest available year rather than a single fixed year.
YEAR_RANGE = "2014:2025"
HEADERS = {"User-Agent": "GlobalLens/1.0 (student analytics project)"}

# short name -> (World Bank indicator code, human label)
INDICATORS: dict[str, tuple[str, str]] = {
    "gdp": ("NY.GDP.MKTP.CD", "GDP (current US$)"),
    "gdp_growth": ("NY.GDP.MKTP.KD.ZG", "GDP growth (annual %)"),
    "gdp_pc": ("NY.GDP.PCAP.CD", "GDP per capita (current US$)"),
    "pop": ("SP.POP.TOTL", "Population, total"),
    "pop_growth": ("SP.POP.GROW", "Population growth (annual %)"),
    "working_age": ("SP.POP.1564.TO.ZS", "Population ages 15-64 (% of total)"),
    "urban": ("SP.URB.TOTL.IN.ZS", "Urban population (% of total)"),
    "internet": ("IT.NET.USER.ZS", "Individuals using the Internet (% of population)"),
    "mobile_subs": ("IT.CEL.SETS.P2", "Mobile cellular subscriptions (per 100 people)"),
    "account_ownership": ("FX.OWN.TOTL.ZS", "Account at a financial institution or mobile-money provider (% age 15+)"),
    "exports_gdp": ("NE.EXP.GNFS.ZS", "Exports of goods and services (% of GDP)"),
    "imports_gdp": ("NE.IMP.GNFS.ZS", "Imports of goods and services (% of GDP)"),
    "exports_usd": ("NE.EXP.GNFS.CD", "Exports of goods and services (current US$)"),
    "imports_usd": ("NE.IMP.GNFS.CD", "Imports of goods and services (current US$)"),
    "fdi_gdp": ("BX.KLT.DINV.WD.GD.ZS", "Foreign direct investment, net inflows (% of GDP)"),
    "inflation": ("FP.CPI.TOTL.ZG", "Inflation, consumer prices (annual %)"),
    "lpi": ("LP.LPI.OVRL.XQ", "Logistics Performance Index (1 = low, 5 = high)"),
    "reg_quality": ("GOV_WGI_RQ.EST", "Regulatory quality (WGI, approx. -2.5 to 2.5)"),
    "pol_stability": ("GOV_WGI_PV.EST", "Political stability & absence of violence (WGI)"),
    "rule_of_law": ("GOV_WGI_RL.EST", "Rule of law (WGI)"),
    "gov_effect": ("GOV_WGI_GE.EST", "Government effectiveness (WGI)"),
    "fuel_exports": ("TX.VAL.FUEL.ZS.UN", "Fuel exports (% of merchandise exports)"),
    "manuf_gdp": ("NV.IND.MANF.ZS", "Manufacturing, value added (% of GDP)"),
    "remittances": ("BX.TRF.PWKR.DT.GD.ZS", "Personal remittances received (% of GDP)"),
}

MIN_INDICATORS = 8  # drop territories with almost no data
MIN_INDICATORS_LOADED = 12  # refuse to build a cache if fewer indicators than this could be fetched

# World Bank codes move between databases. The governance (WGI) series are published in the
# main WDI database as GOV_WGI_*.EST, but older code is "RQ.EST" inside source 3. For anything
# listed here, each (code, source) is tried in order until one works; source None = default.
FALLBACKS: dict[str, list[tuple[str, int | None]]] = {
    "reg_quality": [("GOV_WGI_RQ.EST", None), ("RQ.EST", 3)],
    "pol_stability": [("GOV_WGI_PV.EST", None), ("PV.EST", 3)],
    "rule_of_law": [("GOV_WGI_RL.EST", None), ("RL.EST", 3)],
    "gov_effect": [("GOV_WGI_GE.EST", None), ("GE.EST", 3)],
}


class IndicatorNotFound(RuntimeError):
    """The API answered, but says this indicator does not exist (retrying will not help)."""


def data_dir() -> Path:
    """Cache folder. Overridable so tests can point at synthetic data."""
    return Path(os.environ.get("GLOBALLENS_DATA_DIR", Path(__file__).parent / "data"))


# --------------------------------------------------------------------------- #
# Fetching
# --------------------------------------------------------------------------- #
def _get_all(session: requests.Session, url: str, params: dict, retries: int = 4) -> list[dict]:
    """GET every page of a World Bank list endpoint, with retry + backoff."""
    rows: list[dict] = []
    page, pages = 1, 1
    while page <= pages:
        q = {"format": "json", "per_page": 20000, "page": page, **params}
        for attempt in range(retries):
            try:
                r = session.get(url, params=q, headers=HEADERS, timeout=60)
                r.raise_for_status()
                payload = r.json()
                if payload and isinstance(payload[0], dict) and "message" in payload[0]:
                    # e.g. "The indicator was not found" - a definitive answer, don't retry
                    raise IndicatorNotFound(str(payload[0]["message"])[:200])
                if len(payload) < 2 or payload[1] is None:
                    raise ValueError(f"unexpected response: {str(payload)[:200]}")
                break
            except IndicatorNotFound:
                raise
            except (requests.RequestException, ValueError) as exc:
                if attempt == retries - 1:
                    raise RuntimeError(f"World Bank API failed for {url}: {exc}") from exc
                time.sleep(1.5 * (attempt + 1))
        pages = int(payload[0].get("pages", 1))
        rows.extend(payload[1])
        page += 1
    return rows


def _to_float(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def fetch_countries(session: requests.Session) -> pd.DataFrame:
    """Real countries only (the API also lists regional/income aggregates)."""
    rows = _get_all(session, f"{API}/country", {})
    recs = []
    for r in rows:
        if r["region"]["value"] == "Aggregates":
            continue
        recs.append(
            dict(
                iso3=r["id"],
                name=r["name"],
                region=r["region"]["value"],
                income=r["incomeLevel"]["value"],
                lat=_to_float(r.get("latitude")),
                lon=_to_float(r.get("longitude")),
            )
        )
    return pd.DataFrame(recs)


def fetch_indicator(session: requests.Session, short: str, code: str, source: int | None = None) -> pd.DataFrame:
    """Latest non-empty value per country for one indicator."""
    params = {"date": YEAR_RANGE, "mrnev": 1}
    if source is not None:
        params["source"] = source
    rows = _get_all(session, f"{API}/country/all/indicator/{code}", params)
    recs = []
    for r in rows:
        if r.get("value") is None:
            continue
        recs.append(
            dict(
                iso3=r["countryiso3code"],
                indicator=short,
                value=float(r["value"]),
                year=int(r["date"]),
            )
        )
    return pd.DataFrame(recs, columns=["iso3", "indicator", "value", "year"])


def refresh_cache(verbose: bool = True) -> None:
    """Download everything and rewrite the CSV cache."""
    d = data_dir()
    d.mkdir(parents=True, exist_ok=True)
    with requests.Session() as s:
        countries = fetch_countries(s)
        if verbose:
            print(f"countries: {len(countries)}")
        parts, skipped = [], {}
        for short, (code, label) in INDICATORS.items():
            attempts = FALLBACKS.get(short, [(code, None)])
            part, last_err = None, None
            for c, src in attempts:
                try:
                    part = fetch_indicator(s, short, c, src)
                    code = c
                    break
                except RuntimeError as exc:  # IndicatorNotFound or a persistent API failure
                    last_err = exc
            if part is None:
                skipped[short] = str(last_err)[:160]
                if verbose:
                    print(f"  {short:<18} SKIPPED: {str(last_err)[:90]}")
                continue
            part = part[part["iso3"].isin(countries["iso3"])]
            parts.append(part)
            if verbose:
                print(f"  {short:<18} {code:<22} {len(part):>4} countries")
            time.sleep(0.2)
    if len(parts) < MIN_INDICATORS_LOADED:
        raise RuntimeError(
            f"Only {len(parts)} indicators could be fetched (need {MIN_INDICATORS_LOADED}). "
            f"Problems: {skipped}")
    long = pd.concat(parts, ignore_index=True)
    countries.to_csv(d / "countries.csv", index=False)
    long.to_csv(d / "worldbank_long.csv", index=False)
    meta = {
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "year_range": YEAR_RANGE,
        "source": "World Bank Open Data (WDI, WGI, LPI, Global Findex)",
        "skipped_indicators": skipped,
    }
    (d / "meta.json").write_text(json.dumps(meta, indent=2))
    if verbose:
        print(f"saved {len(long)} observations to {d}")


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _read_long() -> pd.DataFrame:
    d = data_dir()
    if not (d / "worldbank_long.csv").exists():
        refresh_cache()
    return pd.read_csv(d / "worldbank_long.csv")


def load_dataset() -> pd.DataFrame:
    """One row per country, indexed by ISO3, with every indicator as a column."""
    d = data_dir()
    long = _read_long()
    countries = pd.read_csv(d / "countries.csv").set_index("iso3")
    wide = long.pivot_table(index="iso3", columns="indicator", values="value", aggfunc="first")
    df = countries.join(wide, how="inner")
    for col in INDICATORS:
        if col not in df.columns:
            df[col] = np.nan
    df["trade_open"] = df["exports_gdp"] + df["imports_gdp"]
    df["n_indicators"] = df[list(INDICATORS)].notna().sum(axis=1)
    df = df[df["n_indicators"] >= MIN_INDICATORS]
    return df.sort_values("name")


def vintage() -> pd.DataFrame:
    """Per-indicator coverage: how many countries, and which years the values are from."""
    long = _read_long()
    g = long.groupby("indicator")
    out = pd.DataFrame(
        {
            "countries": g["iso3"].nunique(),
            "median_year": g["year"].median(),
            "oldest_year": g["year"].min(),
        }
    )
    out["label"] = [INDICATORS[i][1] if i in INDICATORS else i for i in out.index]
    out["wb_code"] = [INDICATORS[i][0] if i in INDICATORS else "" for i in out.index]
    return out.loc[[k for k in INDICATORS if k in out.index]]


def meta() -> dict:
    p = data_dir() / "meta.json"
    return json.loads(p.read_text()) if p.exists() else {}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="GlobalLens data cache")
    ap.add_argument("--refresh", action="store_true", help="re-download from the World Bank")
    ap.add_argument("--info", action="store_true", help="describe the cached data")
    args = ap.parse_args()
    if args.refresh:
        refresh_cache()
    if args.info or not args.refresh:
        df = load_dataset()
        print(f"{len(df)} countries, cache folder: {data_dir()}")
        print(meta())
        print(vintage()[["countries", "median_year", "label"]].to_string())
    sys.exit(0)

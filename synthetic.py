"""Synthetic stand-in for the World Bank data, used only by the tests.

Real ISO3 codes and plausible (but invented) numbers, written in the exact
on-disk format that data.refresh_cache() produces, so the full load path
(CSV -> pivot -> app) is exercised without network access.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import data

# iso3, name, region, income, lat, lon
COUNTRIES = [
    ("USA", "United States", "North America", "High income", 38.9, -77.0),
    ("CAN", "Canada", "North America", "High income", 45.4, -75.7),
    ("MEX", "Mexico", "Latin America & Caribbean", "Upper middle income", 19.4, -99.1),
    ("BRA", "Brazil", "Latin America & Caribbean", "Upper middle income", -15.8, -47.9),
    ("ARG", "Argentina", "Latin America & Caribbean", "Upper middle income", -34.6, -58.4),
    ("CHL", "Chile", "Latin America & Caribbean", "High income", -33.4, -70.7),
    ("COL", "Colombia", "Latin America & Caribbean", "Upper middle income", 4.6, -74.1),
    ("GBR", "United Kingdom", "Europe & Central Asia", "High income", 51.5, -0.1),
    ("DEU", "Germany", "Europe & Central Asia", "High income", 52.5, 13.4),
    ("FRA", "France", "Europe & Central Asia", "High income", 48.9, 2.3),
    ("ITA", "Italy", "Europe & Central Asia", "High income", 41.9, 12.5),
    ("ESP", "Spain", "Europe & Central Asia", "High income", 40.4, -3.7),
    ("NLD", "Netherlands", "Europe & Central Asia", "High income", 52.4, 4.9),
    ("BEL", "Belgium", "Europe & Central Asia", "High income", 50.8, 4.4),
    ("AUT", "Austria", "Europe & Central Asia", "High income", 48.2, 16.4),
    ("IRL", "Ireland", "Europe & Central Asia", "High income", 53.3, -6.3),
    ("POL", "Poland", "Europe & Central Asia", "High income", 52.2, 21.0),
    ("SWE", "Sweden", "Europe & Central Asia", "High income", 59.3, 18.1),
    ("TUR", "Turkiye", "Europe & Central Asia", "Upper middle income", 39.9, 32.9),
    ("RUS", "Russian Federation", "Europe & Central Asia", "High income", 55.8, 37.6),
    ("UKR", "Ukraine", "Europe & Central Asia", "Lower middle income", 50.5, 30.5),
    ("BLR", "Belarus", "Europe & Central Asia", "Upper middle income", 53.9, 27.6),
    ("KAZ", "Kazakhstan", "Europe & Central Asia", "Upper middle income", 51.1, 71.4),
    ("SAU", "Saudi Arabia", "Middle East & North Africa", "High income", 24.7, 46.7),
    ("ARE", "United Arab Emirates", "Middle East & North Africa", "High income", 24.5, 54.4),
    ("IRQ", "Iraq", "Middle East & North Africa", "Upper middle income", 33.3, 44.4),
    ("KWT", "Kuwait", "Middle East & North Africa", "High income", 29.4, 48.0),
    ("QAT", "Qatar", "Middle East & North Africa", "High income", 25.3, 51.5),
    ("IRN", "Iran, Islamic Rep.", "Middle East & North Africa", "Upper middle income", 35.7, 51.4),
    ("OMN", "Oman", "Middle East & North Africa", "High income", 23.6, 58.6),
    ("BHR", "Bahrain", "Middle East & North Africa", "High income", 26.2, 50.6),
    ("EGY", "Egypt, Arab Rep.", "Middle East & North Africa", "Lower middle income", 30.0, 31.2),
    ("MAR", "Morocco", "Middle East & North Africa", "Lower middle income", 34.0, -6.8),
    ("ISR", "Israel", "Middle East & North Africa", "High income", 31.8, 35.2),
    ("PAK", "Pakistan", "South Asia", "Lower middle income", 33.7, 73.1),
    ("IND", "India", "South Asia", "Lower middle income", 28.6, 77.2),
    ("BGD", "Bangladesh", "South Asia", "Lower middle income", 23.8, 90.4),
    ("LKA", "Sri Lanka", "South Asia", "Lower middle income", 6.9, 79.9),
    ("NPL", "Nepal", "South Asia", "Lower middle income", 27.7, 85.3),
    ("CHN", "China", "East Asia & Pacific", "Upper middle income", 39.9, 116.4),
    ("JPN", "Japan", "East Asia & Pacific", "High income", 35.7, 139.7),
    ("KOR", "Korea, Rep.", "East Asia & Pacific", "High income", 37.6, 127.0),
    ("VNM", "Viet Nam", "East Asia & Pacific", "Lower middle income", 21.0, 105.8),
    ("THA", "Thailand", "East Asia & Pacific", "Upper middle income", 13.8, 100.5),
    ("MYS", "Malaysia", "East Asia & Pacific", "Upper middle income", 3.1, 101.7),
    ("IDN", "Indonesia", "East Asia & Pacific", "Upper middle income", -6.2, 106.8),
    ("PHL", "Philippines", "East Asia & Pacific", "Lower middle income", 14.6, 121.0),
    ("SGP", "Singapore", "East Asia & Pacific", "High income", 1.3, 103.8),
    ("AUS", "Australia", "East Asia & Pacific", "High income", -35.3, 149.1),
    ("NZL", "New Zealand", "East Asia & Pacific", "High income", -41.3, 174.8),
    ("NGA", "Nigeria", "Sub-Saharan Africa", "Lower middle income", 9.1, 7.5),
    ("ZAF", "South Africa", "Sub-Saharan Africa", "Upper middle income", -25.7, 28.2),
    ("KEN", "Kenya", "Sub-Saharan Africa", "Lower middle income", -1.3, 36.8),
    ("ETH", "Ethiopia", "Sub-Saharan Africa", "Low income", 9.0, 38.7),
    ("GHA", "Ghana", "Sub-Saharan Africa", "Lower middle income", 5.6, -0.2),
    ("TZA", "Tanzania", "Sub-Saharan Africa", "Lower middle income", -6.2, 35.7),
    ("SEN", "Senegal", "Sub-Saharan Africa", "Lower middle income", 14.7, -17.5),
    ("CIV", "Cote d'Ivoire", "Sub-Saharan Africa", "Lower middle income", 6.8, -5.3),
    ("AGO", "Angola", "Sub-Saharan Africa", "Lower middle income", -8.8, 13.2),
    ("UGA", "Uganda", "Sub-Saharan Africa", "Low income", 0.3, 32.6),
]


def make_dataset(seed: int = 0) -> pd.DataFrame:
    """Wide frame with the same columns load_dataset() returns."""
    rng = np.random.default_rng(seed)
    rows = []
    for iso, name, region, income, lat, lon in COUNTRIES:
        rich = {"High income": 3, "Upper middle income": 2, "Lower middle income": 1, "Low income": 0}[income]
        pop = float(10 ** rng.uniform(6.8, 8.9))
        gdp_pc = float(10 ** (2.9 + 0.45 * rich + rng.normal(0, 0.15)))
        gdp = pop * gdp_pc
        exp_share = float(rng.uniform(15, 70))
        imp_share = float(np.clip(exp_share + rng.normal(0, 6), 8, 90))
        rows.append(
            dict(
                iso3=iso, name=name, region=region, income=income, lat=lat, lon=lon,
                gdp=gdp, gdp_pc=gdp_pc, pop=pop,
                gdp_growth=rng.normal(3.5, 2.5), pop_growth=rng.normal(1.2, 0.9),
                working_age=rng.uniform(55, 70), urban=rng.uniform(25, 92),
                internet=np.clip(20 + 20 * rich + rng.normal(0, 8), 5, 99),
                mobile_subs=np.clip(70 + 15 * rich + rng.normal(0, 15), 40, 180),
                account_ownership=np.clip(25 + 22 * rich + rng.normal(0, 10), 5, 99),
                exports_gdp=exp_share, imports_gdp=imp_share,
                exports_usd=gdp * exp_share / 100, imports_usd=gdp * imp_share / 100,
                fdi_gdp=rng.normal(2.5, 3), inflation=np.abs(rng.normal(5, 6)),
                lpi=np.clip(2.2 + 0.4 * rich + rng.normal(0, 0.25), 1.5, 4.4),
                reg_quality=-0.9 + 0.7 * rich + rng.normal(0, 0.3),
                pol_stability=-0.8 + 0.5 * rich + rng.normal(0, 0.4),
                rule_of_law=-0.9 + 0.7 * rich + rng.normal(0, 0.3),
                gov_effect=-0.9 + 0.7 * rich + rng.normal(0, 0.3),
                fuel_exports=float(rng.uniform(0, 8)),
                manuf_gdp=rng.uniform(5, 28), remittances=rng.uniform(0, 8),
            )
        )
    df = pd.DataFrame(rows).set_index("iso3")
    # a few oil exporters, so the "energy" scope has something to bite on
    for c in ["SAU", "ARE", "IRQ", "KWT", "QAT", "IRN", "OMN", "BHR", "RUS", "NGA", "AGO"]:
        df.loc[c, "fuel_exports"] = 75.0
    df["trade_open"] = df["exports_gdp"] + df["imports_gdp"]
    df["n_indicators"] = df[list(data.INDICATORS)].notna().sum(axis=1)
    return df


def write_cache(folder: Path, seed: int = 0, drop_some: bool = True) -> pd.DataFrame:
    """Write the synthetic frame in the on-disk format of data.refresh_cache()."""
    df = make_dataset(seed)
    folder.mkdir(parents=True, exist_ok=True)
    df[["name", "region", "income", "lat", "lon"]].reset_index().to_csv(folder / "countries.csv", index=False)
    rng = np.random.default_rng(seed + 1)
    recs = []
    for iso, row in df.iterrows():
        for short in data.INDICATORS:
            # sparse, like the real thing: Findex/LPI are missing for some countries
            if drop_some and short in ("lpi", "account_ownership") and rng.random() < 0.15:
                continue
            recs.append((iso, short, float(row[short]), int(rng.choice([2021, 2022, 2023]))))
    pd.DataFrame(recs, columns=["iso3", "indicator", "value", "year"]).to_csv(folder / "worldbank_long.csv", index=False)
    (folder / "meta.json").write_text(json.dumps({"fetched_at": "synthetic", "source": "synthetic test data"}))
    return df

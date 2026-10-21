"""Supply-chain shock simulator.

Open bilateral trade data sits behind API keys and rate limits, so GlobalLens
*estimates* a bilateral trade matrix with a gravity model and then calibrates it
to the countries' real reported totals:

    T[i, j]  ~  (GDP_i * GDP_j) / distance_ij  * regional_bloc_boost
    then iterative proportional fitting (RAS) scales rows so every country's
    modelled exports equal its reported exports, and columns so every country's
    modelled imports equal its reported imports.

A shock (some countries' exports fall by x%) is then pushed through the matrix
along three channels, each expressed as "trade at risk, % of the country's GDP":

    supply   : imports that the shocked countries no longer deliver, reduced by how
               well the importer can substitute (income + logistics quality)
    demand   : exports lost because the shocked countries' economies contract and
               they import less
    own      : the shocked countries' own lost export revenue

This is a stylised exposure model, not a GDP forecast. See the Method tab.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

EARTH_R_KM = 6371.0

SCENARIOS: dict[str, dict] = {
    "Gulf energy shock": {
        "countries": ["SAU", "ARE", "IRQ", "KWT", "QAT", "IRN", "OMN", "BHR"],
        "drop": 30,
        "scope": "energy",
    },
    "China export slump": {"countries": ["CHN"], "drop": 20, "scope": "all"},
    "Russia-Ukraine style disruption": {"countries": ["RUS", "UKR", "BLR"], "drop": 50, "scope": "all"},
    "Eurozone demand shock": {
        "countries": ["DEU", "FRA", "ITA", "ESP", "NLD", "BEL", "AUT", "IRL"],
        "drop": 15,
        "scope": "all",
    },
    "Custom": {"countries": [], "drop": 25, "scope": "all"},
}


def haversine_matrix(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Great-circle distance in km between every pair of points."""
    la, lo = np.radians(lat)[:, None], np.radians(lon)[:, None]
    a = np.sin((la - la.T) / 2) ** 2 + np.cos(la) * np.cos(la.T) * np.sin((lo - lo.T) / 2) ** 2
    return 2 * EARTH_R_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


@dataclass
class TradeModel:
    iso3: list[str]
    T: np.ndarray  # T[i, j] = modelled exports of i to j, US$
    max_row_error: float  # worst relative gap vs reported exports
    max_col_error: float  # worst relative gap vs (rescaled) reported imports
    iterations: int


def build_trade_model(
    df: pd.DataFrame,
    region_boost: float = 1.5,
    dist_exp: float = 1.0,
    gdp_exp: float = 1.0,
    floor_km: float = 300.0,
    iters: int = 1000,
    tol: float = 1e-9,
) -> TradeModel:
    need = ["gdp", "exports_usd", "imports_usd", "lat", "lon"]
    d = df.dropna(subset=need)
    d = d[(d["exports_usd"] > 0) & (d["imports_usd"] > 0) & (d["gdp"] > 0)]
    iso = list(d.index)
    X = d["exports_usd"].to_numpy(float)
    M = d["imports_usd"].to_numpy(float)
    M = M * X.sum() / M.sum()  # world exports must equal world imports
    G = d["gdp"].to_numpy(float)

    D = np.maximum(haversine_matrix(d["lat"].to_numpy(float), d["lon"].to_numpy(float)), floor_km)
    same_region = d["region"].to_numpy()[:, None] == d["region"].to_numpy()[None, :]
    K = (G[:, None] ** gdp_exp) * (G[None, :] ** gdp_exp) / D**dist_exp
    K = K * np.where(same_region, region_boost, 1.0)
    np.fill_diagonal(K, 0.0)
    K = K / K.max()  # keep numbers well-scaled for the iteration

    a, b = np.ones(len(X)), np.ones(len(X))
    n_it = 0
    for n_it in range(1, iters + 1):
        a = X / (K @ b)
        b_new = M / (K.T @ a)
        done = np.max(np.abs(b_new / b - 1)) < tol
        b = b_new
        if done:
            break
    a = X / (K @ b)
    T = a[:, None] * K * b[None, :]
    row_err = float(np.max(np.abs(T.sum(1) / X - 1)))
    col_err = float(np.max(np.abs(T.sum(0) / M - 1)))
    return TradeModel(iso, T, row_err, col_err, n_it)


def substitutability(df: pd.DataFrame) -> pd.Series:
    """0-1: how easily a country can find replacement suppliers (income + logistics)."""
    inc = np.log10(df["gdp_pc"].where(df["gdp_pc"] > 0)).rank(pct=True)
    lpi = df["lpi"].rank(pct=True)
    s = pd.concat([inc, lpi], axis=1).mean(axis=1, skipna=True)
    return s.fillna(s.median())


def simulate_shock(
    df: pd.DataFrame,
    model: TradeModel,
    shocked: list[str],
    drop_pct: float,
    scope: str = "all",
    demand_spillover: bool = True,
    substitution: float = 0.5,
    gdp_elasticity: float = 0.5,
) -> pd.DataFrame:
    """Run one scenario. Returns a frame indexed by ISO3 of every modelled country."""
    iso = model.iso3
    n = len(iso)
    d = df.loc[iso]
    idx = {c: i for i, c in enumerate(iso)}
    S = [idx[c] for c in shocked if c in idx]
    T, gdp = model.T, d["gdp"].to_numpy(float)

    drop = np.zeros(n)
    base = np.clip(drop_pct, 0, 100) / 100.0
    for i in S:
        factor = 1.0
        if scope == "energy":
            factor = float(np.nan_to_num(d["fuel_exports"].iloc[i], nan=0.0)) / 100.0
        drop[i] = base * np.clip(factor, 0, 1)

    sub = substitutability(d).to_numpy(float)
    resilience = 1.0 - np.clip(substitution, 0, 1) * sub  # share of lost imports NOT replaced

    supply_c = drop[:, None] * T  # [i, j]: j loses imports from shocked i
    supply_c = supply_c * resilience[None, :]
    supply_loss = supply_c.sum(0)

    own_loss = drop * d["exports_usd"].to_numpy(float)

    if demand_spillover and S:
        gdp_hit = np.clip(gdp_elasticity, 0, 2) * drop * (d["exports_gdp"].fillna(0).to_numpy(float) / 100.0)
        import_cut = np.clip(1.5 * gdp_hit, 0, 0.9)  # imports are ~1.5x as cyclical as GDP
        demand_c = T * import_cut[None, :]  # [j, i]: j loses exports to shocked i
        demand_loss = demand_c.sum(1)
    else:
        demand_c = np.zeros_like(T)
        demand_loss = np.zeros(n)

    out = pd.DataFrame(index=pd.Index(iso, name="iso3"))
    out["supply_pct_gdp"] = supply_loss / gdp * 100
    out["demand_pct_gdp"] = demand_loss / gdp * 100
    out["own_pct_gdp"] = own_loss / gdp * 100
    out["trade_at_risk_pct_gdp"] = out[["supply_pct_gdp", "demand_pct_gdp", "own_pct_gdp"]].sum(axis=1)
    # supply and own describe the same flows from the two ends of a trade, so only
    # `own_usd` (exports actually removed) is safe to add up across countries.
    out["own_usd"] = own_loss
    out["supply_usd"] = supply_loss
    out["demand_usd"] = demand_loss
    out["trade_at_risk_usd"] = supply_loss + demand_loss + own_loss
    out["shocked"] = [c in shocked for c in iso]
    out["vulnerability"] = out["trade_at_risk_pct_gdp"].rank(pct=True) * 100
    out["tier"] = pd.cut(
        out["trade_at_risk_pct_gdp"],
        [-np.inf, 0.5, 2, 5, np.inf],
        labels=["Low", "Moderate", "High", "Severe"],
    ).astype(str)

    if S:
        combined = supply_c[S, :] + demand_c[:, S].T  # [shocked, importer/exporter]
        driver = np.array(iso)[np.array(S)][combined.argmax(axis=0)]
        names = d["name"].to_dict()
        out["main_driver"] = [names.get(c, c) if combined[:, k].max() > 0 else "-" for k, c in enumerate(driver)]
    else:
        out["main_driver"] = "-"
    return out

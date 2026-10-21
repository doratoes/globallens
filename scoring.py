"""Market-entry scoring: indicators -> dimensions -> weighted opportunity score.

Pipeline
    1. Each indicator is transformed (log / inflation distance), oriented so that
       higher = better, and converted to a 0-100 percentile rank across countries.
       Percentiles make wildly different units comparable and are robust to
       outliers (one huge economy cannot squash everyone else to zero).
    2. Indicators are averaged into 7 business dimensions.
    3. The user's dimension weights combine them into one opportunity score.
       Missing dimensions are skipped and the weights re-normalised, and countries
       with too little weighted data coverage are not scored at all.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# dimension -> [(column, direction, transform)]; direction -1 means "lower is better"
DIMENSIONS: dict[str, list[tuple[str, int, str]]] = {
    "Market size": [("gdp", 1, "log"), ("pop", 1, "log")],
    "Growth": [("gdp_growth", 1, "none"), ("pop_growth", 1, "none"), ("working_age", 1, "none")],
    "Purchasing power": [("gdp_pc", 1, "log")],
    "Digital & financial access": [
        ("internet", 1, "none"),
        ("mobile_subs", 1, "none"),
        ("account_ownership", 1, "none"),
    ],
    "Business environment": [
        ("reg_quality", 1, "none"),
        ("rule_of_law", 1, "none"),
        ("gov_effect", 1, "none"),
        ("fdi_gdp", 1, "none"),
    ],
    "Stability": [("pol_stability", 1, "none"), ("inflation", -1, "inflation")],
    "Logistics & trade": [("lpi", 1, "none"), ("trade_open", 1, "none")],
}
DIM_NAMES = list(DIMENSIONS)

# 0-10 importance per dimension, in DIM_NAMES order
PRESETS: dict[str, list[int]] = {
    "Balanced": [5, 5, 5, 5, 5, 5, 5],
    "Fintech / payments": [4, 6, 4, 10, 7, 6, 1],
    "E-commerce / retail": [8, 6, 8, 9, 5, 5, 8],
    "Consumer goods (FMCG)": [10, 7, 6, 4, 5, 5, 6],
    "Manufacturing / sourcing": [4, 5, 0, 2, 7, 7, 10],
    "SaaS / B2B software": [5, 6, 8, 9, 8, 6, 1],
}


def preset_weights(name: str) -> dict[str, float]:
    return dict(zip(DIM_NAMES, PRESETS[name]))


def _transform(s: pd.Series, how: str) -> pd.Series:
    if how == "log":
        return np.log10(s.where(s > 0))
    if how == "inflation":
        # distance from a ~2% target, compressed: 2% -> 0, 10% -> ~0.95, 100% -> ~2
        return np.log10(1 + (s - 2).abs())
    return s


def indicator_percentiles(df: pd.DataFrame) -> pd.DataFrame:
    """0-100 percentile per indicator, oriented so higher is always better."""
    out = {}
    for specs in DIMENSIONS.values():
        for col, direction, how in specs:
            v = _transform(df[col].astype(float), how) * direction
            out[col] = v.rank(pct=True) * 100
    return pd.DataFrame(out, index=df.index)


def dimension_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Mean percentile of the available indicators in each dimension (NaN if none)."""
    pct = indicator_percentiles(df)
    cols = {}
    for dim, specs in DIMENSIONS.items():
        cols[dim] = pct[[c for c, _, _ in specs]].mean(axis=1, skipna=True)
    return pd.DataFrame(cols, index=df.index)


def opportunity(dim: pd.DataFrame, weights: dict[str, float], min_coverage: float = 0.6) -> pd.DataFrame:
    """Weighted opportunity score (0-100), coverage and rank.

    coverage = share of the total weight that sits on dimensions with data.
    """
    w = np.array([max(float(weights.get(d, 0)), 0.0) for d in dim.columns])
    if w.sum() == 0:
        w = np.ones_like(w)
    mask = dim.notna().to_numpy()
    num = np.where(mask, dim.to_numpy(dtype=float), 0.0) @ w
    den = mask @ w
    with np.errstate(invalid="ignore", divide="ignore"):
        score = np.where(den > 0, num / den, np.nan)
    coverage = den / w.sum()
    score = np.where(coverage >= min_coverage, score, np.nan)
    out = pd.DataFrame({"score": score, "coverage": coverage}, index=dim.index)
    out["rank"] = out["score"].rank(ascending=False, method="min")
    return out


def weight_sensitivity(
    dim: pd.DataFrame,
    weights: dict[str, float],
    top_k: int = 10,
    runs: int = 500,
    concentration: float = 20.0,
    min_coverage: float = 0.6,
    seed: int = 0,
) -> pd.DataFrame:
    """How stable is the ranking if the weights are only roughly right?

    Draws weight vectors from a Dirichlet distribution centred on the user's
    weights (higher `concentration` = tighter around them), re-scores every
    country each time and reports how often each lands in the top_k.
    """
    base = opportunity(dim, weights, min_coverage)
    eligible = base.index[base["score"].notna()]
    D = dim.loc[eligible]
    mask = D.notna().to_numpy()
    X = np.where(mask, D.to_numpy(dtype=float), 0.0)

    w = np.array([max(float(weights.get(d, 0)), 0.0) for d in D.columns])
    w = w / w.sum() if w.sum() > 0 else np.ones_like(w) / len(w)
    rng = np.random.default_rng(seed)
    W = rng.dirichlet(concentration * w + 0.05, size=runs).T  # (dims, runs)

    num = X @ W
    den = mask @ W
    S = np.where(den > 0, num / np.where(den > 0, den, 1), -np.inf)
    ranks = (-S).argsort(axis=0).argsort(axis=0) + 1  # 1 = best, per run
    k = min(top_k, len(eligible))
    out = pd.DataFrame(
        {
            "in_top_pct": (ranks <= k).mean(axis=1) * 100,
            "rank_p10": np.percentile(ranks, 10, axis=1),
            "rank_median": np.median(ranks, axis=1),
            "rank_p90": np.percentile(ranks, 90, axis=1),
        },
        index=eligible,
    )
    return out


# --------------------------------------------------------------------------- #
# Clustering
# --------------------------------------------------------------------------- #
def cluster_markets(dim: pd.DataFrame, k: int = 5, seed: int = 42) -> pd.DataFrame:
    """KMeans on the 7 dimension scores + a 2-D PCA projection for plotting."""
    ok = dim.notna().sum(axis=1) >= 5
    D = dim.loc[ok]
    X = D.fillna(D.median()).to_numpy(dtype=float)
    Z = StandardScaler().fit_transform(X)
    k = int(max(2, min(k, len(D) - 1)))
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(Z)
    pcs = PCA(n_components=2, random_state=seed).fit_transform(Z)
    out = pd.DataFrame({"cluster": km.labels_, "pc1": pcs[:, 0], "pc2": pcs[:, 1]}, index=D.index)
    labels = _name_clusters(D, out["cluster"])
    out["cluster_label"] = out["cluster"].map(labels)
    return out


def _name_clusters(D: pd.DataFrame, cluster: pd.Series) -> dict[int, str]:
    """Name each cluster after the dimensions where it stands out from the average."""
    overall = D.mean()
    names: dict[int, str] = {}
    used: set[str] = set()
    for c in sorted(cluster.unique()):
        diff = (D[cluster == c].mean() - overall).sort_values(ascending=False)
        top = [d for d, v in diff.items() if v > 4][:2]
        if top:
            name = " + ".join(top)
        elif diff.max() < 0:
            name = "Early-stage markets"
        else:
            name = "Mid-profile markets"
        if D[cluster == c].mean().mean() < overall.mean() - 12:
            name = "Developing: " + name if top else name
        base, i = name, 2
        while name in used:
            name, i = f"{base} ({i})", i + 1
        used.add(name)
        names[c] = name
    return names


# --------------------------------------------------------------------------- #
# Risk adjustment + explanations
# --------------------------------------------------------------------------- #
def risk_adjusted(score: pd.Series, vulnerability: pd.Series, risk_aversion: float) -> pd.Series:
    """Opportunity discounted by exposure: score * (1 - aversion * vulnerability/100)."""
    v = vulnerability.reindex(score.index).fillna(vulnerability.median())
    return score * (1 - float(risk_aversion) * v / 100.0)


def strengths(dim_row: pd.Series, weights: dict[str, float], n: int = 2) -> tuple[list[str], list[str]]:
    """Top-n dimensions that drive a country's score, and its weakest important ones."""
    w = pd.Series({d: weights.get(d, 0) for d in dim_row.index})
    contrib = (dim_row * w).dropna().sort_values(ascending=False)
    top = list(contrib.index[:n])
    weak = [d for d in (dim_row[w > 0].dropna().sort_values().index) if dim_row[d] < 40][:n]
    return top, weak

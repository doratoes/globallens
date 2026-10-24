import numpy as np
import pandas as pd
import pytest

import scoring


@pytest.fixture()
def dim(df):
    return scoring.dimension_scores(df)


def test_percentiles_bounded_and_oriented(df):
    pct = scoring.indicator_percentiles(df)
    assert pct.min().min() >= 0 and pct.max().max() <= 100
    # higher GDP -> higher percentile; higher inflation distance -> LOWER percentile
    big, small = df["gdp"].idxmax(), df["gdp"].idxmin()
    assert pct.loc[big, "gdp"] > pct.loc[small, "gdp"]
    worst_infl = (df["inflation"] - 2).abs().idxmax()
    assert pct.loc[worst_infl, "inflation"] < 20


def test_dimension_scores_cover_all_dimensions(dim):
    assert list(dim.columns) == scoring.DIM_NAMES
    assert dim.notna().mean().min() > 0.9


def test_weights_steer_the_ranking(dim):
    size_only = scoring.opportunity(dim, {d: (10 if d == "Market size" else 0) for d in dim.columns})
    pc_only = scoring.opportunity(dim, {d: (10 if d == "Purchasing power" else 0) for d in dim.columns})
    assert size_only["score"].corr(dim["Market size"]) > 0.999
    assert pc_only["score"].corr(dim["Purchasing power"]) > 0.999
    assert size_only["score"].between(0, 100).all()
    # different weights really do produce different orderings
    assert not size_only["rank"].equals(pc_only["rank"])


def test_zero_weights_fall_back_to_equal_not_nan(dim):
    out = scoring.opportunity(dim, {d: 0 for d in dim.columns})
    assert out["score"].notna().all()


def test_missing_dimension_renormalises_and_low_coverage_is_dropped(dim):
    d = dim.copy()
    d.iloc[0, :4] = np.nan          # lose 4 of 7 dims on the first country
    d.iloc[1, 0] = np.nan           # lose just one on the second
    w = {c: 5 for c in d.columns}
    out = scoring.opportunity(d, w, min_coverage=0.6)
    assert np.isnan(out.iloc[0]["score"])             # 3/7 coverage < 0.6 -> unscored
    assert out.iloc[1]["coverage"] == pytest.approx(6 / 7)
    assert out.iloc[1]["score"] == pytest.approx(d.iloc[1].mean())   # equal weights -> mean of the rest


def test_ranks_are_consistent_with_scores(dim):
    out = scoring.opportunity(dim, scoring.preset_weights("Balanced")).dropna(subset=["score"])
    assert out["rank"].min() == 1
    assert out.sort_values("score", ascending=False)["rank"].is_monotonic_increasing


def test_every_preset_has_a_weight_per_dimension():
    for name, w in scoring.PRESETS.items():
        assert len(w) == len(scoring.DIM_NAMES), name
        assert set(scoring.preset_weights(name)) == set(scoring.DIM_NAMES)


def test_sensitivity_is_deterministic_and_sane(dim):
    w = scoring.preset_weights("Fintech / payments")
    a = scoring.weight_sensitivity(dim, w, top_k=10, runs=200, seed=3)
    b = scoring.weight_sensitivity(dim, w, top_k=10, runs=200, seed=3)
    pd.testing.assert_frame_equal(a, b)
    assert a["in_top_pct"].between(0, 100).all()
    # on average exactly top_k countries are in the top-k per run
    assert a["in_top_pct"].sum() / 100 == pytest.approx(10, abs=1e-6)
    # the base-weight winner should be robustly near the top
    base = scoring.opportunity(dim, w)
    winner = base["score"].idxmax()
    assert a.loc[winner, "in_top_pct"] > 80
    assert (a["rank_p10"] <= a["rank_median"]).all() and (a["rank_median"] <= a["rank_p90"]).all()


def test_tighter_concentration_means_more_stable_top_ten(dim):
    w = scoring.preset_weights("Balanced")
    loose = scoring.weight_sensitivity(dim, w, concentration=2, runs=300, seed=1)
    tight = scoring.weight_sensitivity(dim, w, concentration=200, runs=300, seed=1)
    spread = lambda s: ((s["in_top_pct"] > 5) & (s["in_top_pct"] < 95)).sum()
    assert spread(tight) <= spread(loose)


def test_clusters_partition_and_are_named(dim):
    out = scoring.cluster_markets(dim, k=4)
    assert out["cluster"].nunique() == 4
    assert out["cluster_label"].notna().all()
    assert out.groupby("cluster")["cluster_label"].nunique().max() == 1      # one name per cluster
    assert out["cluster_label"].nunique() == 4                                # names are unique
    again = scoring.cluster_markets(dim, k=4)
    pd.testing.assert_frame_equal(out, again)                                 # reproducible


def test_cluster_count_is_clamped_for_tiny_inputs(dim):
    out = scoring.cluster_markets(dim.head(3), k=6)
    assert out["cluster"].nunique() <= 2


def test_risk_adjustment_math():
    score = pd.Series({"A": 80.0, "B": 80.0, "C": 60.0})
    vul = pd.Series({"A": 100.0, "B": 0.0, "C": 50.0})
    adj = scoring.risk_adjusted(score, vul, 0.5)
    assert adj["A"] == pytest.approx(40.0) and adj["B"] == pytest.approx(80.0) and adj["C"] == pytest.approx(45.0)
    assert scoring.risk_adjusted(score, vul, 0).equals(score)
    # a country with no exposure figure gets the median penalty rather than disappearing
    assert not np.isnan(scoring.risk_adjusted(score, vul.drop("C"), 0.5)["C"])


def test_strengths_picks_weighted_top_and_weak(dim):
    row = pd.Series({d: 50.0 for d in scoring.DIM_NAMES})
    row["Growth"], row["Stability"] = 95.0, 10.0
    w = {d: 5 for d in scoring.DIM_NAMES}
    top, weak = scoring.strengths(row, w, n=1)
    assert top == ["Growth"] and weak == ["Stability"]
    # zero-weight dimension never counts as a strength
    w["Growth"] = 0
    assert "Growth" not in scoring.strengths(row, w, n=2)[0]

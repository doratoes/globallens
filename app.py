"""GlobalLens - where to expand, and what breaks if trade breaks.

Two engines on one World Bank dataset:
  * Shock simulator    (shocks.py)  - who is exposed if a country's exports collapse
  * Market recommender (scoring.py) - which markets fit a given business, with
                                      weight-robustness and clustering
and a combined, risk-adjusted view that joins them.

Run locally:   streamlit run app.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import data
import scoring
import shocks

st.set_page_config(page_title="GlobalLens", page_icon=":globe_with_meridians:", layout="wide")

# --------------------------------------------------------------------------- #
# Visual system. Categorical colours are the first three slots of a palette
# validated for colour-blind safety on all pairs (blue / orange / aqua); single-hue
# ramps carry magnitude; marker shapes carry cluster identity. Charts sit on an
# opaque light surface so they stay legible whatever theme the viewer picks.
# --------------------------------------------------------------------------- #
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
FONT = "system-ui, -apple-system, 'Segoe UI', sans-serif"
BLUE_SEQ = [[0, "#cde2fb"], [0.17, "#9ec5f4"], [0.33, "#5598e7"], [0.5, "#2a78d6"],
            [0.67, "#1c5cab"], [0.83, "#104281"], [1, "#0d366b"]]
ORANGE_SEQ = [[0, "#fde3d6"], [0.25, "#f7b494"], [0.5, "#eb6834"], [0.75, "#b94a1f"], [1, "#7d2f10"]]
CLUSTER_SYMBOLS = ["circle", "square", "diamond", "triangle-up", "x", "cross"]
TIER_ORDER = ["Low", "Moderate", "High", "Severe"]
INCOME_ORDER = ["High income", "Upper middle income", "Lower middle income", "Low income"]


def style(fig: go.Figure, title: str | None = None, height: int = 420) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, x=0, font=dict(color=INK, size=15)) if title else None,
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, height=height,
        font=dict(family=FONT, color=INK2, size=13),
        margin=dict(l=12, r=16, t=88 if title else 24, b=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(color=INK2)),
        hoverlabel=dict(bgcolor=SURFACE, bordercolor=AXIS, font=dict(color=INK, family=FONT)),
    )
    # automargin lets Plotly grow the margin for long tick labels (country names) and axis titles
    for axis in (fig.update_xaxes, fig.update_yaxes):
        axis(gridcolor=GRID, linecolor=AXIS, zerolinecolor=AXIS, tickfont=dict(color=MUTED),
             title_font=dict(color=INK2), automargin=True, title_standoff=12)
    return fig


def show(fig: go.Figure) -> None:
    st.plotly_chart(fig, width="stretch", theme=None, config={"displayModeBar": False})


def table(frame: pd.DataFrame, **kw) -> None:
    st.dataframe(frame, width="stretch", hide_index=True, **kw)


def world_map(frame: pd.DataFrame, col: str, scale, cbar: str, label: str, fmt: str = ".1f",
              zmax: float | None = None, outline: list[str] | None = None, height: int = 430) -> go.Figure:
    zmax = float(zmax if zmax is not None else np.nanmax(frame[col]))
    fig = go.Figure(go.Choropleth(
        locations=frame.index, z=frame[col], text=frame["name"], colorscale=scale,
        zmin=0, zmax=max(zmax, 1e-9), marker_line_color=SURFACE, marker_line_width=0.5,
        colorbar=dict(title=dict(text=cbar, font=dict(color=INK2, size=12)), thickness=12, len=0.7,
                      tickfont=dict(color=MUTED), outlinewidth=0),
        hovertemplate=f"<b>%{{text}}</b><br>{label}: %{{z:{fmt}}}<extra></extra>",
    ))
    if outline:
        fig.add_trace(go.Choropleth(
            locations=outline, z=[0] * len(outline), showscale=False, hoverinfo="skip",
            colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,0,0,0)"]],
            marker_line_color=INK, marker_line_width=1.6))
    fig.update_geos(showframe=False, showcoastlines=False, showland=True, landcolor="#e6e5df",
                    showocean=True, oceancolor=SURFACE, showlakes=False, bgcolor=SURFACE,
                    projection_type="natural earth")
    fig.update_layout(paper_bgcolor=SURFACE, height=height, margin=dict(l=0, r=0, t=8, b=0),
                      font=dict(family=FONT), hoverlabel=dict(bgcolor=SURFACE, font=dict(color=INK)))
    return fig


def usd(x: float) -> str:
    x = float(x)
    if abs(x) >= 1e12:
        return f"${x / 1e12:,.1f}tn"
    if abs(x) >= 1e9:
        return f"${x / 1e9:,.0f}bn"
    return f"${x / 1e6:,.0f}m"


# --------------------------------------------------------------------------- #
# Data (cached)
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Loading World Bank data...")
def load_data() -> pd.DataFrame:
    return data.load_dataset()


@st.cache_data
def get_dims() -> pd.DataFrame:
    return scoring.dimension_scores(load_data())


@st.cache_resource(show_spinner="Calibrating the trade model...")
def get_model() -> shocks.TradeModel:
    return shocks.build_trade_model(load_data())


try:
    df = load_data()
except Exception as exc:  # no cache and no network
    st.error(
        "No cached data yet, and the World Bank API could not be reached.\n\n"
        "Run `python data.py --refresh` once, commit the `data/` folder, and reload.\n\n"
        f"Details: {exc}"
    )
    st.stop()

dim, model = get_dims(), get_model()
names = df["name"]
model_set = set(model.iso3)

# --------------------------------------------------------------------------- #
# Sidebar + header
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("GlobalLens")
    st.caption("Market-entry and trade-shock analytics on World Bank open data.")
    home_options = sorted(names)
    home_name = st.selectbox("Your home country", home_options,
                             index=home_options.index("Pakistan") if "Pakistan" in home_options else 0)
    home = names[names == home_name].index[0]
    st.divider()
    m = data.meta()
    st.caption(f"{len(df)} countries, {len(data.INDICATORS)} indicators.  \n"
               f"Data pulled: {m.get('fetched_at', 'n/a')[:10]}")

st.title("GlobalLens")
st.caption("Where should a business expand next, and which economies break first when trade does?")
T_OVER, T_SHOCK, T_MKT, T_COMB, T_METH = st.tabs(
    ["Overview", "Shock simulator", "Market recommender", "Risk-adjusted view", "Method & data"])

# =========================================================================== #
# SHOCK SIMULATOR
# =========================================================================== #
with T_SHOCK:
    st.subheader("Supply-chain shock simulator")
    st.caption("Pick a scenario, then see how much trade each economy has at risk, as a share of its GDP.")
    metric_row = st.container()  # filled once the scenario below has been run
    left, right = st.columns([1, 2.3], gap="large")
    opts = sorted(model.iso3, key=lambda c: names[c])
    with left:
        preset = st.selectbox("Scenario", list(shocks.SCENARIOS), key="sc_preset")
        sc = shocks.SCENARIOS[preset]
        picked = st.multiselect("Countries whose exports are hit", opts,
                                default=[c for c in sc["countries"] if c in model_set],
                                format_func=lambda c: names[c], key=f"sc_c_{preset}")
        drop = st.slider("Fall in their exports (%)", 5, 100, int(sc["drop"]), 5, key=f"sc_d_{preset}")
        scope = st.radio("Which exports?", ["all", "energy"], horizontal=True,
                         index=0 if sc["scope"] == "all" else 1,
                         format_func={"all": "All exports", "energy": "Energy only"}.get,
                         key=f"sc_s_{preset}",
                         help="Energy-only scales each country's drop by its fuel share of merchandise exports.")
        with st.expander("Model assumptions"):
            spill = st.toggle("Include demand spillover", True,
                              help="Shocked economies shrink, so they import less from their partners.")
            subst = st.slider("How well importers can substitute", 0.0, 1.0, 0.5, 0.05,
                              help="0 = lost imports are never replaced; 1 = richer, better-connected "
                                   "importers fully replace them.")
            elast = st.slider("GDP sensitivity to export loss", 0.1, 1.0, 0.5, 0.05)

    res = shocks.simulate_shock(df, model, picked, drop, scope, spill, subst, elast).join(df[["name", "region"]])
    res["main_driver"] = np.where(res["shocked"], "Direct hit", res["main_driver"])

    if picked:
        with metric_row:
            world_x = df.loc[model.iso3, "exports_usd"].sum()
            removed = res["own_usd"].sum()
            others = res[~res["shocked"]]
            worst = others["trade_at_risk_pct_gdp"].idxmax()
            c = st.columns(4)
            c[0].metric("Exports removed", usd(removed), help=f"{removed / world_x * 100:.1f}% of world exports")
            c[1].metric("Bystanders at high risk", int(others["tier"].isin(["High", "Severe"]).sum()),
                        help="Countries not hit directly whose trade at risk is above 2% of GDP.")
            c[2].metric("Most exposed bystander", names[worst],
                        f"{others.loc[worst, 'trade_at_risk_pct_gdp']:.1f}% of GDP", delta_color="off")
            if home in res.index:
                c[3].metric(f"{home_name}: trade at risk", f"{res.loc[home, 'trade_at_risk_pct_gdp']:.2f}% of GDP",
                            f"{res.loc[home, 'tier']} tier", delta_color="off")
            else:
                c[3].metric(home_name, "n/a", help="Not enough data to model this country.")

    with right:
        if not picked:
            st.info("Select at least one country to hit.")
        else:
            cap = float(np.nanpercentile(res["trade_at_risk_pct_gdp"], 95))
            show(world_map(res, "trade_at_risk_pct_gdp", ORANGE_SEQ, "% of GDP", "Trade at risk", ".2f",
                           zmax=cap, outline=picked))
            st.caption("Darker = more trade at risk relative to the economy (scale capped at the 95th "
                       "percentile). Outlined = hit directly. Grey = no data.")

    if picked:
        a, b = st.columns([1.15, 1], gap="large")
        with a:
            top = res.sort_values("trade_at_risk_pct_gdp", ascending=False).head(12)
            fig = go.Figure()
            for col, label, colour in [("supply_pct_gdp", "Imports lost", BLUE),
                                       ("demand_pct_gdp", "Exports lost", ORANGE),
                                       ("own_pct_gdp", "Own exports", AQUA)]:
                fig.add_bar(y=top["name"], x=top[col], name=label, orientation="h",
                            marker=dict(color=colour, line=dict(color=SURFACE, width=2)),
                            hovertemplate="<b>%{y}</b><br>" + label + ": %{x:.2f}% of GDP<extra></extra>")
            fig.update_layout(barmode="stack", bargap=0.45)
            fig.update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)")
            fig.update_xaxes(title="Trade at risk (% of GDP)")
            show(style(fig, "Most affected economies, by channel", 430))
        with b:
            st.markdown("**Full ranking**")
            tbl = res.sort_values("trade_at_risk_pct_gdp", ascending=False)[
                ["name", "tier", "trade_at_risk_pct_gdp", "main_driver"]]
            tbl.columns = ["Country", "Risk tier", "At risk (% GDP)", "Main driver"]
            table(tbl, height=430, column_config={
                "At risk (% GDP)": st.column_config.NumberColumn(format="%.2f")})

# =========================================================================== #
# MARKET RECOMMENDER
# =========================================================================== #
with T_MKT:
    st.subheader("Market-entry recommender")
    st.caption("Tell it what you are taking abroad and what matters most. It scores every country, "
               "then tests how stable the answer is.")
    mp = st.selectbox("What are you taking abroad?", list(scoring.PRESETS), key="mk_preset")
    weights: dict[str, float] = {}
    with st.expander("Fine-tune what matters (0 = ignore, 10 = critical)"):
        wc = st.columns(4)
        for i, (dn, dv) in enumerate(zip(scoring.DIM_NAMES, scoring.PRESETS[mp])):
            weights[dn] = wc[i % 4].slider(dn, 0, 10, dv, key=f"w_{mp}_{dn}")

    f1, f2, f3, f4 = st.columns(4)
    regions = sorted(df["region"].dropna().unique())
    incomes = [i for i in INCOME_ORDER if i in set(df["income"])] + sorted(set(df["income"]) - set(INCOME_ORDER))
    sel_reg = f1.multiselect("Regions", regions, placeholder="All regions") or regions
    sel_inc = f2.multiselect("Income groups", incomes, placeholder="All income groups") or incomes
    gdp_label = f3.select_slider("Minimum economy size", ["Any", "$1bn", "$10bn", "$50bn", "$100bn", "$500bn"])
    k_clusters = f4.slider("Market clusters", 3, 6, 4)
    gdp_floor = {"Any": 0, "$1bn": 1e9, "$10bn": 1e10, "$50bn": 5e10, "$100bn": 1e11, "$500bn": 5e11}[gdp_label]

    opp = scoring.opportunity(dim, weights)
    mk = df[["name", "region", "income", "gdp"]].join(opp).join(dim)
    keep = (mk["score"].notna() & mk["region"].isin(sel_reg) & mk["income"].isin(sel_inc)
            & (mk["gdp"].fillna(0) >= gdp_floor))
    view = mk[keep].copy()
    view["rank"] = view["score"].rank(ascending=False, method="min")
    view = view.sort_values("rank")

    if len(view) < 5:
        st.warning("Too few countries match these filters. Widen them to see results.")
        view = view.iloc[0:0]
    else:
        cl = scoring.cluster_markets(dim, k_clusters)
        view = view.join(cl[["cluster", "cluster_label", "pc1", "pc2"]])
        sens = scoring.weight_sensitivity(dim.loc[view.index], weights, top_k=10)
        view = view.join(sens)
        S_RANK, S_MAP, S_PROF, S_CLU, S_ROB = st.tabs(
            ["Ranking", "Map", "Profiles", "Clusters", "Robustness"])

        with S_RANK:
            top = view.head(15)
            fig = go.Figure(go.Bar(
                y=top["name"], x=top["score"], orientation="h",
                marker=dict(color=BLUE, line=dict(color=SURFACE, width=2)),
                hovertemplate="<b>%{y}</b><br>Opportunity score: %{x:.1f}<extra></extra>"))
            fig.update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)")
            fig.update_xaxes(range=[0, 100], title="Opportunity score (0-100)")
            fig.update_layout(bargap=0.45)
            show(style(fig, f"Top 15 markets for: {mp}", 460))
            t = view.reset_index()[["rank", "name", "region", "score", "coverage", "cluster_label", "in_top_pct"]]
            t.columns = ["Rank", "Country", "Region", "Score", "Data coverage", "Cluster", "In top 10 (% of runs)"]
            table(t, height=380, column_config={
                "Score": st.column_config.NumberColumn(format="%.1f"),
                "Data coverage": st.column_config.NumberColumn(format="percent"),
                "In top 10 (% of runs)": st.column_config.NumberColumn(format="%.0f"),
                "Rank": st.column_config.NumberColumn(format="%d")})

        with S_MAP:
            show(world_map(view, "score", BLUE_SEQ, "Score", "Opportunity score", ".1f", zmax=100))
            st.caption("Grey = filtered out, or too little data to score.")

        with S_PROF:
            top10 = view.head(10)
            z = top10[scoring.DIM_NAMES]
            fig = go.Figure(go.Heatmap(
                z=z.to_numpy(), x=scoring.DIM_NAMES, y=top10["name"], colorscale=BLUE_SEQ, zmin=0, zmax=100,
                xgap=2, ygap=2, text=z.round(0).to_numpy(), texttemplate="%{text:.0f}",
                hovertemplate="<b>%{y}</b><br>%{x}: %{z:.0f}/100<extra></extra>",
                colorbar=dict(thickness=12, outlinewidth=0, tickfont=dict(color=MUTED))))
            fig.update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)")
            fig.update_xaxes(side="top", gridcolor="rgba(0,0,0,0)", tickangle=-25)
            show(style(fig, "How the top 10 score on each dimension (percentile, 0-100)", 440))
            st.markdown("**Why the top three**")
            for iso in view.index[:3]:
                strong, weak = scoring.strengths(dim.loc[iso], weights)
                line = f"**{names[iso]}** - strongest on {', '.join(s.lower() for s in strong)}."
                if weak:
                    line += f" Weak spots: {', '.join(w.lower() for w in weak)}."
                st.markdown(line)

        with S_CLU:
            fig = go.Figure()
            for i, (lab, g) in enumerate(view.groupby("cluster_label")):
                fig.add_trace(go.Scatter(
                    x=g["pc1"], y=g["pc2"], mode="markers", name=lab, text=g["name"],
                    customdata=g["score"],
                    marker=dict(symbol=CLUSTER_SYMBOLS[i % len(CLUSTER_SYMBOLS)], size=12,
                                color=g["score"], colorscale=BLUE_SEQ, cmin=0, cmax=100,
                                line=dict(color=SURFACE, width=2),
                                colorbar=dict(title=dict(text="Score", font=dict(color=INK2, size=12)),
                                              thickness=12, outlinewidth=0) if i == 0 else None,
                                showscale=(i == 0)),
                    hovertemplate="<b>%{text}</b><br>Score: %{customdata:.1f}<extra>" + lab + "</extra>"))
            fig.update_xaxes(title="Market profile, axis 1", zeroline=False)
            fig.update_yaxes(title="Market profile, axis 2", zeroline=False)
            show(style(fig, "Markets with similar profiles sit together (shape = cluster, colour = score)", 480))
            summ = view.groupby("cluster_label").agg(
                Countries=("name", "count"), Avg_score=("score", "mean"),
                Examples=("name", lambda s: ", ".join(s.head(4)))).reset_index()
            summ.columns = ["Cluster", "Countries", "Avg score", "Examples (top-ranked)"]
            table(summ.sort_values("Avg score", ascending=False),
                  column_config={"Avg score": st.column_config.NumberColumn(format="%.1f")})

        with S_ROB:
            st.caption("The weights are a judgement call, so the app re-scores every country 500 times with "
                       "the weights randomly jittered around yours and counts how often each lands in the top 10.")
            r = view.sort_values(["in_top_pct", "score"], ascending=False).head(15)
            fig = go.Figure(go.Bar(
                y=r["name"], x=r["in_top_pct"], orientation="h",
                marker=dict(color=BLUE, line=dict(color=SURFACE, width=2)),
                customdata=np.stack([r["rank_p10"], r["rank_median"], r["rank_p90"]], axis=1),
                hovertemplate="<b>%{y}</b><br>In top 10: %{x:.0f}% of runs<br>"
                              "Rank range (10th-90th pct): %{customdata[0]:.0f}-%{customdata[2]:.0f}<extra></extra>"))
            fig.update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)")
            fig.update_xaxes(range=[0, 100], title="Share of weight-perturbation runs in the top 10 (%)")
            fig.update_layout(bargap=0.45)
            show(style(fig, "Which recommendations survive changing the weights?", 460))
            solid = view[view["in_top_pct"] >= 80]
            st.markdown(f"**{len(solid)} markets** stay in the top 10 in at least 80% of runs: "
                        + (", ".join(solid.sort_values("rank")["name"].head(10)) or "none") + ".")

# =========================================================================== #
# RISK-ADJUSTED VIEW (joins both engines)
# =========================================================================== #
with T_COMB:
    st.subheader("Risk-adjusted view")
    st.caption("Opportunity from the recommender, discounted by exposure to the scenario set in the "
               "Shock simulator tab.")
    if view.empty or not picked:
        st.info("Set a scenario in **Shock simulator** (at least one country) and make sure the "
                "**Market recommender** filters leave some countries to see the combined view.")
    else:
        risk_av = st.slider("Risk aversion (0 = ignore exposure, 1 = heavily penalise it)", 0.0, 1.0, 0.3, 0.05)
        comb = view[["name", "region", "score", "gdp"]].join(
            res[["trade_at_risk_pct_gdp", "vulnerability", "tier", "shocked"]], how="inner")
        comb["adjusted"] = scoring.risk_adjusted(comb["score"], comb["vulnerability"], risk_av)
        comb["rank_raw"] = comb["score"].rank(ascending=False, method="min")
        comb["rank_adj"] = comb["adjusted"].rank(ascending=False, method="min")
        comb["move"] = (comb["rank_raw"] - comb["rank_adj"]).astype(int)
        med = comb["score"].median()
        comb["quadrant"] = np.select(
            [(comb["score"] >= med) & (comb["vulnerability"] < 50), (comb["score"] >= med)],
            ["Attractive and resilient", "Attractive but exposed"], "Other")
        comb = comb.sort_values("rank_adj")

        fig = go.Figure()
        # label only the six best risk-adjusted markets, rotating the label side so neighbours don't collide
        lab_order = list(comb.head(6).index)
        side = ["top center", "bottom center", "middle right", "middle left"]
        for quad, colour in [("Other", MUTED), ("Attractive but exposed", ORANGE), ("Attractive and resilient", BLUE)]:
            g = comb[comb["quadrant"] == quad]
            fig.add_trace(go.Scatter(
                x=g["score"], y=g["vulnerability"], mode="markers+text", name=quad,
                text=[names[i] if i in lab_order else "" for i in g.index],
                textposition=[side[lab_order.index(i) % 4] if i in lab_order else "top center" for i in g.index],
                textfont=dict(color=INK2, size=11),
                customdata=np.stack([g["name"], g["trade_at_risk_pct_gdp"], g["adjusted"]], axis=1),
                marker=dict(color=colour, size=np.clip(np.sqrt(g["gdp"].fillna(1e9)) / 1.2e5, 7, 34),
                            opacity=0.9, line=dict(color=SURFACE, width=2)),
                hovertemplate="<b>%{customdata[0]}</b><br>Opportunity: %{x:.1f}<br>Exposure percentile: %{y:.0f}"
                              "<br>Trade at risk: %{customdata[1]:.2f}% of GDP<br>Risk-adjusted: %{customdata[2]:.1f}"
                              "<extra></extra>"))
        fig.add_vline(x=med, line=dict(color=AXIS, width=1, dash="dot"))
        fig.add_hline(y=50, line=dict(color=AXIS, width=1, dash="dot"))
        fig.update_xaxes(title="Opportunity score (0-100)")
        fig.update_yaxes(title="Exposure to the scenario (percentile)", range=[-5, 105])
        show(style(fig, "Opportunity vs exposure (bubble size = GDP)", 500))

        a, b = st.columns(2, gap="large")
        best = comb.head(3)
        exposed = comb[comb["quadrant"] == "Attractive but exposed"].sort_values("score", ascending=False).head(3)
        lines = [f"### Decision brief: {mp}", f"*Scenario: {preset}, {drop}% fall in "
                 f"{'energy ' if scope == 'energy' else ''}exports from {len(picked)} countr"
                 f"{'y' if len(picked) == 1 else 'ies'}. Risk aversion {risk_av:.2f}.*", "", "**Best risk-adjusted bets**"]
        for n_, (iso, r_) in enumerate(best.iterrows(), 1):
            strong, _ = scoring.strengths(dim.loc[iso], weights)
            lines.append(f"{n_}. **{r_['name']}** - opportunity {r_['score']:.0f}, "
                         f"{r_['trade_at_risk_pct_gdp']:.1f}% of GDP at risk ({r_['tier'].lower()}); "
                         f"strongest on {', '.join(s.lower() for s in strong)}.")
        if len(exposed):
            lines += ["", "**Attractive, but watch the exposure**"]
            for iso, r_ in exposed.iterrows():
                lines.append(f"- **{r_['name']}** - opportunity {r_['score']:.0f}, "
                             f"{r_['trade_at_risk_pct_gdp']:.1f}% of GDP at risk ({r_['tier'].lower()}).")
        brief = "\n".join(lines)
        with a:
            with st.container(border=True):
                st.markdown(brief)
            st.download_button("Download brief (.md)", brief, "globallens_brief.md", "text/markdown")
        with b:
            out = comb.reset_index()[["rank_adj", "name", "region", "score", "adjusted", "move", "tier"]]
            out.columns = ["Adj. rank", "Country", "Region", "Opportunity", "Risk-adjusted", "Places gained/lost", "Risk tier"]
            table(out, height=360, column_config={
                "Opportunity": st.column_config.NumberColumn(format="%.1f"),
                "Risk-adjusted": st.column_config.NumberColumn(format="%.1f"),
                "Adj. rank": st.column_config.NumberColumn(format="%d")})
            st.download_button("Download ranking (.csv)", out.to_csv(index=False), "globallens_ranking.csv", "text/csv")

# =========================================================================== #
# OVERVIEW
# =========================================================================== #
with T_OVER:
    st.markdown(
        "GlobalLens joins two questions that businesses usually ask separately: "
        "**where should we expand?** and **what happens to those places if trade is disrupted?**")
    c = st.columns(3)
    c[0].metric("Countries", len(df))
    c[1].metric("Indicators per country", len(data.INDICATORS))
    c[2].metric("Modelled trade links", f"{len(model.iso3) * (len(model.iso3) - 1):,}",
                help="Bilateral flows estimated with a gravity model, calibrated to reported totals.")
    st.markdown(f"#### {home_name} at a glance")
    base = scoring.opportunity(dim, scoring.preset_weights("Balanced"))
    c = st.columns(3)
    if home in base.index and pd.notna(base.loc[home, "score"]):
        c[0].metric("Market attractiveness (balanced weights)", f"#{int(base.loc[home, 'rank'])} of {int(base['score'].notna().sum())}",
                    f"score {base.loc[home, 'score']:.0f}/100", delta_color="off")
    else:
        c[0].metric("Market attractiveness", "n/a")
    if home in res.index and picked:
        c[1].metric(f"Exposure: {preset}", f"{res.loc[home, 'trade_at_risk_pct_gdp']:.2f}% of GDP",
                    res.loc[home, "tier"], delta_color="off")
        rk = int(res["trade_at_risk_pct_gdp"].rank(ascending=False, method="min").loc[home])
        c[2].metric("Exposure rank", f"#{rk} of {len(res)}", "1 = most exposed", delta_color="off")
    st.markdown(
        "**Try this**\n"
        "1. Open **Shock simulator**, choose *Gulf energy shock* and see which importers feel it most.\n"
        "2. Open **Market recommender**, switch between *Fintech* and *Manufacturing* and watch the ranking move.\n"
        "3. Open **Risk-adjusted view** to see which attractive markets are also fragile.")
    st.info("The scores and exposure figures are relative, screening-level indicators built from open data. "
            "Read the **Method & data** tab before drawing conclusions from them.")

# =========================================================================== #
# METHOD
# =========================================================================== #
with T_METH:
    st.subheader("Method, assumptions and limits")
    st.markdown("""
**Data.** 24 country indicators from the World Bank (World Development Indicators, Worldwide Governance
Indicators, Logistics Performance Index, Global Findex). For each, the most recent non-empty value from 2014 onward
is used, so the year differs by country (see the table below).

**Market score.** Every indicator is converted to a 0-100 percentile rank (log-transformed where skewed, oriented
so higher is better). Indicators are averaged into seven dimensions, which are combined with your weights. If a
dimension is missing for a country, the weights are re-normalised over what exists; countries with under 60% of
the weight covered are not scored. Clusters come from k-means on the seven dimension scores.

**Robustness.** Weights are redrawn 500 times from a Dirichlet distribution centred on yours; the chart shows how
often each country stays in the top 10.

**Trade model.** Bilateral trade data is not freely available at this scale, so flows are *estimated*:
`T[i,j] ~ GDP_i x GDP_j / distance_ij` (distance between capitals, 1.5x boost within the same World Bank region),
then iterative proportional fitting scales the matrix so each country's modelled exports and imports match its
reported totals.

**Shock channels** (all as % of the country's GDP): imports from hit countries that go missing, discounted by
how easily the importer can substitute (income and logistics quality); exports lost because hit economies
shrink and import less; and the hit countries' own lost export revenue.
""")
    st.warning(
        "**Limits.** Trade flows are modelled, not observed; they get the big picture right but can misstate "
        "any single pair. Exposure is trade at risk, not a GDP forecast: it ignores prices, inventories, services "
        "detail and policy responses. Scores describe national averages, not a specific city or sector, and the "
        "indicators are 1-3 years old. Use the tool to screen and to ask better questions, not to decide.")
    st.markdown(f"Trade-model calibration: worst row error {model.max_row_error:.1e}, worst column error "
                f"{model.max_col_error:.1e} after {model.iterations} iterations, {len(model.iso3)} countries.")
    v = data.vintage().reset_index()[["label", "wb_code", "countries", "median_year", "oldest_year"]]
    v.columns = ["Indicator", "World Bank code", "Countries covered", "Median year", "Oldest year"]
    table(v, column_config={"Median year": st.column_config.NumberColumn(format="%d"),
                            "Oldest year": st.column_config.NumberColumn(format="%d")})

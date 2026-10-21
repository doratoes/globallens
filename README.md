# GlobalLens

**Where should a business expand next, and which economies break first when trade does?**

GlobalLens joins two questions that are usually asked separately, on one open dataset (World Bank, 24 indicators, ~200 countries):

1. **Shock simulator** - hit one or more countries' exports (e.g. a Gulf energy shock, a China slump) and see which economies carry the most *trade at risk*, as a share of their GDP, through three channels: lost imports, lost export demand, and own lost exports.
2. **Market recommender** - pick what you are taking abroad (fintech, e-commerce, manufacturing...), tune what matters, and get a ranked list with clusters of similar markets and a **robustness test** that re-scores every country 500 times with jittered weights.
3. **Risk-adjusted view** - opportunity discounted by exposure to the scenario you set, with a downloadable decision brief.

## Run it

```bash
pip install -r requirements.txt
python data.py --refresh     # one-off: downloads the real World Bank data into ./data
streamlit run app.py
```

Commit the `data/` folder afterwards - the deployed app then starts instantly and doesn't depend on the World Bank API being up. (If `data/` is missing, the app tries to download on first load.)

## Deploy (free)

**Streamlit Community Cloud** (easiest): push to GitHub, go to share.streamlit.io, "New app", choose the repo, main file `app.py`. Python dependencies come from `requirements.txt`.

Alternatives: Hugging Face Spaces (Streamlit template) or Render.

## Tests

```bash
pip install pytest
pytest
```

38 tests cover the API parsing (mocked), scoring maths, the trade-matrix calibration, shock behaviour (monotonicity, channel decomposition, edge cases) and the whole Streamlit app run headlessly. They use synthetic data, so no network is needed.

## How it works

| File | Role |
|---|---|
| `data.py` | World Bank API -> cached tidy CSVs -> one clean DataFrame |
| `scoring.py` | percentile-rank indicators -> 7 dimensions -> weighted score; k-means clusters; Dirichlet weight-sensitivity; risk adjustment |
| `shocks.py` | gravity-model bilateral trade matrix, calibrated with iterative proportional fitting; shock propagation |
| `app.py` | Streamlit interface |

### Key modelling choices (and their limits)

- **Bilateral trade is estimated, not observed.** Open bilateral data sits behind API keys, so flows are modelled as `GDP_i x GDP_j / distance` (plus a same-region boost) and scaled so every country's modelled exports and imports match its reported totals. It gets the big picture right but can misstate any single country pair. *Upgrade path: swap in UN Comtrade data in `build_trade_model`.*
- **Exposure is not a GDP forecast.** It ignores prices, inventories, services detail and policy responses.
- **Percentile scoring** makes different units comparable and resists outliers, but scores are relative, not absolute.
- **Data is 1-3 years old** and comes from different years per country (see the Method tab).

## Data

World Bank Open Data: World Development Indicators, Worldwide Governance Indicators, Logistics Performance Index, Global Findex. Licensed CC BY 4.0.

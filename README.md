# Global Energy Transition Analysis (2000–2020)

End-to-end data analytics project covering 176 countries and 21 years of sustainable energy indicators. From raw data cleaning to clustering, feature engineering, regression, and individual country case studies.

---

## What this project covers

The dataset contains energy indicators for every country from 2000 to 2020: electricity access, renewable energy share, CO₂ emissions, energy intensity, GDP, and more. The goal was to understand how countries differ in their energy profiles, how they evolved over time, and what predicts a successful energy transition.

**Pipeline:**
1. Data cleaning and quality audit
2. Exploratory data analysis (EDA)
3. K-Means clustering — grouping countries by energy profile
4. Temporal dynamics — tracking how clusters changed over 20 years
5. Feature engineering — building 4 composite indicators
6. Regression — predicting energy transition progress
7. Case studies — deep-dive into 5 countries with notable trajectories

---

## Key findings

**The biomass paradox.** Countries with the least electricity access appear to have the highest renewable energy share — because traditional biomass (firewood, agricultural waste) counts as renewable. This is energy poverty, not energy transition. The correlation between electricity access and renewable share is r = −0.79.

**Energy systems have enormous inertia.** In the regression model (R² = 0.82), the single strongest predictor of a country's 2019 energy profile is its low-carbon electricity percentage from the year 2000 — not GDP, not growth rate. Infrastructure decisions made 20+ years ago still define the matrix today.

**42 out of 172 countries changed energy cluster between 2000 and 2019.** Most movements were upward (from energy poverty toward transition or developed), but 4 countries regressed — including Japan, whose low-carbon electricity dropped from 41% to 14% after Fukushima (2011).

**Cambodia is the most remarkable case in the dataset.** Starting from near zero (16% electricity access, GDP $300/capita in 2000), it achieved the highest improvement rate in the world (+2.40 pts/year) by building directly from renewables — no fossil system to dismantle.

**Argentina is the most instructive cautionary tale.** It had a clean energy advantage in 2000 (41% low-carbon electricity) and lost it by 2019 (34%). Repeated macroeconomic crises prevented energy investment. It has wind, solar, and hydro resources to lead a regional transition — but not the policy stability to capitalize on them.

---

## Clusters identified (2019)

| Cluster | Countries | Electricity access | Low-carbon elec | GDP per capita |
|---------|-----------|-------------------|-----------------|----------------|
| Energy poverty | 42 | 45% | 39% | $1,786 |
| Fossil emerging | 62 | 97% | 15% | $10,867 |
| Renewable transition | 43 | 95% | 68% | $8,334 |
| Developed high-consumption | 28 | 100% | 47% | $56,784 |

---

## Engineered features

Four composite variables built from the cleaned dataset:

- **transition_score** — overall energy transition progress (0–100), weighted combination of electricity access, low-carbon electricity, and log(GDP)
- **improvement_rate** — annual velocity of score improvement from 2000 to 2019
- **clean_access_ratio** — quality of electricity access: penalizes dirty electrification
- **fossil_lock_in** — structural dependency on fossil fuels, combining carbon share, energy intensity, and economic constraints

---

## Regression model

**Target:** `transition_score`
**Best model:** Ridge Regression (R² = 0.822, MAE = 5.2 points, 5-fold CV)

| Feature | Importance |
|---------|-----------|
| Low-carbon electricity 2000 | 47.5% |
| Electricity access 2019 | 18.8% |
| Energy per capita (log) | 15.1% |
| GDP per capita (log) | 8.5% |
| Other | 10.1% |

Ridge outperformed Random Forest (R² = 0.768), indicating the relationships are largely linear. The dominance of the 2000 low-carbon variable confirms path dependency — where you start matters more than how fast you grow.

---

## Tech stack

| Tool | Usage |
|------|-------|
| Python 3 | All analysis |
| pandas | Data cleaning, manipulation |
| numpy | Numerical transformations |
| scikit-learn | K-Means clustering, Ridge, Random Forest, cross-validation |
| matplotlib / seaborn | All charts and PDF reports |
| D3.js + TopoJSON | Interactive choropleth map |
| Chart.js | Interactive time series widgets |

---

## Repository structure

```
global-energy-analysis/
│
├── data/
│   ├── global-data-on-sustainable-energy.csv   # original dataset (Kaggle)
│   └── sustainable_energy_clean.csv            # cleaned output
│
├── scripts/
│   ├── 01_clean_sustainable_energy.py
│   ├── 02_eda_sustainable_energy.py
│   ├── 03_clustering_dynamics.py
│   ├── 04_feature_engineering.py
│   ├── 05_regression_analysis.py
│   └── 06_case_studies.py
│
├── outputs/
│   ├── cluster_assignments.csv
│   ├── migration_log.csv
│   ├── features_engineered.csv
│   ├── regression_results.csv
│   └── case_studies_summary.csv
│
├── reports/
│   ├── eda_report.pdf
│   ├── clustering_report.pdf
│   ├── features_report.pdf
│   ├── regression_report.pdf
│   └── case_studies_report.pdf
│
├── logs/
│   └── cleaning_log.json
│
└── project_log.md
```

---

## How to run

Install dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

Run scripts in order from the `scripts/` folder, with `sustainable_energy_clean.csv` in the working directory:

```bash
python 01_clean_sustainable_energy.py
python 02_eda_sustainable_energy.py
python 03_clustering_dynamics.py
python 04_feature_engineering.py
python 05_regression_analysis.py
python 06_case_studies.py
```

Each script prints a summary to the console and saves outputs to the working directory.

---

## Data source

[Global Data on Sustainable Energy — Kaggle](https://www.kaggle.com/datasets/anshtanwar/global-data-on-sustainable-energy)

Covers 176 countries, 2000–2020, with indicators for electricity access, renewable energy, CO₂ emissions, energy intensity, financial flows, and economic growth.

---

## Author

Francisco Tornello — Data Analytics project (2026)

---

## Validation: Out-of-Sample Testing (2021–2023)

The model trained on 2000–2020 data was tested against new data from 2021–2023 using two sources: Our World in Data (energy indicators) and World Bank (electricity access).

### Data sources for validation
- **Our World in Data** — energy dataset (github.com/owid/energy-data), updated through 2024
- **World Bank API** — electricity access indicator (EG.ELC.ACCS.ZS)

### Results

| Metric | Value |
|--------|-------|
| MAE original (2019, CV) | 5.22 points |
| MAE validation — complete data (148 countries) | **9.28 points** |
| MAE validation — all countries (176) | 21.25 points |
| Countries with complete data | 148 / 176 (56%) |

The high MAE across all countries is driven by the 44% without GDP data — for those, the model had to predict without its most important feature. The fair comparison is 9.28 vs 5.22: moderate degradation over a period that included COVID-19 and the 2022 energy crisis.

### Global trend (median transition_score)

| Year | Score |
|------|-------|
| 2019 | 54.5 |
| 2021 | 57.1 |
| 2022 | 57.5 |
| 2023 | 56.2 |

The world continued improving through 2022, with a slight pullback in 2023 as residual effects of the energy crisis persisted.

### 5 reference cases (2019 → 2023)

| Country | 2019 | 2023 | Change |
|---------|------|------|--------|
| Japan | 62.7 | 55.9 | −6.8 |
| Spain | 76.4 | 74.8 | −1.5 |
| Cambodia | 58.8 | 60.0 | +1.2 |
| Argentina | 60.4 | 60.9 | +0.4 |
| India | 49.3 | 52.2 | +2.9 |

**Japan** continued declining — fossil dependency post-Fukushima compounded by the 2022 gas crisis.
**Spain** dipped in 2022 and recovered in 2023 — the crisis was visible but not structural.
**India** kept improving slowly — solar expansion offsetting continued coal dependency.
**Argentina** barely moved — consistent with its historical pattern of macro-driven stagnation.
**Cambodia** slowed down in 2023 after strong growth — possible fossil expansion to sustain economic growth.

### Validation scripts
Scripts 07–10 in the `scripts/` folder cover the full validation pipeline:
`07_validation_step1_explore.py` → `08_validation_step2_align.py` → `08b_validation_step2b_gdp.py` → `09_validation_step3_predict.py` → `10_validation_step4_visualize.py`

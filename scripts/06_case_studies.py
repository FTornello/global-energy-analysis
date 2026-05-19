"""
case_studies.py
===============
Análisis individual de 5 países con trayectorias energéticas destacadas:
  - Japón      (retroceso post-Fukushima)
  - España     (transición deliberada sostenida)
  - India      (electrificación masiva a carbón)
  - Cambodia   (salida limpia desde cero — #1 en improvement_rate)
  - Argentina  (oportunidad perdida — única con base limpia que retrocedió)

Uso:
    python case_studies.py

Inputs:
    - sustainable_energy_clean.csv
    - features_engineered.csv
    - regression_results.csv

Output:
    - case_studies_report.pdf   (10 gráficos — 2 por país)
    - case_studies_summary.csv  (tabla resumen de los 5 países)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
import warnings
warnings.filterwarnings("ignore")

INPUT_CLEAN    = "sustainable_energy_clean.csv"
INPUT_FEATURES = "features_engineered.csv"
INPUT_REG      = "regression_results.csv"
OUTPUT_PDF     = "case_studies_report.pdf"
OUTPUT_CSV     = "case_studies_summary.csv"

CASES = ["Japan", "Spain", "India", "Cambodia", "Argentina"]

CASE_META = {
    "Japan":     {"emoji": "🇯🇵", "color": "#D85A30", "narrative": "El único retroceso — Fukushima 2011"},
    "Spain":     {"emoji": "🇪🇸", "color": "#1D9E75", "narrative": "Transición deliberada y sostenida"},
    "India":     {"emoji": "🇮🇳", "color": "#378ADD", "narrative": "Electrificación masiva, motor fósil"},
    "Cambodia":  {"emoji": "🇰🇭", "color": "#BA7517", "narrative": "#1 mundial en velocidad de mejora"},
    "Argentina": {"emoji": "🇦🇷", "color": "#7F77DD", "narrative": "Base limpia que se perdió"},
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "figure.dpi": 120,
})

# ─────────────────────────────────────────────────────────────────────────────
# CARGA
# ─────────────────────────────────────────────────────────────────────────────
print("Cargando datos...")
df  = pd.read_csv(INPUT_CLEAN)
fe  = pd.read_csv(INPUT_FEATURES)
reg = pd.read_csv(INPUT_REG)

years = list(range(2000, 2021))

def get_series(country, metric):
    sub = df[df["country"] == country].set_index("year")
    return [float(sub.loc[yr, metric]) if yr in sub.index and pd.notna(sub.loc[yr, metric]) else None for yr in years]

def get_fe(country, col):
    row = fe[fe["country"] == country]
    return round(float(row[col].values[0]), 3) if len(row) > 0 else None

def get_reg(country, col):
    row = reg[reg["country"] == country]
    return round(float(row[col].values[0]), 3) if len(row) > 0 else None

# ─────────────────────────────────────────────────────────────────────────────
# RESUMEN EN CONSOLA
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Resumen de los 5 casos ───────────────────────────────────────")
summary_rows = []
for country in CASES:
    row = {
        "country":           country,
        "cluster":           fe[fe["country"]==country]["cluster_name"].values[0] if len(fe[fe["country"]==country]) > 0 else "N/A",
        "transition_score":  get_fe(country, "transition_score"),
        "improvement_rate":  get_fe(country, "improvement_rate"),
        "fossil_lock_in":    get_fe(country, "fossil_lock_in"),
        "clean_access_2019": get_fe(country, "clean_access_ratio"),
        "regression_resid":  get_reg(country, "residual"),
        "lc_2000":           get_series(country, "low_carbon_elec_pct")[0],
        "lc_2019":           get_series(country, "low_carbon_elec_pct")[19],
        "access_2000":       get_series(country, "access_electricity_pct")[0],
        "access_2019":       get_series(country, "access_electricity_pct")[19],
        "gdp_2000":          get_series(country, "gdp_per_capita_usd")[0],
        "gdp_2019":          get_series(country, "gdp_per_capita_usd")[19],
    }
    summary_rows.append(row)
    print(f"\n{country}:")
    print(f"  Cluster:           {row['cluster']}")
    print(f"  Transition score:  {row['transition_score']}")
    print(f"  Improvement rate:  {row['improvement_rate']}")
    print(f"  Low-carbon elec:   {row['lc_2000']:.1f}% (2000) → {row['lc_2019']:.1f}% (2019)")
    print(f"  Acceso electricidad: {row['access_2000']:.1f}% → {row['access_2019']:.1f}%")
    print(f"  Residual regresión: {row['regression_resid']}")

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(OUTPUT_CSV, index=False)
print(f"\n✓ Tabla resumen: {OUTPUT_CSV}")

# ─────────────────────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────────────────────
print(f"\nGenerando reporte → {OUTPUT_PDF}")

with PdfPages(OUTPUT_PDF) as pdf:

    # ── Página 0: Overview comparativo ───────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Comparación de los 5 casos — métricas clave (2019)", fontsize=13, fontweight="bold")

    metrics_overview = [
        ("transition_score",  "Transition score", False),
        ("improvement_rate",  "Improvement rate (pts/año)", True),
        ("fossil_lock_in",    "Fossil lock-in", False),
    ]
    countries_short = CASES
    colors_bar = [CASE_META[c]["color"] for c in CASES]

    for i, (col, title, has_neg) in enumerate(metrics_overview):
        vals = [get_fe(c, col) for c in CASES]
        bars = axes[i].bar(countries_short, vals, color=colors_bar, edgecolor="none")
        axes[i].set_title(title, fontsize=11)
        if has_neg:
            axes[i].axhline(0, color="gray", linewidth=0.8)
        for bar, v in zip(bars, vals):
            axes[i].text(
                bar.get_x() + bar.get_width()/2,
                bar.get_height() + (0.5 if v >= 0 else -1.5),
                f"{v:.2f}", ha="center", va="bottom", fontsize=9
            )
        axes[i].set_xticks(range(len(CASES)))
        axes[i].set_xticklabels(CASES, rotation=20, ha="right", fontsize=9)

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight"); plt.close()
    print("  ✓ Overview comparativo")

    # ── 2 páginas por país ────────────────────────────────────────────────────
    plot_configs = {
        "Japan": [
            ("low_carbon_elec_pct",      "Low-carbon electricity (%) — caída post-Fukushima", "%",   True, 2011),
            ("primary_energy_per_capita_kwh", "Energía per cápita (kWh) — ajuste forzado",    "kWh", False, None),
        ],
        "Spain": [
            ("low_carbon_elec_pct",  "Low-carbon electricity (%) — ascenso sostenido", "%",   True, None),
            ("gdp_per_capita_usd",   "GDP per cápita (USD) — contexto económico",      "$",   False, 2008),
        ],
        "India": [
            ("access_electricity_pct", "Acceso a electricidad (%) — electrificación masiva", "%",  True,  None),
            ("low_carbon_elec_pct",    "Low-carbon electricity (%) — mejora lenta",           "%",  False, None),
        ],
        "Cambodia": [
            ("access_electricity_pct", "Acceso electricidad (%) — desde casi cero", "%", True, None),
            ("gdp_per_capita_usd",     "GDP per cápita — crecimiento simultáneo",   "$", False, None),
        ],
        "Argentina": [
            ("low_carbon_elec_pct", "Low-carbon electricity (%) — retroceso único", "%", True, 2001),
            ("gdp_per_capita_usd",  "GDP per cápita — volatilidad macro",           "$", False, 2001),
        ],
    }

    for country in CASES:
        meta = CASE_META[country]
        configs = plot_configs[country]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f"{meta['emoji']} {country} — {meta['narrative']}", fontsize=13, fontweight="bold")

        for i, (metric, title, unit, fill, event_yr) in enumerate(configs):
            ax = axes[i]
            series = get_series(country, metric)
            clean  = [v for v in series if v is not None]
            yrs_c  = [yr for yr, v in zip(years, series) if v is not None]

            ax.plot(yrs_c, clean, color=meta["color"], linewidth=2.5)
            if fill:
                ax.fill_between(yrs_c, clean, alpha=0.12, color=meta["color"])

            if event_yr:
                ax.axvline(event_yr, color="gray", linestyle="--", linewidth=1, alpha=0.7,
                           label=f"Evento {event_yr}")
                ax.legend(fontsize=8)

            ax.set_title(title, fontsize=10)
            ax.set_xlabel("Año")
            if unit == "%":
                ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=100, decimals=0))
            elif unit == "$":
                ax.yaxis.set_major_formatter(
                    mtick.FuncFormatter(lambda v, _: f"${v/1000:.0f}k" if v >= 1000 else f"${v:.0f}"))
            ax.set_xlim(2000, 2020)

        # Agregar métricas clave como texto
        ts  = get_fe(country, "transition_score")
        ir  = get_fe(country, "improvement_rate")
        fl  = get_fe(country, "fossil_lock_in")
        res = get_reg(country, "residual")
        fig.text(0.5, -0.04,
                 f"Transition score: {ts}  |  Improvement rate: {ir:+.3f} pts/año  |  "
                 f"Fossil lock-in: {fl}  |  Residual regresión: {res:+.2f}",
                 ha="center", fontsize=9, color="gray")

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight"); plt.close()
        print(f"  ✓ {country}")

print(f"\n✓ Reporte guardado: {OUTPUT_PDF}")

# ─────────────────────────────────────────────────────────────────────────────
# NARRATIVAS EN CONSOLA
# ─────────────────────────────────────────────────────────────────────────────
narrativas = {
    "Japan": (
        "La caída de Fukushima (2011) redujo el low-carbon electricity del 41% al 14% en 3 años. "
        "Japón pasó del cluster 'Desarrollados' a 'Emergentes fósiles'. Es el único país del dataset "
        "que retrocedió de cluster por un evento puntual. Su improvement_rate (-0.36) es el peor "
        "entre los países que en 2000 tenían acceso 100%."
    ),
    "Spain": (
        "Política energética sostenida durante 20 años. El low-carbon pasó del 44% al 66%. "
        "La crisis de 2008-2013 frenó temporalmente la inversión pero no revirtió la tendencia. "
        "Hoy es uno de los países con menor fossil_lock_in (26.3) de Europa."
    ),
    "India": (
        "El logro humano más grande del dataset: de 59% a 99% de acceso en 20 años. "
        "El costo: construir el sistema fósil más grande de Asia después de China. "
        "Fossil_lock_in de 53.6 — el desafío de descarbonizar será enorme."
    ),
    "Cambodia": (
        "#1 mundial en improvement_rate (2.40 pts/año). Construyó desde cero apostando a "
        "hidroeléctrica y solar. No tuvo sistema fósil previo que desmantelar. "
        "Residual de regresión +12.4: superó consistentemente sus condiciones iniciales."
    ),
    "Argentina": (
        "Única en el dataset que tenía base limpia (41% en 2000) y la redujo (34% en 2019). "
        "Las crisis macroeconómicas frenaron inversión energética repetidamente. "
        "Vaca Muerta reforzó la apuesta al gas. Tiene todos los recursos para liderar "
        "la transición regional — viento patagónico, sol en Cuyo — pero no la estabilidad."
    ),
}

print("\n── Narrativas de cada caso ───────────────────────────────────────")
for c, txt in narrativas.items():
    print(f"\n{CASE_META[c]['emoji']} {c}:")
    print(f"  {txt}")

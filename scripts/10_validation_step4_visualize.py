"""
10_validation_step4_visualize.py
================================
Paso 4 — Visualizaciones de la validación out-of-sample (2021-2023).

Genera 5 gráficos en un PDF:
  1. MAE original vs validación (con y sin datos completos)
  2. Transition score global 2000-2023 (serie extendida)
  3. Trayectorias de los 5 casos 2019-2023
  4. Predicho vs real (países con datos completos)
  5. Impacto de la crisis 2022 en Europa

Uso:
    python3 10_validation_step4_visualize.py

Inputs:
    - validation_predictions.csv
    - sustainable_energy_clean.csv
    - features_engineered.csv

Output:
    - validation_report.pdf
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
import warnings
warnings.filterwarnings("ignore")

OUTPUT_PDF = "validation_report.pdf"

COLORS = {
    "Argentina": "#7F77DD",
    "Spain":     "#1D9E75",
    "India":     "#378ADD",
    "Japan":     "#D85A30",
    "Cambodia":  "#BA7517",
}

CLUSTER_COLORS = {
    "Emergentes fósiles":         "#378ADD",
    "Desarrollados alto consumo": "#BA7517",
    "Transición renovable":       "#1D9E75",
    "Pobreza energética":         "#D85A30",
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
pred = pd.read_csv("validation_predictions.csv")
orig = pd.read_csv("sustainable_energy_clean.csv")
fe   = pd.read_csv("features_engineered.csv")

W_ACCESS, W_CLEAN, W_GDP = 0.30, 0.45, 0.25
GDP_MAX = np.log1p(orig["gdp_per_capita_usd"].max())
GDP_MIN = np.log1p(orig["gdp_per_capita_usd"].min())

def calc_ts(row):
    access = row["access_electricity_pct"] if pd.notna(row.get("access_electricity_pct")) else 50.0
    lc     = row["low_carbon_elec_pct"]    if pd.notna(row.get("low_carbon_elec_pct"))    else 30.0
    gdp_v  = row["gdp_per_capita_usd"]     if pd.notna(row.get("gdp_per_capita_usd"))     else 5000.0
    gdp_n  = ((np.log1p(gdp_v) - GDP_MIN) / (GDP_MAX - GDP_MIN) * 100).clip(0, 100)
    return round(access * W_ACCESS + lc * W_CLEAN + gdp_n * W_GDP, 2)

orig["ts"] = orig.apply(calc_ts, axis=1)

# Subsets
completos = pred[pred["gdp_per_capita_usd"].notna()].copy()
CASOS = ["Argentina", "Spain", "India", "Japan", "Cambodia"]
YEARS_ALL  = list(range(2000, 2024))
YEARS_ORIG = list(range(2000, 2020))
YEARS_NEW  = [2021, 2022, 2023]

print(f"  Total predicciones: {len(pred)}")
print(f"  Con datos completos: {len(completos)}")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT
# ─────────────────────────────────────────────────────────────────────────────
print(f"\nGenerando {OUTPUT_PDF}...")

with PdfPages(OUTPUT_PDF) as pdf:

    # ── Fig 1: Performance del modelo — MAE comparado ────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Validación out-of-sample: performance del modelo", fontsize=13, fontweight="bold")

    # Barras de MAE
    ax = axes[0]
    labels  = ["Original\n(2019, CV)", "Validación\n(todos,\n2021-23)", "Validación\n(datos\ncompletos)"]
    values  = [5.22, pred["residual"].abs().mean(), completos["residual"].abs().mean()]
    colors_b = ["#1D9E75", "#D85A30", "#BA7517"]
    bars = ax.bar(labels, values, color=colors_b, edgecolor="none", width=0.5)
    ax.set_ylabel("MAE (puntos de transition_score)")
    ax.set_title("Error medio absoluto (MAE)")
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.3,
                f"{v:.2f}", ha="center", va="bottom", fontsize=11, fontweight="500")
    ax.set_ylim(0, 28)
    ax.axhline(5.22, color="#1D9E75", linestyle="--", alpha=0.4, linewidth=1)

    # Distribución de residuales (solo datos completos)
    ax2 = axes[1]
    ax2.hist(completos["residual"], bins=25, color="#378ADD",
             edgecolor="white", linewidth=0.5, alpha=0.85)
    ax2.axvline(0, color="#D85A30", linewidth=1.5, linestyle="--", label="Error = 0")
    ax2.axvline(completos["residual"].mean(), color="#BA7517",
                linewidth=1.5, linestyle=":", label=f"Media = {completos['residual'].mean():.1f}")
    ax2.set_xlabel("Residual (real − predicho)")
    ax2.set_ylabel("Frecuencia")
    ax2.set_title("Distribución de residuales\n(países con datos completos, 2021-2023)")
    ax2.legend(fontsize=9)

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight"); plt.close()
    print("  ✓ Fig 1: Performance del modelo")

    # ── Fig 2: Transition score global — serie extendida 2000-2023 ───────────
    fig, ax = plt.subplots(figsize=(13, 5))

    # Serie original 2000-2019
    ts_orig_yr = orig.groupby("year")["ts"].median()
    ts_orig_yr = ts_orig_yr[ts_orig_yr.index <= 2019]

    # Serie nueva 2021-2023
    ts_new_yr = pred.groupby("year")["transition_score"].median()

    ax.plot(ts_orig_yr.index, ts_orig_yr.values,
            color="#378ADD", linewidth=2.5, marker="o", markersize=4,
            label="Período original (2000–2019)")

    ax.plot([2019, 2021], [ts_orig_yr[2019], ts_new_yr[2021]],
            color="#378ADD", linewidth=1.5, linestyle="--", alpha=0.5)

    ax.plot(ts_new_yr.index, ts_new_yr.values,
            color="#1D9E75", linewidth=2.5, marker="s", markersize=5,
            label="Período de validación (2021–2023)")

    ax.axvspan(2020, 2021.5, alpha=0.07, color="#D85A30", label="COVID-19")
    ax.axvspan(2021.5, 2022.5, alpha=0.07, color="#BA7517", label="Crisis energética 2022")

    ax.set_xlabel("Año")
    ax.set_ylabel("Transition score (mediana global)")
    ax.set_title("Evolución del transition score global — serie extendida 2000–2023",
                 fontsize=12)
    ax.legend(fontsize=9)
    ax.set_xlim(1999, 2024)

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight"); plt.close()
    print("  ✓ Fig 2: Serie global extendida")

    # ── Fig 3: Trayectorias de los 5 casos ───────────────────────────────────
    fig, ax = plt.subplots(figsize=(13, 6))

    for caso in CASOS:
        # Serie original
        sub_orig = orig[orig["country"] == caso].sort_values("year")
        ts_caso_orig = sub_orig.apply(calc_ts, axis=1).values
        yrs_orig = sub_orig["year"].values

        # Serie nueva
        sub_new = pred[pred["country"] == caso].sort_values("year")
        ts_caso_new = sub_new["transition_score"].values
        yrs_new = sub_new["year"].values

        color = COLORS[caso]

        # Línea original sólida
        ax.plot(yrs_orig, ts_caso_orig, color=color, linewidth=2,
                label=caso)
        # Extensión 2019-2021 punteada
        if len(ts_caso_orig) > 0 and len(ts_caso_new) > 0:
            ax.plot([yrs_orig[-1], yrs_new[0]],
                    [ts_caso_orig[-1], ts_caso_new[0]],
                    color=color, linewidth=1.5, linestyle="--", alpha=0.6)
        # Línea nueva
        ax.plot(yrs_new, ts_caso_new, color=color, linewidth=2,
                linestyle="--", marker="o", markersize=5)

    # Anotaciones finales
    for caso in CASOS:
        sub_new = pred[(pred["country"] == caso) & (pred["year"] == 2023)]
        if len(sub_new):
            val_2023 = sub_new["transition_score"].values[0]
            val_2019 = fe[fe["country"] == caso]["transition_score"].values[0]
            cambio = val_2023 - val_2019
            sign = "+" if cambio >= 0 else ""
            ax.annotate(
                f"{caso}\n{val_2023:.0f} ({sign}{cambio:.1f})",
                xy=(2023, val_2023),
                xytext=(8, 0), textcoords="offset points",
                fontsize=8, color=COLORS[caso], va="center"
            )

    ax.axvline(2019.5, color="gray", linestyle=":", alpha=0.5, linewidth=1)
    ax.text(2019.6, ax.get_ylim()[0] + 2, "Fin período\noriginal",
            fontsize=8, color="gray")

    ax.set_xlabel("Año")
    ax.set_ylabel("Transition score")
    ax.set_title("Trayectorias energéticas 2000–2023 — 5 casos de referencia",
                 fontsize=12)
    ax.legend(fontsize=9, loc="upper left")
    ax.set_xlim(1999, 2025)

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight"); plt.close()
    print("  ✓ Fig 3: Trayectorias 5 casos")

    # ── Fig 4: Scatter predicho vs real (datos completos) ────────────────────
    fig, ax = plt.subplots(figsize=(9, 8))

    for yr, color, marker in [(2021,"#378ADD","o"), (2022,"#BA7517","s"), (2023,"#1D9E75","^")]:
        sub = completos[completos["year"] == yr]
        ax.scatter(sub["transition_score"], sub["y_pred"],
                   color=color, alpha=0.6, s=35,
                   marker=marker, label=str(yr), zorder=3)

    # Diagonal perfecta
    ax.plot([10, 100], [10, 100], "--", color="gray", alpha=0.4,
            linewidth=1, label="Predicción perfecta")

    # Etiquetar los 5 casos
    for caso in CASOS:
        for yr in [2021, 2022, 2023]:
            sub = completos[(completos["country"]==caso) & (completos["year"]==yr)]
            if len(sub):
                ax.annotate(
                    f"{caso[:3]}.{yr-2000}",
                    (sub["transition_score"].values[0], sub["y_pred"].values[0]),
                    textcoords="offset points", xytext=(4, 3),
                    fontsize=7.5, color=COLORS[caso]
                )

    ax.set_xlabel("Transition score real")
    ax.set_ylabel("Transition score predicho (Ridge)")
    ax.set_title(f"Predicho vs Real — países con datos completos\n"
                 f"MAE = {completos['residual'].abs().mean():.2f} pts | "
                 f"n = {len(completos)} observaciones", fontsize=11)
    ax.legend(fontsize=9)
    ax.set_xlim(10, 105); ax.set_ylim(10, 105)

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight"); plt.close()
    print("  ✓ Fig 4: Scatter predicho vs real")

    # ── Fig 5: Impacto de la crisis 2022 en Europa ───────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Impacto de la crisis energética de 2022 en Europa",
                 fontsize=13, fontweight="bold")

    # Países europeos con datos completos
    eu_countries = [
        "Germany", "France", "Spain", "Italy", "Poland",
        "Netherlands", "Belgium", "Sweden", "Denmark", "Norway",
        "Finland", "Austria", "Portugal", "Czech Republic", "Hungary",
        "Romania", "Bulgaria", "Greece", "United Kingdom", "Switzerland"
    ]

    eu_data = pred[pred["country"].isin(eu_countries) &
                   pred["gdp_per_capita_usd"].notna()].copy()

    # Low-carbon electricity por año en Europa
    ax1 = axes[0]
    eu_lc = eu_data.groupby("year")["low_carbon_elec_pct"].median()

    # Agregar 2019 desde datos originales
    lc_2019_eu = orig[
        (orig["country"].isin(eu_countries)) & (orig["year"]==2019)
    ]["low_carbon_elec_pct"].median()

    all_yrs  = [2019] + list(eu_lc.index)
    all_vals = [lc_2019_eu] + list(eu_lc.values)

    ax1.plot(all_yrs, all_vals, color="#D85A30", linewidth=2.5,
             marker="o", markersize=6)
    ax1.axvspan(2021.5, 2022.5, alpha=0.1, color="#D85A30",
                label="Crisis gas (Ucrania)")
    ax1.set_xlabel("Año")
    ax1.set_ylabel("Low-carbon electricity (%)")
    ax1.set_title("Low-carbon electricity en Europa\n(mediana, países seleccionados)")
    ax1.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=100, decimals=0))
    ax1.legend(fontsize=9)
    ax1.set_xlim(2018.5, 2023.5)

    # Transition score 2019 vs 2022 por país europeo
    ax2 = axes[1]

    ts_2019_eu = fe[fe["country"].isin(eu_countries)][["country","transition_score"]].copy()
    ts_2022_eu = pred[
        (pred["country"].isin(eu_countries)) &
        (pred["year"]==2022) &
        pred["gdp_per_capita_usd"].notna()
    ][["country","transition_score"]].copy()

    eu_comp = ts_2019_eu.merge(
        ts_2022_eu, on="country", suffixes=("_2019","_2022")
    )
    eu_comp["cambio"] = eu_comp["transition_score_2022"] - eu_comp["transition_score_2019"]
    eu_comp = eu_comp.sort_values("cambio")

    colors_eu = ["#D85A30" if v < 0 else "#1D9E75" for v in eu_comp["cambio"]]
    ax2.barh(eu_comp["country"], eu_comp["cambio"],
             color=colors_eu, edgecolor="none")
    ax2.axvline(0, color="gray", linewidth=0.8)
    ax2.set_xlabel("Cambio en transition_score (2019 → 2022)")
    ax2.set_title("¿Quién subió y quién bajó en Europa\nentre 2019 y 2022?")
    for i, (_, row) in enumerate(eu_comp.iterrows()):
        v = row["cambio"]
        ax2.text(v + 0.1 if v >= 0 else v - 0.1, i,
                 f"{v:+.1f}", va="center",
                 ha="left" if v >= 0 else "right", fontsize=8)

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight"); plt.close()
    print("  ✓ Fig 5: Crisis 2022 en Europa")

print(f"\n✓ Reporte guardado: {OUTPUT_PDF}")

# ─────────────────────────────────────────────────────────────────────────────
# RESUMEN CONSOLA
# ─────────────────────────────────────────────────────────────────────────────
mae_completos = completos["residual"].abs().mean()
mae_todos     = pred["residual"].abs().mean()

print(f"""
{"="*65}
VALIDACIÓN OUT-OF-SAMPLE — RESUMEN FINAL
{"="*65}

PERFORMANCE:
  MAE original (2019, CV):              5.22 puntos
  MAE validación (datos completos):     {mae_completos:.2f} puntos  ← comparación justa
  MAE validación (todos los países):    {mae_todos:.2f} puntos  ← incluye 44% sin GDP

  Veredicto: el modelo degradó moderadamente (+{mae_completos-5.22:.1f} pts de error).
  Causas probables: distorsión COVID 2021, crisis energética 2022,
  diferencias metodológicas entre OWID y el dataset original de Kaggle.

TENDENCIA GLOBAL:
  El transition_score global (mediana) siguió subiendo:
  2019: 54.5 → 2021: 57.1 → 2022: 57.5 → 2023: 56.2
  Leve caída en 2023 — efecto residual de la crisis.

5 CASOS:
  Japón:     -6.8 pts  (peor caída — fósiles post-Fukushima + crisis)
  España:    -1.5 pts  (bajó en 2022, se recuperó en 2023)
  Cambodia:  +1.2 pts  (sigue mejorando aunque desaceleró en 2023)
  Argentina: +0.4 pts  (prácticamente estancada — patrón histórico)
  India:     +2.9 pts  (mejora lenta pero consistente)

ARCHIVOS GENERADOS:
  validation_report.pdf
""")

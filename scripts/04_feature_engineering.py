"""
feature_engineering.py
=======================
Construye 4 variables nuevas sobre el dataset limpio:

  1. transition_score    — avance total en la transición energética (0–100)
  2. improvement_rate   — velocidad de mejora anual (2000–2019)
  3. clean_access_ratio — qué tan limpio es el acceso a electricidad (0–100)
  4. fossil_lock_in     — qué tan atrapado está un país en los fósiles (0–100)

Uso:
    python feature_engineering.py

Inputs:
    - sustainable_energy_clean.csv
    - cluster_assignments.csv

Output:
    - features_engineered.csv   (175 países × features + clusters)
    - features_report.pdf       (visualizaciones)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.backends.backend_pdf import PdfPages
import warnings
warnings.filterwarnings("ignore")

INPUT_CLEAN    = "sustainable_energy_clean.csv"
INPUT_CLUSTERS = "cluster_assignments.csv"
OUTPUT_CSV     = "features_engineered.csv"
OUTPUT_PDF     = "features_report.pdf"

COLORS = {
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
    "grid.alpha": 0.3,
    "figure.dpi": 120,
})

# ─────────────────────────────────────────────────────────────────────────────
# CARGA
# ─────────────────────────────────────────────────────────────────────────────
print("Cargando datos...")
df       = pd.read_csv(INPUT_CLEAN)
clusters = pd.read_csv(INPUT_CLUSTERS)[["country", "cluster", "cluster_name"]]

# Constantes de normalización del GDP (calculadas sobre todo el dataset)
GDP_MAX = np.log1p(df["gdp_per_capita_usd"].max())
GDP_MIN = np.log1p(df["gdp_per_capita_usd"].min())

W_ACCESS = 0.30
W_CLEAN  = 0.45
W_GDP    = 0.25

# ─────────────────────────────────────────────────────────────────────────────
# FUNCIÓN: transition_score para cualquier fila del dataset
# ─────────────────────────────────────────────────────────────────────────────
def transition_score(row):
    access = row["access_electricity_pct"] if pd.notna(row["access_electricity_pct"]) else 50.0
    lc     = row["low_carbon_elec_pct"]    if pd.notna(row["low_carbon_elec_pct"])    else 30.0
    gdp_l  = np.log1p(row["gdp_per_capita_usd"]) if pd.notna(row["gdp_per_capita_usd"]) else (GDP_MIN + GDP_MAX) / 2
    gdp_n  = ((gdp_l - GDP_MIN) / (GDP_MAX - GDP_MIN) * 100).clip(0, 100)
    return round(access * W_ACCESS + lc * W_CLEAN + gdp_n * W_GDP, 2)

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE 1 y 2: transition_score + improvement_rate
# ─────────────────────────────────────────────────────────────────────────────
print("Calculando transition_score...")
df["transition_score"] = df.apply(transition_score, axis=1)

# Improvement rate: diferencia entre 2019 y 2000, dividida por 19 años
ts_2000 = df[df["year"] == 2000][["country", "transition_score"]].rename(columns={"transition_score": "score_2000"})
ts_2019 = df[df["year"] == 2019][["country", "transition_score"]].rename(columns={"transition_score": "score_2019"})
scores  = ts_2000.merge(ts_2019, on="country")
scores["score_change"]    = (scores["score_2019"] - scores["score_2000"]).round(2)
scores["improvement_rate"] = (scores["score_change"] / 19).round(3)

print("Distribución improvement_rate:")
print(scores["improvement_rate"].describe().round(3))

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE 3: clean_access_ratio
# ─────────────────────────────────────────────────────────────────────────────
print("\nCalculando clean_access_ratio...")
df_2019 = df[df["year"] == 2019].copy()

df_2019["clean_access_ratio"] = (
    (df_2019["access_electricity_pct"].fillna(0) / 100) *
    (df_2019["low_carbon_elec_pct"].fillna(0) / 100) * 100
).round(2)

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE 4: fossil_lock_in
# ─────────────────────────────────────────────────────────────────────────────
print("Calculando fossil_lock_in...")
df_2019["gdp_log_norm"] = (
    (np.log1p(df_2019["gdp_per_capita_usd"]) - GDP_MIN) /
    (GDP_MAX - GDP_MIN) * 100
).clip(0, 100)

df_2019["fossil_share_raw"] = 1 - df_2019["low_carbon_elec_pct"].fillna(
    df_2019["low_carbon_elec_pct"].median()) / 100

ei = df_2019["energy_intensity_mj_gdp"].fillna(df_2019["energy_intensity_mj_gdp"].median())
df_2019["ei_norm"] = (ei - ei.min()) / (ei.max() - ei.min())

df_2019["fossil_lock_in"] = (
    df_2019["fossil_share_raw"] * 0.45 +
    df_2019["ei_norm"]          * 0.35 +
    (1 - df_2019["gdp_log_norm"] / 100) * 0.20
).mul(100).round(2)

# ─────────────────────────────────────────────────────────────────────────────
# CONSOLIDAR
# ─────────────────────────────────────────────────────────────────────────────
features = df_2019[[
    "country", "access_electricity_pct", "low_carbon_elec_pct",
    "gdp_per_capita_usd", "transition_score",
    "clean_access_ratio", "fossil_lock_in"
]].merge(scores[["country", "score_2000", "score_2019", "score_change", "improvement_rate"]], on="country", how="left")
features = features.merge(clusters, on="country", how="left")

features.to_csv(OUTPUT_CSV, index=False)
print(f"\n✓ Features guardadas: {OUTPUT_CSV} ({len(features)} países)")

# ─────────────────────────────────────────────────────────────────────────────
# RESUMEN EN CONSOLA
# ─────────────────────────────────────────────────────────────────────────────
for col, label in [
    ("transition_score",   "Transition score"),
    ("improvement_rate",   "Improvement rate"),
    ("clean_access_ratio", "Clean access ratio"),
    ("fossil_lock_in",     "Fossil lock-in"),
]:
    s = features[col].dropna()
    print(f"\n── {label}")
    print(f"   Mediana: {s.median():.2f}  |  Media: {s.mean():.2f}  |  Rango: {s.min():.2f} – {s.max():.2f}")
    print(f"   Top 3:    {', '.join(features.nlargest(3,col)['country'].tolist())}")
    print(f"   Bottom 3: {', '.join(features.nsmallest(3,col)['country'].tolist())}")

# ─────────────────────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────────────────────
print(f"\nGenerando reporte → {OUTPUT_PDF}")

with PdfPages(OUTPUT_PDF) as pdf:

    # ── Fig 1: Scatter cuadrante transition_score vs improvement_rate ────────
    fig, ax = plt.subplots(figsize=(12, 8))
    for cname, ccolor in COLORS.items():
        sub = features[features["cluster_name"] == cname].dropna(
            subset=["transition_score", "improvement_rate"])
        ax.scatter(sub["transition_score"], sub["improvement_rate"],
                   color=ccolor, alpha=0.7, s=50, label=cname, zorder=3)
        # Etiquetar casos notables
        notable = features[
            (features["cluster_name"] == cname) &
            (features["improvement_rate"].abs() > 1.0)
        ].dropna(subset=["transition_score", "improvement_rate"])
        for _, r in notable.iterrows():
            ax.annotate(r["country"], (r["transition_score"], r["improvement_rate"]),
                        textcoords="offset points", xytext=(5, 4), fontsize=8, color=ccolor)

    ax.axvline(features["transition_score"].median(), color="gray", linestyle="--",
               alpha=0.5, linewidth=1, label="Mediana score")
    ax.axhline(features["improvement_rate"].median(), color="gray", linestyle=":",
               alpha=0.5, linewidth=1, label="Mediana rate")
    ax.set_xlabel("Transition score (posición 2019)")
    ax.set_ylabel("Improvement rate (pts/año, 2000–2019)")
    ax.set_title("¿Dónde están y hacia dónde van? Transition score vs velocidad de mejora", fontsize=12)
    ax.legend(fontsize=9)

    # Etiquetas de cuadrante
    xm, ym = features["transition_score"].median(), features["improvement_rate"].median()
    ax.text(xm + 1, ax.get_ylim()[1] * 0.92, "Avanzados y acelerando", fontsize=8, color="gray", style="italic")
    ax.text(0 + 1, ax.get_ylim()[1] * 0.92, "Rezagados pero mejorando", fontsize=8, color="gray", style="italic")

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight"); plt.close()
    print("  ✓ Fig 1: Scatter cuadrante")

    # ── Fig 2: Rankings horizontales para las 4 features ────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Rankings globales — top 10 por variable engineerada", fontsize=13, fontweight="bold")
    axes = axes.flatten()

    plot_configs = [
        ("transition_score",   "Transition score (0–100)",         True),
        ("improvement_rate",   "Improvement rate (pts/año)",        True),
        ("clean_access_ratio", "Clean access ratio (0–100)",        True),
        ("fossil_lock_in",     "Fossil lock-in (0=libre, 100=atrapado)", True),
    ]

    for i, (col, title, do_top) in enumerate(plot_configs):
        ax = axes[i]
        top10 = features.nlargest(10, col)[["country", col, "cluster_name"]].dropna()
        colors_bar = [COLORS.get(c, "#888") for c in top10["cluster_name"]]
        ax.barh(top10["country"], top10[col], color=colors_bar, edgecolor="none")
        ax.set_title(title, fontsize=11)
        ax.invert_yaxis()
        for j, (_, row) in enumerate(top10.iterrows()):
            v = row[col]
            fmt = f"{v:.1f}" if abs(v) >= 1 else f"{v:.2f}"
            ax.text(v * 1.01, j, fmt, va="center", fontsize=9)
        ax.set_xlim(0, top10[col].max() * 1.18)

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight"); plt.close()
    print("  ✓ Fig 2: Rankings")

    # ── Fig 3: Distribución de features por cluster ──────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Distribución de variables engineeradas por cluster", fontsize=13, fontweight="bold")
    axes = axes.flatten()

    for i, col in enumerate(["transition_score", "improvement_rate", "clean_access_ratio", "fossil_lock_in"]):
        ax = axes[i]
        for cname, ccolor in COLORS.items():
            sub = features[features["cluster_name"] == cname][col].dropna()
            if len(sub) > 0:
                ax.hist(sub, bins=15, color=ccolor, alpha=0.5, label=cname, edgecolor="none")
        ax.set_title(col.replace("_", " ").title(), fontsize=11)
        ax.legend(fontsize=8)

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight"); plt.close()
    print("  ✓ Fig 3: Distribuciones")

print(f"\n✓ Reporte guardado: {OUTPUT_PDF}")

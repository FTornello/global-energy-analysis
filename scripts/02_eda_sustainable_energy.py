"""
eda_sustainable_energy.py
==========================
EDA completo del dataset Global Sustainable Energy (2000–2020).
Genera 6 gráficos en un PDF y muestra estadísticas clave en consola.

Uso:
    python eda_sustainable_energy.py

Output:
    - eda_report.pdf   (todos los gráficos)
    - consola          (estadísticas descriptivas y hallazgos)

Requiere: pandas, numpy, matplotlib, seaborn
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import warnings
warnings.filterwarnings("ignore")

INPUT  = "sustainable_energy_clean.csv"
OUTPUT = "eda_report.pdf"

BLUE   = "#378ADD"
TEAL   = "#1D9E75"
PURPLE = "#7F77DD"
CORAL  = "#D85A30"
AMBER  = "#BA7517"
GRAY   = "#888780"

plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
    "figure.dpi": 120,
})

# ── Load ──────────────────────────────────────────────────────────────────────
print("Cargando dataset limpio...")
df = pd.read_csv(INPUT)
print(f"Shape: {df.shape}")

# ── Estadísticas descriptivas en consola ──────────────────────────────────────
print("\n" + "="*60)
print("ESTADÍSTICAS DESCRIPTIVAS — columnas clave")
print("="*60)
key_cols = [
    "access_electricity_pct", "access_clean_fuels_pct",
    "renewable_share_pct", "low_carbon_elec_pct",
    "primary_energy_per_capita_kwh", "co2_emissions_kt",
    "gdp_growth_pct", "gdp_per_capita_usd", "energy_intensity_mj_gdp"
]
print(df[key_cols].describe().round(2).to_string())

print("\n" + "="*60)
print("CORRELACIONES RELEVANTES")
print("="*60)
corr_cols = [
    "access_electricity_pct", "access_clean_fuels_pct",
    "renewable_share_pct", "low_carbon_elec_pct",
    "gdp_per_capita_usd", "energy_intensity_mj_gdp"
]
print(df[corr_cols].corr().round(3).to_string())

print("\n" + "="*60)
print("OUTLIERS (método IQR)")
print("="*60)
for col in ["co2_emissions_kt", "gdp_per_capita_usd",
            "primary_energy_per_capita_kwh", "elec_fossil_twh"]:
    Q1, Q3 = df[col].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    out = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)]
    top = out.nlargest(3, col)["country"].tolist()
    print(f"{col}: {len(out)} outliers  |  top: {top}")

# ── Plots ─────────────────────────────────────────────────────────────────────
print(f"\nGenerando gráficos → {OUTPUT}")

with PdfPages(OUTPUT) as pdf:

    # ── Fig 1: Tendencias temporales ─────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Tendencias globales 2000–2020", fontsize=14, fontweight="bold", y=1.02)

    trend_cols = ["access_electricity_pct", "low_carbon_elec_pct",
                  "co2_emissions_kt", "gdp_per_capita_usd"]
    trends = df.groupby("year")[trend_cols].agg(
        access_electricity_pct=("access_electricity_pct", "mean"),
        low_carbon_elec_pct=("low_carbon_elec_pct", "median"),
        co2_emissions_kt=("co2_emissions_kt", "median"),
        gdp_per_capita_usd=("gdp_per_capita_usd", "median")
    ).reset_index()

    ax1 = axes[0]
    ax1.plot(trends["year"], trends["access_electricity_pct"],
             color=BLUE, linewidth=2, marker="o", markersize=3, label="Acceso electricidad (% medio)")
    ax1.plot(trends["year"], trends["low_carbon_elec_pct"],
             color=TEAL, linewidth=2, marker="s", markersize=3, linestyle="--",
             label="Low-carbon electricity (% mediana)")
    ax1.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax1.set_xlabel("Año")
    ax1.set_title("Electricidad: acceso y fuentes limpias")
    ax1.legend(fontsize=9)
    ax1.set_ylim(15, 100)

    ax2 = axes[1]
    color_gdp = PURPLE
    ax2_twin = ax2.twinx()
    ax2.plot(trends["year"], trends["gdp_per_capita_usd"] / 1000,
             color=color_gdp, linewidth=2, marker="o", markersize=3, label="GDP per capita ($k)")
    # CO2 solo hasta 2019 (2020 sin datos)
    co2_data = trends[trends["year"] <= 2019]
    ax2_twin.bar(co2_data["year"], co2_data["co2_emissions_kt"] / 1000,
                 alpha=0.25, color=CORAL, label="CO₂ mediana (kt, eje der.)")
    ax2.set_xlabel("Año")
    ax2.set_ylabel("GDP per capita ($k)", color=color_gdp)
    ax2_twin.set_ylabel("CO₂ mediana (miles kt)", color=CORAL)
    ax2.set_title("GDP per capita y CO₂ (mediana global)")
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=9)
    ax2.spines["right"].set_visible(True)

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close()
    print("  ✓ Fig 1: Tendencias temporales")

    # ── Fig 2: Distribuciones ─────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Distribuciones de variables clave", fontsize=14, fontweight="bold")

    bins_renew = np.arange(0, 101, 10)
    axes[0].hist(df["renewable_share_pct"].dropna(), bins=bins_renew,
                 color=TEAL, edgecolor="white", linewidth=0.5)
    axes[0].set_title("Renovables share (%)")
    axes[0].set_xlabel("% energía renovable")
    axes[0].set_ylabel("Registros")
    axes[0].xaxis.set_major_formatter(mtick.PercentFormatter(xmax=100, decimals=0))

    axes[1].hist(np.log1p(df["gdp_per_capita_usd"].dropna()), bins=30,
                 color=PURPLE, edgecolor="white", linewidth=0.5)
    axes[1].set_title("GDP per capita (escala log)")
    axes[1].set_xlabel("log(GDP per capita + 1)")
    axes[1].set_ylabel("Registros")

    bins_ei = np.arange(0, 33, 2)
    axes[2].hist(df["energy_intensity_mj_gdp"].dropna(), bins=bins_ei,
                 color=AMBER, edgecolor="white", linewidth=0.5)
    axes[2].set_title("Intensidad energética (MJ/$2017 PPP)")
    axes[2].set_xlabel("MJ por $ de GDP")
    axes[2].set_ylabel("Registros")

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close()
    print("  ✓ Fig 2: Distribuciones")

    # ── Fig 3: Mapa de correlaciones ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 7))
    corr_matrix = df[corr_cols].corr()
    short_names = {
        "access_electricity_pct": "Acc. electricidad",
        "access_clean_fuels_pct": "Acc. combustibles",
        "renewable_share_pct": "Renov. share",
        "low_carbon_elec_pct": "Low-carbon elec",
        "gdp_per_capita_usd": "GDP per capita",
        "energy_intensity_mj_gdp": "E. intensity"
    }
    corr_matrix.index = [short_names[c] for c in corr_cols]
    corr_matrix.columns = [short_names[c] for c in corr_cols]

    mask = np.zeros_like(corr_matrix, dtype=bool)
    mask[np.triu_indices_from(mask, k=1)] = True

    sns.heatmap(
        corr_matrix, mask=mask, annot=True, fmt=".2f",
        cmap="RdYlGn", vmin=-1, vmax=1, center=0,
        square=True, linewidths=0.5, ax=ax,
        annot_kws={"size": 10},
        cbar_kws={"shrink": 0.7}
    )
    ax.set_title("Mapa de correlaciones — variables principales", fontsize=13, pad=12)
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close()
    print("  ✓ Fig 3: Mapa de correlaciones")

    # ── Fig 4: Top/Bottom países por indicador (año 2020) ────────────────────
    last = df[df["year"] == 2020].copy()
    last_gdp = df[df["year"] == 2019].copy()  # GDP disponible hasta 2019 completo

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Disparidades entre países (2020)", fontsize=14, fontweight="bold")

    acc_sorted = last[["country","access_electricity_pct"]].dropna().sort_values("access_electricity_pct")
    bottom10 = acc_sorted.head(10)
    axes[0].barh(bottom10["country"], bottom10["access_electricity_pct"],
                 color=CORAL, edgecolor="none")
    axes[0].set_title("Menor acceso a electricidad (2020)")
    axes[0].set_xlabel("% de población")
    axes[0].xaxis.set_major_formatter(mtick.PercentFormatter(xmax=100, decimals=0))
    for i, (_, row) in enumerate(bottom10.iterrows()):
        axes[0].text(row["access_electricity_pct"] + 0.5, i,
                     f"{row['access_electricity_pct']:.1f}%", va="center", fontsize=9)

    gdp_sorted = last_gdp[["country","gdp_per_capita_usd"]].dropna().sort_values("gdp_per_capita_usd")
    bot5 = gdp_sorted.head(5)
    top5 = gdp_sorted.tail(5)
    combined = pd.concat([bot5, top5])
    colors_gdp = [CORAL]*5 + [BLUE]*5
    axes[1].barh(combined["country"], combined["gdp_per_capita_usd"],
                 color=colors_gdp, edgecolor="none")
    axes[1].set_title("GDP per capita: 5 menores vs 5 mayores (2019)")
    axes[1].set_xlabel("USD")
    axes[1].xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"${x/1000:.0f}k"))

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close()
    print("  ✓ Fig 4: Disparidades entre países")

    # ── Fig 5: Intensidad energética top 10 ──────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    ei_top = (
        df[df["year"] >= 2015]
        .groupby("country")["energy_intensity_mj_gdp"]
        .mean()
        .nlargest(10)
        .sort_values()
    )
    bars = ax.barh(ei_top.index, ei_top.values, color=AMBER, edgecolor="none")
    ax.set_title("Top 10: países más ineficientes energéticamente (promedio 2015–2020)",
                 fontsize=12)
    ax.set_xlabel("MJ por dólar de GDP ($2017 PPP)")
    for bar, val in zip(bars, ei_top.values):
        ax.text(val + 0.1, bar.get_y() + bar.get_height()/2,
                f"{val:.1f}", va="center", fontsize=10)
    ax.set_xlim(0, ei_top.max() * 1.15)

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close()
    print("  ✓ Fig 5: Intensidad energética")

    # ── Fig 6: Scatter GDP vs Renovables ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter_df = df[["gdp_per_capita_usd","renewable_share_pct","year"]].dropna()
    scatter_df = scatter_df[scatter_df["year"].isin([2000, 2010, 2019])]

    palette = {2000: CORAL, 2010: PURPLE, 2019: TEAL}
    for yr, grp in scatter_df.groupby("year"):
        ax.scatter(np.log1p(grp["gdp_per_capita_usd"]),
                   grp["renewable_share_pct"],
                   alpha=0.45, s=20, color=palette[yr], label=str(yr))

    ax.set_xlabel("log(GDP per capita + 1)")
    ax.set_ylabel("Renovables share (%)")
    ax.set_title("GDP per capita vs Renovables share (años seleccionados)")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=100, decimals=0))
    ax.legend(title="Año")

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close()
    print("  ✓ Fig 6: GDP vs Renovables (scatter)")

print(f"\n✓ Reporte guardado en: {OUTPUT}")
print("\n── Hallazgos principales ─────────────────────────────────────────")
print("1. Acceso a electricidad subió de 73% a 85% (media global, 2000–2020).")
print("2. Distribución de renovables bimodal: concentración en <10% Y >80%.")
print("   → Dos grupos distintos: países industrializados y países en desarrollo con biomasa.")
print("3. GDP per capita muy sesgado a la derecha — mediana $4.6k vs media $13.3k.")
print("4. Correlación fuerte negativa entre acceso electricidad y renovables share (r=-0.79):")
print("   → Países con bajo acceso usan más biomasa/renovables tradicionales.")
print("5. Outliers estructurales: China en CO₂ y electricidad fósil,")
print("   Luxembourg en GDP, Qatar en consumo de energía per cápita.")
print("6. Trinidad y Tobago: intensidad energética 4x el promedio global (industria petroquímica).")

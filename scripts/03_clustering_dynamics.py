"""
clustering_dynamics.py
=======================
Clustering de países por perfil energético (K-Means, k=4)
y análisis de evolución dinámica 2000–2019.

Uso:
    python clustering_dynamics.py

Inputs:
    - sustainable_energy_clean.csv

Outputs:
    - clustering_report.pdf    (scatter PCA, perfiles, evolución, migraciones)
    - cluster_assignments.csv  (país, cluster, año)
    - migration_log.csv        (países que cambiaron de cluster)

Requiere: pandas, numpy, matplotlib, seaborn, scikit-learn
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mtick
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from matplotlib.backends.backend_pdf import PdfPages
import warnings
warnings.filterwarnings("ignore")

INPUT    = "sustainable_energy_clean.csv"
OUT_PDF  = "clustering_report.pdf"
OUT_CSV  = "cluster_assignments.csv"
OUT_MIG  = "migration_log.csv"

# ── Paleta y nombres ──────────────────────────────────────────────────────────
COLORS = {0: "#378ADD", 1: "#BA7517", 2: "#1D9E75", 3: "#D85A30"}
NAMES  = {
    0: "Emergentes fósiles",
    1: "Desarrollados alto consumo",
    2: "Transición renovable",
    3: "Pobreza energética",
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
    "figure.dpi": 120,
})

FEATURES = [
    "access_electricity_pct",
    "access_clean_fuels_pct",
    "renewable_share_pct",
    "low_carbon_elec_pct",
    "primary_energy_per_capita_kwh",
    "energy_intensity_mj_gdp",
    "gdp_per_capita_usd",
]

FEATURE_LABELS = {
    "access_electricity_pct":      "Acceso electricidad (%)",
    "access_clean_fuels_pct":      "Acceso combustibles limpios (%)",
    "renewable_share_pct":         "Renovables share (%)",
    "low_carbon_elec_pct":         "Low-carbon electricity (%)",
    "primary_energy_per_capita_kwh": "Energía per cápita (kWh)",
    "energy_intensity_mj_gdp":     "Intensidad energética (MJ/$GDP)",
    "gdp_per_capita_usd":          "GDP per cápita (USD)",
}

# ─────────────────────────────────────────────────────────────────────────────
# 1. CARGA Y PREPARACIÓN
# ─────────────────────────────────────────────────────────────────────────────
print("Cargando datos...")
df = pd.read_csv(INPUT)

def prepare_year(df, year, scaler=None, fit=False):
    sub = df[df["year"] == year].copy()
    sub = sub[["country"] + FEATURES].copy()
    sub["null_count"] = sub[FEATURES].isnull().sum(axis=1)
    sub = sub[sub["null_count"] <= 2].drop(columns="null_count")
    for col in FEATURES:
        sub[col] = sub[col].fillna(sub[col].median())
    if fit:
        X = scaler.fit_transform(sub[FEATURES])
    else:
        X = scaler.transform(sub[FEATURES])
    return sub, X

scaler = StandardScaler()
base, X_base = prepare_year(df, 2019, scaler, fit=True)

# ─────────────────────────────────────────────────────────────────────────────
# 2. ELECCIÓN DE K
# ─────────────────────────────────────────────────────────────────────────────
print("Evaluando k óptimo...")
inertias, silhouettes = [], []
ks = range(2, 10)
for k in ks:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labs = km.fit_predict(X_base)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_base, labs))
    print(f"  k={k}  inertia={km.inertia_:.1f}  silhouette={silhouette_score(X_base, labs):.3f}")

K = 4
km_final = KMeans(n_clusters=K, random_state=42, n_init=20)
base["cluster"] = km_final.fit_predict(X_base)

# ─────────────────────────────────────────────────────────────────────────────
# 3. PCA PARA VISUALIZACIÓN
# ─────────────────────────────────────────────────────────────────────────────
pca = PCA(n_components=2, random_state=42)
coords = pca.fit_transform(X_base)
base["pca1"] = coords[:, 0]
base["pca2"] = coords[:, 1]
print(f"\nVarianza explicada PCA: PC1={pca.explained_variance_ratio_[0]:.1%}, PC2={pca.explained_variance_ratio_[1]:.1%}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. CENTROS EN VALORES ORIGINALES
# ─────────────────────────────────────────────────────────────────────────────
centers_orig = scaler.inverse_transform(km_final.cluster_centers_)
centers_df = pd.DataFrame(centers_orig, columns=FEATURES)
print("\nCentros de cluster (valores originales):")
print(centers_df.round(2).to_string())

# ─────────────────────────────────────────────────────────────────────────────
# 5. EVOLUCIÓN TEMPORAL (labels fijos de 2019)
# ─────────────────────────────────────────────────────────────────────────────
country_cluster = dict(zip(base["country"], base["cluster"]))
df["cluster"] = df["country"].map(country_cluster)
df_c = df[df["cluster"].notna()].copy()
df_c["cluster"] = df_c["cluster"].astype(int)

trend_metrics = [
    "access_electricity_pct",
    "low_carbon_elec_pct",
    "renewable_share_pct",
    "gdp_per_capita_usd",
]

years_range = list(range(2000, 2020))
trends = {}
for c in range(K):
    g = df_c[df_c["cluster"] == c]
    trends[c] = g.groupby("year")[trend_metrics].median()

# ─────────────────────────────────────────────────────────────────────────────
# 6. MIGRACIONES (clustering por año)
# ─────────────────────────────────────────────────────────────────────────────
print("\nCalculando migraciones por año...")
year_clusters = {}
years_check = [2000, 2005, 2010, 2015, 2019]

for yr in years_check:
    sub_yr, X_yr = prepare_year(df, yr, scaler)
    labs = km_final.predict(X_yr)
    sub_yr["cluster"] = labs
    year_clusters[yr] = dict(zip(sub_yr["country"], sub_yr["cluster"]))

common = set(year_clusters[2000]) & set(year_clusters[2019])
movers = []
for country in common:
    c2000 = year_clusters[2000][country]
    c2019 = year_clusters[2019][country]
    if c2000 != c2019:
        movers.append({
            "country": country,
            "cluster_2000": c2000,
            "cluster_2019": c2019,
            "from_name": NAMES[c2000],
            "to_name": NAMES[c2019],
        })

movers_df = pd.DataFrame(movers).sort_values(["cluster_2000", "cluster_2019"])
print(f"Países que migraron: {len(movers_df)}")
print(movers_df[["country", "from_name", "to_name"]].to_string(index=False))

cluster_sizes = {
    yr: {c: list(year_clusters[yr].values()).count(c) for c in range(K)}
    for yr in years_check
}

# ─────────────────────────────────────────────────────────────────────────────
# 7. EXPORTAR CSVs
# ─────────────────────────────────────────────────────────────────────────────
cluster_export = base[["country", "cluster"] + FEATURES].copy()
cluster_export["cluster_name"] = cluster_export["cluster"].map(NAMES)
cluster_export.to_csv(OUT_CSV, index=False)
print(f"\n✓ Asignaciones guardadas: {OUT_CSV}")

movers_df.to_csv(OUT_MIG, index=False)
print(f"✓ Log de migraciones: {OUT_MIG}")

# ─────────────────────────────────────────────────────────────────────────────
# 8. PLOTS
# ─────────────────────────────────────────────────────────────────────────────
print(f"\nGenerando informe → {OUT_PDF}")
legend_patches = [
    mpatches.Patch(color=COLORS[c], label=f"Cluster {c}: {NAMES[c]}")
    for c in range(K)
]

with PdfPages(OUT_PDF) as pdf:

    # ── Fig 1: Elbow + Silhouette ────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Selección de k óptimo", fontsize=13, fontweight="bold")
    axes[0].plot(list(ks), inertias, marker="o", color="#378ADD", linewidth=2)
    axes[0].axvline(K, color="#D85A30", linestyle="--", alpha=0.7, label=f"k={K} seleccionado")
    axes[0].set_xlabel("k"); axes[0].set_title("Inercia (método del codo)")
    axes[0].legend()
    axes[1].plot(list(ks), silhouettes, marker="s", color="#1D9E75", linewidth=2)
    axes[1].axvline(K, color="#D85A30", linestyle="--", alpha=0.7, label=f"k={K} seleccionado")
    axes[1].set_xlabel("k"); axes[1].set_title("Silhouette score")
    axes[1].legend()
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight"); plt.close()
    print("  ✓ Fig 1: Elección de k")

    # ── Fig 2: Scatter PCA ───────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 8))
    for c in range(K):
        sub = base[base["cluster"] == c]
        ax.scatter(sub["pca1"], sub["pca2"],
                   color=COLORS[c], alpha=0.7, s=50,
                   label=f"{NAMES[c]} ({len(sub)})", zorder=3)
        # Etiquetas países notables
        notable = {
            0: ["Argentina", "China", "India", "Mexico", "Saudi Arabia"],
            1: ["United States", "Germany", "Qatar", "Norway", "Iceland"],
            2: ["Brazil", "Spain", "Costa Rica", "Nepal", "Uruguay"],
            3: ["Nigeria", "Ethiopia", "Chad", "South Sudan"],
        }
        for _, row in sub[sub["country"].isin(notable[c])].iterrows():
            ax.annotate(row["country"], (row["pca1"], row["pca2"]),
                        textcoords="offset points", xytext=(5, 4),
                        fontsize=7.5, color=COLORS[c], alpha=0.85)

    var1 = pca.explained_variance_ratio_[0]
    var2 = pca.explained_variance_ratio_[1]
    ax.set_xlabel(f"Componente principal 1 ({var1:.1%} varianza)")
    ax.set_ylabel(f"Componente principal 2 ({var2:.1%} varianza)")
    ax.set_title("Clustering de países por perfil energético (PCA 2D, datos 2019)", fontsize=13)
    ax.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight"); plt.close()
    print("  ✓ Fig 2: Scatter PCA")

    # ── Fig 3: Perfiles de cluster (barras normalizadas) ─────────────────────
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle("Perfil promedio por cluster — variables originales", fontsize=13, fontweight="bold")
    axes = axes.flatten()
    for i, (feat, label) in enumerate(FEATURE_LABELS.items()):
        vals = [centers_df.loc[c, feat] for c in range(K)]
        bars = axes[i].bar(range(K), vals,
                           color=[COLORS[c] for c in range(K)],
                           edgecolor="white", linewidth=0.5)
        axes[i].set_title(label, fontsize=10)
        axes[i].set_xticks(range(K))
        axes[i].set_xticklabels([f"C{c}" for c in range(K)], fontsize=9)
        for bar, v in zip(bars, vals):
            axes[i].text(bar.get_x() + bar.get_width()/2,
                         bar.get_height() * 1.02,
                         f"{v:.0f}", ha="center", va="bottom", fontsize=8)
        axes[i].grid(axis="y")
        axes[i].spines["bottom"].set_visible(False)

    fig.legend(handles=legend_patches, loc="lower center",
               ncol=2, fontsize=9, bbox_to_anchor=(0.5, -0.01))
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight"); plt.close()
    print("  ✓ Fig 3: Perfiles de cluster")

    # ── Fig 4: Evolución temporal — 4 métricas ───────────────────────────────
    metric_titles = {
        "access_electricity_pct":  ("Acceso a electricidad (%)", "%"),
        "low_carbon_elec_pct":     ("Low-carbon electricity (%)", "%"),
        "renewable_share_pct":     ("Renovables share (%)", "%"),
        "gdp_per_capita_usd":      ("GDP per cápita (USD)", "$"),
    }
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Evolución de indicadores por cluster (mediana, 2000–2019)", fontsize=13, fontweight="bold")
    axes = axes.flatten()
    for i, (metric, (title, unit)) in enumerate(metric_titles.items()):
        ax = axes[i]
        for c in range(K):
            series = trends[c][metric].reindex(years_range)
            ls = "--" if c == 1 else "-"
            ax.plot(years_range, series.values,
                    color=COLORS[c], linewidth=2, linestyle=ls,
                    label=NAMES[c])
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Año")
        if unit == "%":
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=100, decimals=0))
        else:
            ax.yaxis.set_major_formatter(mtick.FuncFormatter(
                lambda v, _: f"${v/1000:.0f}k" if v >= 1000 else f"${v:.0f}"))
        ax.set_xlim(2000, 2019)

    fig.legend(handles=legend_patches, loc="lower center",
               ncol=2, fontsize=9, bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight"); plt.close()
    print("  ✓ Fig 4: Evolución temporal")

    # ── Fig 5: Tamaño de clusters por año (stacked bar) ──────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    bottom = np.zeros(len(years_check))
    for c in [3, 0, 2, 1]:
        vals = np.array([cluster_sizes[yr][c] for yr in years_check])
        ax.bar(range(len(years_check)), vals, bottom=bottom,
               color=COLORS[c], label=NAMES[c],
               edgecolor="white", linewidth=0.5)
        bottom += vals
    ax.set_xticks(range(len(years_check)))
    ax.set_xticklabels(years_check)
    ax.set_ylabel("Número de países")
    ax.set_title("Distribución de países por cluster a lo largo del tiempo", fontsize=12)
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight"); plt.close()
    print("  ✓ Fig 5: Tamaño de clusters")

    # ── Fig 6: Migraciones (matriz de flujo) ─────────────────────────────────
    migration_matrix = pd.DataFrame(0, index=range(K), columns=range(K))
    for _, row in movers_df.iterrows():
        migration_matrix.loc[row["cluster_2000"], row["cluster_2019"]] += 1

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(migration_matrix.values, annot=True, fmt="d",
                cmap="Blues", ax=ax,
                xticklabels=[f"→ {NAMES[c]}" for c in range(K)],
                yticklabels=[f"Desde {NAMES[c]}" for c in range(K)],
                linewidths=0.5, cbar_kws={"label": "países"})
    ax.set_title("Matriz de migraciones entre clusters (2000 → 2019)", fontsize=12)
    plt.xticks(rotation=25, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight"); plt.close()
    print("  ✓ Fig 6: Matriz de migraciones")

print(f"\n✓ Reporte PDF guardado: {OUT_PDF}")

# ─────────────────────────────────────────────────────────────────────────────
# 9. RESUMEN EN CONSOLA
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Hallazgos del clustering ──────────────────────────────────────")
for c in range(K):
    countries = base[base["cluster"] == c]["country"].tolist()
    print(f"\nCluster {c} — {NAMES[c]} ({len(countries)} países)")
    print(f"  Acceso electricidad : {centers_df.loc[c,'access_electricity_pct']:.1f}%")
    print(f"  Renovables share    : {centers_df.loc[c,'renewable_share_pct']:.1f}%")
    print(f"  Low-carbon elec     : {centers_df.loc[c,'low_carbon_elec_pct']:.1f}%")
    print(f"  GDP per cápita      : ${centers_df.loc[c,'gdp_per_capita_usd']:.0f}")
    print(f"  Países: {', '.join(sorted(countries)[:8])}...")

print(f"\n── Migraciones 2000 → 2019 ───────────────────────────────────────")
print(f"  Total: {len(movers_df)} países cambiaron de cluster")
for _, row in movers_df.iterrows():
    arrow = "↑" if row["cluster_2000"] > row["cluster_2019"] else ("↓" if row["cluster_2000"] == 1 and row["cluster_2019"] == 0 else "↑")
    print(f"  {row['country']:30s}  {row['from_name']} → {row['to_name']}")

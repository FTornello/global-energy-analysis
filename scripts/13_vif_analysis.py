"""
13_vif_analysis.py
==================
Análisis de multicolinealidad mediante VIF (Variance Inflation Factor)
y auditoría del salto R² Linear vs Ridge reportado en Etapa 8.

Hallazgos:
  1. Multicolinealidad severa confirmada:
     - log_energy y log_gdp: VIF > 300 (r=0.914 entre sí)
     - access_electricity_pct y acc_2000: VIF > 29 (r=0.867 entre sí)
  2. El salto R²=0.33→0.82 reportado era un bug de implementación
     (escalado o features distintas entre modelos).
     Con mismo escalado y mismas features: Linear=0.823, Ridge=0.824, Δ=+0.001
  3. La multicolinealidad NO explica la diferencia — los dos son problemas distintos.

Uso:
    python3 13_vif_analysis.py

Input:
    - sustainable_energy_clean.csv

Output:
    - vif_report.pdf
    - vif_results.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
from matplotlib.backends.backend_pdf import PdfPages
import warnings
warnings.filterwarnings("ignore")

INPUT    = "sustainable_energy_clean.csv"
OUT_PDF  = "vif_report.pdf"
OUT_CSV  = "vif_results.csv"

BLUE, GREEN, RED, AMBER = "#378ADD", "#1D9E75", "#D85A30", "#BA7517"
GRAY = "#888780"

plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "figure.dpi": 120,
})

# ─────────────────────────────────────────────────────────────────────────────
# PREPARACIÓN DE DATOS
# ─────────────────────────────────────────────────────────────────────────────
print("Cargando datos...")
orig = pd.read_csv(INPUT)
GDP_MAX = np.log1p(orig["gdp_per_capita_usd"].max())
GDP_MIN = np.log1p(orig["gdp_per_capita_usd"].min())

df_2019 = orig[orig["year"] == 2019].copy()
df_2000 = orig[orig["year"] == 2000][
    ["country","access_electricity_pct","low_carbon_elec_pct"]
].rename(columns={"access_electricity_pct":"acc_2000",
                  "low_carbon_elec_pct":"lc_2000"})

train = df_2019.merge(df_2000, on="country", how="left")
train["log_gdp"]      = np.log1p(train["gdp_per_capita_usd"])
train["log_energy"]   = np.log1p(train["primary_energy_per_capita_kwh"])
train["gdp_log_norm"] = ((train["log_gdp"] - GDP_MIN) /
                          (GDP_MAX - GDP_MIN) * 100).clip(0, 100)
train["transition_score"] = (
    train["access_electricity_pct"].fillna(50) * 0.30 +
    train["low_carbon_elec_pct"].fillna(30)    * 0.45 +
    train["gdp_log_norm"]                       * 0.25
).round(2)

FEATURES = ["access_electricity_pct", "renewable_share_pct", "log_gdp",
            "energy_intensity_mj_gdp", "log_energy", "gdp_growth_pct",
            "acc_2000", "lc_2000"]

FEAT_LABELS = {
    "access_electricity_pct":  "Acceso electricidad 2019",
    "renewable_share_pct":     "Renovables share",
    "log_gdp":                 "GDP per cápita (log)",
    "energy_intensity_mj_gdp": "Intensidad energética",
    "log_energy":              "Energía per cápita (log)",
    "gdp_growth_pct":          "Crecimiento GDP",
    "acc_2000":                "Acceso electricidad 2000",
    "lc_2000":                 "Low-carbon electricity 2000",
}

train_clean = train[FEATURES + ["transition_score"]].dropna()
X_df = train_clean[FEATURES]
X    = X_df.values
y    = train_clean["transition_score"].values
n    = len(X_df)
print(f"Muestra: {n} países")

# ─────────────────────────────────────────────────────────────────────────────
# VIF
# ─────────────────────────────────────────────────────────────────────────────
print("\n── VIF ───────────────────────────────────────────────────────")
vif = pd.DataFrame({
    "feature": FEATURES,
    "label":   [FEAT_LABELS[f] for f in FEATURES],
    "VIF":     [variance_inflation_factor(X, i) for i in range(X.shape[1])]
}).sort_values("VIF", ascending=False)

print(f"{'Feature':<30} {'VIF':>8}  Nivel")
print("-" * 52)
for _, row in vif.iterrows():
    level = "🔴 SEVERO" if row.VIF > 10 else ("🟡 MODERADO" if row.VIF > 5 else "🟢 BAJO")
    print(f"  {row.feature:<28} {row.VIF:>8.1f}  {level}")

print(f"\nFeatures con VIF > 10: {(vif['VIF'] > 10).sum()} / {len(vif)}")
print(f"Features con VIF > 5:  {(vif['VIF'] > 5).sum()} / {len(vif)}")

# ─────────────────────────────────────────────────────────────────────────────
# NÚMERO DE CONDICIÓN
# ─────────────────────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_sc = scaler.fit_transform(X)
kappa = np.linalg.cond(X_sc.T @ X_sc)
print(f"\n── Número de condición (κ) ───────────────────────────────────")
print(f"  κ = {kappa:.1f}")
print(f"  < 100: bajo | 100–1000: moderado | > 1000: severo")
print(f"  → {'Moderado' if kappa < 1000 else 'Severo'} — consistente con VIF")

# ─────────────────────────────────────────────────────────────────────────────
# CORRELACIONES ENTRE PARES SOSPECHOSOS
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Correlaciones entre features de alto VIF ─────────────────")
pairs = [
    ("log_gdp",                "log_energy"),
    ("access_electricity_pct", "acc_2000"),
    ("log_gdp",                "access_electricity_pct"),
    ("log_energy",             "access_electricity_pct"),
]
for f1, f2 in pairs:
    r = X_df[f1].corr(X_df[f2])
    print(f"  {f1} ↔ {f2}: r = {r:.3f}")

# ─────────────────────────────────────────────────────────────────────────────
# AUDITORÍA LINEAR vs RIDGE
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Auditoría R² Linear vs Ridge ──────────────────────────────")
lr = LinearRegression()
rg = Ridge(alpha=2.0)

# Con todas las features, mismo escalado
r2_lr_full = cross_val_score(lr, X_sc, y, cv=5, scoring="r2").mean()
r2_rg_full = cross_val_score(rg, X_sc, y, cv=5, scoring="r2").mean()
print(f"  Todas las features (escaladas):")
print(f"    Linear: {r2_lr_full:.3f}  |  Ridge: {r2_rg_full:.3f}  |  Δ = {r2_rg_full-r2_lr_full:+.4f}")

# Sin features de alto VIF
high_vif = vif[vif["VIF"] > 10]["feature"].tolist()
feats_red = [f for f in FEATURES if f not in high_vif]
X_red_sc  = StandardScaler().fit_transform(X_df[feats_red].values)
r2_lr_red = cross_val_score(lr, X_red_sc, y, cv=5, scoring="r2").mean()
r2_rg_red = cross_val_score(rg, X_red_sc, y, cv=5, scoring="r2").mean()
print(f"\n  Sin features VIF > 10 ({high_vif}):")
print(f"    Linear: {r2_lr_red:.3f}  |  Ridge: {r2_rg_red:.3f}  |  Δ = {r2_rg_red-r2_lr_red:+.4f}")

print(f"""
── Diagnóstico ───────────────────────────────────────────────
  El salto R²=0.33→0.82 reportado en Etapa 8 era un bug
  de implementación (distinto escalado o features entre modelos).
  
  Con mismo escalado: Linear={r2_lr_full:.3f}, Ridge={r2_rg_full:.3f}, Δ={r2_rg_full-r2_lr_full:+.4f}
  
  La multicolinealidad severa ES real:
    - log_gdp ↔ log_energy: r=0.914  (misma información)
    - access_2019 ↔ access_2000: r=0.867 (misma variable en 2 años)
  
  Pero la multicolinealidad NO explica el salto reportado.
  Son dos problemas distintos: uno es bug de código,
  el otro es una característica del dataset.
""")

# ─────────────────────────────────────────────────────────────────────────────
# EXPORTAR CSV
# ─────────────────────────────────────────────────────────────────────────────
vif_out = vif.copy()
vif_out["severity"] = vif_out["VIF"].apply(
    lambda v: "severo" if v > 10 else ("moderado" if v > 5 else "bajo"))
vif_out["r2_impact"] = "ver diagnóstico"
vif_out.to_csv(OUT_CSV, index=False)
print(f"✓ CSV: {OUT_CSV}")

# ─────────────────────────────────────────────────────────────────────────────
# GRÁFICOS
# ─────────────────────────────────────────────────────────────────────────────
with PdfPages(OUT_PDF) as pdf:

    # ── Fig 1: VIF barras + correlación matrix ────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Análisis de multicolinealidad — VIF y correlaciones",
                 fontsize=13, fontweight="bold")

    # Barras VIF
    colors_vif = [RED if v > 10 else (AMBER if v > 5 else GREEN)
                  for v in vif["VIF"]]
    labels_short = [FEAT_LABELS[f].replace(" ", "\n") for f in vif["feature"]]
    bars = axes[0].barh(labels_short, vif["VIF"],
                        color=colors_vif, edgecolor="none")
    axes[0].axvline(10, color=RED,   linestyle="--", alpha=0.6,
                    linewidth=1, label="VIF = 10 (severo)")
    axes[0].axvline(5,  color=AMBER, linestyle=":",  alpha=0.6,
                    linewidth=1, label="VIF = 5 (moderado)")
    for bar, v in zip(bars, vif["VIF"]):
        axes[0].text(v + 5, bar.get_y() + bar.get_height()/2,
                     f"{v:.0f}", va="center", fontsize=9)
    axes[0].set_xlabel("VIF")
    axes[0].set_title("Variance Inflation Factor por feature")
    axes[0].legend(fontsize=8)
    axes[0].invert_yaxis()

    # Heatmap de correlaciones
    corr_matrix = X_df.corr()
    short_names = [FEAT_LABELS[f][:12] for f in FEATURES]
    im = axes[1].imshow(corr_matrix.abs().values,
                        cmap="RdYlGn_r", vmin=0, vmax=1)
    axes[1].set_xticks(range(len(FEATURES)))
    axes[1].set_yticks(range(len(FEATURES)))
    axes[1].set_xticklabels(short_names, rotation=45, ha="right", fontsize=7)
    axes[1].set_yticklabels(short_names, fontsize=7)
    for i in range(len(FEATURES)):
        for j in range(len(FEATURES)):
            v = corr_matrix.iloc[i, j]
            axes[1].text(j, i, f"{abs(v):.2f}",
                         ha="center", va="center", fontsize=6.5,
                         color="white" if abs(v) > 0.7 else "black")
    plt.colorbar(im, ax=axes[1], shrink=0.8, label="|correlación|")
    axes[1].set_title("Correlación entre features (|r|)")
    axes[1].grid(False)

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight"); plt.close()
    print("  ✓ Fig 1: VIF + correlaciones")

    # ── Fig 2: Auditoría Linear vs Ridge ─────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Auditoría: R² Linear vs Ridge — diagnóstico del salto reportado",
                 fontsize=13, fontweight="bold")

    # Barras comparativas
    configs = [
        ("Reportado\nen Etapa 8\n(bug)", 0.331, 0.822),
        ("Corregido\n(mismo escalado,\nmismas features)",
         r2_lr_full, r2_rg_full),
        (f"Sin features\nVIF>10\n({len(feats_red)} features)",
         r2_lr_red, r2_rg_red),
    ]
    x_pos = np.arange(len(configs))
    w = 0.3
    for i, (label, r2_lr_val, r2_rg_val) in enumerate(configs):
        axes[0].bar(i - w/2, r2_lr_val, w, color=BLUE,
                    edgecolor="none", label="Linear" if i == 0 else "")
        axes[0].bar(i + w/2, r2_rg_val, w, color=GREEN,
                    edgecolor="none", label="Ridge" if i == 0 else "")
        axes[0].text(i - w/2, r2_lr_val + 0.01, f"{r2_lr_val:.3f}",
                     ha="center", fontsize=8)
        axes[0].text(i + w/2, r2_rg_val + 0.01, f"{r2_rg_val:.3f}",
                     ha="center", fontsize=8)
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels([c[0] for c in configs], fontsize=9)
    axes[0].set_ylabel("R² (CV 5-fold)")
    axes[0].set_title("R² Linear vs Ridge en tres escenarios")
    axes[0].set_ylim(0, 1.0)
    axes[0].legend(fontsize=9)

    # Scatter log_gdp vs log_energy (el par más correlacionado)
    axes[1].scatter(X_df["log_gdp"], X_df["log_energy"],
                    color=BLUE, alpha=0.5, s=25)
    z = np.polyfit(X_df["log_gdp"], X_df["log_energy"], 1)
    xline = np.linspace(X_df["log_gdp"].min(), X_df["log_gdp"].max(), 100)
    axes[1].plot(xline, np.poly1d(z)(xline),
                 color=RED, linewidth=1.5, linestyle="--",
                 label=f"r = {X_df['log_gdp'].corr(X_df['log_energy']):.3f}")
    axes[1].set_xlabel("GDP per cápita (log)")
    axes[1].set_ylabel("Energía per cápita (log)")
    axes[1].set_title("Par más correlacionado: log_gdp ↔ log_energy\n(VIF: 376 y 513 respectivamente)")
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight"); plt.close()
    print("  ✓ Fig 2: Auditoría Linear vs Ridge")

print(f"\n✓ PDF: {OUT_PDF}")

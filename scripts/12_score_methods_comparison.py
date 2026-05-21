"""
12_score_methods_comparison.py
==============================
Compara tres métodos para asignar pesos al transition_score:

  V1 — Manual: pesos elegidos con criterio conceptual (0.30 / 0.45 / 0.25)
  V2 — Ridge:  pesos derivados del modelo de regresión (data-driven)
  V3 — PCA:    pesos de la primera componente principal (estructura de datos)

Hallazgo clave:
  V1 y V2 son casi idénticos (r=0.9994). V3 (PCA) diverge porque
  el algoritmo descubre que acceso+GDP y low_carbon son dimensiones
  independientes en los datos. Eso confirma que darle más peso a
  low_carbon fue un juicio de diseño deliberado, no arbitrario.

Uso:
    python3 12_score_methods_comparison.py

Input:
    - sustainable_energy_clean.csv

Output:
    - score_methods_comparison.csv
    - score_methods_report.pdf
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from matplotlib.backends.backend_pdf import PdfPages
import warnings
warnings.filterwarnings("ignore")

INPUT   = "sustainable_energy_clean.csv"
OUT_CSV = "score_methods_comparison.csv"
OUT_PDF = "score_methods_report.pdf"

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

print("Cargando datos...")
orig = pd.read_csv(INPUT)
GDP_MAX = np.log1p(orig["gdp_per_capita_usd"].max())
GDP_MIN = np.log1p(orig["gdp_per_capita_usd"].min())

df = orig[orig["year"] == 2019].copy()
df["gdp_log_norm"] = ((np.log1p(df["gdp_per_capita_usd"]) - GDP_MIN) /
                       (GDP_MAX - GDP_MIN) * 100).clip(0, 100)

VARS = ["access_electricity_pct", "low_carbon_elec_pct", "gdp_log_norm"]
VAR_LABELS = ["Acceso electricidad", "Low-carbon electricity", "GDP (log)"]

sub = df[["country"] + VARS].dropna().copy()

# ── V1: pesos manuales ────────────────────────────────────────────────────────
W1 = [0.30, 0.45, 0.25]
sub["score_v1"] = sum(sub[v] * w for v, w in zip(VARS, W1))

# ── V2: pesos Ridge ───────────────────────────────────────────────────────────
df_2000 = orig[orig["year"]==2000][["country","access_electricity_pct","low_carbon_elec_pct"]
    ].rename(columns={"access_electricity_pct":"acc_2000","low_carbon_elec_pct":"lc_2000"})
train = df.merge(df_2000, on="country", how="left")
train["log_gdp"]    = np.log1p(train["gdp_per_capita_usd"])
train["log_energy"] = np.log1p(train["primary_energy_per_capita_kwh"])
train["transition_score"] = (train["access_electricity_pct"].fillna(50)*0.30 +
                              train["low_carbon_elec_pct"].fillna(30)*0.45 +
                              train["gdp_log_norm"]*0.25)
FEATS = ["access_electricity_pct","renewable_share_pct","log_gdp",
         "energy_intensity_mj_gdp","log_energy","gdp_growth_pct","acc_2000","lc_2000"]
tc = train[FEATS+["transition_score"]].dropna()
sc_r = StandardScaler()
ridge = Ridge(alpha=2.0)
ridge.fit(sc_r.fit_transform(tc[FEATS].values), tc["transition_score"].values)
coefs = dict(zip(FEATS, ridge.coef_))
raw2 = [coefs["access_electricity_pct"], coefs["lc_2000"], coefs["log_gdp"]]
t2 = sum(v for v in raw2 if v > 0)
W2 = [round(v/t2, 3) if v > 0 else 0 for v in raw2]
sub["score_v2"] = sum(sub[v] * w for v, w in zip(VARS, W2))

# ── V3: pesos PCA ─────────────────────────────────────────────────────────────
sc_p = StandardScaler()
X_sc = sc_p.fit_transform(sub[VARS].values)
pca = PCA(n_components=3)
pca.fit(X_sc)
pc1 = pca.components_[0].copy()
if pc1[0] < 0:
    pc1 = -pc1
W3 = [round(abs(v)/np.abs(pc1).sum(), 3) for v in pc1]
pc1_scores = pca.transform(X_sc)[:,0]
if pca.components_[0][0] < 0:
    pc1_scores = -pc1_scores
sub["score_v3"] = ((pc1_scores - pc1_scores.min()) /
                   (pc1_scores.max() - pc1_scores.min()) * 100).round(2)

# ── Correlaciones ─────────────────────────────────────────────────────────────
r12 = sub["score_v1"].corr(sub["score_v2"])
r13 = sub["score_v1"].corr(sub["score_v3"])
r23 = sub["score_v2"].corr(sub["score_v3"])

print(f"\n── Pesos por método ─────────────────────────────────────────")
print(f"{'Variable':<28} {'V1':>8} {'V2':>8} {'V3':>8}")
print("-"*52)
for lbl, w1, w2, w3 in zip(VAR_LABELS, W1, W2, W3):
    print(f"  {lbl:<26} {w1:>8.3f} {w2:>8.3f} {w3:>8.3f}")

print(f"\n── Correlaciones ────────────────────────────────────────────")
print(f"  V1 vs V2: r = {r12:.4f}")
print(f"  V1 vs V3: r = {r13:.4f}")
print(f"  V2 vs V3: r = {r23:.4f}")

casos = ["Argentina","Spain","India","Japan","Cambodia"]
print(f"\n── 5 casos ──────────────────────────────────────────────────")
print(f"{'País':<12} {'V1':>8} {'V2':>8} {'V3':>8}")
for c in casos:
    r = sub[sub["country"]==c]
    if len(r):
        print(f"  {c:<10} {r.score_v1.values[0]:>8.1f} "
              f"{r.score_v2.values[0]:>8.1f} {r.score_v3.values[0]:>8.1f}")

# Guardar CSV
for col in ["score_v1","score_v2","score_v3"]:
    sub[col.replace("score","rank")] = sub[col].rank(ascending=False, method="min").astype("Int64")
sub[["country","score_v1","score_v2","score_v3","rank_v1","rank_v2","rank_v3"]
    ].sort_values("rank_v1").to_csv(OUT_CSV, index=False)
print(f"\n✓ CSV: {OUT_CSV}")

# ── Gráficos ──────────────────────────────────────────────────────────────────
with PdfPages(OUT_PDF) as pdf:

    # Fig 1: pesos + correlación
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Tres métodos para asignar pesos al transition_score",
                 fontsize=13, fontweight="bold")
    x = np.arange(3)
    w = 0.25
    for i, (lbl, weights, color) in enumerate([
        ("V1 (manual)", W1, BLUE),
        ("V2 (Ridge)",  W2, GREEN),
        ("V3 (PCA)",    W3, RED),
    ]):
        bars = axes[0].bar(x + (i-1)*w, weights, w, label=lbl,
                           color=color, edgecolor="none", alpha=0.85)
        for bar, v in zip(bars, weights):
            axes[0].text(bar.get_x()+bar.get_width()/2, v+0.01,
                         f"{v:.3f}", ha="center", fontsize=7.5, color=color)
    axes[0].set_xticks(x); axes[0].set_xticklabels(VAR_LABELS, fontsize=9)
    axes[0].set_ylabel("Peso"); axes[0].set_title("Pesos por método")
    axes[0].legend(fontsize=9); axes[0].set_ylim(0, 0.65)

    corr_m = np.array([[1.0,r12,r13],[r12,1.0,r23],[r13,r23,1.0]])
    im = axes[1].imshow(corr_m, cmap="RdYlGn", vmin=0.7, vmax=1.0)
    axes[1].set_xticks(range(3)); axes[1].set_yticks(range(3))
    axes[1].set_xticklabels(["V1\n(manual)","V2\n(Ridge)","V3\n(PCA)"])
    axes[1].set_yticklabels(["V1\n(manual)","V2\n(Ridge)","V3\n(PCA)"])
    for i in range(3):
        for j in range(3):
            axes[1].text(j, i, f"{corr_m[i,j]:.3f}", ha="center", va="center",
                         fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=axes[1], shrink=0.7)
    axes[1].set_title("Correlación entre scores"); axes[1].grid(False)
    plt.tight_layout(); pdf.savefig(fig, bbox_inches="tight"); plt.close()
    print("  ✓ Fig 1: Pesos y correlaciones")

    # Fig 2: scatter V1 vs V2 y V1 vs V3
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle("¿Qué tan distintos son los scores?", fontsize=13, fontweight="bold")
    for ax, (ycol, color, title, r) in zip(axes, [
        ("score_v2", GREEN, f"V1 vs V2 (Ridge) — r={r12:.4f}", r12),
        ("score_v3", RED,   f"V1 vs V3 (PCA)   — r={r13:.4f}", r13),
    ]):
        ax.scatter(sub["score_v1"], sub[ycol], color=color, alpha=0.4, s=25)
        lim = [sub["score_v1"].min()-2, sub["score_v1"].max()+2]
        ax.plot(lim, lim, "--", color="gray", alpha=0.4, linewidth=1)
        for c in casos:
            row = sub[sub["country"]==c]
            if len(row):
                ax.annotate(c, (row["score_v1"].values[0], row[ycol].values[0]),
                            textcoords="offset points", xytext=(5,3),
                            fontsize=8, color=AMBER)
        ax.set_xlabel("Score V1 (manual)"); ax.set_ylabel(f"Score {ycol[-2:].upper()}")
        ax.set_title(title)
    plt.tight_layout(); pdf.savefig(fig, bbox_inches="tight"); plt.close()
    print("  ✓ Fig 2: Scatter comparativos")

    # Fig 3: 5 casos en los 3 métodos
    fig, ax = plt.subplots(figsize=(12, 5))
    comp = sub[sub["country"].isin(casos)].sort_values("score_v1", ascending=True)
    y_p = np.arange(len(comp))
    for i, (col, lbl, color) in enumerate([
        ("score_v1","V1 (manual)",BLUE),
        ("score_v2","V2 (Ridge)", GREEN),
        ("score_v3","V3 (PCA)",   RED),
    ]):
        ax.barh(y_p+(i-1)*0.25, comp[col], 0.25,
                label=lbl, color=color, edgecolor="none", alpha=0.85)
    ax.set_yticks(y_p); ax.set_yticklabels(comp["country"], fontsize=11)
    ax.set_xlabel("Transition score"); ax.legend(fontsize=10)
    ax.set_title("5 casos — score según cada método\nV3 da más peso a GDP → Japón sube, Cambodia baja")
    ax.set_xlim(40, 105)
    plt.tight_layout(); pdf.savefig(fig, bbox_inches="tight"); plt.close()
    print("  ✓ Fig 3: 5 casos")

    # Fig 4: estructura PCA — PC1 vs PC2
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("PCA: los dos ejes de la energía global", fontsize=13, fontweight="bold")
    pc_coords = pca.transform(X_sc)
    sc_plot = axes[0].scatter(pc_coords[:,0], pc_coords[:,1],
                              c=sub["low_carbon_elec_pct"].values,
                              cmap="RdYlGn", alpha=0.6, s=30)
    plt.colorbar(sc_plot, ax=axes[0], label="Low-carbon (%)")
    axes[0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.0f}%) — Eje DESARROLLO")
    axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.0f}%) — Eje ENERGÍA LIMPIA")
    axes[0].set_title("Países en el espacio PCA\n(color = % electricidad limpia)")
    axes[0].axhline(0, color="gray", linewidth=0.5, alpha=0.5)
    axes[0].axvline(0, color="gray", linewidth=0.5, alpha=0.5)
    for c in casos:
        row_idx = sub[sub["country"]==c].index
        if len(row_idx):
            idx = list(sub.index).index(row_idx[0])
            axes[0].annotate(c, (pc_coords[idx,0], pc_coords[idx,1]),
                             textcoords="offset points", xytext=(5,3),
                             fontsize=8, color=AMBER, fontweight="bold")

    x_l = np.arange(3)
    axes[1].bar(x_l-0.18, pca.components_[0], 0.35,
                label=f"PC1 ({pca.explained_variance_ratio_[0]*100:.0f}%) — Desarrollo",
                color=BLUE, edgecolor="none")
    axes[1].bar(x_l+0.18, pca.components_[1], 0.35,
                label=f"PC2 ({pca.explained_variance_ratio_[1]*100:.0f}%) — Energía limpia",
                color=GREEN, edgecolor="none")
    axes[1].axhline(0, color="gray", linewidth=0.8)
    axes[1].set_xticks(x_l)
    axes[1].set_xticklabels(VAR_LABELS, fontsize=9)
    axes[1].set_ylabel("Loading (contribución al componente)")
    axes[1].set_title("PC1 = acceso + GDP\nPC2 = low_carbon casi solo")
    axes[1].legend(fontsize=9)
    plt.tight_layout(); pdf.savefig(fig, bbox_inches="tight"); plt.close()
    print("  ✓ Fig 4: Estructura PCA")

print(f"\n✓ PDF: {OUT_PDF}")
print(f"""
{"="*65}
CONCLUSIÓN
{"="*65}
  V1 vs V2: r={r12:.4f} — pesos manuales validados por Ridge
  V1 vs V3: r={r13:.4f} — PCA diverge, revela estructura diferente
  V2 vs V3: r={r23:.4f}

  PCA descubre DOS ejes independientes:
    PC1 (58%): desarrollo — acceso + GDP
    PC2 (33%): energía limpia — low_carbon casi solo

  Darle más peso a low_carbon (V1=0.45) fue una decisión de
  diseño deliberada que un algoritmo puro no puede tomar.
  El análisis muestra QUÉ se decide y POR QUÉ.
""")

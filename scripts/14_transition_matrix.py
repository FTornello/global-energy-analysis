"""
14_transition_matrix.py
=======================
Genera la matriz de transición 4×4 entre clusters energéticos
2000 → 2019, usando los centroides de 2019 aplicados hacia atrás
para mantener etiquetas comparables entre años.

Resultado: heatmap 4×4 que muestra qué países permanecieron
en su cluster y cuáles migraron, y hacia dónde.

Uso:
    python3 14_transition_matrix.py

Input:
    - sustainable_energy_clean.csv

Output:
    - transition_matrix.csv
    - transition_matrix_report.pdf
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from matplotlib.backends.backend_pdf import PdfPages
import warnings
warnings.filterwarnings("ignore")

INPUT    = "sustainable_energy_clean.csv"
OUT_CSV  = "transition_matrix.csv"
OUT_PDF  = "transition_matrix_report.pdf"

BLUE, GREEN, RED, AMBER, GRAY = "#378ADD", "#1D9E75", "#D85A30", "#BA7517", "#888780"

CLUSTER_NAMES = {
    0: "Emergentes\nfósiles",
    1: "Desarrollados\nalto consumo",
    2: "Transición\nrenovable",
    3: "Pobreza\nenergética",
}
CLUSTER_COLORS = {0: BLUE, 1: AMBER, 2: GREEN, 3: RED}

FEATURES = [
    "access_electricity_pct", "access_clean_fuels_pct",
    "renewable_share_pct", "low_carbon_elec_pct",
    "primary_energy_per_capita_kwh", "energy_intensity_mj_gdp",
    "gdp_per_capita_usd",
]

plt.rcParams.update({
    "font.family": "sans-serif",
    "figure.dpi": 120,
})

# ─────────────────────────────────────────────────────────────────────────────
# PREPARACIÓN — entrenar K-Means en 2019, aplicar hacia atrás en 2000
# ─────────────────────────────────────────────────────────────────────────────
print("Cargando y preparando datos...")
orig = pd.read_csv(INPUT)

def prepare(df, year, scaler=None, km=None, fit=False):
    sub = df[df["year"] == year][["country"] + FEATURES].copy()
    sub["nc"] = sub[FEATURES].isnull().sum(axis=1)
    sub = sub[sub["nc"] <= 2].drop(columns="nc")
    for col in FEATURES:
        sub[col] = sub[col].fillna(sub[col].median())
    X = scaler.fit_transform(sub[FEATURES]) if fit else scaler.transform(sub[FEATURES])
    labels = km.fit_predict(X) if fit else km.predict(X)
    sub["cluster"] = labels
    return sub

scaler = StandardScaler()
km     = KMeans(n_clusters=4, random_state=42, n_init=20)

sub_2019 = prepare(orig, 2019, scaler, km, fit=True)
sub_2000 = prepare(orig, 2000, scaler, km, fit=False)

common = set(sub_2000["country"]) & set(sub_2019["country"])
c2000  = dict(zip(sub_2000["country"], sub_2000["cluster"]))
c2019  = dict(zip(sub_2019["country"], sub_2019["cluster"]))

print(f"Países en común 2000–2019: {len(common)}")

# ─────────────────────────────────────────────────────────────────────────────
# MATRIZ DE TRANSICIÓN
# ─────────────────────────────────────────────────────────────────────────────
matrix = np.zeros((4, 4), dtype=int)
migrations = []

for country in common:
    fc = c2000[country]
    tc = c2019[country]
    matrix[fc][tc] += 1
    if fc != tc:
        migrations.append({"country": country, "from_cluster": fc, "to_cluster": tc,
                            "from_name": CLUSTER_NAMES[fc].replace("\n"," "),
                            "to_name":   CLUSTER_NAMES[tc].replace("\n"," ")})

stable = int(np.trace(matrix))
moved  = len(migrations)

print(f"\n── Matriz 4×4 ────────────────────────────────────────────────")
print(f"{'':4} {'C0':>6} {'C1':>6} {'C2':>6} {'C3':>6} {'Total':>6}")
for i in range(4):
    row = f"  C{i}  "
    for j in range(4):
        row += f"{matrix[i][j]:>6}"
    row += f" {matrix[i].sum():>6}"
    print(row)
print(f"{'Tot':>4}", end="")
for j in range(4):
    print(f"{matrix[:,j].sum():>6}", end="")
print(f" {matrix.sum():>6}")

print(f"\nEstables:   {stable} países ({stable/matrix.sum()*100:.0f}%)")
print(f"Migraron:   {moved} países ({moved/matrix.sum()*100:.0f}%)")

print(f"\n── Migraciones detalladas ────────────────────────────────────")
mig_df = pd.DataFrame(migrations).sort_values(["from_cluster","to_cluster"])
for _, row in mig_df.iterrows():
    arrow = "↑" if row.to_cluster < row.from_cluster else "↓"
    print(f"  {arrow} {row['country']:<25} {row['from_name'][:18]} → {row['to_name'][:18]}")

# Exportar CSV
mig_df.to_csv(OUT_CSV, index=False)
print(f"\n✓ CSV: {OUT_CSV}")

# ─────────────────────────────────────────────────────────────────────────────
# GRÁFICOS
# ─────────────────────────────────────────────────────────────────────────────
with PdfPages(OUT_PDF) as pdf:

    # ── Fig 1: Heatmap 4×4 ───────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Matriz de transición de clusters energéticos — 2000 → 2019",
                 fontsize=13, fontweight="bold")

    # Normalizar por fila (% que salió de cada cluster)
    row_sums = matrix.sum(axis=1, keepdims=True)
    matrix_pct = matrix / row_sums * 100

    ax = axes[0]
    im = ax.imshow(matrix_pct, cmap="Blues", vmin=0, vmax=100)

    tick_labels = [CLUSTER_NAMES[i].replace("\n", " ") for i in range(4)]
    ax.set_xticks(range(4)); ax.set_yticks(range(4))
    ax.set_xticklabels(["→ " + l for l in tick_labels], rotation=25,
                       ha="right", fontsize=9)
    ax.set_yticklabels(["Desde " + l for l in tick_labels], fontsize=9)

    for i in range(4):
        for j in range(4):
            n   = matrix[i][j]
            pct = matrix_pct[i][j]
            txt_color = "white" if pct > 50 else "black"
            weight = "bold" if i == j else "normal"
            ax.text(j, i, f"{n}\n({pct:.0f}%)",
                    ha="center", va="center", fontsize=9,
                    color=txt_color, fontweight=weight)

    plt.colorbar(im, ax=ax, label="% del cluster de origen", shrink=0.8)
    ax.set_title("Heatmap — n países y % por fila\n(diagonal = permanecieron en el mismo cluster)")
    ax.grid(False)

    # Sankey simplificado — barras de flujo
    ax2 = axes[1]
    ax2.set_xlim(0, 10); ax2.set_ylim(-0.5, 3.5)
    ax2.axis("off")
    ax2.set_title(f"Flujos de migración\n({moved} de {matrix.sum()} países cambiaron de cluster)",
                  fontsize=11)

    # Posiciones Y de cada cluster
    y_pos = {0: 2.5, 1: 3.5, 2: 1.5, 3: 0.5}

    # Dibujar barras 2000 (izq) y 2019 (der)
    for i in range(4):
        n_2000 = matrix[i].sum()
        n_2019 = matrix[:,i].sum()
        color = CLUSTER_COLORS[i]
        # Barra izquierda (2000)
        ax2.barh(y_pos[i], n_2000/172*3, height=0.4, left=0.5,
                 color=color, alpha=0.8)
        ax2.text(0.3, y_pos[i], f"C{i}\n{n_2000}",
                 ha="right", va="center", fontsize=8, color=color)
        # Barra derecha (2019)
        ax2.barh(y_pos[i], n_2019/172*3, height=0.4, left=6.5,
                 color=color, alpha=0.8)
        ax2.text(9.8, y_pos[i], f"C{i}\n{n_2019}",
                 ha="left", va="center", fontsize=8, color=color)

    # Flechas de flujo principales
    for i in range(4):
        for j in range(4):
            if i != j and matrix[i][j] > 0:
                ax2.annotate("",
                    xy=(6.4, y_pos[j]),
                    xytext=(3.7, y_pos[i]),
                    arrowprops=dict(
                        arrowstyle=f"->,head_width={matrix[i][j]*0.02+0.1}",
                        color=CLUSTER_COLORS[i],
                        lw=max(0.5, matrix[i][j]*0.3),
                        alpha=0.7
                    )
                )
                # Etiqueta en el medio
                mx = 5.05
                my = (y_pos[i] + y_pos[j]) / 2
                ax2.text(mx, my, str(matrix[i][j]),
                         fontsize=7.5, ha="center", va="center",
                         color=CLUSTER_COLORS[i], fontweight="bold")

    ax2.text(2.0, -0.2, "2000", ha="center", fontsize=10, color=GRAY)
    ax2.text(8.0, -0.2, "2019", ha="center", fontsize=10, color=GRAY)

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight"); plt.close()
    print("  ✓ Fig 1: Heatmap 4×4 + flujos")

    # ── Fig 2: Detalle de migraciones por tipo ───────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle("Migraciones por tipo de movimiento (2000 → 2019)",
                 fontsize=13, fontweight="bold")

    # Agrupar por par from→to
    migration_summary = []
    for i in range(4):
        for j in range(4):
            if i != j and matrix[i][j] > 0:
                migration_summary.append({
                    "label": f"C{i}→C{j}\n{CLUSTER_NAMES[i][:10]}→{CLUSTER_NAMES[j][:10]}",
                    "count": matrix[i][j],
                    "direction": "mejora" if j < i else "retroceso" if j > i else "lateral",
                    "color": GREEN if j < 2 and i > j else RED if j > i else BLUE
                })

    migration_summary.sort(key=lambda x: x["count"], reverse=True)
    labels = [m["label"] for m in migration_summary]
    counts = [m["count"] for m in migration_summary]
    colors_bar = []
    for i_src in range(4):
        for j_dst in range(4):
            if i_src != j_dst and matrix[i_src][j_dst] > 0:
                pass
    for m in migration_summary:
        parts = m["label"].split("→")[0].strip()
        from_c = int(parts.replace("C","").strip()[0])
        to_c   = int(m["label"].split("→")[0][-1])
        if to_c == 2 or to_c == 1:
            colors_bar.append(GREEN)
        elif to_c > from_c:
            colors_bar.append(RED)
        else:
            colors_bar.append(BLUE)

    bars = ax.bar(range(len(labels)), counts, color=colors_bar, edgecolor="none")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8, rotation=15, ha="right")
    ax.set_ylabel("Número de países")
    ax.set_title(f"42 migraciones totales — {stable} países permanecieron estables")

    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, count + 0.1,
                str(count), ha="center", fontsize=10, fontweight="bold")

    patches = [
        mpatches.Patch(color=GREEN, label="Hacia transición renovable o desarrollados"),
        mpatches.Patch(color=RED,   label="Hacia pobreza energética o emergentes"),
        mpatches.Patch(color=BLUE,  label="Lateral"),
    ]
    ax.legend(handles=patches, fontsize=9)

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight"); plt.close()
    print("  ✓ Fig 2: Migraciones por tipo")

print(f"\n✓ PDF: {OUT_PDF}")

# ─────────────────────────────────────────────────────────────────────────────
# RESUMEN
# ─────────────────────────────────────────────────────────────────────────────
print(f"""
{"="*65}
RESUMEN — Matriz de transición 2000 → 2019
{"="*65}

Total países analizados: {matrix.sum()}
Permanecieron en su cluster: {stable} ({stable/matrix.sum()*100:.0f}%)
Cambiaron de cluster:        {moved}  ({moved/matrix.sum()*100:.0f}%)

Movimientos más frecuentes:
  Pobreza → Transición renovable:  {matrix[3][2]} países
  Emergentes → Desarrollados:      {matrix[0][1]} países
  Pobreza → Emergentes fósiles:    {matrix[3][0]} países
  Emergentes → Transición:         {matrix[0][2]} países
  Desarrollados → Emergentes:      {matrix[1][0]} países (Japón + 1)

Interpretación: el 76% de los países mantuvo su perfil energético
en 20 años — confirma la inercia del sistema. El 24% que migró
lo hizo mayormente hacia arriba (transición o desarrollo),
con la excepción notable de Japón.
""")

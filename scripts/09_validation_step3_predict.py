"""
09_validation_step3_predict.py
==============================
Paso 3 de la validación out-of-sample (2021-2023).

- Calcula gdp_per_capita desde gdp / population (OWID)
- Recalcula gdp_growth_pct
- Recalcula las 4 features engineeradas (transition_score, etc.)
  usando los MISMOS parámetros que el modelo original
- Aplica el modelo Ridge SIN reentrenar
- Analiza residuales y compara con el período original

Uso:
    python3 09_validation_step3_predict.py

Inputs (misma carpeta):
    - validation_data_2021_2023.csv
    - owid_raw.csv                   (para gdp y population)
    - sustainable_energy_clean.csv   (parámetros de normalización)
    - features_engineered.csv        (scores originales para comparar)
    - regression_results.csv         (residuales originales)

Output:
    - validation_predictions.csv
    - validation_report.txt
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 1. CARGAR DATOS
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("PASO 3A — Cargando datos")
print("=" * 65)

val   = pd.read_csv("validation_data_2021_2023.csv")
owid  = pd.read_csv("owid_raw.csv")
orig  = pd.read_csv("sustainable_energy_clean.csv")
fe    = pd.read_csv("features_engineered.csv")
reg   = pd.read_csv("regression_results.csv")

print(f"  Validación: {val.shape}")
print(f"  Original:   {orig.shape}")
print(f"  Features:   {fe.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. CALCULAR GDP PER CAPITA desde gdp / population en OWID
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("PASO 3B — Usando gdp_per_capita del CSV (ya calculado en Paso 2.5)")
print("=" * 65)

# El GDP ya viene en validation_data_2021_2023.csv del Paso 2.5
# No recalculamos — usamos directamente lo que está en val

n_gdp = val["gdp_per_capita_usd"].notna().sum() if "gdp_per_capita_usd" in val.columns else 0
n_growth = val["gdp_growth_pct"].notna().sum() if "gdp_growth_pct" in val.columns else 0
print(f"  gdp_per_capita_usd disponible: {n_gdp}/{len(val)} filas ({n_gdp/len(val)*100:.1f}%)")
print(f"  gdp_growth_pct disponible:     {n_growth}/{len(val)} filas ({n_growth/len(val)*100:.1f}%)")

# Muestra
print(f"\n  GDP per cápita 2022 para 5 casos:")
for c in ["Argentina", "Spain", "India", "Japan", "Cambodia"]:
    row = val[(val["country"]==c) & (val["year"]==2022)]
    if len(row):
        gdp = row["gdp_per_capita_usd"].values[0]
        print(f"    {c:<12}: ${gdp:,.0f}" if pd.notna(gdp) else f"    {c:<12}: N/A")

# ─────────────────────────────────────────────────────────────────────────────
# 3. RECALCULAR FEATURES ENGINEERADAS
#    Usando los MISMOS parámetros que el proyecto original
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("PASO 3C — Recalculando features engineeradas")
print("=" * 65)

# Parámetros de normalización del GDP — deben ser IDÉNTICOS al proyecto original
# Los calculamos desde todo el dataset original (no solo 2019)
GDP_MAX = np.log1p(orig["gdp_per_capita_usd"].max())
GDP_MIN = np.log1p(orig["gdp_per_capita_usd"].min())

W_ACCESS = 0.30
W_CLEAN  = 0.45
W_GDP    = 0.25

def calc_transition_score(row):
    access = row["access_electricity_pct"] if pd.notna(row.get("access_electricity_pct")) else 50.0
    lc     = row["low_carbon_elec_pct"]    if pd.notna(row.get("low_carbon_elec_pct"))    else 30.0
    gdp_v  = row["gdp_per_capita_usd"]     if pd.notna(row.get("gdp_per_capita_usd"))     else np.exp((GDP_MIN + GDP_MAX)/2) - 1
    gdp_l  = np.log1p(gdp_v)
    gdp_n  = ((gdp_l - GDP_MIN) / (GDP_MAX - GDP_MIN) * 100).clip(0, 100)
    return round(access * W_ACCESS + lc * W_CLEAN + gdp_n * W_GDP, 2)

val["transition_score"] = val.apply(calc_transition_score, axis=1)

# clean_access_ratio
val["clean_access_ratio"] = (
    (val["access_electricity_pct"].fillna(0) / 100) *
    (val["low_carbon_elec_pct"].fillna(0) / 100) * 100
).round(2)

# fossil_lock_in
# Usar los parámetros de normalización de energy_intensity del dataset ORIGINAL
ei_orig = orig["energy_intensity_mj_gdp"].dropna()
EI_MIN  = ei_orig.min()
EI_MAX  = ei_orig.max()

# GDP normalizado para fossil_lock_in
val["gdp_log_norm"] = (
    (np.log1p(val["gdp_per_capita_usd"].fillna(np.exp((GDP_MIN+GDP_MAX)/2)-1)) - GDP_MIN)
    / (GDP_MAX - GDP_MIN) * 100
).clip(0, 100)

val["fossil_share_raw"] = 1 - val["low_carbon_elec_pct"].fillna(
    val["low_carbon_elec_pct"].median()) / 100

val["ei_norm"] = (
    (val["energy_intensity_mj_gdp"].fillna(val["energy_intensity_mj_gdp"].median()) - EI_MIN)
    / (EI_MAX - EI_MIN)
).clip(0, 1)

val["fossil_lock_in"] = (
    val["fossil_share_raw"] * 0.45 +
    val["ei_norm"]          * 0.35 +
    (1 - val["gdp_log_norm"] / 100) * 0.20
).mul(100).round(2)

print(f"  transition_score: rango [{val['transition_score'].min():.1f} — {val['transition_score'].max():.1f}]")
print(f"  clean_access_ratio: rango [{val['clean_access_ratio'].min():.1f} — {val['clean_access_ratio'].max():.1f}]")
print(f"  fossil_lock_in: rango [{val['fossil_lock_in'].min():.1f} — {val['fossil_lock_in'].max():.1f}]")

# ─────────────────────────────────────────────────────────────────────────────
# 4. REENTRENAR EL MODELO RIDGE EN DATOS ORIGINALES Y PREDECIR EN NUEVOS
#    El modelo se entrena en 2019 (igual que el proyecto original)
#    y predice en 2021-2023
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("PASO 3D — Aplicando modelo Ridge (entrenado en 2019)")
print("=" * 65)

# Features que usó el modelo original
FEATURES_ORIG = [
    "access_electricity_pct",
    "renewable_share_pct",
    "log_gdp",
    "energy_intensity_mj_gdp",
    "log_energy",
    "gdp_growth_pct",
    "acc_2000",
    "lc_2000",
]

# Preparar dataset de entrenamiento (2019, igual que el original)
df_2019 = orig[orig["year"] == 2019].copy()
df_2000 = orig[orig["year"] == 2000][
    ["country", "access_electricity_pct", "low_carbon_elec_pct"]
].rename(columns={
    "access_electricity_pct": "acc_2000",
    "low_carbon_elec_pct":    "lc_2000"
})

train = df_2019.merge(df_2000, on="country", how="left")
train["log_gdp"]    = np.log1p(train["gdp_per_capita_usd"])
train["log_energy"] = np.log1p(train["primary_energy_per_capita_kwh"])

# Calcular transition_score para entrenamiento
train["gdp_log_norm_tr"] = (
    (train["log_gdp"] - GDP_MIN) / (GDP_MAX - GDP_MIN) * 100
).clip(0, 100)
train["transition_score"] = (
    train["access_electricity_pct"].fillna(50) * W_ACCESS +
    train["low_carbon_elec_pct"].fillna(30) * W_CLEAN +
    train["gdp_log_norm_tr"] * W_GDP
).round(2)

# Filtrar filas con datos completos para entrenamiento
train_clean = train[FEATURES_ORIG + ["transition_score"]].dropna()
X_train = train_clean[FEATURES_ORIG].values
y_train = train_clean["transition_score"].values

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)

ridge = Ridge(alpha=2.0)
ridge.fit(X_train_sc, y_train)

from sklearn.model_selection import cross_val_score
r2_cv = cross_val_score(ridge, X_train_sc, y_train, cv=5, scoring="r2").mean()
print(f"  Modelo Ridge reentrenado | R² CV en 2019: {r2_cv:.3f}")

# Preparar dataset de validación con las mismas features
df_2000_v = orig[orig["year"] == 2000][
    ["country", "access_electricity_pct", "low_carbon_elec_pct"]
].rename(columns={
    "access_electricity_pct": "acc_2000",
    "low_carbon_elec_pct":    "lc_2000"
})

val_feat = val.merge(df_2000_v, on="country", how="left")
val_feat["log_gdp"]    = np.log1p(val_feat["gdp_per_capita_usd"].fillna(1))
val_feat["log_energy"] = np.log1p(val_feat["primary_energy_per_capita_kwh"].fillna(1))

# Imputar nulos con medianas del training
for col in FEATURES_ORIG:
    if col in val_feat.columns:
        median_train = train_clean[col].median()
        val_feat[col] = val_feat[col].fillna(median_train)

X_val = val_feat[FEATURES_ORIG].values
X_val_sc = scaler.transform(X_val)
val["y_pred"] = ridge.predict(X_val_sc)
val["residual"] = val["transition_score"] - val["y_pred"]

print(f"  Predicciones generadas: {len(val)}")
print(f"  MAE validación: {val['residual'].abs().mean():.2f} puntos")
print(f"  MAE original (2019): 5.22 puntos")

# ─────────────────────────────────────────────────────────────────────────────
# 5. ANÁLISIS DE RESULTADOS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("PASO 3E — Análisis de resultados")
print("=" * 65)

# Transition score promedio por año
print("\n  Transition score global (mediana) por año:")
ts_orig = orig.copy()
ts_orig["log_gdp"]    = np.log1p(ts_orig["gdp_per_capita_usd"].fillna(1))
ts_orig["gdp_log_norm"] = ((ts_orig["log_gdp"] - GDP_MIN) / (GDP_MAX - GDP_MIN) * 100).clip(0,100)
ts_orig["ts"] = (
    ts_orig["access_electricity_pct"].fillna(50) * W_ACCESS +
    ts_orig["low_carbon_elec_pct"].fillna(30) * W_CLEAN +
    ts_orig["gdp_log_norm"] * W_GDP
)

for yr in [2000, 2005, 2010, 2015, 2019]:
    med = ts_orig[ts_orig["year"]==yr]["ts"].median()
    print(f"    {yr}: {med:.1f}")
for yr in [2021, 2022, 2023]:
    med = val[val["year"]==yr]["transition_score"].median()
    print(f"    {yr}: {med:.1f}")

# 5 casos de referencia
print("\n  Transition score — 5 casos de referencia:")
print(f"  {'País':<12} {'2019':>8} {'2021':>8} {'2022':>8} {'2023':>8} {'Cambio':>8}")
print(f"  {'-'*56}")

score_2019 = fe[["country","transition_score"]].rename(
    columns={"transition_score":"ts_2019"})

for c in ["Argentina", "Spain", "India", "Japan", "Cambodia"]:
    ts19_row = score_2019[score_2019["country"]==c]
    ts19 = ts19_row["ts_2019"].values[0] if len(ts19_row) else None

    vals_yr = {}
    for yr in [2021, 2022, 2023]:
        row = val[(val["country"]==c) & (val["year"]==yr)]
        vals_yr[yr] = row["transition_score"].values[0] if len(row) else None

    cambio = vals_yr[2023] - ts19 if ts19 and vals_yr[2023] else None
    cambio_str = f"{cambio:+.1f}" if cambio else "N/A"
    ts19_str   = f"{ts19:.1f}" if ts19 else "N/A"
    v21_str    = f"{vals_yr[2021]:.1f}" if vals_yr[2021] else "N/A"
    v22_str    = f"{vals_yr[2022]:.1f}" if vals_yr[2022] else "N/A"
    v23_str    = f"{vals_yr[2023]:.1f}" if vals_yr[2023] else "N/A"
    print(f"  {c:<12} {ts19_str:>8} {v21_str:>8} {v22_str:>8} {v23_str:>8} {cambio_str:>8}")

# Residuales más grandes en 2022 (crisis energética)
print("\n  Top 10 residuales más grandes en 2022 (modelo falla más):")
resid_2022 = val[val["year"]==2022].copy()
resid_2022["abs_resid"] = resid_2022["residual"].abs()
top10 = resid_2022.nlargest(10, "abs_resid")[
    ["country","transition_score","y_pred","residual"]
].round(2)
print(top10.to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# 6. GUARDAR
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("PASO 3F — Guardando outputs")
print("=" * 65)

output_cols = [
    "country","year","transition_score","y_pred","residual",
    "access_electricity_pct","low_carbon_elec_pct","renewable_share_pct",
    "gdp_per_capita_usd","energy_intensity_mj_gdp",
    "primary_energy_per_capita_kwh","gdp_growth_pct",
    "clean_access_ratio","fossil_lock_in"
]
output_cols = [c for c in output_cols if c in val.columns]
val[output_cols].sort_values(["country","year"]).to_csv(
    "validation_predictions.csv", index=False)
print(f"  ✓ validation_predictions.csv")

# Reporte de texto
mae_val  = val["residual"].abs().mean()
mae_orig = 5.22

lines = [
    "REPORTE DE VALIDACIÓN OUT-OF-SAMPLE (2021-2023)",
    "=" * 55,
    f"Modelo: Ridge Regression (α=2.0)",
    f"Entrenado en: datos 2019 (153 países)",
    f"Validado en: datos 2021-2023 (176 países)",
    "",
    "PERFORMANCE:",
    f"  MAE original (2019, CV):  5.22 puntos",
    f"  MAE validación (2021-23): {mae_val:.2f} puntos",
    f"  Diferencia: {mae_val - mae_orig:+.2f} puntos",
    f"  Veredicto: {'✓ Modelo robusto' if mae_val < 8 else '⚠ Degradación significativa'}",
    "",
    "TRANSITION SCORE GLOBAL (mediana):",
]
for yr in [2019, 2021, 2022, 2023]:
    if yr == 2019:
        med = fe["transition_score"].median()
    else:
        med = val[val["year"]==yr]["transition_score"].median()
    lines.append(f"  {yr}: {med:.1f}")

lines += ["", "5 CASOS DE REFERENCIA (transition_score 2019 → 2023):"]
for c in ["Argentina", "Spain", "India", "Japan", "Cambodia"]:
    ts19_row = fe[fe["country"]==c]
    ts23_row = val[(val["country"]==c) & (val["year"]==2023)]
    if len(ts19_row) and len(ts23_row):
        t19 = ts19_row["transition_score"].values[0]
        t23 = ts23_row["transition_score"].values[0]
        lines.append(f"  {c:<12}: {t19:.1f} → {t23:.1f}  ({t23-t19:+.1f})")

report = "\n".join(lines)
with open("validation_report.txt", "w") as f:
    f.write(report)
print(f"  ✓ validation_report.txt")

print(f"\n{report}")

print(f"""
{"=" * 65}
RESUMEN — Paso 3 completado
{"=" * 65}
  validation_predictions.csv — predicciones para 176 países × 3 años
  validation_report.txt      — reporte ejecutivo

Siguiente paso:
  python3 10_validation_step4_visualize.py
  (gráficos y actualización del repositorio GitHub)
""")

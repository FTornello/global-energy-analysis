"""
08_validation_step2_align.py
============================
Paso 2 de la validación out-of-sample (2021-2023).
Versión 2 — detección automática de nombres de columnas OWID.
"""

import pandas as pd
import numpy as np
import json
import warnings
warnings.filterwarnings("ignore")

YEARS_NEW = [2021, 2022, 2023]

# ─────────────────────────────────────────────────────────────────────────────
# 1. CARGAR LOS TRES DATASETS
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("PASO 2A — Cargando datasets")
print("=" * 65)

owid = pd.read_csv("owid_raw.csv")
wb   = pd.read_csv("worldbank_access.csv")
orig = pd.read_csv("sustainable_energy_clean.csv")

print(f"  OWID:       {owid.shape[0]:,} filas × {owid.shape[1]} columnas")
print(f"  World Bank: {len(wb):,} filas × {wb.shape[1]} columnas")
print(f"  Original:   {len(orig):,} filas × {orig.shape[1]} columnas")

# ─────────────────────────────────────────────────────────────────────────────
# 2. DETECTAR NOMBRES REALES DE COLUMNAS EN OWID
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("PASO 2B — Detectando nombres de columnas disponibles en OWID")
print("=" * 65)

all_cols = owid.columns.tolist()

# Buscar columna de low-carbon electricity
def find_col(keywords, columns, exclude=[]):
    """Busca la primera columna que contenga todas las keywords."""
    for col in columns:
        col_lower = col.lower()
        if all(k in col_lower for k in keywords) and not any(e in col_lower for e in exclude):
            return col
    return None

# Detección automática por keywords
detected = {
    "low_carbon_elec_pct":          find_col(["low_carbon", "share", "elec"], all_cols),
    "renewable_share_pct":           find_col(["renew", "share", "elec"], all_cols, exclude=["low_carbon"]),
    "primary_energy_per_capita_kwh": find_col(["energy", "per_capita"], all_cols, exclude=["low_carbon","renew","fossil","gdp","demand"]),
    "energy_intensity_mj_gdp":       find_col(["energy", "gdp"], all_cols),
    "gdp_per_capita_usd":            find_col(["gdp", "per_capita"], all_cols),
}

print(f"\n  {'Variable modelo':<35} {'Columna detectada en OWID'}")
print(f"  {'-'*70}")
for var, col in detected.items():
    status = f"✓ {col}" if col else "✗ NO ENCONTRADA"
    print(f"  {var:<35} {status}")

# Fallback manual si la detección automática falla
FALLBACKS = {
    "primary_energy_per_capita_kwh": [
        "primary_energy_per_capita", "energy_per_capita",
        "primary_energy_cons_per_capita"
    ],
    "gdp_per_capita_usd": [
        "gdp_per_capita", "gdp__ppp__constant_2017_international",
        "gdp_per_capita__ppp"
    ],
}

for var, options in FALLBACKS.items():
    if detected[var] is None:
        for opt in options:
            if opt in all_cols:
                detected[var] = opt
                print(f"  → Fallback para {var}: '{opt}'")
                break

# Mostrar columnas disponibles para debug si algo falta
missing = [v for v, c in detected.items() if c is None]
if missing:
    print(f"\n  ⚠ No detectadas: {missing}")
    print(f"\n  Columnas OWID con 'gdp':")
    for c in all_cols:
        if "gdp" in c.lower(): print(f"    {c}")
    print(f"\n  Columnas OWID con 'energy' y 'capita':")
    for c in all_cols:
        if "energy" in c.lower() and "capita" in c.lower(): print(f"    {c}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. EXTRAER Y RENOMBRAR COLUMNAS DE OWID
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("PASO 2C — Preparando OWID (2021-2023)")
print("=" * 65)

base_cols = ["country", "year"]
if "iso_code" in all_cols:
    base_cols.append("iso_code")

# Solo columnas que existen
cols_to_take  = base_cols + [c for c in detected.values() if c and c not in base_cols]
rename_map    = {v: k for k, v in detected.items() if v}

owid_sub = (
    owid[owid["year"].isin(YEARS_NEW)][cols_to_take]
    .rename(columns=rename_map)
    .copy()
)

# Excluir regiones agregadas (sin iso_code)
if "iso_code" in owid_sub.columns:
    owid_sub = owid_sub[owid_sub["iso_code"].notna() & (owid_sub["iso_code"].str.len() == 3)]

print(f"  Filas: {len(owid_sub):,} | Países: {owid_sub['country'].nunique()}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. CALCULAR GDP GROWTH desde gdp_per_capita
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("PASO 2D — Calculando gdp_growth_pct")
print("=" * 65)

gdp_col_owid = detected.get("gdp_per_capita_usd")

if gdp_col_owid and gdp_col_owid in owid.columns:
    gdp_hist = (
        owid[owid["year"].isin([2020, 2021, 2022, 2023])]
        [["country", "year", gdp_col_owid]]
        .copy()
        .sort_values(["country", "year"])
    )
    gdp_hist["gdp_growth_pct"] = (
        gdp_hist.groupby("country")[gdp_col_owid]
        .pct_change() * 100
    )
    gdp_growth_new = gdp_hist[gdp_hist["year"].isin(YEARS_NEW)][
        ["country", "year", "gdp_growth_pct"]
    ]
    owid_sub = owid_sub.merge(gdp_growth_new, on=["country", "year"], how="left")
    n = owid_sub["gdp_growth_pct"].notna().sum()
    print(f"  gdp_growth_pct calculado para {n}/{len(owid_sub)} filas")
else:
    owid_sub["gdp_growth_pct"] = np.nan
    print("  ⚠ gdp_per_capita no disponible — gdp_growth_pct = NaN")

# ─────────────────────────────────────────────────────────────────────────────
# 5. MERGE CON WORLD BANK (acceso electricidad)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("PASO 2E — Merge con World Bank")
print("=" * 65)

wb_sub = wb[wb["year"].isin(YEARS_NEW)][
    ["country_code", "year", "access_electricity_pct"]
].rename(columns={"country_code": "iso_code"}).copy()

if "iso_code" in owid_sub.columns:
    owid_sub["iso_code"] = owid_sub["iso_code"].str.upper().str.strip()
    wb_sub["iso_code"]   = wb_sub["iso_code"].str.upper().str.strip()
    merged = owid_sub.merge(wb_sub, on=["iso_code", "year"], how="left")
else:
    # Merge por nombre de país si no hay iso_code
    wb_sub2 = wb[wb["year"].isin(YEARS_NEW)][
        ["country_wb", "year", "access_electricity_pct"]
    ].rename(columns={"country_wb": "country"})
    merged = owid_sub.merge(wb_sub2, on=["country", "year"], how="left")

n_con = merged["access_electricity_pct"].notna().sum()
n_sin = merged["access_electricity_pct"].isna().sum()
print(f"  Con acceso electricidad: {n_con} ({n_con/len(merged)*100:.1f}%)")
print(f"  Sin acceso electricidad: {n_sin} — imputando con valor 2019...")

# Imputar nulos con valor 2019 del dataset original
if n_sin > 0:
    acc_2019 = (
        orig[orig["year"] == 2019][["country", "access_electricity_pct"]]
        .rename(columns={"access_electricity_pct": "acc_proxy"})
    )
    merged = merged.merge(acc_2019, on="country", how="left")
    mask = merged["access_electricity_pct"].isna()
    merged.loc[mask, "access_electricity_pct"] = merged.loc[mask, "acc_proxy"]
    merged = merged.drop(columns=["acc_proxy"], errors="ignore")
    print(f"  Nulos restantes: {merged['access_electricity_pct'].isna().sum()}")

# ─────────────────────────────────────────────────────────────────────────────
# 6. FILTRAR PAÍSES DEL DATASET ORIGINAL
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("PASO 2F — Filtrar países del dataset original")
print("=" * 65)

paises_orig = set(orig["country"].unique())
final = merged[merged["country"].isin(paises_orig)].copy()
faltantes = paises_orig - set(final["country"].unique())

print(f"  Filas finales: {len(final):,} | Países: {final['country'].nunique()}")
if faltantes:
    print(f"  Sin datos nuevos ({len(faltantes)}): {sorted(faltantes)}")

# ─────────────────────────────────────────────────────────────────────────────
# 7. CALIDAD Y MUESTRA FINAL
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("PASO 2G — Calidad del dataset final")
print("=" * 65)

COLS_MODELO = [
    "access_electricity_pct", "low_carbon_elec_pct",
    "renewable_share_pct", "gdp_per_capita_usd",
    "energy_intensity_mj_gdp", "primary_energy_per_capita_kwh",
    "gdp_growth_pct",
]

print(f"\n  {'Columna':<35} {'Disponible':<12} {'% nulos'}")
print(f"  {'-'*58}")
for col in COLS_MODELO:
    if col in final.columns:
        pct = final[col].isna().mean() * 100
        print(f"  {col:<35} {'✓':<12} {pct:.1f}%")
    else:
        print(f"  {col:<35} {'✗ FALTA':<12}")

casos = ["Argentina", "Spain", "India", "Japan", "Cambodia"]
print(f"\n  Muestra 5 casos (2022):")
cols_vis = ["country","year","low_carbon_elec_pct","access_electricity_pct","gdp_per_capita_usd"]
cols_vis = [c for c in cols_vis if c in final.columns]
print(final[(final["country"].isin(casos)) & (final["year"]==2022)][cols_vis].to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# 8. GUARDAR
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("PASO 2H — Guardando outputs")
print("=" * 65)

col_order = ["country","year"] + [c for c in COLS_MODELO if c in final.columns]
extra = [c for c in final.columns if c not in col_order and c not in ["iso_code"]]
final_out = final[col_order + extra].sort_values(["country","year"])

final_out.to_csv("validation_data_2021_2023.csv", index=False)
print(f"  ✓ validation_data_2021_2023.csv  ({len(final_out):,} filas × {len(final_out.columns)} cols)")

log = {
    "rows": int(len(final_out)),
    "countries": int(final_out["country"].nunique()),
    "years": YEARS_NEW,
    "columns": list(final_out.columns),
    "detected_owid_columns": {k: v for k,v in detected.items()},
    "missing_countries": sorted(list(faltantes)),
}
with open("alignment_log.json","w",encoding="utf-8") as f:
    json.dump(log, f, ensure_ascii=False, indent=2)
print(f"  ✓ alignment_log.json")

print(f"""
{"=" * 65}
RESUMEN — Paso 2 completado
{"=" * 65}
  Filas:   {len(final_out):,}
  Países:  {final_out['country'].nunique()}
  Años:    {YEARS_NEW}

Siguiente paso:
  python3 09_validation_step3_predict.py
""")

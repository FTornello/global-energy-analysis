"""
08b_validation_step2b_gdp.py
============================
Paso 2.5 — Completar GDP per cápita desde World Bank API.

El dataset OWID no tiene gdp_per_capita directamente.
Calcularlo desde gdp/population solo cubre el 56% de los países.
Este script descarga NY.GDP.PCAP.CD del World Bank (cobertura ~95%)
y lo agrega al dataset de validación.

Uso:
    python3 08b_validation_step2b_gdp.py

Inputs:
    - validation_data_2021_2023.csv
    - owid_raw.csv
    - sustainable_energy_clean.csv

Output:
    - validation_data_2021_2023.csv  (actualizado con GDP completo)
    - worldbank_gdp.csv              (GDP raw del World Bank)
"""

import pandas as pd
import numpy as np
import requests
import json
import warnings
warnings.filterwarnings("ignore")

YEARS_NEW = [2021, 2022, 2023]

# ─────────────────────────────────────────────────────────────────────────────
# 1. CARGAR DATASET DE VALIDACIÓN ACTUAL
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("PASO 2.5A — Cargando dataset de validación actual")
print("=" * 65)

val  = pd.read_csv("validation_data_2021_2023.csv")
orig = pd.read_csv("sustainable_energy_clean.csv")
owid = pd.read_csv("owid_raw.csv")

# Estado actual del GDP
n_gdp_actual = val["gdp_per_capita_usd"].notna().sum() if "gdp_per_capita_usd" in val.columns else 0
print(f"  Filas totales:           {len(val)}")
print(f"  Con GDP actualmente:     {n_gdp_actual} ({n_gdp_actual/len(val)*100:.1f}%)")
print(f"  Sin GDP (a completar):   {len(val) - n_gdp_actual}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. CALCULAR GDP DESDE OWID (gdp / population) — primer intento
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("PASO 2.5B — Calculando GDP desde OWID (gdp / population)")
print("=" * 65)

gdp_owid = owid[owid["year"].isin(YEARS_NEW)].copy()

if "gdp" in gdp_owid.columns and "population" in gdp_owid.columns:
    gdp_owid["gdp_per_capita_owid"] = gdp_owid["gdp"] / gdp_owid["population"]
    gdp_owid = gdp_owid[["country", "year", "gdp_per_capita_owid"]].dropna()
    print(f"  Filas con GDP desde OWID: {len(gdp_owid)}")
    print(f"  Países únicos:            {gdp_owid['country'].nunique()}")
else:
    gdp_owid = pd.DataFrame(columns=["country", "year", "gdp_per_capita_owid"])
    print("  ⚠ Columnas gdp/population no disponibles en OWID")

# ─────────────────────────────────────────────────────────────────────────────
# 3. DESCARGAR GDP PER CÁPITA DESDE WORLD BANK API
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("PASO 2.5C — Descargando GDP per cápita (World Bank)")
print("=" * 65)

# Indicador NY.GDP.PCAP.CD = GDP per cápita en USD corrientes
WB_GDP_URL = (
    "https://api.worldbank.org/v2/country/all/indicator/NY.GDP.PCAP.CD"
    "?date=2018:2023&format=json&per_page=2000"
)

records_gdp = []
try:
    resp = requests.get(WB_GDP_URL, timeout=30)
    wb_data = resp.json()
    total_pages = wb_data[0]["pages"]
    print(f"  Páginas disponibles: {total_pages}")

    for page in range(1, total_pages + 1):
        if page == 1:
            items = wb_data[1]
        else:
            r = requests.get(WB_GDP_URL + f"&page={page}", timeout=30)
            items = r.json()[1]
        for item in items:
            if item["value"] is not None:
                records_gdp.append({
                    "country_wb":   item["country"]["value"],
                    "country_code": item["countryiso3code"],
                    "year":         int(item["date"]),
                    "gdp_per_capita_wb": round(float(item["value"]), 2)
                })

    wb_gdp = pd.DataFrame(records_gdp)
    wb_gdp.to_csv("worldbank_gdp.csv", index=False)

    print(f"  ✓ Descargado correctamente")
    print(f"  Registros totales: {len(wb_gdp):,}")
    print(f"  Países únicos:     {wb_gdp['country_wb'].nunique()}")
    print(f"  Años cubiertos:    {sorted(wb_gdp['year'].unique())}")
    print(f"  ✓ Guardado: worldbank_gdp.csv")

    wb_available = True

except Exception as e:
    print(f"  ✗ Error: {e}")
    wb_available = False
    wb_gdp = pd.DataFrame()

# ─────────────────────────────────────────────────────────────────────────────
# 4. MERGE WORLD BANK GDP CON ISO CODES DE OWID
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("PASO 2.5D — Mapeando países por ISO code")
print("=" * 65)

if wb_available and len(wb_gdp) > 0:
    # Obtener ISO codes de OWID para el merge
    iso_map = owid[["country", "iso_code"]].dropna().drop_duplicates()
    iso_map["iso_code"] = iso_map["iso_code"].str.upper().str.strip()

    wb_gdp_new = wb_gdp[wb_gdp["year"].isin(YEARS_NEW)].copy()
    wb_gdp_new["country_code"] = wb_gdp_new["country_code"].str.upper().str.strip()

    # Merge: WB por iso_code → nombre de país de OWID
    wb_gdp_mapped = wb_gdp_new.merge(
        iso_map,
        left_on="country_code",
        right_on="iso_code",
        how="inner"
    )[["country", "year", "gdp_per_capita_wb"]]

    print(f"  Países mapeados via ISO: {wb_gdp_mapped['country'].nunique()}")
    print(f"  Filas 2021-2023:         {len(wb_gdp_mapped)}")

    # Muestra para los 5 casos
    casos = ["Argentina", "Spain", "India", "Japan", "Cambodia"]
    print(f"\n  GDP per cápita (WB) para 5 casos — 2022:")
    for c in casos:
        row = wb_gdp_mapped[
            (wb_gdp_mapped["country"]==c) & (wb_gdp_mapped["year"]==2022)
        ]
        if len(row):
            print(f"    {c:<12}: ${row['gdp_per_capita_wb'].values[0]:,.0f}")
        else:
            print(f"    {c:<12}: sin dato")

# ─────────────────────────────────────────────────────────────────────────────
# 5. COMBINAR: OWID + WORLD BANK, priorizar WB (más cobertura)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("PASO 2.5E — Combinando fuentes de GDP")
print("=" * 65)

# Empezar desde el dataset de validación
if "gdp_per_capita_usd" in val.columns:
    val = val.drop(columns=["gdp_per_capita_usd"], errors="ignore")

# Merge con OWID gdp/population
val = val.merge(gdp_owid, on=["country", "year"], how="left")
val = val.rename(columns={"gdp_per_capita_owid": "gdp_per_capita_usd"})

# Para los que siguen sin GDP, completar con World Bank
if wb_available and len(wb_gdp_mapped) > 0:
    val = val.merge(wb_gdp_mapped, on=["country", "year"], how="left")
    mask_missing = val["gdp_per_capita_usd"].isna() & val["gdp_per_capita_wb"].notna()
    val.loc[mask_missing, "gdp_per_capita_usd"] = val.loc[mask_missing, "gdp_per_capita_wb"]
    val = val.drop(columns=["gdp_per_capita_wb"], errors="ignore")

    n_owid = (~gdp_owid.set_index(["country","year"]).index.duplicated()).sum()
    n_wb   = mask_missing.sum()
    print(f"  GDP desde OWID (gdp/pop): {gdp_owid['country'].nunique()} países")
    print(f"  GDP completado con WB:    {n_wb} filas adicionales")

n_final = val["gdp_per_capita_usd"].notna().sum()
print(f"\n  GDP disponible FINAL: {n_final}/{len(val)} filas ({n_final/len(val)*100:.1f}%)")
print(f"  Mejora vs antes:      {n_gdp_actual} → {n_final} filas (+{n_final-n_gdp_actual})")

# ─────────────────────────────────────────────────────────────────────────────
# 6. CALCULAR GDP GROWTH desde la serie combinada
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("PASO 2.5F — Calculando gdp_growth_pct")
print("=" * 65)

# Necesitamos el GDP del año anterior — tomarlo de OWID o dataset original
gdp_2020_orig = orig[orig["year"]==2020][["country","gdp_per_capita_usd"]].rename(
    columns={"gdp_per_capita_usd": "gdp_2020"}
)

# Construir serie: 2020 + 2021 + 2022 + 2023
gdp_series = (
    val[["country","year","gdp_per_capita_usd"]]
    .merge(gdp_2020_orig, on="country", how="left")
    .sort_values(["country","year"])
)

# Para 2021: (gdp_2021 - gdp_2020) / gdp_2020 * 100
# Para 2022: (gdp_2022 - gdp_2021) / gdp_2021 * 100
# etc.
gdp_series["gdp_growth_pct"] = gdp_series.groupby("country")["gdp_per_capita_usd"].pct_change() * 100

# Para 2021, el pct_change usa 2020 del loop — pero 2020 no está en val
# Calcularlo manualmente para 2021
gdp_2021 = val[val["year"]==2021][["country","gdp_per_capita_usd"]].rename(
    columns={"gdp_per_capita_usd":"gdp_2021"})
growth_2021 = gdp_2020_orig.merge(gdp_2021, on="country", how="inner")
growth_2021["gdp_growth_pct_2021"] = (
    (growth_2021["gdp_2021"] - growth_2021["gdp_2020"]) / growth_2021["gdp_2020"] * 100
)

# Merge growth 2021 al dataset
gdp_growth_calc = gdp_series[gdp_series["year"].isin([2022,2023])][
    ["country","year","gdp_growth_pct"]
].copy()
growth_2021_final = growth_2021[["country","gdp_growth_pct_2021"]].rename(
    columns={"gdp_growth_pct_2021":"gdp_growth_pct"}
)
growth_2021_final["year"] = 2021

all_growth = pd.concat([growth_2021_final, gdp_growth_calc], ignore_index=True)

# Eliminar gdp_growth_pct viejo y reemplazar
if "gdp_growth_pct" in val.columns:
    val = val.drop(columns=["gdp_growth_pct"])
val = val.merge(all_growth, on=["country","year"], how="left")

n_growth = val["gdp_growth_pct"].notna().sum()
print(f"  gdp_growth_pct disponible: {n_growth}/{len(val)} filas ({n_growth/len(val)*100:.1f}%)")

# ─────────────────────────────────────────────────────────────────────────────
# 7. CALIDAD FINAL Y GUARDAR
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("PASO 2.5G — Calidad final del dataset")
print("=" * 65)

COLS_MODELO = [
    "access_electricity_pct", "low_carbon_elec_pct",
    "renewable_share_pct", "gdp_per_capita_usd",
    "energy_intensity_mj_gdp", "primary_energy_per_capita_kwh",
    "gdp_growth_pct",
]

print(f"\n  {'Columna':<35} {'% nulos'}")
print(f"  {'-'*45}")
for col in COLS_MODELO:
    if col in val.columns:
        pct = val[col].isna().mean() * 100
        status = "✓" if pct < 20 else "⚠"
        print(f"  {status} {col:<33} {pct:.1f}%")
    else:
        print(f"  ✗ {col:<33} FALTA")

# Muestra 5 casos
casos = ["Argentina", "Spain", "India", "Japan", "Cambodia"]
print(f"\n  Muestra 5 casos (2022):")
cols_v = ["country","year","gdp_per_capita_usd","low_carbon_elec_pct","gdp_growth_pct"]
cols_v = [c for c in cols_v if c in val.columns]
print(val[(val["country"].isin(casos)) & (val["year"]==2022)][cols_v].to_string(index=False))

# Guardar (sobreescribe el archivo anterior)
col_order = ["country","year"] + [c for c in COLS_MODELO if c in val.columns]
extra = [c for c in val.columns if c not in col_order]
val[col_order + extra].sort_values(["country","year"]).to_csv(
    "validation_data_2021_2023.csv", index=False
)

print(f"\n  ✓ validation_data_2021_2023.csv actualizado")
print(f"    {len(val):,} filas × {len(val.columns)} columnas")

print(f"""
{"=" * 65}
RESUMEN — Paso 2.5 completado
{"=" * 65}
  GDP per cápita: {n_gdp_actual/len(val)*100:.0f}% → {n_final/len(val)*100:.0f}% de cobertura
  gdp_growth_pct: {n_growth/len(val)*100:.0f}% de cobertura

Siguiente paso:
  Volver a correr 09_validation_step3_predict.py
  El MAE debería mejorar significativamente.
""")

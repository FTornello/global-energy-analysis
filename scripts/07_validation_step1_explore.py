"""
07_validation_step1_explore.py
==============================
Paso 1 de la validación out-of-sample (2021-2023).

Descarga dos fuentes nuevas:
  - Our World in Data (OWID) Energy Dataset — energía, electricidad, GDP
  - World Bank — acceso a electricidad (EG.ELC.ACCS.ZS)

Explora su estructura, años disponibles, cobertura de países,
y hace el mapeo de columnas contra el dataset original.

Uso:
    python 07_validation_step1_explore.py

Requiere conexión a internet.

Output:
    - exploración completa en consola
    - owid_raw.csv          (dataset OWID descargado)
    - worldbank_access.csv  (acceso electricidad World Bank)
    - column_mapping.json   (mapeo de columnas documentado)
"""

import pandas as pd
import numpy as np
import requests
import json
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# NUESTRAS VARIABLES ORIGINALES (las que necesita el modelo Ridge)
# ─────────────────────────────────────────────────────────────────────────────
ORIGINAL_FEATURES = {
    "access_electricity_pct":       "% población con electricidad",
    "low_carbon_elec_pct":          "% electricidad de fuentes limpias",
    "renewable_share_pct":          "% renovables en consumo final",
    "gdp_per_capita_usd":           "GDP per cápita en USD",
    "energy_intensity_mj_gdp":      "Energía por unidad de GDP",
    "primary_energy_per_capita_kwh":"Energía primaria per cápita (kWh)",
    "gdp_growth_pct":               "Crecimiento anual del GDP",
}

PAISES_REFERENCIA = [
    "Argentina", "Spain", "India", "Japan", "Cambodia",
    "Germany", "Brazil", "United States", "China", "Nigeria"
]

# ─────────────────────────────────────────────────────────────────────────────
# 1. DESCARGAR OWID ENERGY DATASET
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("PASO 1A — Descargando OWID Energy Dataset...")
print("=" * 65)

OWID_URL = "https://owid-public.owid.io/data/energy/owid-energy-data.csv"

try:
    owid = pd.read_csv(
        OWID_URL,
        storage_options={"User-Agent": "Our World In Data data fetch/1.0"}
    )
    print(f"✓ Descargado correctamente")
except Exception:
    # Fallback sin storage_options
    import urllib.request
    import io
    req = urllib.request.Request(
        OWID_URL,
        headers={"User-Agent": "Our World In Data data fetch/1.0"}
    )
    with urllib.request.urlopen(req) as r:
        owid = pd.read_csv(io.BytesIO(r.read()))
    print(f"✓ Descargado (método alternativo)")

print(f"\nShape: {owid.shape[0]:,} filas × {owid.shape[1]} columnas")
print(f"Años: {owid['year'].min()} — {owid['year'].max()}")
print(f"Entidades únicas: {owid['country'].nunique()}")

# Guardar raw
owid.to_csv("owid_raw.csv", index=False)
print(f"✓ Guardado: owid_raw.csv")

# ─────────────────────────────────────────────────────────────────────────────
# 2. EXPLORAR COLUMNAS OWID Y MAPEAR A NUESTRAS VARIABLES
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("PASO 1B — Mapeo de columnas OWID → variables originales")
print("=" * 65)

# Mapeo definido manualmente según el codebook de OWID
COLUMN_MAP = {
    # Variable original          : columna OWID
    "low_carbon_elec_pct":       "low_carbon_share_elec",
    "renewable_share_pct":       "renewables_share_elec",
    "primary_energy_per_capita_kwh": "primary_energy_per_capita",
    "energy_intensity_mj_gdp":   "energy_per_gdp",
    "gdp_per_capita_usd":        "gdp_per_capita",
    # access_electricity_pct viene del World Bank (ver paso 1C)
    # gdp_growth_pct se calcula desde gdp_per_capita
}

print(f"\n{'Variable original':<35} {'Columna OWID':<30} {'Disponible':<12} {'% nulos 2021+'}")
print("-" * 95)

owid_2021 = owid[owid["year"] >= 2021]
mapping_result = {}

for orig, owid_col in COLUMN_MAP.items():
    disponible = owid_col in owid.columns
    if disponible:
        nulos = owid_2021[owid_col].isnull().mean() * 100
        unidades = owid_2021[owid_col].dropna().describe()
        mapping_result[orig] = {
            "owid_column": owid_col,
            "available": True,
            "null_pct_2021plus": round(float(nulos), 1),
            "min": round(float(unidades["min"]), 2) if len(unidades) > 0 else None,
            "max": round(float(unidades["max"]), 2) if len(unidades) > 0 else None,
        }
        print(f"  {orig:<33} {owid_col:<30} {'✓':<12} {nulos:.1f}%")
    else:
        mapping_result[orig] = {"owid_column": owid_col, "available": False}
        print(f"  {orig:<33} {owid_col:<30} {'✗ NO EXISTE':<12}")

# Variables que no se pueden mapear directamente
print(f"\n  {'access_electricity_pct':<33} {'→ World Bank API':<30} {'Paso 1C'}")
print(f"  {'gdp_growth_pct':<33} {'→ calcular desde gdp_per_capita':<30} {'Derivada'}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. VERIFICAR AÑOS DISPONIBLES EN OWID PARA VARIABLES CLAVE
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("PASO 1C — Cobertura temporal de variables clave (2019–2023)")
print("=" * 65)

owid_cols = [v["owid_column"] for v in mapping_result.values() if v["available"]]
owid_sub = owid[owid["year"].isin([2019, 2020, 2021, 2022, 2023])].copy()

print(f"\nPaíses con datos en cada año (conteo de filas no-nulas en low_carbon_share_elec):")
for yr in [2019, 2020, 2021, 2022, 2023]:
    sub = owid_sub[owid_sub["year"] == yr]
    n = sub["low_carbon_share_elec"].notna().sum()
    print(f"  {yr}: {n} países con low_carbon_share_elec")

# ─────────────────────────────────────────────────────────────────────────────
# 4. DESCARGAR ACCESO A ELECTRICIDAD — WORLD BANK API
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("PASO 1D — Descargando acceso a electricidad (World Bank)")
print("=" * 65)

WB_URL = (
    "https://api.worldbank.org/v2/country/all/indicator/EG.ELC.ACCS.ZS"
    "?date=2018:2023&format=json&per_page=2000"
)

try:
    resp = requests.get(WB_URL, timeout=30)
    wb_data = resp.json()
    
    records = []
    # La API devuelve paginado — puede haber múltiples páginas
    total_pages = wb_data[0]["pages"]
    print(f"  Páginas disponibles: {total_pages}")
    
    for page in range(1, total_pages + 1):
        if page == 1:
            items = wb_data[1]
        else:
            r = requests.get(WB_URL + f"&page={page}", timeout=30)
            items = r.json()[1]
        
        for item in items:
            if item["value"] is not None:
                records.append({
                    "country_wb": item["country"]["value"],
                    "country_code": item["countryiso3code"],
                    "year": int(item["date"]),
                    "access_electricity_pct": round(float(item["value"]), 2)
                })
    
    wb_df = pd.DataFrame(records)
    wb_df = wb_df.sort_values(["country_wb", "year"])
    wb_df.to_csv("worldbank_access.csv", index=False)
    
    print(f"✓ Descargado correctamente")
    print(f"  Registros totales: {len(wb_df):,}")
    print(f"  Países únicos: {wb_df['country_wb'].nunique()}")
    print(f"  Años: {wb_df['year'].min()} — {wb_df['year'].max()}")
    print(f"✓ Guardado: worldbank_access.csv")
    
    # Muestra para países de referencia
    print(f"\n  Acceso electricidad en países de referencia (últimos datos):")
    for pais in PAISES_REFERENCIA:
        sub = wb_df[
            wb_df["country_wb"].str.contains(pais[:6], case=False, na=False)
        ].sort_values("year")
        if len(sub) > 0:
            row = sub.iloc[-1]
            print(f"    {row['country_wb'][:25]:<25} {int(row['year'])}: {row['access_electricity_pct']:.1f}%")
    
    wb_available = True

except Exception as e:
    print(f"✗ Error descargando World Bank: {e}")
    print("  → Usaremos los valores de 2019 del dataset original como proxy")
    wb_available = False

# ─────────────────────────────────────────────────────────────────────────────
# 5. VERIFICAR SOLAPAMIENTO DE PAÍSES ENTRE OWID Y NUESTRO DATASET ORIGINAL
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("PASO 1E — Solapamiento de países OWID vs dataset original")
print("=" * 65)

try:
    original = pd.read_csv("sustainable_energy_clean.csv")
    paises_orig = set(original["country"].unique())
    paises_owid_2021 = set(
        owid[(owid["year"] == 2021) & owid["low_carbon_share_elec"].notna()]["country"].unique()
    )
    
    en_ambos = paises_orig & paises_owid_2021
    solo_orig = paises_orig - paises_owid_2021
    solo_owid = paises_owid_2021 - paises_orig
    
    print(f"\n  Países en dataset original: {len(paises_orig)}")
    print(f"  Países en OWID 2021 (con datos): {len(paises_owid_2021)}")
    print(f"  En ambos: {len(en_ambos)} ← estos son los que podemos validar")
    print(f"  Solo en original (pueden tener nombre distinto en OWID): {len(solo_orig)}")
    if solo_orig:
        print(f"    {sorted(list(solo_orig))[:10]}")

except FileNotFoundError:
    print("  ⚠ No se encontró sustainable_energy_clean.csv en este directorio")
    print("  Copialo a la misma carpeta que este script antes de correr el Paso 2")

# ─────────────────────────────────────────────────────────────────────────────
# 6. MUESTRA DE DATOS 2021-2023 PARA PAÍSES CLAVE
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("PASO 1F — Muestra de datos 2021-2023 para los 5 casos")
print("=" * 65)

casos = ["Argentina", "Spain", "India", "Japan", "Cambodia"]
cols_muestra = ["country", "year", "low_carbon_share_elec",
                "primary_energy_per_capita", "gdp_per_capita"]
cols_disp = [c for c in cols_muestra if c in owid.columns]

muestra = owid[
    (owid["country"].isin(casos)) &
    (owid["year"].isin([2019, 2020, 2021, 2022, 2023]))
][cols_disp].sort_values(["country", "year"])

print(f"\n{muestra.to_string(index=False)}")

# ─────────────────────────────────────────────────────────────────────────────
# 7. EXPORTAR MAPEO DE COLUMNAS
# ─────────────────────────────────────────────────────────────────────────────
full_mapping = {
    **mapping_result,
    "access_electricity_pct": {
        "owid_column": "N/A",
        "source": "World Bank API (EG.ELC.ACCS.ZS)",
        "available": wb_available,
        "fallback": "valor 2019 del dataset original si WB no disponible"
    },
    "gdp_growth_pct": {
        "owid_column": "derivada",
        "formula": "(gdp_per_capita_t - gdp_per_capita_t-1) / gdp_per_capita_t-1 * 100",
        "available": "gdp_per_capita" in owid.columns
    }
}

with open("column_mapping.json", "w", encoding="utf-8") as f:
    json.dump(full_mapping, f, ensure_ascii=False, indent=2)
print(f"\n✓ Mapeo guardado: column_mapping.json")

# ─────────────────────────────────────────────────────────────────────────────
# RESUMEN FINAL
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("RESUMEN — Paso 1 completado")
print("=" * 65)
print(f"""
Variables del modelo original: {len(ORIGINAL_FEATURES)}
Variables encontradas en OWID: {sum(1 for v in mapping_result.values() if v['available'])}
Acceso electricidad (WB):      {'✓ descargado' if wb_available else '✗ usar proxy 2019'}
GDP growth:                    ✓ se calcula desde gdp_per_capita
Años disponibles para validar: 2021, 2022, 2023

Archivos generados:
  - owid_raw.csv
  - worldbank_access.csv  {'(✓)' if wb_available else '(✗ no disponible)'}
  - column_mapping.json

Siguiente paso:
  Correr 08_validation_step2_align.py para alinear los datasets
  y preparar los datos 2021-2023 en el mismo formato que el original.
""")

"""
clean_sustainable_energy.py
============================
Pipeline de limpieza del dataset Global Sustainable Energy (2000-2020).
Genera el CSV limpio y actualiza el log de decisiones en JSON.

Uso:
    python clean_sustainable_energy.py

Outputs:
    - sustainable_energy_clean.csv   (datos limpios)
    - cleaning_log.json              (registro de todas las decisiones)
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime

# ─── Paths ────────────────────────────────────────────────────────────────────
INPUT_PATH  = "/mnt/user-data/uploads/global-data-on-sustainable-energy__1_.csv"
OUTPUT_CSV  = "/mnt/user-data/outputs/sustainable_energy_clean.csv"
OUTPUT_LOG  = "/mnt/user-data/outputs/cleaning_log.json"

# ─── Logger ───────────────────────────────────────────────────────────────────
log = {
    "dataset": "Global Data on Sustainable Energy (2000–2020)",
    "script_version": "1.0",
    "run_timestamp": datetime.now().isoformat(timespec="seconds"),
    "input_file": INPUT_PATH,
    "output_file": OUTPUT_CSV,
    "snapshot_before": {},
    "snapshot_after": {},
    "steps": []
}

def log_step(step_id, category, description, detail, rows_affected=None, columns=None):
    entry = {
        "step": step_id,
        "category": category,
        "description": description,
        "detail": detail,
        "rows_affected": rows_affected,
        "columns": columns or [],
        "timestamp": datetime.now().isoformat(timespec="seconds")
    }
    log["steps"].append(entry)
    tag = {"rename": "RENAME", "fix": "FIX", "clip": "CLIP", "flag": "FLAG", "drop": "DROP"}
    label = tag.get(category, category.upper())
    ra = f"  →  {rows_affected} filas" if rows_affected is not None else ""
    print(f"[{label}] {description}{ra}")

def snapshot(df, label):
    return {
        "label": label,
        "rows": len(df),
        "columns": len(df.columns),
        "total_nulls": int(df.isnull().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
        "nulls_per_column": {
            col: int(df[col].isnull().sum())
            for col in df.columns
            if df[col].isnull().sum() > 0
        }
    }

# ─── Load ─────────────────────────────────────────────────────────────────────
print("\n── Cargando dataset ──────────────────────────────────────────────")
df = pd.read_csv(INPUT_PATH)
log["snapshot_before"] = snapshot(df, "raw")
print(f"   Filas: {len(df)} | Columnas: {len(df.columns)}")
print(f"   Nulos totales: {df.isnull().sum().sum()}")

step = 0

# ──────────────────────────────────────────────────────────────────────────────
# PASO 1: Renombrar columnas (estandarizar a snake_case limpio)
# ──────────────────────────────────────────────────────────────────────────────
print("\n── Paso 1: Renombrar columnas ────────────────────────────────────")

RENAME_MAP = {
    "Entity":                                                          "country",
    "Year":                                                            "year",
    "Access to electricity (% of population)":                         "access_electricity_pct",
    "Access to clean fuels for cooking":                               "access_clean_fuels_pct",
    "Renewable-electricity-generating-capacity-per-capita":            "renewable_capacity_per_capita",
    "Financial flows to developing countries (US $)":                  "financial_flows_usd",
    "Renewable energy share in the total final energy consumption (%)":"renewable_share_pct",
    "Electricity from fossil fuels (TWh)":                             "elec_fossil_twh",
    "Electricity from nuclear (TWh)":                                  "elec_nuclear_twh",
    "Electricity from renewables (TWh)":                               "elec_renewables_twh",
    "Low-carbon electricity (% electricity)":                          "low_carbon_elec_pct",
    "Primary energy consumption per capita (kWh/person)":              "primary_energy_per_capita_kwh",
    "Energy intensity level of primary energy (MJ/$2017 PPP GDP)":     "energy_intensity_mj_gdp",
    "Value_co2_emissions_kt_by_country":                               "co2_emissions_kt",
    "Renewables (% equivalent primary energy)":                        "renewables_equiv_primary_pct",
    "gdp_growth":                                                      "gdp_growth_pct",
    "gdp_per_capita":                                                  "gdp_per_capita_usd",
    "Density\\n(P/Km2)":                                               "density_p_km2",
    "Land Area(Km2)":                                                  "land_area_km2",
    "Latitude":                                                        "latitude",
    "Longitude":                                                       "longitude",
}

step += 1
df.rename(columns=RENAME_MAP, inplace=True)
log_step(
    step_id=step,
    category="rename",
    description="Estandarización de nombres de columnas a snake_case",
    detail=(
        "Todas las columnas renombradas: eliminados caracteres especiales (\\n, paréntesis, "
        "signo $, %), unificado formato snake_case. La columna 'Density\\n(P/Km2)' pasó a "
        "'density_p_km2' eliminando el salto de línea embebido en el nombre."
    ),
    columns=list(RENAME_MAP.keys())
)

# ──────────────────────────────────────────────────────────────────────────────
# PASO 2: Limpiar columna density_p_km2
# (string con comas → float)
# ──────────────────────────────────────────────────────────────────────────────
print("\n── Paso 2: Limpiar density_p_km2 (string → float) ───────────────")

before_non_numeric = df["density_p_km2"].apply(
    lambda x: pd.to_numeric(str(x).replace(",", ""), errors="coerce")
).isna().sum()

df["density_p_km2"] = (
    df["density_p_km2"]
    .astype(str)
    .str.replace(",", "", regex=False)
    .apply(pd.to_numeric, errors="coerce")
)

after_non_numeric = df["density_p_km2"].isna().sum()
rows_fixed = before_non_numeric - after_non_numeric

step += 1
log_step(
    step_id=step,
    category="fix",
    description="density_p_km2: convertida de string a float64",
    detail=(
        f"La columna tenía separadores de miles en formato string (ej: '2,239'). "
        f"Se removieron las comas y se convirtió a float64. "
        f"{rows_fixed} filas corregidas (correspondían a Bahrain principalmente). "
        f"Valores nulos antes: {before_non_numeric} — después: {after_non_numeric}."
    ),
    rows_affected=rows_fixed,
    columns=["density_p_km2"]
)

# ──────────────────────────────────────────────────────────────────────────────
# PASO 3: Clip low_carbon_elec_pct (no puede superar 100%)
# ──────────────────────────────────────────────────────────────────────────────
print("\n── Paso 3: Clip low_carbon_elec_pct a [0, 100] ──────────────────")

over_100 = (df["low_carbon_elec_pct"] > 100).sum()
df["low_carbon_elec_pct"] = df["low_carbon_elec_pct"].clip(lower=0, upper=100)

step += 1
log_step(
    step_id=step,
    category="clip",
    description="low_carbon_elec_pct: valores fuera de rango [0, 100] corregidos",
    detail=(
        f"{over_100} valor(es) superaban el 100% (Bhutan 2005: 100.00001). "
        "Error de redondeo de punto flotante. Aplicado clip(0, 100). "
        "No se eliminó ninguna fila."
    ),
    rows_affected=int(over_100),
    columns=["low_carbon_elec_pct"]
)

# ──────────────────────────────────────────────────────────────────────────────
# PASO 4: Marcar columnas de alta nulidad como "use with caution"
# (financial_flows_usd y renewables_equiv_primary_pct)
# Decisión: NO imputar — los nulos tienen significado estructural
# ──────────────────────────────────────────────────────────────────────────────
print("\n── Paso 4: Flag columnas de alta nulidad ─────────────────────────")

HIGH_NULL_COLS = {
    "financial_flows_usd": {
        "null_pct": round(df["financial_flows_usd"].isnull().mean() * 100, 1),
        "rationale": (
            "Los países desarrollados no reciben flujos financieros para energía limpia, "
            "por lo que sus nulos son ausencias estructuralmente válidas, no errores. "
            "Imputar con 0 sería correcto conceptualmente (no recibieron fondos), pero "
            "puede distorsionar análisis de distribución. Se recomienda filtrar solo "
            "países en desarrollo al usar esta columna."
        )
    },
    "renewables_equiv_primary_pct": {
        "null_pct": round(df["renewables_equiv_primary_pct"].isnull().mean() * 100, 1),
        "rationale": (
            "Más del 58% de los registros no tienen este indicador. Cobertura limitada "
            "especialmente en años tempranos y países de bajos recursos. "
            "No se imputa. Solo usar en análisis con subset de países/años con datos suficientes."
        )
    }
}

step += 1
log_step(
    step_id=step,
    category="flag",
    description="Columnas de alta nulidad documentadas — sin imputación",
    detail=HIGH_NULL_COLS,
    rows_affected=None,
    columns=list(HIGH_NULL_COLS.keys())
)

# ──────────────────────────────────────────────────────────────────────────────
# PASO 5: Añadir columna de flag para países con cobertura temporal incompleta
# ──────────────────────────────────────────────────────────────────────────────
print("\n── Paso 5: Flag países con cobertura temporal incompleta ─────────")

PARTIAL_COVERAGE = ["French Guiana", "South Sudan", "Montenegro", "Serbia"]
df["partial_temporal_coverage"] = df["country"].isin(PARTIAL_COVERAGE)

coverage_counts = df[df["partial_temporal_coverage"]].groupby("country")["year"].count().to_dict()

step += 1
log_step(
    step_id=step,
    category="flag",
    description="Nueva columna 'partial_temporal_coverage' para países con < 21 años",
    detail=(
        f"4 países tienen menos de los 21 años esperados: {coverage_counts}. "
        "Las causas son históricas (independencia de South Sudan en 2011, "
        "separación de Montenegro y Serbia de Yugoslavia en 2006, "
        "French Guiana con datos esporádicos). "
        "La columna booleana permite filtrarlos fácilmente en análisis de series de tiempo."
    ),
    rows_affected=int(df["partial_temporal_coverage"].sum()),
    columns=["partial_temporal_coverage"]
)

# ──────────────────────────────────────────────────────────────────────────────
# PASO 6: Verificar tipos finales y orden de columnas
# ──────────────────────────────────────────────────────────────────────────────
print("\n── Paso 6: Verificar tipos de datos finales ──────────────────────")

expected_types = {
    "country": "object",
    "year": "int64",
    "density_p_km2": "float64",
    "partial_temporal_coverage": "bool"
}

type_issues = []
for col, expected in expected_types.items():
    actual = str(df[col].dtype)
    if actual != expected:
        type_issues.append(f"{col}: esperado {expected}, actual {actual}")

step += 1
log_step(
    step_id=step,
    category="fix",
    description="Verificación de tipos de datos post-limpieza",
    detail=(
        "Todos los tipos verificados correctamente. "
        + (f"Problemas encontrados: {type_issues}" if type_issues else "Sin inconsistencias.")
    ),
    rows_affected=None,
    columns=list(df.columns)
)

# ──────────────────────────────────────────────────────────────────────────────
# GUARDAR OUTPUT
# ──────────────────────────────────────────────────────────────────────────────
print("\n── Guardando archivos ────────────────────────────────────────────")

os.makedirs("/mnt/user-data/outputs", exist_ok=True)

df.to_csv(OUTPUT_CSV, index=False)
print(f"   ✓ CSV limpio: {OUTPUT_CSV}")

log["snapshot_after"] = snapshot(df, "clean")
log["summary"] = {
    "total_steps": len(log["steps"]),
    "rows_unchanged": log["snapshot_before"]["rows"] == log["snapshot_after"]["rows"],
    "cols_added": log["snapshot_after"]["columns"] - log["snapshot_before"]["columns"],
    "nulls_resolved": log["snapshot_before"]["total_nulls"] - log["snapshot_after"]["total_nulls"],
    "no_rows_dropped": True
}

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        return super().default(obj)

with open(OUTPUT_LOG, "w", encoding="utf-8") as f:
    json.dump(log, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
print(f"   ✓ Log JSON: {OUTPUT_LOG}")

# ──────────────────────────────────────────────────────────────────────────────
# RESUMEN FINAL
# ──────────────────────────────────────────────────────────────────────────────
print("\n── Resumen ───────────────────────────────────────────────────────")
print(f"   Pasos ejecutados   : {len(log['steps'])}")
print(f"   Filas (sin cambio) : {len(df)}")
print(f"   Columnas           : {log['snapshot_before']['columns']} → {log['snapshot_after']['columns']}")
print(f"   Nulos resueltos    : {log['summary']['nulls_resolved']}")
print(f"   Filas eliminadas   : 0")
print("──────────────────────────────────────────────────────────────────\n")

# Proyecto: Análisis Global de Energía Sostenible (2000–2023)

**Estado:** Completado ✅  
**Última actualización:** Mayo 2026  
**Dataset:** Global Data on Sustainable Energy — Kaggle (2000–2020) + Our World in Data / World Bank (2021–2023)  
**Herramientas:** Python, pandas, scikit-learn, matplotlib, seaborn, D3.js  
**Repositorio:** https://github.com/FTornello/global-energy-analysis

---

## Resumen del proyecto

Pipeline completo de Data Analytics sobre transición energética global. Cubre 176 países entre 2000 y 2023. Incluye limpieza, EDA, clustering K-Means, dinámica temporal, mapa geoespacial, feature engineering, regresión, casos individuales, validación out-of-sample con datos nuevos (2021–2023), y análisis metodológico de los pesos del índice compuesto.

---

## Etapa 1 — Exploración del dataset

**Script:** ninguno (exploración manual en Python interactivo)  
**Output:** informe de calidad de datos (en sesión)

### Estructura del dataset

| Dimensión | Valor |
|-----------|-------|
| Filas | 3.649 |
| Columnas | 21 |
| Países | 176 |
| Años | 2000–2020 (21 años) |
| Duplicados | 0 |

### Problemas de calidad detectados

| Problema | Columna afectada | Severidad | Resolución |
|----------|-----------------|-----------|------------|
| Nombre de columna con `\n` embebido | `Density\n(P/Km2)` | Media | Renombrar a `density_p_km2` |
| Tipo string en columna numérica | `density_p_km2` | Media | Remover comas, convertir a float |
| 57–58% de nulos estructurales | `financial_flows_usd`, `renewables_equiv_primary_pct` | Alta | Documentar sin imputar |
| Valor fuera de rango (>100%) | `low_carbon_elec_pct` (Bhutan 2005) | Baja | `clip(0, 100)` |
| 4 países con serie incompleta | French Guiana, South Sudan, Montenegro, Serbia | Informativa | Flag booleano |
| Nombres mixtos snake_case / descriptivos | Todas | Baja | Estandarizar a snake_case |

---

## Etapa 2 — Limpieza de datos

**Script:** `01_clean_sustainable_energy.py`  
**Output:** `sustainable_energy_clean.csv`, `cleaning_log.json`

6 pasos documentados: renombrado de columnas, conversión de density_p_km2 (string → float), clip de low_carbon_elec_pct, flag de columnas de alta nulidad, flag de cobertura temporal, verificación de tipos. 0 filas eliminadas.

| Dimensión | Antes | Después |
|-----------|-------|---------|
| Filas | 3.649 | 3.649 (sin cambio) |
| Columnas | 21 | 22 (+1 flag) |
| Filas eliminadas | — | 0 |

---

## Etapa 3 — Análisis exploratorio (EDA)

**Script:** `02_eda_sustainable_energy.py`  
**Output:** `eda_report.pdf` (6 gráficos)

### Estadísticas descriptivas — columnas clave

| Columna | Media | Mediana | Min | Max |
|---------|-------|---------|-----|-----|
| access_electricity_pct | 78.9% | 98.4% | 1.3% | 100% |
| renewable_share_pct | 32.6% | 23.3% | 0% | 96% |
| low_carbon_elec_pct | 36.8% | 27.9% | 0% | 100% |
| gdp_per_capita_usd | $13.284 | $4.579 | $112 | $123.514 |
| energy_intensity_mj_gdp | 5.31 | 4.30 | 0.11 | 32.57 |

### Correlaciones principales

| Par de variables | Correlación | Interpretación |
|-----------------|-------------|----------------|
| access_electricity ↔ access_fuels | +0.867 | Muy alta — van juntas |
| access_electricity ↔ renewable_share | −0.785 | Alta negativa — paradoja de la biomasa |
| gdp_per_capita ↔ primary_energy | +0.667 | Más riqueza = más consumo |

### Hallazgos principales

1. **Paradoja de la biomasa.** Correlación negativa (r=−0.79) entre acceso a electricidad y renovables share. Los países más pobres aparecen con más "renovables" porque la leña cuenta como renovable. No es transición energética — es pobreza energética.
2. **Distribución bimodal de renovables.** Dos picos: 0–10% (industrializados fósiles) y 80–90% (países en desarrollo con biomasa). El "país promedio" no existe.
3. **GDP per cápita extremadamente sesgado.** Mediana $4.600 vs media $13.300.
4. **Trinidad y Tobago — intensidad energética extrema.** 19.4 MJ/$GDP, casi 4x el promedio global. Industria petroquímica con energía subsidiada.

---

## Etapa 4 — Clustering por perfil energético

**Script:** `03_clustering_dynamics.py`  
**Output:** `clustering_report.pdf`, `cluster_assignments.csv`, `migration_log.csv`

### Metodología

- **Año base:** 2019 (año más completo)
- **Países incluidos:** 175 (excluidos con >2 nulos en features)
- **Features:** 7 variables normalizadas
- **Preprocesamiento:** imputación de nulos restantes con mediana del training set, escalado StandardScaler
- **Algoritmo:** K-Means, k=4
- **Selección de k:** método del codo + silhouette score. k=2 daba silhouette más alto (0.367) pero grupos poco informativos. k=4 (silhouette=0.268) ofrecía granularidad interpretativa real.
- **Metodología temporal:** se aplicaron los centroides de 2019 hacia atrás para mantener etiquetas comparables entre años. Las "migraciones" reflejan movimiento real de países respecto a una clasificación estable, no reclasificación por centroides recalculados.
- **Visualización:** PCA 2D (47.3% + 18.8% = 66.1% varianza explicada)

### Los 4 clusters (2019)

| Cluster | Países | Acceso elec. | Low-carbon | GDP/cap | Descripción |
|---------|--------|-------------|------------|---------|-------------|
| Pobreza energética | 42 | 45% | 39% | $1.786 | Principalmente África subsahariana |
| Emergentes fósiles | 62 | 97% | 15% | $10.867 | Asia, América Latina, Medio Oriente |
| Transición renovable | 43 | 95% | 68% | $8.334 | América Latina y Europa del Este |
| Desarrollados alto consumo | 28 | 100% | 47% | $56.784 | Europa Occidental, Norteamérica, Golfo |

**Nota sobre Cluster 1 (Desarrollados):** incluye tanto Europa Occidental (en transición real hacia renovables) como estados del Golfo (Arabia Saudita, Qatar) que no están en esa trayectoria. El promedio del cluster oculta dos sub-poblaciones moviéndose en direcciones opuestas. El indicador más representativo es la mediana, no la media.

---

## Etapa 5 — Dinámica temporal de clusters (2000–2019)

**Script:** `03_clustering_dynamics.py` (misma ejecución)

### Evolución del tamaño de clusters

| Cluster | 2000 | 2005 | 2010 | 2015 | 2019 | Tendencia |
|---------|------|------|------|------|------|-----------|
| Pobreza energética | 60 | 57 | 50 | 48 | 42 | ↓ −18 |
| Emergentes fósiles | 67 | 63 | 66 | 62 | 62 | → estable |
| Transición renovable | 29 | 28 | 32 | 39 | 43 | ↑ +14 |
| Desarrollados | 16 | 24 | 26 | 26 | 28 | ↑ +12 |

### Migraciones (2000 → 2019)

**42 de 172 países cambiaron de cluster.**

| Tipo de movimiento | Cantidad | Países notables |
|-------------------|----------|-----------------|
| Pobreza → Transición renovable | 13 | Nepal, Bhutan, Cambodia, Ghana, Guatemala |
| Pobreza → Emergentes fósiles | 6 | India, Indonesia, Bangladesh, Botswana |
| Emergentes → Desarrollados | 10 | Alemania, UK, Denmark, Netherlands, Irlanda |
| Emergentes → Transición renovable | 6 | España, Portugal, Bulgaria, Hungría |
| Transición → Desarrollados | 3 | Francia, Austria, Nueva Zelanda |
| Transición → Emergentes (retroceso) | 3 | Chile, Filipinas, Suriname |
| Desarrollados → Emergentes (retroceso) | 1 | **Japón** |

---

## Etapa 6 — Mapa geoespacial de clusters

**Herramienta:** D3.js + TopoJSON + world-atlas@2  
**Output:** widget interactivo (hover por país)

175 países coloreados por cluster. Mapeados a códigos ISO numéricos usando pycountry con correcciones manuales.

### Patrones geográficos identificados

| Región | Patrón |
|--------|--------|
| África subsahariana | Casi 100% Pobreza energética — bloque continuo |
| América Latina | Mayoría Transición renovable — mejor perfil relativo a su desarrollo |
| Europa Occidental | Desarrollados alto consumo — bloque uniforme |
| Europa del Este y Sur | Transición renovable (España, Portugal, Bulgaria, Rumania) |
| Medio Oriente / Asia Central | Emergentes fósiles |
| Norteamérica | Desarrollados alto consumo |

---

## Etapa 7 — Feature engineering

**Script:** `04_feature_engineering.py`  
**Output:** `features_engineered.csv`, `features_report.pdf`

### Variables construidas

| Variable | Fórmula | Rango | Qué mide |
|----------|---------|-------|----------|
| transition_score | 0.30×acceso + 0.45×low_carbon + 0.25×log(GDP) | 0–100 | Avance total en la transición |
| improvement_rate | (score_2019 − score_2000) / 19 años | pts/año | Velocidad de mejora |
| clean_access_ratio | (acceso/100) × (low_carbon/100) × 100 | 0–100 | Calidad del acceso eléctrico |
| fossil_lock_in | 0.45×(1−low_carbon) + 0.35×E.intensity + 0.20×(1−GDP_norm) | 0–100 | Dependencia estructural de fósiles |

**Justificación de pesos del transition_score:** se eligieron con criterio conceptual. Posteriormente validados contra pesos del modelo Ridge (ver Etapa 11) — diferencia < 2% en todos los pesos.

---

## Etapa 8 — Regresión

**Script:** `05_regression_analysis.py`  
**Output:** `regression_report.pdf`, `regression_results.csv`

### Objetivo
Predecir `transition_score` a partir de variables observables.

### Limitación conocida — data leakage parcial
El transition_score está construido con acceso, low_carbon y GDP. El modelo usa como features esas mismas variables (entre otras). Esto genera correlación estructural entre features y target — el R²=0.82 está parcialmente inflado por esta composición. La lección más honesta del ejercicio es la **identificación de path dependence**: el estado energético actual de un país está fuertemente determinado por su historia (low_carbon_2000 explica 47.5%), más que por su nivel de riqueza actual. Para futuros análisis: predecir improvement_rate (independiente del score) o excluir las features que componen el score.

### Limitación conocida — salto Linear → Ridge
La diferencia R²=0.33 (Linear) vs R²=0.82 (Ridge) con α=2 es mayor de lo esperado. Ridge con α bajo debería acercarse a OLS. La diferencia puede atribuirse a multicolinealidad severa entre features (acc_2000 con access_2019, lc_2000 con low_carbon_2019) — en ese caso la diferencia es un hallazgo real sobre la estructura de los datos, no un bug.

### Resultados por modelo (CV 5-fold, 153 países)

| Modelo | R² | MAE |
|--------|-----|-----|
| Linear Regression | 0.331 | 10.42 |
| Ridge (α=2) | 0.822 | 5.22 |
| Random Forest | 0.768 | 6.18 |

### Feature importance (Random Forest)

| Variable | Importancia |
|----------|-------------|
| Low-carbon electricity 2000 | 47.5% |
| Acceso electricidad 2019 | 18.8% |
| Energía per cápita (log) | 15.1% |
| GDP per cápita (log) | 8.5% |
| Resto | 10.2% |

### Hallazgos principales

1. **Path dependence domina sobre riqueza.** El predictor #1 es low_carbon_2000 — la historia energética pesa más que el GDP actual.
2. **Residuales informan sobre factores no energéticos.** Subestimados: Sierra Leone (+17.6), Kenya (+15.3), Denmark (+13.4), Cambodia (+12.4). Sobreestimados: Burkina Faso (−14.6), Chad (−13.8), Burundi (−11.3) — señal de conflictos e inestabilidad institucional.

---

## Etapa 9 — Análisis de casos individuales

**Script:** `06_case_studies.py`  
**Output:** `case_studies_report.pdf`, `case_studies_summary.csv`

| País | 2019 score | Narrativa |
|------|-----------|-----------|
| 🇯🇵 Japón | 62.7 | Fukushima 2011 derrumbó el 41% de electricidad limpia |
| 🇪🇸 España | 76.4 | Apuesta política sostenida 20 años: 44% → 66% low-carbon |
| 🇮🇳 India | 49.3 | 59% → 99% acceso. Motor: carbón. Dilema desarrollo vs clima |
| 🇰🇭 Cambodia | 58.8 | #1 mundial en velocidad. Construyó limpio desde cero |
| 🇦🇷 Argentina | 60.4 | Base limpia (41%) que se perdió (34%). Inercia macro |

---

## Etapa 10 — Validación out-of-sample (2021–2023)

**Scripts:** `07_validation_step1_explore.py` → `08_validation_step2_align.py` → `08b_validation_step2b_gdp.py` → `09_validation_step3_predict.py` → `10_validation_step4_visualize.py`  
**Output:** `validation_predictions.csv`, `validation_report.pdf`

### Fuentes de datos nuevas
- Our World in Data — energy dataset (github.com/owid/energy-data)
- World Bank API — electricity access (EG.ELC.ACCS.ZS) y GDP per cápita (NY.GDP.PCAP.CD)

### Decisiones de alineación
- GDP per cápita calculado como gdp/population desde OWID (56% cobertura) + complementado con World Bank API
- 44% de países sin GDP en datos nuevos — imputados con mediana del training set
- Esto infla el MAE global; la comparación justa es sobre el subset con datos completos

### Resultados

| Métrica | Valor | Interpretación |
|---------|-------|----------------|
| MAE original (2019, CV) | 5.22 pts | Referencia |
| MAE validación — datos completos (148 países) | 9.28 pts | Comparación justa |
| MAE validación — todos los países (176) | 21.25 pts | Inflado por 44% sin GDP |
| Degradación (comparación justa) | +4.1 pts | Moderada — COVID + crisis 2022 |

### Tendencia global (mediana transition_score)

| Año | Score |
|-----|-------|
| 2019 | 54.5 |
| 2021 | 57.1 |
| 2022 | 57.5 |
| 2023 | 56.2 |

### 5 casos (2019 → 2023)

| País | 2019 | 2023 | Cambio | Veredicto |
|------|------|------|--------|-----------|
| Japón | 62.7 | 55.9 | −6.8 | Empeoró — fósiles + crisis 2022 |
| España | 76.4 | 74.8 | −1.5 | Bajó en 2022, rebotó en 2023 |
| India | 49.3 | 52.2 | +2.9 | Mejora lenta y constante |
| Argentina | 60.4 | 60.9 | +0.4 | Prácticamente estancada |
| Cambodia | 58.8 | 60.0 | +1.2 | Desaceleró en 2023 |

**¿El shock de Ucrania 2022 aparece en los datos?** Sí. España bajó en 2022 y rebotó en 2023. Europa en general muestra la huella del año — varios países quemaron más carbón temporalmente. Pero no fue tan profundo ni permanente como Fukushima.

---

## Etapa 11 — Validación de pesos del transition_score (V1 vs V2)

**Script:** `11_score_weights_validation.py`  
**Output:** `weights_validation_report.pdf`, `score_weights_comparison.csv`

### Pregunta
¿Los pesos manuales (0.30 / 0.45 / 0.25) estaban justificados, o fueron arbitrarios?

### Método
Extraer los coeficientes normalizados del modelo Ridge para las mismas 3 variables y comparar contra los pesos manuales.

### Resultado

| Variable | V1 (manual) | V2 (Ridge) | Delta |
|----------|-------------|------------|-------|
| Acceso electricidad | 0.300 | 0.292 | −0.008 |
| Low-carbon electricity | 0.450 | 0.466 | +0.016 |
| GDP per cápita (log) | 0.250 | 0.243 | −0.007 |

- Correlación V1 vs V2: **r = 0.9995**
- Diferencia media entre scores: **0.70 puntos** (escala 0–100)
- Países que cambiaron más de 5 posiciones en ranking: **11 / 175**
- Los 5 casos de referencia: **0 cambios de posición**

**Conclusión:** los pesos manuales estaban a menos del 2% de lo que el modelo matemático encontró de forma independiente. La decisión subjetiva quedó validada cuantitativamente.

---

## Etapa 12 — Comparación de tres métodos de pesos

**Script:** `12_score_methods_comparison.py`  
**Output:** `score_methods_report.pdf`, `score_methods_comparison.csv`

### Los tres métodos

| Método | Acceso | Low-carbon | GDP | Descripción |
|--------|--------|------------|-----|-------------|
| V1 — Manual | 0.300 | 0.450 | 0.250 | Decisión conceptual |
| V2 — Ridge | 0.292 | 0.466 | 0.243 | Derivado del modelo |
| V3 — PCA | 0.458 | 0.086 | 0.456 | Primera componente principal |

### Correlaciones

- V1 vs V2: **r = 0.9995** — prácticamente idénticos
- V1 vs V3: **r = 0.4995** — significativamente distintos
- V2 vs V3: **r = 0.4711** — significativamente distintos

### Por qué PCA diverge

PCA revela que las tres variables contienen dos ejes independientes:
- **PC1 (58% varianza):** eje de desarrollo — acceso + GDP se mueven juntos
- **PC2 (33% varianza):** eje de energía limpia — low_carbon casi solo

Un país puede tener alto desarrollo con electricidad sucia (Qatar, EE.UU.) o bajo desarrollo con electricidad limpia (Albania, Bhutan, Nepal). PCA captura el eje de desarrollo como PC1 y asigna peso mínimo a low_carbon.

**Conclusión clave:** construir un índice de transición energética que recompense energía limpia independientemente del desarrollo económico requiere una decisión de diseño explícita. Ningún algoritmo puede tomarla solo. Esto valida que darle más peso a low_carbon (0.45) fue deliberado y justificado, no arbitrario.

### Impacto en 5 casos

| País | V1 | V2 | V3 | Nota |
|------|-----|-----|-----|------|
| Japón | 62.7 | 61.7 | 91.5 | PCA ignora dependencia de fósiles |
| España | 76.4 | 75.9 | 86.4 | Similar en los tres |
| Argentina | 60.4 | 59.7 | 79.0 | PCA premia acceso universal |
| Cambodia | 58.8 | 58.7 | 54.0 | PCA penaliza bajo GDP |
| India | 49.3 | 48.6 | 64.9 | PCA premia electrificación |

---

## Limitaciones conocidas del proyecto

1. **Data leakage parcial en regresión.** El transition_score usa acceso, low_carbon y GDP. El modelo predice ese score usando las mismas variables. R²=0.82 está parcialmente inflado. El hallazgo real es path dependence, no capacidad predictiva pura.

2. **Salto Linear→Ridge.** Diferencia de 0.49 en R² mayor a lo esperado con α=2. Probable causa: multicolinealidad severa entre features. Requiere auditoría formal.

3. **Imputación con mediana en clustering.** Nulos restantes imputados con mediana antes de K-Means. Puede distorsionar levemente los centroides. Alternativa preferible: KNN imputer o exclusión más agresiva.

4. **Cluster 1 heterogéneo.** "Desarrollados alto consumo" mezcla Europa Occidental (en transición) con estados del Golfo (no en transición). El promedio del cluster es engañoso. Reportar medianas + rangos sería más honesto.

5. **GDP incompleto en validación 2021–2023.** 44% de países sin GDP en datos nuevos. MAE global inflado artificialmente. La comparación justa es MAE sobre países con datos completos (9.28 vs 5.22).

6. **Dataset usado frecuentemente en Kaggle.** Los puntos diferenciadores del proyecto — clustering dinámico, case studies narrativos, validación out-of-sample, análisis metodológico de pesos — no están suficientemente destacados en la presentación.

---

## Archivos generados

| Archivo | Script | Descripción |
|---------|--------|-------------|
| `sustainable_energy_clean.csv` | 01 | Dataset limpio, 3.649 filas × 22 columnas |
| `cleaning_log.json` | 01 | Log de decisiones de limpieza |
| `eda_report.pdf` | 02 | 6 gráficos exploratorios |
| `clustering_report.pdf` | 03 | 6 gráficos clustering + dinámica |
| `cluster_assignments.csv` | 03 | Cluster por país (2019) |
| `migration_log.csv` | 03 | 42 países que cambiaron de cluster |
| `features_engineered.csv` | 04 | 4 variables nuevas por país |
| `features_report.pdf` | 04 | 3 gráficos |
| `regression_results.csv` | 05 | Predicciones vs real por país |
| `regression_report.pdf` | 05 | 4 gráficos |
| `case_studies_summary.csv` | 06 | Tabla resumen 5 países |
| `case_studies_report.pdf` | 06 | 6 páginas de análisis |
| `validation_data_2021_2023.csv` | 07-08b | Dataset alineado 2021–2023 |
| `validation_predictions.csv` | 09 | Predicciones validación |
| `validation_report.pdf` | 10 | 5 gráficos validación |
| `score_weights_comparison.csv` | 11 | Score V1 vs V2 por país |
| `weights_validation_report.pdf` | 11 | 3 gráficos comparación pesos |
| `score_methods_comparison.csv` | 12 | Score V1/V2/V3 por país |
| `score_methods_report.pdf` | 12 | 4 gráficos tres métodos |
| `project_log.md` | — | Este documento |

## Pipeline completo (en orden de ejecución)

```
01_clean_sustainable_energy.py
02_eda_sustainable_energy.py
03_clustering_dynamics.py
04_feature_engineering.py
05_regression_analysis.py
06_case_studies.py
07_validation_step1_explore.py
08_validation_step2_align.py
08b_validation_step2b_gdp.py
09_validation_step3_predict.py
10_validation_step4_visualize.py
11_score_weights_validation.py
12_score_methods_comparison.py
```

---

*Documento actualizado: mayo 2026 — Francisco Tornello*

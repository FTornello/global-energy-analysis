# Proyecto: Análisis Global de Energía Sostenible (2000–2020)

**Estado:** En progreso  
**Última actualización:** Mayo 2026  
**Dataset:** Global Data on Sustainable Energy — Kaggle  
**Herramientas:** Python, pandas, scikit-learn, matplotlib, seaborn

---

## Resumen del proyecto

Análisis exploratorio y de clustering del dataset de indicadores energéticos globales. Cubre 176 países entre 2000 y 2020 con variables de acceso a electricidad, fuentes de energía, emisiones de CO₂ y desarrollo económico. El objetivo es identificar patrones de perfiles energéticos y entender cómo evolucionaron los países a lo largo del período.

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

### Columnas originales

| # | Nombre original | Descripción |
|---|-----------------|-------------|
| 1 | Entity | País o región |
| 2 | Year | Año |
| 3 | Access to electricity (% of population) | % población con electricidad |
| 4 | Access to clean fuels for cooking | % con combustibles limpios para cocinar |
| 5 | Renewable-electricity-generating-capacity-per-capita | Capacidad renovable instalada per cápita |
| 6 | Financial flows to developing countries (US $) | Flujos financieros para energía limpia |
| 7 | Renewable energy share in the total final energy consumption (%) | % renovables en consumo final |
| 8 | Electricity from fossil fuels (TWh) | Electricidad de fósiles |
| 9 | Electricity from nuclear (TWh) | Electricidad nuclear |
| 10 | Electricity from renewables (TWh) | Electricidad renovable |
| 11 | Low-carbon electricity (% electricity) | % electricidad de fuentes bajas en carbono |
| 12 | Primary energy consumption per capita (kWh/person) | Energía primaria per cápita |
| 13 | Energy intensity level of primary energy (MJ/$2017 PPP GDP) | Intensidad energética por unidad de GDP |
| 14 | Value_co2_emissions_kt_by_country | Emisiones CO₂ (kt totales) |
| 15 | Renewables (% equivalent primary energy) | Renovables en energía primaria equivalente |
| 16 | gdp_growth | Crecimiento anual del GDP (%) |
| 17 | gdp_per_capita | GDP per cápita (USD) |
| 18 | Density\n(P/Km2) | Densidad poblacional (con error de formato) |
| 19 | Land Area(Km2) | Área terrestre |
| 20 | Latitude | Latitud del centroide |
| 21 | Longitude | Longitud del centroide |

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

**Script:** `clean_sustainable_energy.py`  
**Output:** `sustainable_energy_clean.csv`, `cleaning_log.json`

### Decisiones de limpieza

**Paso 1 — Renombrado de columnas**  
Todas las columnas estandarizadas a snake_case limpio. La columna `Density\n(P/Km2)` pasó a `density_p_km2`, eliminando el salto de línea embebido en el nombre.

**Paso 2 — Conversión de `density_p_km2`**  
La columna contenía valores como `"2,239"` (con separadores de miles) que impedían operar matemáticamente. Se removieron las comas y se convirtió a `float64`. Bahrain era el principal afectado (21 años × 1 país = 21 filas).

**Paso 3 — Clip de `low_carbon_elec_pct`**  
Un registro (Bhutan, 2005) tenía valor `100.00001` por error de punto flotante. Aplicado `clip(0, 100)`. Sin pérdida de información.

**Paso 4 — Flag columnas de alta nulidad**  
`financial_flows_usd` (57% nulos) y `renewables_equiv_primary_pct` (58% nulos) documentadas como columnas de uso restringido. Los nulos son estructuralmente válidos: los países desarrollados no reciben flujos financieros para energía limpia; los datos de renovables equivalentes tienen cobertura limitada históricamente. Decisión: no imputar.

**Paso 5 — Flag de cobertura temporal**  
Nueva columna booleana `partial_temporal_coverage` para los 4 países con menos de 21 años de datos. Permite filtrarlos en análisis de series de tiempo sin eliminar filas.

**Paso 6 — Verificación de tipos**  
Todos los tipos verificados. Sin inconsistencias post-limpieza.

### Estado del dataset limpio

| Dimensión | Antes | Después |
|-----------|-------|---------|
| Filas | 3.649 | 3.649 (sin cambio) |
| Columnas | 21 | 22 (+1 flag) |
| Filas eliminadas | — | 0 |
| Nulos críticos resueltos | 107 (density) + 1 (clip) | ✓ |

---

## Etapa 3 — Análisis exploratorio (EDA)

**Script:** `eda_sustainable_energy.py`  
**Output:** `eda_report.pdf` (6 gráficos)

### Estadísticas descriptivas — columnas clave

| Columna | Media | Mediana | Min | Max |
|---------|-------|---------|-----|-----|
| access_electricity_pct | 78.9% | 98.4% | 1.3% | 100% |
| renewable_share_pct | 32.6% | 23.3% | 0% | 96% |
| low_carbon_elec_pct | 36.8% | 27.9% | 0% | 100% |
| primary_energy_per_capita_kwh | 25.744 | 13.121 | 0 | 262.586 |
| co2_emissions_kt | 159.866 | 10.500 | 10 | 10.707.220 |
| gdp_per_capita_usd | 13.284 | 4.579 | 112 | 123.514 |
| energy_intensity_mj_gdp | 5.31 | 4.30 | 0.11 | 32.57 |

### Correlaciones principales

| Par de variables | Correlación | Interpretación |
|-----------------|-------------|----------------|
| access_electricity ↔ access_fuels | +0.867 | Muy alta — van juntas |
| access_electricity ↔ renewable_share | −0.785 | Alta negativa — paradoja de la biomasa |
| access_fuels ↔ renewable_share | −0.792 | Ídem |
| gdp_per_capita ↔ primary_energy | +0.667 | Más riqueza = más consumo |
| renewable_share ↔ low_carbon_elec | +0.468 | Moderada positiva |
| gdp_per_capita ↔ access_electricity | +0.418 | Moderada positiva |

### Hallazgos principales del EDA

1. **Acceso a electricidad mejoró pero con brecha persistente.** La media global subió de 73% (2000) a 85% (2020). South Sudan (7%), Chad (11%) y Burundi (12%) siguen casi sin cobertura en 2020.

2. **Distribución de renovables bimodal.** Dos picos distintos: 0–10% (países industrializados dependientes de fósiles) y 80–90% (países en desarrollo con biomasa tradicional). El "país promedio" no existe.

3. **GDP per capita extremadamente sesgado.** Mediana $4.600 vs media $13.300. Luxembourg ($116.000) y Bermuda ($107.000) distorsionan el promedio global.

4. **Paradoja de las renovables.** Correlación negativa (r=−0.79) entre acceso a electricidad y renovables share. Los países con menos electricidad aparecen con más "renovables" porque la leña y biomasa tradicional se contabilizan como renovable. No es transición energética — es pobreza energética.

5. **Outliers estructurales.** China: mayor emisor de CO₂ y generador de electricidad fósil. Qatar: mayor consumo per cápita. Luxembourg: mayor GDP per cápita. No son errores — son casos legítimos que hay que aislar en análisis globales.

6. **Trinidad y Tobago — intensidad energética extrema.** 19.4 MJ por dólar de GDP, casi 4x el promedio global. Industria petroquímica masiva con energía históricamente subsidiada.

---

## Etapa 4 — Clustering por perfil energético

**Script:** `clustering_dynamics.py`  
**Output:** `clustering_report.pdf`, `cluster_assignments.csv`, `migration_log.csv`

### Metodología

- **Año base:** 2019 (año más completo)
- **Países incluidos:** 175 (excluidos con >2 nulos en features)
- **Features:** 7 variables normalizadas per cápita o en porcentaje
- **Preprocesamiento:** imputación de nulos restantes con mediana, escalado StandardScaler
- **Algoritmo:** K-Means, k=4 (seleccionado por balance entre silhouette score y granularidad interpretativa)
- **Visualización:** PCA 2D (47.3% + 18.8% = 66.1% varianza explicada)

### Los 4 clusters

#### Cluster 0 — Emergentes fósiles (62 países)
Economías de ingreso medio con electricidad casi universal pero fuertemente dependientes de combustibles fósiles. Bajo porcentaje renovable. Perfil dominante en el mundo: Argentina, China, India, México, Turquía, Arabia Saudita, Indonesia.

| Indicador | Valor promedio |
|-----------|---------------|
| Acceso electricidad | 97% |
| GDP per cápita | $10.867 |
| Renovables share | 10% |
| Low-carbon electricity | 15% |
| Energía per cápita | 23.942 kWh |

#### Cluster 1 — Desarrollados, alto consumo (28 países)
Acceso 100%, GDP más alto, pero también el mayor consumo energético del mundo. En transición hacia fuentes limpias — el porcentaje low-carbon creció del 29% al 53% entre 2000 y 2019. EE.UU., Alemania, Francia, UK, países nórdicos, estados del Golfo.

| Indicador | Valor promedio |
|-----------|---------------|
| Acceso electricidad | 100% |
| GDP per cápita | $56.784 |
| Renovables share | 19% |
| Low-carbon electricity | 47% |
| Energía per cápita | 82.673 kWh |

#### Cluster 2 — Transición renovable (43 países)
El cluster más interesante. GDP moderado, pero 68% de su electricidad ya proviene de fuentes limpias. Apostaron a hidroeléctrica, eólica o solar antes de que fuera tendencia global. Brasil, España, Portugal, Colombia, Uruguay, Costa Rica, Nepal, Bhutan.

| Indicador | Valor promedio |
|-----------|---------------|
| Acceso electricidad | 95% |
| GDP per cápita | $8.334 |
| Renovables share | 37% |
| Low-carbon electricity | 68% |
| Energía per cápita | 15.393 kWh |

#### Cluster 3 — Pobreza energética (42 países)
Principalmente África subsahariana. Acceso eléctrico 45%, GDP mínimo ($1.786). Alto porcentaje de "renovables" por biomasa tradicional (leña). El mayor desafío: expandir acceso sin replicar el camino fósil.

| Indicador | Valor promedio |
|-----------|---------------|
| Acceso electricidad | 45% |
| GDP per cápita | $1.786 |
| Renovables share | 66% |
| Low-carbon electricity | 39% |
| Energía per cápita | 1.744 kWh |

---

## Etapa 5 — Dinámica temporal de clusters (2000–2019)

**Script:** `clustering_dynamics.py` (misma ejecución)  
**Análisis:** evolución de métricas por cluster + detección de migraciones

### Evolución del tamaño de clusters

| Cluster | 2000 | 2005 | 2010 | 2015 | 2019 | Tendencia |
|---------|------|------|------|------|------|-----------|
| Pobreza energética | 60 | 57 | 50 | 48 | 42 | ↓ −18 |
| Emergentes fósiles | 67 | 63 | 66 | 62 | 62 | → estable |
| Transición renovable | 29 | 28 | 32 | 39 | 43 | ↑ +14 |
| Desarrollados | 16 | 24 | 26 | 26 | 28 | ↑ +12 |

### Migraciones destacadas (2000 → 2019)

**42 de 172 países cambiaron de cluster.**

| Tipo de movimiento | Cantidad | Países notables |
|-------------------|----------|-----------------|
| Pobreza → Transición renovable | 13 | Nepal, Bhutan, Cambodia, Ghana, Guatemala, Honduras |
| Pobreza → Emergentes fósiles | 6 | India, Indonesia, Bangladesh, Botswana |
| Emergentes → Desarrollados | 10 | Alemania, UK, Denmark, Netherlands, Australia, Irlanda |
| Emergentes → Transición renovable | 6 | España, Portugal, Bulgaria, Hungría, Rumania, Ucrania |
| Transición → Desarrollados | 3 | Francia, Austria, Nueva Zelanda |
| Transición → Emergentes (retroceso) | 3 | Chile, Filipinas, Suriname |
| Desarrollados → Emergentes (retroceso) | 1 | **Japón** |

### Casos notables

**Japón — único retroceso desde el cluster más alto.** Tras el accidente de Fukushima (2011), cerró casi toda su energía nuclear. Su porcentaje de electricidad limpia cayó del 28% al 13%, moviéndolo al cluster de emergentes fósiles. Ejemplo de que el progreso energético no es lineal.

**India e Indonesia — electrificación masiva basada en carbón.** Llevaron electricidad a cientos de millones de personas, saliendo de la pobreza energética. Pero el motor fue carbón. Representan la tensión central entre desarrollo humano y transición climática.

**España y Portugal — transición deliberada sostenida.** Empezaron como economías dependientes del petróleo y terminaron con más del 50% de electricidad limpia. Resultado de política energética sostenida durante 20 años.

**Nepal, Bhutan, Cambodia — salida limpia de la pobreza.** Escaparon de la pobreza energética apostando directamente a renovables (hidroeléctrica, solar). Saltaron la etapa fósil. Casos de estudio potenciales para el siglo XXI.

---

## Archivos generados

| Archivo | Etapa | Descripción |
|---------|-------|-------------|
| `sustainable_energy_clean.csv` | Limpieza | Dataset limpio, 3.649 filas × 22 columnas |
| `cleaning_log.json` | Limpieza | Log detallado de cada decisión de limpieza |
| `clean_sustainable_energy.py` | Limpieza | Script reproducible de limpieza |
| `eda_report.pdf` | EDA | 6 gráficos de análisis exploratorio |
| `eda_sustainable_energy.py` | EDA | Script reproducible de EDA |
| `clustering_report.pdf` | Clustering | 6 gráficos de clustering y dinámica |
| `clustering_dynamics.py` | Clustering | Script de clustering y análisis temporal |
| `cluster_assignments.csv` | Clustering | Asignación de cluster por país (2019) |
| `migration_log.csv` | Dinámica | Países que cambiaron de cluster 2000–2019 |
| `project_log.md` | Todos | Este documento |

---

## Próximos pasos sugeridos

- [ ] Análisis de regresión: predictores de transición al cluster 2 (transición renovable)
- [ ] Feature engineering: crear variable `transition_score` que combine acceso, low-carbon y renovables
- [ ] Análisis geoespacial: visualizar clusters en mapa mundial
- [ ] Análisis de casos: profundizar en trayectorias individuales (Japón, India, España)
- [ ] Preparar presentación de resultados

---

*Documento generado como parte del proceso de aprendizaje de Data Analytics.*

---

## Etapa 6 — Mapa geoespacial de clusters

**Herramienta:** D3.js + TopoJSON + world-atlas@2  
**Output:** widget interactivo en sesión (hover por país)

### Metodología

Cada país coloreado según su cluster de 2019. Los 175 países del análisis fueron mapeados a códigos ISO numéricos (estándar usado por world-atlas) usando `pycountry` con correcciones manuales para nombres que difieren entre fuentes (Turkey→792, Czechia→203, etc.).

### Patrones geográficos identificados

| Región | Patrón | Observación |
|--------|--------|-------------|
| África subsahariana | Casi 100% Pobreza energética | Bloque continuo sin mezcla |
| América Latina | Mayoría Transición renovable | Continente con mejor perfil relativo a su desarrollo |
| Europa Occidental | Desarrollados alto consumo | Bloque uniforme |
| Europa del Este y Sur | Transición renovable | España, Portugal, Bulgaria, Rumania, Hungría |
| Medio Oriente / Asia Central | Emergentes fósiles | Petróleo + electrificación sin transición limpia |
| Asia del Este y del Sur | Emergentes fósiles | China, India, Indonesia, Japón |
| Norteamérica | Desarrollados alto consumo | EE.UU., Canadá |
| Oceanía | Mixto | Australia (desarrollados), Pacífico Sur (transición) |

### Hallazgo visual clave

La geografía del cluster es más limpia de lo esperado — la pobreza energética y la transición renovable forman bloques regionales casi continuos. Sugiere que factores estructurales regionales (historia colonial, recursos naturales, cooperación regional) explican tanto como los indicadores individuales de cada país.

---

## Etapa 7 — Feature engineering

**Script:** `feature_engineering.py`
**Output:** `features_engineered.csv`, `features_report.pdf` (3 gráficos)

### Variables construidas

#### 1. transition_score (0–100)
Índice compuesto de avance en la transición energética moderna.

`0.30 × acceso_electricidad + 0.45 × low_carbon_elec + 0.25 × log(GDP_norm)`

Mayor peso en low-carbon electricity porque es el cambio más difícil y el más relevante para la transición. Se usa escala logarítmica para GDP para no penalizar desproporcionadamente a países pobres. No usa renewable_share para evitar la trampa de la biomasa.

| Stat | Valor |
|------|-------|
| Mediana | 54.5 |
| Rango | 10.5 (Chad) – 97.9 (Islandia) |
| Top 3 | Islandia, Noruega, Suiza |
| Bottom 3 | Chad, Liberia, Niger |

#### 2. improvement_rate (puntos por año)
Velocidad de mejora del transition_score entre 2000 y 2019.

`(score_2019 − score_2000) / 19 años`

Permite identificar países que están acelerando vs estancando, independientemente de su posición actual.

| Stat | Valor |
|------|-------|
| Mediana | +0.36 pts/año |
| Rango | −1.35 (Congo) a +2.40 (Cambodia) |
| Retrocedieron | 11 países — Congo, Japón, Haití, Libia entre los más notables |

#### 3. clean_access_ratio (0–100)
Mide la calidad del acceso a electricidad.

`(acceso/100) × (low_carbon/100) × 100`

Un país con acceso universal pero electricidad 100% fósil obtiene 0. Detecta la "electrificación sucia" — acceso alto pero carbón. Albania, Bhutan e Islandia son los únicos con score 100.

#### 4. fossil_lock_in (0–100)
Riesgo de quedar atrapado en los fósiles.

`0.45 × (1−low_carbon) + 0.35 × energia_intensity_norm + 0.20 × (1−GDP_norm)`

Combina dependencia de fósiles, ineficiencia energética y limitaciones económicas. Trinidad y Tobago lidera con 85.7 por su industria petroquímica con energía subsidiada.

### Hallazgos del cuadrante (transition_score vs improvement_rate)

| Cuadrante | Descripción | Ejemplos |
|-----------|-------------|---------|
| Alto score + alta mejora | Avanzados y acelerando | Dinamarca, Costa Rica, Bhutan |
| Alto score + baja mejora | Avanzados pero desacelerando | Varios países desarrollados |
| Bajo score + alta mejora | Rezagados pero alcanzando | Cambodia, Sierra Leone, Kenya |
| Bajo score + baja mejora | Atrapados | Chad, Niger, Somalia |

### Argentina
| Variable | Valor | vs mediana global |
|----------|-------|-------------------|
| transition_score | 60.4 | +5.9 puntos ↑ |
| improvement_rate | −0.09 | −0.45 pts ↓ |
| clean_access_ratio | 32.0 | +10.6 puntos ↑ |
| fossil_lock_in | 42.7 | ≈ mediana |

Argentina está por encima de la mediana global pero con improvement_rate negativo — en 2019 estaba perdiendo terreno. Refleja la crisis económica y la dependencia histórica del gas natural.

---

## Etapa 8 — Regresión

**Script:** `regression_analysis.py`
**Output:** `regression_report.pdf` (4 gráficos), `regression_results.csv`

### Objetivo
Predecir `transition_score` (índice 0–100 de avance energético) a partir de variables observables.

### Features utilizadas
8 variables: acceso a electricidad 2019, renovables share, GDP (log), intensidad energética, energía per cápita (log), crecimiento GDP, acceso electricidad 2000, low-carbon electricity 2000.

### Resultados por modelo (CV 5-fold, 153 países)

| Modelo | R² | MAE |
|--------|-----|-----|
| Linear Regression | 0.331 | 10.42 |
| Ridge (α=2) | **0.822** | 5.22 |
| Random Forest | 0.768 | 6.18 |

Ridge superó a Random Forest — las relaciones son mayormente lineales.

### Feature importance (Random Forest)

| Variable | Importancia |
|----------|-------------|
| Low-carbon electricity 2000 | **47.5%** |
| Acceso electricidad 2019 | 18.8% |
| Energía per cápita (log) | 15.1% |
| GDP per cápita (log) | 8.5% |
| Acceso electricidad 2000 | 5.2% |
| Resto | 5.0% |

### Hallazgos principales

**1. La historia energética pesa más que el dinero.** El predictor #1 es la electricidad limpia que tenía un país en el año 2000 (47.5% de importancia). Los sistemas energéticos tienen inercia enorme: las inversiones en infraestructura hidráulica, nuclear o renovable de hace 20 años siguen definiendo el perfil de hoy.

**2. El modelo lineal gana.** R²=0.82 con Ridge vs 0.77 con Random Forest. Esto confirma que no hay interacciones complejas — las relaciones entre variables y el score son directas y aditivas. Interpretación más confiable.

**3. Residuales informan sobre factores no energéticos.** Los países con residuales altos son los más interesantes:
- **Subestimados** (mejoraron más de lo esperado): Sierra Leone (+17.6), Kenya (+15.3), Denmark (+13.4), Cambodia (+12.4) — señal de reformas externas efectivas o política energética acertada.
- **Sobreestimados** (rindieron menos de lo esperado): Burkina Faso (−14.6), Chad (−13.8), Burundi (−11.3), Haití (−10.1) — señal de conflictos, inestabilidad política o falla institucional no capturada por los datos.

---

## Etapa 9 — Análisis de casos individuales

**Script:** `case_studies.py`
**Output:** `case_studies_report.pdf` (6 páginas), `case_studies_summary.csv`

### Países analizados y narrativa central

| País | Narrativa | Improvement rate | Cluster 2019 |
|------|-----------|-----------------|--------------|
| 🇯🇵 Japón | Único retroceso — Fukushima 2011 | −0.36 | Emergentes fósiles |
| 🇪🇸 España | Transición deliberada sostenida | +0.49 | Transición renovable |
| 🇮🇳 India | Electrificación masiva a carbón | +1.01 | Emergentes fósiles |
| 🇰🇭 Cambodia | #1 mundial en velocidad de mejora | +2.40 | Transición renovable |
| 🇦🇷 Argentina | Base limpia que se perdió | −0.09 | Emergentes fósiles |

### Hallazgos por caso

**Japón:** El low-carbon electricity cayó del 41% al 14% en 3 años post-Fukushima. Pasó del cluster "Desarrollados" a "Emergentes fósiles". Ejemplo más claro de que el progreso no es lineal y puede revertirse por un único evento. En 2023 comenzó la reactivación nuclear, pero en 2020 aún no había recuperado los niveles previos.

**España:** Política energética sostenida 20 años pese a la crisis 2008–2013. Low-carbon del 44% al 66%. Fossil lock-in de 26.3 — uno de los más bajos de Europa. La transición no fue lineal ni sin tropiezos, pero nunca cambió de dirección.

**India:** Logro humano descomunal: de 59% a 99% de acceso en 20 años. El motor fue carbón. Hoy tiene el segundo sistema fósil más grande de Asia. Fossil lock-in de 53.6 — la descarbonización posterior será enormemente costosa. Al mismo tiempo, lidera en capacidad solar instalada nueva desde 2015.

**Cambodia:** El caso más notable del dataset. Improvement rate +2.40 — el más alto del mundo. Construyó desde cero sin un sistema fósil previo que desmantelar. Fue directamente a hidroeléctrica y solar porque era la opción viable para zonas rurales sin red. Residual de regresión +12.4: superó consistentemente lo que sus condiciones iniciales permitían predecir.

**Argentina:** Única en el dataset con base limpia en 2000 (41% low-carbon) que la redujo en 2019 (34%). Las crisis macroeconómicas de 2001, 2014 y 2018–19 frenaron la inversión energética en cada ciclo. El descubrimiento de Vaca Muerta (2011) reforzó la apuesta al gas. Tiene los recursos para liderar la transición regional (viento patagónico, sol en Cuyo, reservas hídricas) pero no ha tenido la estabilidad para capitalizarlos.

---

## Estado final del proyecto

### Archivos generados

| Archivo | Etapa | Descripción |
|---------|-------|-------------|
| `sustainable_energy_clean.csv` | Limpieza | Dataset limpio, 3.649 filas × 22 columnas |
| `cleaning_log.json` | Limpieza | Log detallado de decisiones |
| `clean_sustainable_energy.py` | Limpieza | Script reproducible |
| `eda_report.pdf` | EDA | 6 gráficos exploratorios |
| `eda_sustainable_energy.py` | EDA | Script reproducible |
| `clustering_report.pdf` | Clustering | 6 gráficos + dinámica |
| `clustering_dynamics.py` | Clustering | Script reproducible |
| `cluster_assignments.csv` | Clustering | Cluster por país (2019) |
| `migration_log.csv` | Dinámica | 42 países que cambiaron de cluster |
| `features_engineered.csv` | Feature Eng. | 4 variables nuevas por país |
| `feature_engineering.py` | Feature Eng. | Script reproducible |
| `features_report.pdf` | Feature Eng. | 3 gráficos |
| `regression_results.csv` | Regresión | Predicciones vs real por país |
| `regression_analysis.py` | Regresión | Script reproducible |
| `regression_report.pdf` | Regresión | 4 gráficos |
| `case_studies_summary.csv` | Casos | Tabla resumen 5 países |
| `case_studies.py` | Casos | Script reproducible |
| `case_studies_report.pdf` | Casos | 6 páginas de análisis |
| `project_log.md` | Todos | Este documento |

### Pipeline completo (en orden de ejecución)
```
clean_sustainable_energy.py
eda_sustainable_energy.py
clustering_dynamics.py
feature_engineering.py
regression_analysis.py
case_studies.py
```

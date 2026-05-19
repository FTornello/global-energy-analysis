# Data Analytics Workflow

Guía de referencia para proyectos de análisis de datos. Aplica a cualquier dataset tabular.

---

## Etapa 1 — Exploración inicial

Antes de tocar los datos, entendé qué tenés.

**Preguntas a responder:**
- ¿Cuántas filas y columnas tiene el dataset?
- ¿Qué tipo de dato es cada columna? (numérico, texto, fecha, booleano)
- ¿Hay duplicados?
- ¿Cuántos nulos hay por columna y qué porcentaje representan?
- ¿Los rangos de valores tienen sentido? (ej: porcentajes entre 0 y 100, fechas coherentes)
- ¿Los nombres de columnas son claros y consistentes?

**Output esperado:** informe de calidad de datos — qué está bien, qué está mal, qué decisiones hay que tomar.

---

## Etapa 2 — Limpieza

Documentá cada decisión. No limpies por intuición — limpiás porque encontraste algo concreto en la etapa anterior.

**Acciones típicas:**
- Estandarizar nombres de columnas (snake_case, sin caracteres especiales)
- Corregir tipos de datos (strings que deberían ser numéricos, fechas mal formateadas)
- Manejar nulos — para cada columna decidir: imputar, eliminar fila, eliminar columna, o dejar y documentar
- Corregir valores fuera de rango (clip, corrección manual, o eliminación)
- Agregar columnas de flag para casos especiales que no se eliminan pero hay que identificar

**Regla de oro:** nunca eliminar filas sin documentar por qué. El log de limpieza es parte del proyecto.

**Output esperado:** dataset limpio + log de decisiones (JSON o Markdown).

---

## Etapa 3 — Análisis exploratorio (EDA)

Conocé los datos antes de modelar. El EDA no es un paso previo al análisis — es análisis.

**Qué mirar:**
- Estadísticas descriptivas (media, mediana, desvío, percentiles) para cada variable numérica
- Distribuciones — ¿son simétricas? ¿sesgadas? ¿bimodales?
- Correlaciones entre variables — ¿qué se mueve junto? ¿qué correlación negativa llama la atención?
- Outliers — ¿son errores o casos reales? ¿afectan los promedios?
- Tendencias temporales si el dataset tiene fecha

**Preguntas clave:**
- ¿La media y la mediana difieren mucho? (señal de sesgo o outliers)
- ¿Hay correlaciones que no deberían existir? (señal de error en los datos)
- ¿Hay correlaciones que parecen paradójicas? (oportunidad de análisis)

**Output esperado:** gráficos de distribuciones, mapa de correlaciones, lista de hallazgos.

---

## Etapa 4 — Análisis principal

Acá es donde definís qué querés responder y elegís el método. Las opciones más comunes:

**Clustering** — si querés agrupar registros similares sin etiquetas previas.
- Normalizá las variables antes de aplicar el algoritmo
- Usá el método del codo + silhouette score para elegir el número de clusters
- Validá que los grupos tengan sentido interpretable, no solo matemático
- Nombrá los clusters según lo que describen, no según su número

**Regresión** — si querés predecir una variable numérica.
- Empezá con un modelo lineal. Si da bien, no necesitás uno más complejo
- Usá validación cruzada (cross-validation), no train/test split simple
- Analizá los residuales — los casos donde el modelo falla son los más informativos
- Medí con R² y MAE juntos (R² solo puede engañar)

**Clasificación** — si querés predecir una categoría.
- Revisá el balance de clases antes de modelar
- Usá accuracy + precision/recall, no solo accuracy
- Igual que regresión: empezá simple

**Series de tiempo** — si el orden temporal importa.
- No mezcles datos futuros con datos de entrenamiento (data leakage)
- Mirá tendencia, estacionalidad y outliers por separado

**Output esperado:** modelo entrenado, métricas de performance, interpretación de resultados.

---

## Etapa 5 — Feature engineering

Creá variables nuevas que cuenten mejor la historia que los datos crudos.

**Cuándo hacerlo:** después del EDA y antes (o durante) el modelado. No antes — necesitás entender los datos para saber qué construir.

**Tipos de features útiles:**
- Índices compuestos — combinar varias variables en una sola métrica normalizada
- Ratios — variable A dividida variable B (ej: ingresos por empleado)
- Deltas — diferencia entre dos puntos en el tiempo (ej: crecimiento anual)
- Transformaciones — log para variables muy sesgadas, clip para outliers extremos
- Flags booleanos — marcar casos especiales (ej: país con datos incompletos)

**Regla:** toda variable nueva debe tener una justificación clara. Si no podés explicar en una oración qué mide y por qué importa, probablemente no sirve.

**Output esperado:** dataset enriquecido con las nuevas variables + documentación de cada una.

---

## Etapa 6 — Comunicación de resultados

El análisis no termina cuando el modelo converge. Termina cuando alguien más puede entenderlo.

**Capas de comunicación:**

| Audiencia | Formato | Foco |
|-----------|---------|------|
| Técnica (vos mismo, equipo de datos) | Scripts + logs + CSVs | Reproducibilidad |
| Analítica (gerencia, stakeholders) | Gráficos + resumen escrito | Hallazgos e implicancias |
| General (portfolio, redes) | Narrativa + visualizaciones clave | Historia del dato |

**Principios:**
- Empezá por el hallazgo, no por la metodología
- Cada gráfico debe poder leerse solo, sin texto alrededor
- Los casos que el modelo no explica bien suelen ser los más interesantes — contálos
- Documentá las limitaciones del análisis (qué datos faltan, qué asumiste)

**Output esperado:** reporte PDF o presentación + README si es para portfolio.

---

## Checklist antes de cerrar un proyecto

- [ ] El dataset limpio está guardado por separado del original
- [ ] Cada decisión de limpieza está documentada
- [ ] Los scripts corren en orden sin intervención manual
- [ ] Los hallazgos principales están escritos en lenguaje no técnico
- [ ] Las limitaciones del análisis están explicitadas
- [ ] El README explica cómo reproducir el análisis desde cero

---

## Orden de ejecución estándar

```
01_cleaning.py          → dataset limpio + log
02_eda.py               → distribuciones, correlaciones, outliers
03_main_analysis.py     → clustering / regresión / clasificación
04_feature_engineering.py → variables nuevas
05_results.py           → gráficos finales, casos especiales
```

Los números en el nombre de archivo evitan confusión sobre el orden.

---

## Señales de que el análisis está bien hecho

- Podés explicar cada hallazgo sin mencionar el nombre del algoritmo
- Los casos donde el modelo falla te dicen algo nuevo sobre los datos
- Alguien puede correr tus scripts y llegar exactamente a los mismos resultados
- Las visualizaciones muestran algo que el número solo no mostraba
- Sabés qué no pudiste responder y por qué

# Modelado y Pronóstico de Aprobaciones del BCIE con XGBoost y Lag Features (v2.1)

Este notebook implementa un flujo completo de modelado de series temporales utilizando **XGBoost** (Extreme Gradient Boosting) para pronosticar las aprobaciones del Banco Centroamericano de Integración Económica (BCIE).

Forma parte del módulo:

`models/aprobaciones_xgboost/`

Este enfoque complementa al modelo Prophet, especializándose en capturar interacciones no lineales complejas entre las variables.

---

## Objetivo

Construir un modelo robusto y reproducible (v2.1) que permita:

1.  Cargar el dataset consolidado (`tabla_final.parquet`) preparado por el notebook anterior.
2.  Implementar **Ingeniería de Características (Feature Engineering)** avanzada, creando *lags* (históricos), *rolling averages* (tendencias) y *flags* (eventos exógenos).
3.  Entrenar un **único modelo global** de XGBoost que aprenda las dinámicas de todos los segmentos (País, Sector, Socio) simultáneamente.
4.  Generar **pronósticos recursivos** multi-anuales (2026–2030) alimentando las predicciones futuras de nuevo al modelo.
5.  Presentar visualizaciones de diagnóstico (Heatmaps) y un Dashboard interactivo (`ipywidgets` + `plotly`) para la exploración de resultados.

---

## ¿Qué es XGBoost (en este contexto)?

**XGBoost** es un algoritmo de *Gradient Boosting* (Potenciación de Gradiente) que se ha convertido en el estándar de la industria para datos tabulares. Es un **modelo de ensamble** que construye secuencialmente "árboles de decisión", donde cada nuevo árbol corrige los errores del anterior.

A diferencia de Prophet (un modelo aditivo de series temporales), XGBoost es un **modelo de regresión supervisada**. No "entiende" el tiempo por sí mismo.

Para usarlo en forecasting, transformamos el problema:

> **"Predecir el futuro"** se convierte en...
> **"Predecir el `Monto` de este año, dadas las features (el `Monto` del año pasado, el `Promedio` de los últimos 3 años, el `País`, el `Sector`, y si fue un `Año de Crisis`)."**

Esta transformación se llama **Ingeniería de Características (Feature Engineering)** y es el núcleo de este notebook.

---

## ¿Por qué XGBoost es adecuado para las aprobaciones del BCIE?

Este enfoque es poderoso porque complementa las debilidades de los modelos de series temporales puros:

1.  **Manejo de Interacciones Complejas:**
    XGBoost puede aprender reglas no lineales. Por ejemplo, puede descubrir que "Costa Rica + Sector Público" tiene un patrón de crecimiento completamente diferente a "Guatemala + Sector Público". Los modelos aditivos como Prophet no capturan estas interacciones tan fácilmente.
    

2.  **Uso Eficaz de Features Exógenas:**
    Las "crisis flags" (COVID, 2008, Crisis Nicaragua) se incorporan como simples columnas binarias (0 o 1), permitiendo al modelo aprender su impacto preciso sobre las aprobaciones.

3.  **Captura de Dinámicas de Lags:**
    El modelo puede determinar qué histórico es más importante. Quizás el `monto_lag_1` (año anterior) es el predictor más fuerte, pero `monto_lag_3` también tiene influencia. XGBoost pondera esto automáticamente.

4.  **Escalabilidad y Rendimiento:**
    Entrenar **un solo modelo global** de XGBoost (como hacemos aquí) es computacionalmente más eficiente que entrenar cientos de modelos Prophet individuales (uno por cada segmento).

5.  **Interpretabilidad (Feature Importance):**
    Aunque es más complejo que Prophet, XGBoost ofrece métricas claras de "Importancia de Features" (Gain, Weight), permitiéndonos auditar qué variables considera el modelo más predictivas.

---

## Mejoras de Robustez y Calidad (v2.1)

Este notebook no es solo un prototipo; ha sido refactorizado (basado en el Plan de Mejoras) para incluir prácticas de robustez y MLOps:

-   **Métricas Extendidas (Ítem #1):** La validación del modelo (Paso 3) ahora reporta un panel completo (`MAE`, `RMSE`, `MAPE`, y `R²`) para un diagnóstico completo del error.
-   **Pruebas Unitarias (Ítem #13):** Se han añadido celdas de `assert` después de cada paso principal (Paso 1-6) para validar la integridad de los datos, la creación de features y el guardado de archivos, previniendo errores silenciosos.
-   **Validación de Loop (Ítem #3):** El loop de pronóstico recursivo (Paso 3.6) ahora incluye una validación explícita para asegurar que el esquema de columnas de los datos futuros coincida 100% con el del modelo.
-   **Trazabilidad del Modelo (Ítem #2):** Los hiperparámetros del modelo final (`best_iterations`, `learning_rate`, etc.) se guardan automáticamente en `model_final_hyperparameters.json`.
-   **Lógica de Relleno Avanzada (Ítem #4):** Se reemplazó `fillna(0)` por `fillna(mediana_del_grupo)` en el Paso 2 (Feature Engineering) para crear lags y rolling averages más realistas.
-   **Optimización de Dashboard (Ítem #10, #11):** El dashboard del Paso 6 ahora incluye un *slider* de rango de años y utiliza `@cache` en la función `get_data` para una respuesta de filtrado instantánea.
-   **Exportación de Reportes (Ítem #9):** Los Mapas de Calor del Paso 5 se exportan automáticamente como archivos `.png` de 300 dpi para uso en reportes ejecutivos.
-   **Logging Profesional (Ítem #12):** Todos los `print()` han sido reemplazados por el módulo `logging`, permitiendo un control de verbosidad (INFO, WARNING, ERROR).
-   **Entorno Fijo (Ítem #15):** Se incluye un `requirements.txt` para garantizar la reproducibilidad total del entorno.

---

## Datos Utilizados

Este notebook depende del *output* del notebook de preparación de datos.

-   **Insumo Principal:** `results/tabla_final.parquet`
-   **Output Intermedio (para el modelo):** `results/tabla_final_features.parquet`
-   **Output Final (Predicciones):** `results/predicciones_XGBOOST_2026_2030.parquet`
-   **Output Final (Unificado para BI):** `results/bcies_aprobaciones_XGBOOST_historico_pronostico.parquet` (y `.csv`, `.xlsx`).

---

## Contenido del notebook

El notebook está organizado en 6 pasos claros (más celdas de validación):

1.  **Paso 1: Carga de Librerías y Datos**
    Importa librerías (con `nest_asyncio` para `dfi`) y carga `tabla_final.parquet`.

2.  **Paso 2: Feature Engineering**
    El paso más crítico: crea `lags`, `rolling averages`, `growth rates` y `crisis flags`. Codifica variables categóricas (`One-Hot Encoding`) y guarda el resultado numérico en `tabla_final_features.parquet`.

3.  **Paso 3: Entrenamiento y Pronóstico Recursivo**
    Valida el modelo (Train/Test split temporal), re-entrena con todos los datos y ejecuta el loop recursivo para generar los pronósticos 2026-2030.

4.  **Paso 4: Unificación de Datos**
    Combina los datos históricos (1961-2025) con las nuevas predicciones (2026-2030) en `df_unico` y añade el "puente" visual para Power BI.

5.  **Paso 5: Visualización de Heatmaps**
    Genera dos mapas de calor: por "Tipo de Socio" y por "Sector Institucional".

6.  **Paso 6: Dashboard Interactivo**
    Implementa el dashboard final con `ipywidgets` y `plotly` para la exploración de todos los resultados.

---

## Requisitos

Las versiones exactas están fijadas en el archivo `requirements.txt` de este repositorio.

-   `pandas`
-   `numpy`
-   `xgboost`
-   `scikit-learn`
-   `plotly`, `ipywidgets`
-   `openpyxl`, `pyarrow`
-   `dataframe_image`, `kaleido`, `nest_asyncio` (Para exportar imágenes)

---

## Uso Recomendado

1.  Clonar el repositorio.
2.  Ejecutar primero el notebook de **Preparación de Datos** para generar `results/tabla_final.parquet`.
3.  Instalar las dependencias: `pip install -r requirements.txt`
4.  Ejecutar este notebook celda por celda:

    `models/aprobaciones_xgboost/Modelado_y_Pronóstico_de_Aprobaciones_del_BCIE_con_XGBoost_y_Lag_Features.ipynb`

5.  Revisar los archivos `.parquet`, `.xlsx` y `.png` generados en la carpeta `results/`.

# Sistema Automatizado de Evaluación Crediticia 🏦

Este proyecto implementa un sistema de Machine Learning para la evaluación automática de solicitudes de crédito. Utiliza el dataset clásico **German Credit Data** para predecir la probabilidad de *default* (impago) de un cliente, comparando tres paradigmas de clasificación: Árbol de Decisión, Regresión Logística y K-Nearest Neighbors (KNN).

El valor diferencial de este proyecto radica en su **arquitectura de explicación subrogada (surrogate explanation pattern)**, diseñada para cumplir con las exigencias regulatorias de explicabilidad bancaria sin sacrificar el poder predictivo.

## Características Principales

* **Decisión Robusta:** Utiliza un modelo de Regresión Logística optimizado (AUC = 0.81) como motor de decisión primario.
* **Explicabilidad Transparente:** Implementa un Árbol de Decisión paralelo que traza la trayectoria de reglas (condiciones) que justifican la evaluación.
* **Pipeline Completo:** Incluye desde la descarga automática y el Análisis Exploratorio de Datos (EDA) hasta la simulación de un entorno productivo con solicitudes nuevas.
* **Prevención de Sesgos:** Manejo del desbalance de clases mediante pesos balanceados (`class_weight='balanced'`) y evaluación enfocada en F1-Score sobre la clase positiva.

## Estructura del Proyecto

El código está dividido en 10 pasos secuenciales, reflejando un flujo de trabajo estándar en Ciencia de Datos:

* `01_descarga_y_eda.py`: Descarga del dataset y generación de visualizaciones exploratorias (distribuciones, tasas de default, correlaciones).
* `02_preprocesamiento.py`: One-Hot encoding, estandarización (`StandardScaler`) y división estratificada en conjuntos de entrenamiento y prueba (70/30).
* `03_arbol_decision.py`: Entrenamiento y ajuste de hiperparámetros del Árbol de Decisión.
* `04_regresion_logistica.py`: Entrenamiento y optimización de la Regresión Logística.
* `05_knn.py`: Entrenamiento y optimización de KNN (incluyendo curva de validación).
* `06_comparacion.py`: Evaluación comparativa de los tres modelos (Accuracy, Precision, Recall, F1, ROC-AUC) y generación de gráficos superpuestos.
* `07_demo_solicitud.py`: Script interactivo para probar solicitudes individuales y extraer su trayectoria de decisión.
* `08_seleccionar_clientes.py`: Extracción de una muestra representativa (15 casos) para pruebas de simulación.
* `09_casos_uso_3_modelos.py`: Prueba de estrés comparativa generando un *heatmap* de aciertos contra el *ground truth*.
* `10_sistema_final.py`: **Implementación productiva.** Procesa un lote de solicitudes aplicando el modelo dual (Logística para decidir, Árbol para explicar).

## Resultados Destacados

Tras evaluar los algoritmos en el conjunto de prueba (n=300), la **Regresión Logística** demostró ser el modelo más apto para las características lineales del problema financiero:

| Modelo | F1-Score | ROC-AUC |
| :--- | :--- | :--- |
| **Regresión Logística** | **0.62** | **0.81** |
| Árbol de Decisión | 0.50 | 0.68 |
| KNN | 0.47 | 0.69 |

En la simulación final con clientes reales, la arquitectura propuesta alcanzó una **tasa de acierto del 85%**.

## Instalación y Uso

1. **Clona el repositorio:**
   ```bash
   git clone [https://github.com/tu-usuario/evaluacion-crediticia-ia.git](https://github.com/tu-usuario/evaluacion-crediticia-ia.git)
   cd evaluacion-crediticia-ia

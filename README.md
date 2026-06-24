# Proyecto de Clasificación — Predicción de Aprobación de Préstamos (Loan Risk Prediction)

## 📝 Introducción y Objetivos
Este proyecto aborda un problema de **clasificación binaria**: predecir si una solicitud de préstamo será aprobada (`1`) o rechazada (`0`) a partir de las características socioeconómicas del solicitante. 

El objetivo final es doble:
* **Analítico:** Construir, comparar y optimizar varios modelos de clasificación, manteniendo el *overfitting* por debajo del **5%** (requisito central del proyecto).
* **Productivo:** Dejar el modelo final exportado y listo para ser consumido por una aplicación web en **Streamlit** que reciba los datos de un cliente nuevo y devuelva la predicción en tiempo real, junto con otros componentes avanzados (validación cruzada, tuning de hiperparámetros, sistema de feedback, ingesta de nuevos datos, dockerización, base de datos y tests unitarios).

---

## 📊 Dataset y Variables
El proyecto utiliza un conjunto de datos que cuenta con **5.000 solicitudes de préstamo**, pasando por las siguientes etapas de almacenamiento:
* `data/loan_risk_prediction_dataset.csv` (Dataset original)
* `data/dataset_clean.csv` (Dataset limpio tras el tratamiento de outliers)

### Características del Dataset:

| Columna | Descripción |
| :--- | :--- |
| **Age** | Edad del solicitante |
| **Income** | Ingresos anuales (USD) |
| **LoanAmount** | Importe del préstamo solicitado (USD) |
| **CreditScore** | Puntuación crediticia (300-850) |
| **YearsExperience** | Años de experiencia laboral |
| **Gender** | Género |
| **Education** | Nivel educativo |
| **City** | Ciudad de residencia |
| **EmploymentType** | Situación laboral |
| **LoanApproved** | **Variable objetivo:** 1 = aprobado, 0 = rechazado |

---

## 🛠️ Estructura del Ciclo de Vida del Proyecto

### PARTE 01: Análisis y Preparación
* **FASE 1 - Carga y Auditoría de Datos:** Inspección inicial de las dimensiones, tipos de datos y verificación de la calidad de la información.
* **FASE 2 - Limpieza de Datos y Outliers:** Identificación y tratamiento de valores atípicos o nulos para asegurar la estabilidad de los algoritmos.
* **FASE 3 - EDA Multivariado:** Análisis Exploratorio de Datos enfocado en las correlaciones con el objetivo de negocio. Se identificó una fuerte correlación positiva con `CreditScore` e `Income`, y un impacto drástico de `EmploymentType` (donde los desempleados presentan una tasa de aprobación de apenas ~3% frente al ~33% de asalariados o autónomos).

### PARTE 02: Modelado y Puesta en Producción
* **FASE 4 - Train/Test Split y Pipeline de Preprocesado:** División estratificada 80/20 (preservando la proporción original de la variable objetivo: 77% / 23%). Construcción de un `ColumnTransformer` que aplica codificación (*One-Hot Encoding* para variables nominales y *Label Encoding* para ordinales como `Education`) acoplado con un escalado estándar (`StandardScaler`) para evitar el *data leakage*.
* **FASE 5 - Comparativa de Modelos de Clasificación:** Evaluación inicial empleando algoritmos como `LogisticRegression`, `DecisionTreeClassifier`, `RandomForestClassifier` y `GradientBoostingClassifier`.
* **FASE 6 - Validación Cruzada y Optimización de Hiperparámetros:** Afinación fina automatizada mediante la librería de optimización **Optuna** junto con `StratifiedKFold`.
* **FASE 7 - Modelo Final y Control de Overfitting:** Selección del modelo óptimo asegurando estrictamente que el diferencial de rendimiento entre entrenamiento y prueba no supere el límite exigido del **5%**.
* **FASE 8 - Importancia de Variables:** Extracción e interpretación de los pesos de las características para auditoría del modelo.
* **FASE 9 - Exportación del Modelo y Metadata:** Almacenamiento del Pipeline final preprocesado y entrenado mediante la librería `joblib` en la carpeta `models/`.
* **FASE 10 - Sistema de Feedback e Ingesta de Datos Nuevos:** Diseño de la lógica para el reentrenamiento del modelo y la captura de nuevas muestras ingresadas desde la interfaz de usuario.

---

## 📁 Estructura del Repositorio
El pipeline del proyecto está configurado para generar y organizar de forma automática los siguientes directorios:

```plaintext
│
├── 📁 data/
│   ├── dataset_clean.csv            # Tu dataset limpio histórico principal
│   └── feedback_loans.csv           # El CSV de logs operativos (Fase 10)
│
├── 📁 models/
│   ├── best_model.pkl               # Modelo definitivo en producción
│   ├── model_metadata.json          # Hiperparámetros, métricas y features del framework
│   └── preprocessor.pkl             # Transformaciones/Scalers si van desacoplados
│
├── 📁 notebooks/                    # Opcional: Para mover tus análisis fuera de la raíz
│   ├── DA_project_classification_parte01_...
│   └── DA_project_classification_parte02_...
│
├── 📁 scripts/
│   └── export_model_comparison.py   # Script de ingeniería para generar el cuadro comparativo
│
├── 📁 utils/
│   ├── __init__.py
│   └── predictor.py                 # Funciones lógicas de carga e inferencia (desacopladas)
│
├── 📄 .gitignore                    # Para evitar subir archivos .pkl o .csv pesados a GitHub
├── 📄 app.py                        # Tu interfaz de usuario e interacción en Streamlit
├── 📄 LICENSE
├── 📄 README.md                     # Documentación técnica del proyecto
└── 📄 requirements.txt              # Librerías necesarias (Streamlit, Pandas, Joblib...)
```


## ⚙️ Estructura del Equipo

![Tablero Kanban](assets/Tablero%20Kanban.png)

Para garantizar un desarrollo ágil, coordinado y eficiente en esta fase de productivización y despliegue, el equipo ha implementado un marco de trabajo basado en la metodología **Kanban**. A través de un tablero digital organizado en sprints dinámicos, gestionamos el ciclo de vida de cada tarea mediante los siguientes estados:

* **Backlog / Por hacer (To Do):** Listado priorizado de requisitos, refactorización de código y tareas de despliegue pendientes.
* **En proceso (In Progress):** Tareas activas asignadas a los miembros del equipo para evitar la sobrecarga de trabajo en paralelo (WIP).
* **En revisión / Testing (In Review):** Código en fase de control de calidad, pruebas locales de la aplicación y revisión de *Pull Requests* (PRs).
* **Finalizado (Done):** Funcionalidades e integraciones totalmente completadas, validadas y listas para su fusión en la rama principal.

> 💡 **Nota del equipo:** Esta estructura, sumada a una comunicación constante, nos permite mantener una visibilidad total sobre los cuellos de botella. Al afrontar las incidencias y errores como un bloque unitario, aseguramos soluciones más rápidas, la continuidad del flujo de trabajo y un incremento del producto completamente estable en cada iteración.

---

## 🚀 Requisitos e Instalación

El entorno del proyecto utiliza principalmente las siguientes librerías de Python:

* **pandas** y **numpy** para la manipulación de datos.
* **scikit-learn** para la construcción de pipelines, preprocesamiento y algoritmos de Machine Learning.
* **optuna** para la optimización de hiperparámetros.
* **seaborn** y **matplotlib** para la generación de gráficos del EDA y evaluación.
* **streamlit** para el despliegue de la interfaz de usuario.
* **joblib** para la persistencia del modelo.

---

## 👥 Autores
* Sonia Navarro
* Manuel Macarro
* Noelia Sánchez
* Irene Condado

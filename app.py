import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
from pathlib import Path
import plotly.express as px
from datetime import datetime
from utils import load_production_artifacts, load_available_models, predict_loan

# Configuración de página premium y ancha
st.set_page_config(
    page_title="Credit Risk IA Premium", 
    page_icon="",
    layout="wide"
)

# =====================================================================
# CAPA DE DISEÑO VISUAL: ESTILO CLARO, VIVO Y ANIMADO (MODERNO/BI)
# =====================================================================
st.markdown("""
    <style>
    .stApp {
        background-color: #f8fafc;
        color: #1e293b !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    .stMarkdown, p, span, label, .stCaption {
        color: #334155 !important;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #0f172a !important;
        font-weight: 700 !important;
    }
    .kpi-card-premium {
        background: #ffffff;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(148, 163, 184, 0.12);
        text-align: center;
        margin-bottom: 15px;
        border-top: 5px solid #1068DA;
        border-left: 1px solid #e2e8f0; border-right: 1px solid #e2e8f0; border-bottom: 1px solid #e2e8f0;
    }
    .kpi-label-premium {
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: #64748b !important;
        font-weight: 600;
        margin-bottom: 8px;
    }
    .kpi-value-premium {
        font-size: 28px;
        font-weight: 700;
        color: #1068DA !important;
    }
    .chart-header {
        font-size: 18px;
        font-weight: 600;
        color: #0f172a !important;
        margin-top: 25px;
        margin-bottom: 12px;
        border-left: 4px solid #1068DA;
        padding-left: 12px;
    }
    .chart-card-premium {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 14px 14px 6px 14px;
        box-shadow: 0 6px 20px rgba(148, 163, 184, 0.10);
        margin-bottom: 16px;
    }
    .chart-card-premium .chart-subtitle {
        font-size: 13px;
        color: #64748b !important;
        margin-bottom: 8px;
    }
    [data-testid="stSidebar"] {
        background-color: #1e293b !important;
        border-right: 1px solid #334155;
    }
    [data-testid="stSidebar"] [data-testid="stSidebarHeader"] h1,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h1 span,
    [data-testid="stSidebar"] h1 p {
        color: #cbd5e1 !important; /* Gris claro premium con alto contraste */
        font-weight: 700 !important;
    }
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] span, [data-testid="stSidebar"] label {
        color: #f1f5f9 !important;
    }
    [data-testid="stSidebar"] div[role="box"] {
        background-color: #334155 !important;
    }
    [data-testid="stSidebar"] div[data-testid="stExpander"] {
        background-color: #243242 !important; /* Un azul ligeramente más claro para hacer capas */
        border: 1px solid #334155 !important;
        border-radius: 8px;
    }
    div[data-testid="stRadio"] > div {
        gap: 10px;
    }
    div[data-testid="stRadio"] label {
        background-color: #f1f5f9;
        padding: 10px 20px;
        border-radius: 8px;
        border: 1px solid #e2e8f0;
        transition: all 0.2s ease;
        cursor: pointer;
    }
    div[data-testid="stRadio"] label:hover {
        border-color: #10CDDA;
        color: #1068DA !important;
    }
    div[data-testid="stRadio"] label[data-checked="true"] {
        background-color: #1068DA !important;
        color: white !important;
        border-color: #1068DA;
        box-shadow: 0 4px 12px rgba(16, 104, 218, 0.25);
    }
    div[data-testid="stRadio"] label[data-checked="true"] p {
        color: white !important;
    }
    div[data-baseweb="select"] > div {
        border-color: #cbd5e1;
    }
    div[data-baseweb="select"] > div:hover {
        border-color: #10CDDA;
    }
    [data-testid="stSlider"] [role="slider"] {
        background-color: #1068DA;
        border-color: #1068DA;
    }
    [data-testid="stSlider"] [data-baseweb="slider"] > div > div {
        background: linear-gradient(90deg, #1068DA, #10CDDA, #10DA82);
    }
    </style>
""", unsafe_allow_html=True)

# =====================================================================
# DETECTOR AUTOMÁTICO DE RUTAS
# =====================================================================
BASE_DIR = Path(__file__).resolve().parent
RUTA_PIPELINE = BASE_DIR / "models" / "pipeline.pkl"
RUTA_JSON = BASE_DIR / "models" / "model_metadata.json"
FEEDBACK_FILE = BASE_DIR /"data" / "feedback_log.csv"

# Se asume que tu CSV limpio se llama dataset_clean.csv dentro de la carpeta data
NOMBRE_CSV = "dataset_clean.csv"
RUTA_CSV = BASE_DIR / "data" / NOMBRE_CSV

pipeline_ia, meta_ia = load_production_artifacts()
modelos_disponibles = load_available_models()

# Carga del CSV del histórico limpio
df_clean = None
if RUTA_CSV.exists():
    df_clean = pd.read_csv(RUTA_CSV)

# =====================================================================
# ENCABEZADO Y NAVEGACIÓN PRINCIPAL
# =====================================================================
st.title("Sistema Inteligente de Riesgo Crediticio")
st.write("Plataforma Analítica BI & Motor Predictivo de Concesión de Préstamos Financieros.")

seccion_activa = st.radio(
    "Navegación del Sistema:",
    ["Dashboard de Negocio", "Métricas del Modelo y Estimador de Riesgo", "Simulador de Riesgo en Tiempo Real", "Ingesta de Datos y Feedback"],
    horizontal=True,
    label_visibility="collapsed"
)

st.markdown("<br>", unsafe_allow_html=True)

# =====================================================================
# SIDEBAR INFORMATIVO
# =====================================================================
st.sidebar.title("Panel de Control")
st.sidebar.markdown("Filtros globales y estado de auditoría del scoring de riesgo.")

COLOR_PRIMARY = "#1068DA"
COLOR_SECONDARY = "#10CDDA"
COLOR_APPROVED = "#10DA82"
COLOR_DENIED = "#1068DA"
TRAINED_MODEL_FILES = {
    "Logistic Regression": "logistic_regression",
    "Decision Tree": "decision_tree",
    "Random Forest": "random_forest",
    "Gradient Boosting": "gradient_boosting",
    "RF Optimizado": "rf_optimizado",
}

MODEL_COMPARISON_PATH = BASE_DIR / "assets" / "model_comparison_summary.csv"


def build_model_comparison_df():
    if MODEL_COMPARISON_PATH.exists():
        df_models = pd.read_csv(MODEL_COMPARISON_PATH)
        if "Modelo" in df_models.columns:
            return df_models
    return pd.DataFrame()


def style_dashboard_figure(fig, title, y_title=None):
    fig.update_layout(
        template="plotly_white",
        title={"text": title, "x": 0.02, "xanchor": "left"},
        font={"family": "Arial, sans-serif", "color": "#0f172a"},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#ffffff",
        margin={"l": 25, "r": 20, "t": 60, "b": 25},
        legend={"orientation": "h", "y": -0.18, "x": 0},
    )
    if y_title is not None:
        fig.update_yaxes(title_text=y_title, gridcolor="#e2e8f0")
    fig.update_xaxes(gridcolor="#e2e8f0")
    return fig

# =====================================================================
# RENDERIZADO DE CONTENIDOS SEGÚN SECCIÓN SELECCIONADA
# =====================================================================

# --- SECCIÓN 1: DASHBOARD DE NEGOCIO ---
if seccion_activa == "Dashboard de Negocio":
    if df_clean is not None:
        st.subheader("Panel Analítico de Solicitudes de Crédito")

        gender_labels = {0: "Femenino", 1: "Masculino"}
        education_labels = {
            0: "Nivel 0",
            1: "Nivel 1",
            2: "Nivel 2",
            3: "Nivel 3",
            4: "Nivel 4",
        }
        
        # Filtros interactivos en el Sidebar
        with st.sidebar.expander("Filtros del Histórico", expanded=True):
            edad_min = int(df_clean["Age"].min()) if "Age" in df_clean.columns else 18
            edad_max = int(df_clean["Age"].max()) if "Age" in df_clean.columns else 90

            filtro_edad = st.slider(
                "Rango de Edad:",
                min_value=edad_min,
                max_value=edad_max,
                value=(edad_min, edad_max),
            )

            filtro_genero = st.selectbox(
                "Género:",
                ["Todos"] + list(gender_labels.values()),
            )

            filtro_educacion = st.selectbox(
                "Educación:",
                ["Todas"] + list(education_labels.values()),
            )
        
        df_filtrado = df_clean.copy()
        if "Age" in df_filtrado.columns:
            df_filtrado = df_filtrado[
                (df_filtrado["Age"] >= filtro_edad[0]) & (df_filtrado["Age"] <= filtro_edad[1])
            ]
        if filtro_genero != "Todos" and "Gender" in df_filtrado.columns:
            genero_codigo = 0 if filtro_genero == "Femenino" else 1
            df_filtrado = df_filtrado[df_filtrado["Gender"] == genero_codigo]
        if filtro_educacion != "Todas" and "Education" in df_filtrado.columns:
            educacion_codigo = next((k for k, v in education_labels.items() if v == filtro_educacion), None)
            if educacion_codigo is not None:
                df_filtrado = df_filtrado[df_filtrado["Education"] == educacion_codigo]

        # Fila de KPIs Premium
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        with kpi1:
            st.markdown(f'<div class="kpi-card-premium"><div class="kpi-label-premium">Solicitudes Analizadas</div><div class="kpi-value-premium">{len(df_filtrado):,}</div></div>', unsafe_allow_html=True)
        with kpi2:
            # Si tu columna target mapeada en el CSV es 'LoanApproved'
            porc_aprobados = (df_filtrado['LoanApproved'] == 1).mean() * 100 if 'LoanApproved' in df_filtrado.columns else 0.0
            st.markdown(f'<div class="kpi-card-premium"><div class="kpi-label-premium">% Aprobación</div><div class="kpi-value-premium">{porc_aprobados:.1f}%</div></div>', unsafe_allow_html=True)
        with kpi3:
            st.markdown(f'<div class="kpi-card-premium"><div class="kpi-label-premium">Variables de Entrada</div><div class="kpi-value-premium">{len(df_filtrado.columns)}</div></div>', unsafe_allow_html=True)
        with kpi4:
            st.markdown(f'<div class="kpi-card-premium"><div class="kpi-label-premium">Registros Totales Dataset</div><div class="kpi-value-premium">{len(df_clean):,}</div></div>', unsafe_allow_html=True)

        st.markdown("---")
        
        # Gráficos Dinámicos con Plotly en una matriz 2x2 homogénea
        row1_col1, row1_col2 = st.columns(2)
        row2_col1, row2_col2 = st.columns(2)

        with row1_col1:
            st.markdown("<div class='chart-header'>Distribución de Aprobación</div>", unsafe_allow_html=True)
            st.caption("Proporción global de solicitudes aprobadas y denegadas.")
            if "LoanApproved" in df_filtrado.columns and not df_filtrado.empty:
                approval_counts = (
                    df_filtrado["LoanApproved"]
                    .map({0: "Denegado", 1: "Aprobado"})
                    .value_counts()
                    .rename_axis("Estado")
                    .reset_index(name="Cantidad")
                )
                fig_pie = px.pie(
                    approval_counts,
                    names="Estado",
                    values="Cantidad",
                    color="Estado",
                    color_discrete_map={"Denegado": COLOR_DENIED, "Aprobado": COLOR_APPROVED},
                    hole=0.48,
                )
                fig_pie = style_dashboard_figure(fig_pie, "")
                fig_pie.update_traces(textinfo="percent+label", textfont_size=13, marker=dict(line=dict(color="#ffffff", width=2)))
                fig_pie.update_layout(showlegend=False)
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.info("No hay datos suficientes para mostrar la distribución de aprobación.")

        with row1_col2:
            st.markdown("<div class='chart-header'>Edad por Resultado</div>", unsafe_allow_html=True)
            st.caption("Comparativa de edad entre aprobados y denegados.")
            if "Age" in df_filtrado.columns and "LoanApproved" in df_filtrado.columns and not df_filtrado.empty:
                df_chart = df_filtrado.copy()
                df_chart["Resultado"] = df_chart["LoanApproved"].map({0: "Denegado", 1: "Aprobado"})
                fig_bar = px.histogram(
                    df_chart,
                    x="Age",
                    color="Resultado",
                    nbins=15,
                    barmode="overlay",
                    opacity=0.78,
                    color_discrete_map={"Denegado": COLOR_DENIED, "Aprobado": COLOR_APPROVED},
                )
                fig_bar = style_dashboard_figure(fig_bar, "", "Frecuencia")
                st.plotly_chart(fig_bar, use_container_width=True)
            else:
                st.info("No hay datos suficientes para mostrar la distribución por edad.")

        with row2_col1:
            st.markdown("<div class='chart-header'>Ingreso por Resultado</div>", unsafe_allow_html=True)
            st.caption("Dispersión del ingreso anual según la decisión del modelo.")
            if "Income" in df_filtrado.columns and "LoanApproved" in df_filtrado.columns and not df_filtrado.empty:
                df_income = df_filtrado.copy()
                df_income["Resultado"] = df_income["LoanApproved"].map({0: "Denegado", 1: "Aprobado"})
                fig_income = px.box(
                    df_income,
                    x="Resultado",
                    y="Income",
                    color="Resultado",
                    color_discrete_map={"Denegado": COLOR_DENIED, "Aprobado": COLOR_APPROVED},
                    points="outliers",
                )
                fig_income = style_dashboard_figure(fig_income, "", "Ingreso anual")
                fig_income.update_layout(showlegend=False)
                st.plotly_chart(fig_income, use_container_width=True)
            else:
                st.info("No hay datos suficientes para mostrar la distribución de ingresos.")

        with row2_col2:
            st.markdown("<div class='chart-header'>Score Crediticio por Resultado</div>", unsafe_allow_html=True)
            st.caption("Distribución del score crediticio según la decisión del modelo.")
            if "CreditScore" in df_filtrado.columns and "LoanApproved" in df_filtrado.columns and not df_filtrado.empty:
                df_score = df_filtrado.copy()
                df_score["Resultado"] = df_score["LoanApproved"].map({0: "Denegado", 1: "Aprobado"})
                fig_score = px.box(
                    df_score,
                    y="CreditScore",
                    x="Resultado",
                    color="Resultado",
                    points="outliers",
                    color_discrete_map={"Denegado": COLOR_DENIED, "Aprobado": COLOR_APPROVED},
                )
                fig_score = style_dashboard_figure(fig_score, "", "Credit Score")
                fig_score.update_layout(showlegend=False)
                st.plotly_chart(fig_score, use_container_width=True)
            else:
                st.info("No hay datos suficientes para mostrar la relación con el score crediticio.")

        st.markdown("#### Vista Previa del Dataset Limpio (`df_clean`)")
        st.dataframe(df_filtrado.head(100), use_container_width=True)
    else:
        st.error(f"No se encontró el archivo histórico '{NOMBRE_CSV}' en la ruta 'data/'.")

# --- SECCIÓN 2: MÉTRICAS DEL MODELO ---
elif seccion_activa == "Métricas del Modelo y Estimador de Riesgo":
    st.subheader("Métricas del Modelo y Estimador de Riesgo")
    
    if meta_ia is not None:
        metric_config = [
            ("Accuracy", "accuracy", "percent"),
            ("Precision", "precision", "percent"),
            ("Recall (Sensibilidad)", "recall", "percent"),
            ("F1-Score", "f1_score", "decimal"),
            ("ROC AUC", "roc_auc", "decimal"),
        ]

        with st.sidebar.expander("Filtros de la Sección 2", expanded=True):
            metricas_visibles = st.multiselect(
                "Métricas a mostrar:",
                options=[label for label, _, _ in metric_config],
                default=[label for label, _, _ in metric_config],
            )
            filtro_parametros = st.text_input("Buscar hiperparámetro:", value="").strip().lower()
            filtro_features = st.text_input("Buscar feature:", value="").strip().lower()

        m = meta_ia.get("metricas_test", {})
        metricas_seleccionadas = [cfg for cfg in metric_config if cfg[0] in metricas_visibles]

        if metricas_seleccionadas:
            columnas_metricas = st.columns(len(metricas_seleccionadas))
            for col, (label, key, kind) in zip(columnas_metricas, metricas_seleccionadas):
                valor = m.get(key, 0)
                if kind == "percent":
                    col.metric(label, f"{valor*100:.2f}%")
                else:
                    col.metric(label, f"{valor:.4f}")
        else:
            st.info("Selecciona al menos una métrica en el filtro lateral para visualizarla.")
        
        st.markdown("---")
        st.markdown("#### Hiperparámetros de Producción Aplicados")
        params = meta_ia.get("hiperparametros", {})
        df_hp = pd.DataFrame(list(params.items()), columns=["Hiperparámetro", "Valor Óptimo"])
        if filtro_parametros:
            df_hp = df_hp[
                df_hp["Hiperparámetro"].astype(str).str.lower().str.contains(filtro_parametros, na=False)
            ]
        st.dataframe(df_hp, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        st.markdown("#### Estructura de Características Esperadas")
        features = meta_ia.get("features_entrada", [])
        df_feats = pd.DataFrame({"Posición": range(len(features)), "Feature Requerida": features})
        if filtro_features:
            df_feats = df_feats[
                df_feats["Feature Requerida"].astype(str).str.lower().str.contains(filtro_features, na=False)
            ]
        st.dataframe(df_feats, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("#### Comparativa final de modelos entrenados")
        df_modelos = build_model_comparison_df()

        with st.sidebar.expander("Comparativa de modelos entrenados", expanded=True):
            metricas_disponibles = ["ROC AUC", "F1-Score", "Recall", "Precision", "Acc Test", "Brecha (gap)", "CV ROC AUC"]
            metrica_orden = st.selectbox("Ordenar comparativa por:", metricas_disponibles, index=0)
            modelos_seleccionados = st.multiselect(
                "Modelos a mostrar:",
                options=df_modelos["Modelo"].tolist(),
                default=df_modelos["Modelo"].tolist(),
            )
            solo_sin_overfitting = st.checkbox("Solo modelos con gap < 5%", value=False)
            modelo_detalle = st.selectbox("Ver detalle de un modelo:", options=df_modelos["Modelo"].tolist())

        df_modelos_filtrado = df_modelos[df_modelos["Modelo"].isin(modelos_seleccionados)].copy()
        if solo_sin_overfitting:
            df_modelos_filtrado = df_modelos_filtrado[df_modelos_filtrado["Brecha (gap)"] < 0.05]

        if not df_modelos_filtrado.empty:
            df_modelos_filtrado = df_modelos_filtrado.sort_values(by=metrica_orden, ascending=False, na_position="last")

            mejor_modelo = df_modelos_filtrado.iloc[0]
            st.info(
                f"Modelo líder según {metrica_orden}: {mejor_modelo['Modelo']} "
                f"(valor = {mejor_modelo[metrica_orden]:.4f})" if pd.notna(mejor_modelo[metrica_orden]) else f"Modelo líder según {metrica_orden}: {mejor_modelo['Modelo']}")

            metricas_graficables = ["Acc Test", "Precision", "Recall", "F1-Score", "ROC AUC"]
            df_plot_modelos = df_modelos_filtrado[["Modelo"] + metricas_graficables].melt(
                id_vars="Modelo",
                var_name="Métrica",
                value_name="Valor",
            )
            fig_modelos = px.bar(
                df_plot_modelos,
                x="Modelo",
                y="Valor",
                color="Métrica",
                barmode="group",
                color_discrete_sequence=[COLOR_PRIMARY, COLOR_SECONDARY, COLOR_APPROVED, "#60A5FA", "#0F766E"],
            )
            fig_modelos = style_dashboard_figure(fig_modelos, "Comparativa de métricas por modelo", "Valor")
            fig_modelos.update_layout(xaxis_title="Modelo", yaxis=dict(range=[0, 1.05], showgrid=False, zeroline=False, showline=False),paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)")
            
            fig_modelos.update_yaxes(showgrid=False)
            fig_modelos.update_xaxes(showgrid=False, zeroline=False)

            st.plotly_chart(fig_modelos, use_container_width=True)

            st.dataframe(df_modelos_filtrado, use_container_width=True, hide_index=True)

            st.markdown("##### Detalle del modelo seleccionado")
            detalle = df_modelos[df_modelos["Modelo"] == modelo_detalle].iloc[0]
            det_c1, det_c2, det_c3, det_c4 = st.columns(4)
            det_c1.metric("Acc Test", f"{detalle['Acc Test']:.4f}")
            det_c2.metric("F1-Score", f"{detalle['F1-Score']:.4f}")
            det_c3.metric("ROC AUC", f"{detalle['ROC AUC']:.4f}")
            det_c4.metric("Gap Train/Test", f"{detalle['Brecha (gap)']:.4f}")

            st.caption(
                f"Overfitting OK: {detalle['Overfitting OK']} | "
                f"Acc Train: {detalle['Acc Train']:.4f} | "
                f"Precision: {detalle['Precision']:.4f} | Recall: {detalle['Recall']:.4f}"
            )
        else:
            st.warning("No hay modelos para mostrar con los filtros actuales.")
    else:
        st.error("⚠️ El archivo 'model_metadata.json' no se encuentra en 'models/'.")

# --- SECCIÓN 3: SIMULADOR DE PREDICCIONES ---
elif seccion_activa == "Simulador de Riesgo en Tiempo Real":
    st.subheader("Formulario de Scoring de Crédito Automatizado")
    
    if not modelos_disponibles:
        st.error("⚠️ No hay modelos disponibles en la carpeta 'models/' para ejecutar inferencia.")
    else:
        st.write("Introduzca los datos del solicitante en la barra lateral izquierda y envíe el formulario.")
        st.markdown("---")
        
        # Inputs del simulador estructurados en la barra lateral como tu guía
        with st.sidebar.expander("Perfil del Solicitante", expanded=True):
            modelos_para_simulador = {
                label: modelos_disponibles[file_stem]
                for label, file_stem in TRAINED_MODEL_FILES.items()
                if file_stem in modelos_disponibles
            }
            if not modelos_para_simulador:
                st.error("⚠️ No se encontraron los cinco modelos entrenados en la carpeta 'models/'.")
                st.stop()
            nombres_modelos = list(modelos_para_simulador.keys())
            modelo_seleccionado = st.selectbox("Modelo para inferencia:", nombres_modelos, index=0)
            comparar_modelos = st.checkbox("Comparar con todos los modelos entrenados", value=False)

            age = st.slider("Edad del Solicitante:", 18, 90, 35)
            income = st.number_input("Ingresos Anuales ($):", min_value=0.0, value=55000.0, step=1000.0)
            loan_amount = st.number_input("Monto del Préstamo Solicitado ($):", min_value=0.0, value=15000.0, step=500.0)
            credit_score = st.slider("Score Crediticio (Bureau):", 300, 850, 720)
            experience = st.slider("Años de Experiencia Laboral:", 0, 50, 8)
            gender = st.selectbox("Género:", [0, 1], format_func=lambda x: "Femenino" if x == 0 else "Masculino")
            education = st.selectbox("Nivel de Estudios:", [1, 2, 3, 4], format_func=lambda x: f"Nivel {x}")
            city = st.selectbox("Ciudad de Residencia:", ["New York", "Houston", "San Francisco", "Other"])
            emp_type = st.selectbox("Tipo de Empleo:", ["Self-Employed", "Unemployed", "Employed"])

        if st.button("Procesar Análisis de Riesgo Crediticio", type="primary", use_container_width=True):
            # Formateamos los datos para pasarlos idénticos a como los espera tu Pipeline
            cliente_input = {
                'Age': age, 'Income': income, 'LoanAmount': loan_amount,
                'CreditScore': credit_score, 'YearsExperience': experience,
                'Gender': gender, 'Education': education,
                'City_Houston': 1 if city == "Houston" else 0,
                'City_New York': 1 if city == "New York" else 0,
                'City_San Francisco': 1 if city == "San Francisco" else 0,
                'EmploymentType_Self-Employed': 1 if emp_type == "Self-Employed" else 0,
                'EmploymentType_Unemployed': 1 if emp_type == "Unemployed" else 0
            }
            
            with st.spinner("Evaluando viabilidad financiera en los servidores..."):
                modelo_activo = modelos_para_simulador[modelo_seleccionado]
                pred, proba = predict_loan(modelo_activo, cliente_input)
            
            res_col1, res_col2 = st.columns([1, 2])
            with res_col1:
                decision = "APROBADO" if pred == 1 else "DENEGADO"
                st.metric(label="Dictamen del Algoritmo", value=decision)
                st.metric(label="Modelo utilizado", value=modelo_seleccionado)
                if proba is not None:
                    st.metric(label="Probabilidad de Viabilidad", value=f"{proba*100:.2f}%")
                else:
                    st.metric(label="Probabilidad de Viabilidad", value="N/D")
            with res_col2:
                if pred == 1:
                    st.success("🎉 **Solicitud Viable:** El perfil cumple satisfactoriamente con los umbrales de riesgo exigidos por la entidad.")
                else:
                    st.error("❌ **Solicitud Denegada:** Los niveles de riesgo e ingresos proyectados superan el umbral tolerable establecido.")

            if comparar_modelos and len(modelos_para_simulador) > 1:
                comparativa = []
                for nombre, modelo in modelos_para_simulador.items():
                    try:
                        pred_m, proba_m = predict_loan(modelo, cliente_input)
                        comparativa.append({
                            "Modelo": nombre,
                            "Dictamen": "APROBADO" if pred_m == 1 else "DENEGADO",
                            "Probabilidad de viabilidad": round(float(proba_m), 4) if proba_m is not None else np.nan,
                            "Confianza": round(abs(float(proba_m) - 0.5) * 2, 4) if proba_m is not None else np.nan,
                        })
                    except Exception:
                        continue

                if comparativa:
                    df_comp = pd.DataFrame(comparativa).sort_values(by="Confianza", ascending=False, na_position="last")
                    st.markdown("#### Comparativa de modelos para este solicitante")
                    st.dataframe(df_comp, use_container_width=True, hide_index=True)
                    if not df_comp.empty:
                        mejor_modelo = df_comp.iloc[0]["Modelo"]
                        st.info(f"Modelo más confiado para este caso: {mejor_modelo}")
#SECCION 4 INGESTA DE DATOS
elif seccion_activa == "Ingesta de Datos y Feedback":
    st.subheader("Ingesta Manual y Registro de Feedback Operacional")
    st.markdown("""
    Esta consola permite a los analistas de riesgo almacenar los resultados reales de los préstamos otorgados 
    para reentrenar el modelo en futuros ciclos (Mapeo de la **Fase 10** del proyecto).
    """)
    
    st.markdown("#### Registrar un Caso Cerrado Real")
    
    with st.expander("Formulario de Registro en Base de Datos", expanded=True):
        f_col1, f_col2, f_col3 = st.columns(3)
        with f_col1:
            fb_score = st.number_input("Credit Score del Cliente", 300, 850, 600, key="fb_cs")
            fb_income = st.number_input("Ingresos Anuales (USD)", 0, 500000, 50000, key="fb_inc")
        with f_col2:
            fb_amount = st.number_input("Monto Prestado (USD)", 0, 300000, 20000, key="fb_amt")
            fb_pred = st.selectbox("Predicción Original del Sistema", ["Rechazado (0)", "Aprobado (1)"])
        with f_col3:
            fb_real = st.selectbox("Estatus Real Final (Feedback)", ["Pagado Correctamente", "Incurrió en Default / Impago"])
            
        save_feedback = st.button("Guardar en Registro Histórico", use_container_width=True, type="primary")
        
        if save_feedback:
            new_row = pd.DataFrame([{
                'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'CreditScore': fb_score,
                'Income': fb_income,
                'LoanAmount': fb_amount,
                'ModelPrediction': 0 if "Rechazado" in fb_pred else 1,
                'RealOutcome': 1 if "Pagado" in fb_real else 0
            }])
            
            if not FEEDBACK_FILE.parent.exists():
                os.makedirs(FEEDBACK_FILE.parent, exist_ok=True)
                
            if not FEEDBACK_FILE.exists():
                new_row.to_csv(FEEDBACK_FILE, index=False)
            else:
                new_row.to_csv(FEEDBACK_FILE, mode='a', header=False, index=False)
                
            st.success(f"Registro almacenado correctamente en '{FEEDBACK_FILE.name}'. Listo para el pipeline de reentrenamiento.")

    st.markdown("---")
    st.markdown("#### Historial de logs acumulados (Producción)")
    if FEEDBACK_FILE.exists():
        df_fb_saved = pd.read_csv(FEEDBACK_FILE)
        st.dataframe(df_fb_saved, use_container_width=True)
    else:
        st.caption("No hay registros guardados en este ciclo operativo aún.")
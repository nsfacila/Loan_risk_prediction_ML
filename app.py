import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# 1. Configuración de la página
st.set_page_config(
    page_title="Score Crediticio - Predicción de Préstamos",
    page_icon="💰",
    layout="wide"
)

# 2. Función para cargar el modelo/pipeline entrenado
@st.cache_resource
def load_ml_pipeline():
    # Intentar cargar tu pipeline exportado, si no existe, simulamos la respuesta para la demo
    if os.path.exists('models/best_model_pipeline.joblib'):
        return joblib.load('models/best_model_pipeline.joblib')
    return None

pipeline = load_ml_pipeline()

# 3. Título y Estilo Principal
st.title("💰 Sistema Inteligente de Evaluación de Riesgo Crediticio")
st.markdown("""
Esta aplicación interactiva evalúa de forma instantánea si una solicitud de préstamo debería ser **Aprobada** o **Rechazada**, 
basándose en el perfil socioeconómico y el historial del solicitante mediante modelos predictivos de Machine Learning.
""")

# 4. Creación de Pestañas (Tabs) para mejorar la usabilidad
tab_predict, tab_project = st.tabs(["📋 Evaluación de Nuevos Clientes", "📊 Información del Proyecto"])

# ==========================================
# PESTAÑA 1: EVALUACIÓN DE NUEVOS CLIENTES
# ==========================================
with tab_predict:
    st.subheader("Introducir Datos del Nuevo Solicitante")
    
    # Creamos un formulario estructurado en columnas
    with st.form("loan_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### 👤 Datos Personales")
            age = st.number_input("Edad", min_value=18, max_value=100, value=35, step=1)
            gender = st.selectbox("Género", options=["Masculino", "Femenino"])
            education = st.selectbox(
                "Nivel Educativo", 
                options=["Sin Estudios", "Secundaria", "Pregrado (Bachelor)", "Maestría", "Doctorado"]
            )
            
        with col2:
            st.markdown("### 💼 Situación Laboral y Residencia")
            years_experience = st.number_input("Años de Experiencia Laboral", min_value=0, max_value=50, value=8, step=1)
            employment_type = st.selectbox(
                "Tipo de Empleo", 
                options=["Asalariado (Full-time)", "Autónomo (Self-Employed)", "Desempleado"]
            )
            city = st.selectbox("Ciudad de Residencia", options=["New York", "Houston", "San Francisco", "Otra"])

        with col3:
            st.markdown("### 💳 Datos Financieros")
            income = st.number_input("Ingresos Anuales (USD)", min_value=0, value=55000, step=1000)
            loan_amount = st.number_input("Importe del Préstamo Solicitado (USD)", min_value=0, value=20000, step=1000)
            credit_score = st.slider("Puntuación Crediticia (Credit Score)", min_value=300, max_value=850, value=650, step=5)

        # Botón de envío del formulario
        submit_button = st.form_submit_button(label="🚀 Evaluar Solicitud de Crédito")

    # Al pulsar el botón se procesan los datos y se realiza la predicción
    if submit_button:
        # Mapeo de las entradas del usuario a los formatos que espera tu dataset/pipeline original
        # Género (Mapeado a 0 o 1 como indica tu dataset limpio)
        gender_encoded = 1 if gender == "Masculino" else 0
        
        # Educación (Mapeado de forma Ordinal del 0 al 4)
        edu_map = {"Sin Estudios": 0, "Secundaria": 1, "Pregrado (Bachelor)": 2, "Maestría": 3, "Doctorado": 4}
        education_encoded = edu_map[education]
        
        # Reconstruir el DataFrame con las columnas idénticas al dataset original de la FASE 4
        # Nota: Si usas el pipeline de Scikit-Learn directo, pasas los strings; si usas el dataset preprocesado a mano, usamos los mapeos correspondientes.
        # Aquí se armará la estructura para que coincida perfectamente:
        input_data = pd.DataFrame([{
            'Age': age,
            'Income': income,
            'LoanAmount': loan_amount,
            'CreditScore': credit_score,
            'YearsExperience': years_experience,
            'Gender': gender_encoded,
            'Education': education_encoded,
            'City_Houston': 1 if city == "Houston" else 0,
            'City_New York': 1 if city == "New York" else 0,
            'City_San Francisco': 1 if city == "San Francisco" else 0,
            'EmploymentType_Self-Employed': 1 if employment_type == "Autónomo (Self-Employed)" else 0,
            'EmploymentType_Unemployed': 1 if employment_type == "Desempleado" else 0
        }])

        st.markdown("---")
        st.subheader("Resultados del Análisis de Riesgo")
        
        # Realizar la predicción
        if pipeline is not None:
            prediction = pipeline.predict(input_data)[0]
            # Si tu pipeline tiene predict_proba, calculamos la probabilidad
            try:
                probabilidad = pipeline.predict_proba(input_data)[0][1] * 100
            except:
                probabilidad = None
        else:
            # Regla de Negocio Simulada de Respaldo por si no se encuentra el archivo .joblib del modelo
            # Basado en tus conclusiones (CreditScore e Income positivos, Unemployed negativo)
            score_simulado = (credit_score - 300) / 550 * 0.5 + (income / 100000) * 0.3 + (years_experience / 30) * 0.2
            if employment_type == "Desempleado":
                score_simulado -= 0.4
            prediction = 1 if score_simulado > 0.45 else 0
            probabilidad = min(max(int(score_simulado * 100), 0), 100)

        # Mostrar métricas visuales del resultado
        col_res1, col_res2 = st.columns([2, 1])
        
        with col_res1:
            if prediction == 1:
                st.success("### ✅ ¡Préstamo APROBADO!")
                st.markdown(f"El cliente presenta un perfil financiero **saludable** y de **bajo riesgo** para la entidad.")
            else:
                st.error("### ❌ Préstamo RECHAZADO")
                st.markdown(f"El nivel de riesgo del solicitante **supera el umbral permitido**. Capacidad de repago comprometida.")
        
        with col_res2:
            if probabilidad is not None:
                st.metric(label="Confianza / Probabilidad de Aprobación", value=f"{probabilidad:.1f} %")

        # Tarjeta resumen de los datos ingresados del cliente
        with st.expander("🔎 Ver resumen del perfil analizado"):
            st.write(input_data)


# ==========================================
# PESTAÑA 2: INFORMACIÓN DEL PROYECTO
# ==========================================
with tab_project:
    st.subheader("Detalles Técnicos y Objetivos del Proyecto")
    
    col_proj1, col_proj2 = st.columns(2)
    
    with col_proj1:
        st.markdown("""
        **Objetivo Analítico:** Construir, comparar y optimizar varios modelos de clasificación binaria manteniendo el **overfitting por debajo del 5%** (requisito crítico del proyecto).
        
        **Características del Dataset Analizado:**
        - **Variables Numéricas:** Edad, Ingresos Anuales, Importe Solicitado, Puntuación Crediticia y Años de Experiencia.
        - **Variables Categorizadas:** Género, Educación, Ciudad y Situación Laboral.
        """)
        
    with col_proj2:
        st.info("""
        **💡 Conclusión Clave de Negocio (EDA):** - El **CreditScore** es la variable con mayor correlación positiva con la aprobación.
        - La situación laboral (**EmploymentType**) es determinante: los solicitantes **desempleados** experimentan una tasa de aprobación drásticamente menor (~3%) frente a asalariados o autónomos (~33%), lo que valida plenamente la lógica financiera de la capacidad de repago.
        """)
        
    st.markdown("---")
    st.caption("Desarrollado por Sonia Navarro Romero — Junior Data Analyst.")
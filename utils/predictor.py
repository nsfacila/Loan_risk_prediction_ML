import joblib
import json
import pandas as pd
import streamlit as st
from pathlib import Path
import numpy as np

# Encontramos la raíz del proyecto de forma automática
BASE_DIR = Path(__file__).resolve().parent.parent


def get_models_signature() -> str:
    """Devuelve una firma simple del contenido de /models para invalidar cachés cuando cambian los artefactos."""
    models_dir = BASE_DIR / "models"
    if not models_dir.exists():
        return "missing"

    parts = []
    for model_path in sorted(models_dir.glob("*.pkl")):
        try:
            parts.append(f"{model_path.name}:{model_path.stat().st_mtime_ns}")
        except OSError:
            continue
    return "|".join(parts)

@st.cache_resource
def load_production_artifacts():
    """Carga de forma segura el pipeline y la metadata usando la caché de Streamlit."""
    ruta_pipeline = BASE_DIR / "models" / "pipeline.pkl"
    ruta_json = BASE_DIR / "models" / "model_metadata.json"
    
    pipeline = joblib.load(ruta_pipeline) if ruta_pipeline.exists() else None
    metadata = None
    
    if ruta_json.exists():
        with open(ruta_json, "r", encoding="utf-8") as f:
            metadata = json.load(f)
            
    return pipeline, metadata


@st.cache_resource
def load_available_models(models_signature: str | None = None):
    """Carga todos los modelos .pkl utilizables para inferencia desde /models."""
    _ = models_signature
    models_dir = BASE_DIR / "models"
    loaded_models = {}

    if not models_dir.exists():
        return loaded_models

    for model_path in sorted(models_dir.glob("*.pkl")):
        if model_path.stem.lower() == "preprocessor":
            continue
        try:
            model_obj = joblib.load(model_path)
            if hasattr(model_obj, "predict"):
                loaded_models[model_path.stem] = model_obj
        except Exception:
            # Si un artefacto no se puede cargar, lo omitimos para no romper la app.
            continue

    return loaded_models

def predict_loan(pipeline_ia, data: dict) -> tuple:
    """Recibe el pipeline y los datos del formulario, devolviendo la predicción y probabilidad."""
    df_input = pd.DataFrame([data])
    pred = pipeline_ia.predict(df_input)[0]
    if hasattr(pipeline_ia, "predict_proba"):
        proba = pipeline_ia.predict_proba(df_input)[0][1]
    elif hasattr(pipeline_ia, "decision_function"):
        score = pipeline_ia.decision_function(df_input)[0]
        proba = 1 / (1 + np.exp(-score))
    else:
        proba = None
    return pred, proba
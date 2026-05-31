"""Service untuk load model dan prediksi sentimen."""

import joblib
import streamlit as st

from src.config import MODELS_DIR


@st.cache_resource
def load_model():
    path = MODELS_DIR / "best_model.pkl"
    if path.exists():
        return joblib.load(path)
    return None


@st.cache_resource
def load_model_info():
    path = MODELS_DIR / "model_info.pkl"
    if path.exists():
        return joblib.load(path)
    return None


def predict_sentiment(texts, model):
    from src.preprocessing import preprocess

    cleaned = [preprocess(text, use_stemming=False) for text in texts]
    preds = model.predict(cleaned)
    try:
        probas = model.predict_proba(cleaned)
        return preds, probas, model.classes_
    except Exception:
        return preds, None, None


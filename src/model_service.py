"""Model per sesi pengguna dan prediksi sentimen."""

import streamlit as st


SESSION_MODEL_KEY = "sentipro_model"
SESSION_MODEL_INFO_KEY = "sentipro_model_info"


def get_session_model():
    return st.session_state.get(SESSION_MODEL_KEY)


def get_session_model_info():
    return st.session_state.get(SESSION_MODEL_INFO_KEY)


def set_session_model(model, model_info: dict) -> None:
    st.session_state[SESSION_MODEL_KEY] = model
    st.session_state[SESSION_MODEL_INFO_KEY] = model_info


def clear_session_model() -> None:
    st.session_state.pop(SESSION_MODEL_KEY, None)
    st.session_state.pop(SESSION_MODEL_INFO_KEY, None)


def predict_sentiment(texts, model, use_stemming: bool = False):
    from src.preprocessing import preprocess

    cleaned = [preprocess(text, use_stemming=use_stemming) for text in texts]
    if any(not text for text in cleaned):
        raise ValueError(
            "Terdapat teks yang kosong setelah preprocessing dan tidak dapat diprediksi."
        )

    preds = model.predict(cleaned)
    try:
        probas = model.predict_proba(cleaned)
        return preds, probas, model.classes_
    except (AttributeError, NotImplementedError):
        return preds, None, None

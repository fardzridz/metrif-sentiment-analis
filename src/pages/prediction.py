"""Tab prediksi sentimen."""

import pandas as pd
import streamlit as st

from src.config import SENTIMENT_EMOJI
from src.data_service import (
    DatasetValidationError,
    read_uploaded_dataset,
    validate_prediction_dataset,
)
from src.model_service import (
    get_session_model,
    get_session_model_info,
    predict_sentiment,
)
from src.preprocessing import preprocess
from src.ui import section_header


def render_prediction_page() -> None:
    section_header("4. Prediksi Sentimen")
    model = get_session_model()
    if not model:
        st.warning(
            "Prediksi dikunci. Selesaikan upload, validasi, preprocessing, dan training "
            "pada dataset sesi ini terlebih dahulu."
        )
        return

    model_info = get_session_model_info() or {}
    _render_model_overview(model, model_info)

    mode = st.radio("Mode prediksi:", ["Teks Tunggal", "Batch (Upload File)"], horizontal=True, key="pred_mode")
    if mode == "Teks Tunggal":
        _render_single_prediction(model, model_info)
    else:
        _render_batch_prediction(model, model_info)


def _render_model_overview(model, model_info) -> None:
    best_name = _get_best_model_name(model_info, model)
    accuracy = _get_metric(model_info, best_name, "accuracy")
    f1_score = _get_metric(model_info, best_name, "f1_macro")
    classes = _get_model_classes(model, model_info)
    vectorizer_desc, classifier_desc = _describe_pipeline(model)

    st.markdown("#### 🤖 Model Prediksi Aktif")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model", best_name)
    c2.metric("Akurasi", _format_percent(accuracy))
    c3.metric("Macro F1", _format_percent(f1_score))
    c4.metric("Kelas", ", ".join(classes) if classes else "-")

    with st.expander("Detail model dan alur prediksi", expanded=False):
        st.markdown(
            f"""
**Pipeline:** teks ulasan → preprocessing → TF-IDF → classifier → label sentimen

**Stemming:** {"aktif" if model_info.get("use_stemming", False) else "nonaktif"}

**Vectorizer:** {vectorizer_desc}

**Classifier:** {classifier_desc}

**Output:** label `positif`, `netral`, atau `negatif`
            """
        )


def _render_single_prediction(model, model_info) -> None:
    input_text = st.text_area(
        "Teks Ulasan",
        height=120,
        placeholder="Contoh: Produk ini sangat bagus, pengiriman cepat!",
    )
    if not st.button("🔮 Prediksi", type="primary", key="pred_btn"):
        return
    if not input_text.strip():
        st.warning("Masukkan teks terlebih dahulu.")
        return

    try:
        with st.spinner("Menganalisis..."):
            use_stemming = model_info.get("use_stemming", False)
            preds, probas, classes = predict_sentiment(
                [input_text], model, use_stemming=use_stemming
            )
            label = preds[0]
            clean_text = preprocess(input_text, use_stemming=use_stemming)
    except ValueError as exc:
        st.error(str(exc))
        return
    st.markdown("---")
    cr1, cr2 = st.columns([1, 2])
    with cr1:
        cmap = {"positif": "#d4edda", "netral": "#fff3cd", "negatif": "#f8d7da"}
        tmap = {"positif": "#155724", "netral": "#856404", "negatif": "#721c24"}
        confidence = _get_confidence(label, probas, classes)
        confidence_text = _format_percent(confidence)
        st.markdown(
            f"""
            <div style="background:{cmap[label]};border-radius:16px;
                        padding:30px;text-align:center;">
                <div style="font-size:3rem">{SENTIMENT_EMOJI.get(label,'')}</div>
                <div style="font-size:1.8rem;font-weight:700;color:{tmap[label]};
                            margin-top:8px">{label.upper()}</div>
                <div style="color:{tmap[label]};margin-top:4px;font-size:0.9rem">
                    Hasil Prediksi Sentimen</div>
                <div style="color:{tmap[label]};margin-top:10px;font-size:0.95rem">
                    Confidence: <b>{confidence_text}</b></div>
            </div>""",
            unsafe_allow_html=True,
        )
    with cr2:
        best_name = _get_best_model_name(model_info, model)
        st.markdown("**Ringkasan Prediksi:**")
        st.dataframe(
            pd.DataFrame(
                [
                    {"Item": "Model yang digunakan", "Nilai": best_name},
                    {"Item": "Teks setelah preprocessing", "Nilai": clean_text or "(kosong)"},
                    {"Item": "Label hasil prediksi", "Nilai": label},
                    {"Item": "Confidence", "Nilai": confidence_text},
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )
        if probas is not None:
            st.markdown("**Probabilitas per Kelas:**")
            prob_df = _build_probability_dataframe(classes, probas[0])
            st.dataframe(prob_df, use_container_width=True, hide_index=True)
            for cls, prob in zip(prob_df["Kelas"], prob_df["Probabilitas"]):
                st.markdown(f"**{cls.capitalize()}**")
                st.progress(float(prob), text=f"{prob*100:.1f}%")
        else:
            st.info("Model ini tidak menyediakan probabilitas kelas. Confidence ditampilkan jika tersedia dari model.")


def _render_batch_prediction(model, model_info) -> None:
    batch_file = st.file_uploader("File untuk prediksi", type=["csv", "xlsx"], key="batch_pred")
    if not batch_file:
        return

    try:
        raw_batch = read_uploaded_dataset(batch_file)
    except DatasetValidationError as exc:
        st.error(str(exc))
        return
    df_batch, validation = validate_prediction_dataset(raw_batch)
    if not validation.is_valid:
        for message in validation.errors:
            st.error(message)
        st.error("File batch ditolak. Perbaiki seluruh error sebelum prediksi.")
        return

    try:
        with st.spinner(f"Memprediksi {len(df_batch):,} ulasan..."):
            use_stemming = model_info.get("use_stemming", False)
            preds, probas, classes = predict_sentiment(
                df_batch["review_text"].astype(str).tolist(),
                model,
                use_stemming=use_stemming,
            )
            df_batch["prediksi_sentimen"] = preds
            df_batch["clean_text"] = df_batch["review_text"].astype(str).apply(
                lambda text: preprocess(text, use_stemming=use_stemming)
            )
            if probas is not None:
                for i, cls in enumerate(classes):
                    df_batch[f"prob_{cls}"] = probas[:, i].round(4)
                df_batch["confidence"] = probas.max(axis=1).round(4)
    except ValueError as exc:
        st.error(str(exc))
        return
    st.success(f"✅ Selesai! {len(df_batch):,} ulasan diprediksi.")
    sc2 = pd.Series(preds).value_counts()
    b1, b2, b3, b4 = st.columns(4)
    b1.metric("😊 Positif", sc2.get("positif", 0))
    b2.metric("😐 Netral", sc2.get("netral", 0))
    b3.metric("😞 Negatif", sc2.get("negatif", 0))
    if probas is not None:
        b4.metric("Rata-rata Confidence", f"{df_batch['confidence'].mean()*100:.1f}%")
    else:
        b4.metric("Model", _get_best_model_name(model_info, model))

    summary_df = (
        df_batch["prediksi_sentimen"]
        .value_counts()
        .rename_axis("Sentimen")
        .reset_index(name="Jumlah")
    )
    summary_df["Persentase"] = (summary_df["Jumlah"] / len(df_batch) * 100).round(2)
    st.markdown("**Ringkasan Hasil Batch:**")
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    display_columns = [
        col
        for col in ["review_text", "clean_text", "prediksi_sentimen", "confidence", "prob_positif", "prob_netral", "prob_negatif"]
        if col in df_batch.columns
    ]
    st.dataframe(df_batch[display_columns].head(100), use_container_width=True)
    st.download_button(
        "⬇️ Download Hasil CSV",
        data=df_batch.to_csv(index=False).encode("utf-8"),
        file_name="hasil_prediksi.csv",
        mime="text/csv",
    )


def _get_best_model_name(model_info, model) -> str:
    if model_info and "best_model_name" in model_info:
        return model_info["best_model_name"]
    return _describe_model_object(model)


def _get_metric(model_info, model_name: str, metric_name: str):
    if not model_info:
        return None
    return model_info.get("results", {}).get(model_name, {}).get(metric_name)


def _get_model_classes(model, model_info) -> list[str]:
    if model_info and model_info.get("label_classes"):
        return model_info["label_classes"]
    classes = getattr(model, "classes_", None)
    if classes is not None:
        return [str(cls) for cls in classes]
    return []


def _describe_pipeline(model) -> tuple[str, str]:
    if not hasattr(model, "named_steps"):
        return "-", _describe_model_object(model)

    tfidf = model.named_steps.get("tfidf")
    clf = model.named_steps.get("clf")
    vectorizer_desc = "-"
    if tfidf is not None:
        max_features = getattr(tfidf, "max_features", None)
        ngram_range = getattr(tfidf, "ngram_range", None)
        vectorizer_desc = f"TF-IDF, max_features={max_features}, ngram_range={ngram_range}"
    return vectorizer_desc, _describe_model_object(clf)


def _describe_model_object(model) -> str:
    if model is None:
        return "-"
    if model.__class__.__name__ == "CalibratedClassifierCV":
        estimator = getattr(model, "estimator", None) or getattr(model, "base_estimator", None)
        if estimator is not None:
            return f"Calibrated {estimator.__class__.__name__}"
    return model.__class__.__name__


def _format_percent(value) -> str:
    if value is None:
        return "-"
    return f"{float(value) * 100:.2f}%"


def _get_confidence(label, probas, classes):
    if probas is None or classes is None:
        return None
    prob_map = {cls: prob for cls, prob in zip(classes, probas[0])}
    return prob_map.get(label)


def _build_probability_dataframe(classes, probabilities) -> pd.DataFrame:
    prob_df = pd.DataFrame(
        {
            "Kelas": classes,
            "Probabilitas": probabilities,
            "Persentase": [f"{prob*100:.2f}%" for prob in probabilities],
        }
    )
    return prob_df.sort_values("Probabilitas", ascending=False).reset_index(drop=True)

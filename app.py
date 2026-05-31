"""
app.py — Entry point dashboard SentiPro.
Jalankan: python -m streamlit run app.py
"""

import sys
import warnings

import streamlit as st

from src.config import BASE_DIR
from src.data_service import preprocess_data, read_uploaded_dataset
from src.model_service import load_model, load_model_info, predict_sentiment
from src.pages.brand import render_brand_page
from src.pages.category import render_category_page
from src.pages.prediction import render_prediction_page
from src.pages.preprocessing_view import render_preprocessing_page
from src.pages.product import render_product_page
from src.pages.summary import render_summary_page
from src.pages.training import render_training_page
from src.pages.trend import render_trend_page
from src.ui import configure_page, inject_custom_css

warnings.filterwarnings("ignore")

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


def render_sidebar():
    with st.sidebar:
        st.markdown("## 📊 SentiPro")
        st.markdown("*Analisis Sentimen Ulasan Produk*")
        st.divider()

        st.markdown("### 📁 Upload Dataset")
        uploaded = st.file_uploader("CSV atau Excel", type=["csv", "xlsx", "xls"])

        st.divider()
        st.markdown("### ⚙️ Pengaturan")
        use_model_predict = st.toggle(
            "Gunakan Model ML untuk Prediksi",
            value=False,
            help="Jika OFF, label diambil dari kolom Rating",
        )
        show_raw = st.toggle("Tampilkan Data Mentah", value=False)

        st.divider()
        model_info = load_model_info()
        if model_info:
            st.markdown("### 🤖 Model Aktif")
            st.success(f"**{model_info['best_model_name']}**")
            best_acc = model_info["results"][model_info["best_model_name"]]["accuracy"]
            st.metric("Akurasi", f"{best_acc*100:.1f}%")
        else:
            st.warning("Model belum dilatih.\nGunakan tab 🤖 Training Model")

        st.divider()
        st.caption("SentiPro v2.0 | Bahasa Indonesia")

    return uploaded, use_model_predict, show_raw


def load_dashboard_data(uploaded):
    if not uploaded:
        st.info("👈 Upload dataset CSV/Excel di sidebar untuk memulai analisis.")
        st.markdown(
            """
| Kolom | Keterangan |
|-------|-----------|
| Product name | Nama produk |
| Brand | Merek produk |
| Category | Kategori produk |
| Rating | Nilai 1–5 |
| Review date | Tanggal ulasan |
| Review text | Isi teks ulasan |
            """
        )
        st.stop()

    try:
        raw = read_uploaded_dataset(uploaded)
        with st.spinner("Memproses dataset..."):
            df = preprocess_data(raw)
        st.success(f"✅ Dataset dimuat: **{len(df):,} ulasan** | Kolom: {list(df.columns)}")
        return df
    except Exception as exc:
        st.error(f"Gagal membaca file: {exc}")
        st.stop()


def apply_model_prediction_if_requested(df, use_model_predict: bool):
    model = load_model()
    if use_model_predict and model:
        with st.spinner("Memprediksi sentimen dengan model ML..."):
            preds, _, _ = predict_sentiment(df["review_text"].tolist(), model)
            df = df.copy()
            df["sentiment"] = preds
        st.info("🤖 Sentimen diprediksi menggunakan model ML")
    elif use_model_predict and not model:
        st.warning("Model belum tersedia. Menggunakan rating sebagai label.")
    return df


def render_header() -> None:
    st.markdown("# 📊 SentiPro — Dashboard Analisis Sentimen Produk")
    st.markdown(
        "Analisis mendalam sentimen ulasan produk: perbandingan brand, kategori, produk, dan tren waktu."
    )
    st.divider()


def main() -> None:
    configure_page()
    inject_custom_css()

    uploaded, use_model_predict, show_raw = render_sidebar()
    render_header()

    df = load_dashboard_data(uploaded)
    df = apply_model_prediction_if_requested(df, use_model_predict)

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(
        [
            "🏠 Ringkasan",
            "🏷️ Brand",
            "📦 Produk",
            "🗂️ Kategori",
            "📅 Tren Waktu",
            "🔍 Prediksi",
            "⚙️ Preprocessing",
            "🤖 Training Model",
        ]
    )

    with tab1:
        render_summary_page(df, show_raw)
    with tab2:
        render_brand_page(df)
    with tab3:
        render_product_page(df)
    with tab4:
        render_category_page(df)
    with tab5:
        render_trend_page(df)
    with tab6:
        render_prediction_page()
    with tab7:
        render_preprocessing_page(df)
    with tab8:
        render_training_page(df)


if __name__ == "__main__":
    main()


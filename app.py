"""
app.py — Entry point dashboard SentiPro.
Jalankan: python -m streamlit run app.py
"""

import hashlib
import sys
import warnings

import streamlit as st

from src.config import BASE_DIR
from src.data_service import (
    DatasetValidationError,
    preprocess_data,
    read_uploaded_dataset,
    validate_dataset,
)
from src.model_service import (
    clear_session_model,
    get_session_model_info,
)
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
        uploaded = st.file_uploader("CSV atau Excel", type=["csv", "xlsx"])

        st.divider()
        st.markdown("### ⚙️ Pengaturan")
        show_raw = st.toggle("Tampilkan Data Mentah", value=False)

        st.divider()
        st.markdown("### ✅ Status Alur")
        dataset_ready = st.session_state.get("dataset_ready", False)
        st.write(f"{'✅' if uploaded else '⬜'} 1. Upload dataset")
        st.write(f"{'✅' if dataset_ready else '⬜'} 2. Validasi & preprocessing")

        model_info = get_session_model_info()
        if model_info:
            st.write("✅ 3. Training model")
            st.write("✅ 4. Prediksi aktif")
            st.success(f"**{model_info['best_model_name']}**")
            best_result = model_info["results"][model_info["best_model_name"]]
            st.metric("Macro F1", f"{best_result['f1_macro']*100:.1f}%")
        else:
            st.write("⬜ 3. Training model")
            st.write("🔒 4. Prediksi")

        st.divider()
        st.caption("Model dan dataset hanya tersedia selama sesi pengguna aktif.")

    return uploaded, show_raw


def load_dashboard_data(uploaded):
    if not uploaded:
        st.session_state["dataset_ready"] = False
        st.session_state.pop("dataset_fingerprint", None)
        clear_session_model()
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

    fingerprint = hashlib.sha256(uploaded.getvalue()).hexdigest()
    if fingerprint != st.session_state.get("dataset_fingerprint"):
        clear_session_model()
        st.session_state["dataset_ready"] = False
        st.session_state["dataset_fingerprint"] = fingerprint

    try:
        raw = read_uploaded_dataset(uploaded)
        normalized, report = validate_dataset(raw)
        _render_validation_report(report)
        if not report.is_valid:
            st.session_state["dataset_ready"] = False
            st.error("Dataset ditolak. Perbaiki semua error sebelum melanjutkan.")
            st.stop()

        with st.spinner("Memproses dataset..."):
            df = preprocess_data(normalized, use_stemming=False)
        st.session_state["dataset_ready"] = True
        st.success(
            f"Dataset valid dan selesai diproses: **{len(df):,} ulasan**. "
            "Label analisis dibuat dari rating."
        )
        return df
    except DatasetValidationError as exc:
        st.session_state["dataset_ready"] = False
        st.error(str(exc))
        st.stop()
    except Exception:
        st.session_state["dataset_ready"] = False
        st.error(
            "Terjadi kegagalan saat memproses dataset. Periksa kembali format file dan isi kolom."
        )
        st.stop()


def _render_validation_report(report) -> None:
    with st.expander("Hasil validasi dataset", expanded=bool(report.errors or report.warnings)):
        if report.errors:
            for message in report.errors:
                st.error(message)
        else:
            st.success("Struktur dan nilai wajib dataset valid.")
        for message in report.warnings:
            st.warning(message)
        if report.stats:
            st.json(report.stats)


def render_header() -> None:
    st.markdown("# 📊 SentiPro — Dashboard Analisis Sentimen Produk")
    st.markdown(
        "Alur wajib: upload → validasi & preprocessing → training → prediksi."
    )
    st.caption(
        "Dashboard analisis menggunakan label dari rating. Prediksi ML baru aktif setelah model dilatih pada sesi ini."
    )
    st.divider()


def main() -> None:
    configure_page()
    inject_custom_css()

    uploaded, show_raw = render_sidebar()
    render_header()

    df = load_dashboard_data(uploaded)

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(
        [
            "1️⃣ Ringkasan",
            "2️⃣ Preprocessing",
            "3️⃣ Training Model",
            "4️⃣ Prediksi",
            "🏷️ Brand",
            "📦 Produk",
            "🗂️ Kategori",
            "📅 Tren Waktu",
        ]
    )

    with tab1:
        render_summary_page(df, show_raw)
    with tab2:
        render_preprocessing_page(df)
    with tab3:
        render_training_page(df)
    with tab4:
        render_prediction_page()
    with tab5:
        render_brand_page(df)
    with tab6:
        render_product_page(df)
    with tab7:
        render_category_page(df)
    with tab8:
        render_trend_page(df)


if __name__ == "__main__":
    main()

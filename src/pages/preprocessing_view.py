"""Tab visualisasi preprocessing."""

from collections import Counter

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.preprocessing import SLANG_DICT, STOP_WORDS, preprocess_steps
from src.ui import section_header


def render_preprocessing_page(df) -> None:
    section_header("⚙️ Visualisasi Pipeline Preprocessing")

    st.markdown("#### 🧪 Coba Teks Sendiri")
    demo_text = st.text_area(
        "Masukkan teks:",
        value="Produk ini SANGAT bagus bgt!! Pengiriman cepet banget, recommended deh 👍 harga 50000",
        height=80,
        key="pp_demo",
    )
    use_stem_demo = st.checkbox("Aktifkan Stemming", value=True, key="pp_stem")

    if st.button("▶️ Jalankan Preprocessing", type="primary", key="pp_run"):
        _render_preprocessing_steps(demo_text, use_stem_demo)

    st.divider()
    if "clean_text" in df.columns:
        _render_dataset_preprocessing_stats(df)

    st.divider()
    st.markdown("#### 📖 Kamus Normalisasi Slang")
    st.caption(f"Total {len(SLANG_DICT)} entri")
    st.dataframe(pd.DataFrame(list(SLANG_DICT.items()), columns=["Slang", "Bentuk Baku"]), use_container_width=True, height=300)

    st.markdown("#### 🚫 Daftar Stopwords")
    st.caption(f"Total {len(STOP_WORDS)} stopwords")
    sw_list = sorted(STOP_WORDS)
    sw_cols = st.columns(5)
    chunk = len(sw_list) // 5 + 1
    for i, col in enumerate(sw_cols):
        col.write("\n".join(sw_list[i * chunk : (i + 1) * chunk]))


def _render_preprocessing_steps(demo_text: str, use_stem_demo: bool) -> None:
    steps = preprocess_steps(demo_text, use_stemming=use_stem_demo)
    step_meta = {
        "original": ("📝 Teks Asli", "#6c757d", "Teks input sebelum diproses"),
        "case_folding": ("🔡 Case Folding", "#0d6efd", "Semua huruf diubah ke lowercase"),
        "cleaning": ("🧹 Cleaning", "#6610f2", "Hapus URL, mention, angka, tanda baca, emoji"),
        "normalisasi": ("📖 Normalisasi Slang", "#0dcaf0", "Kata slang/singkatan → bentuk baku"),
        "tokenisasi": ("✂️ Tokenisasi", "#198754", "Teks dipecah menjadi token (kata)"),
        "stopword_removal": ("🚫 Stopword Removal", "#ffc107", "Kata tidak bermakna dihapus"),
        "stemming": ("🌱 Stemming", "#dc3545", "Kata dikembalikan ke bentuk dasar"),
        "hasil_akhir": ("✅ Hasil Akhir", "#20c997", "Teks siap untuk training/prediksi"),
    }
    for key, (lbl, color, desc) in step_meta.items():
        if key not in steps:
            continue
        val = steps[key]
        display = val if not isinstance(val, list) else (" | ".join(val) if val else "*(kosong)*")
        n_tok = len(val) if isinstance(val, list) else len(str(val).split())
        st.markdown(
            f"""
            <div style="border-left:4px solid {color};background:#f8f9fa;border-radius:8px;
                        padding:12px 16px;margin-bottom:10px;">
                <div style="font-weight:700;color:{color};font-size:0.95rem">{lbl}</div>
                <div style="font-size:0.78rem;color:#6c757d;margin-bottom:6px">{desc}</div>
                <div style="font-family:monospace;font-size:0.9rem;color:#212529;background:white;
                            padding:8px;border-radius:4px;border:1px solid #dee2e6">{display}</div>
                <div style="font-size:0.75rem;color:#6c757d;margin-top:4px">
                    Jumlah token: <b>{n_tok}</b></div>
            </div>""",
            unsafe_allow_html=True,
        )


def _render_dataset_preprocessing_stats(df) -> None:
    st.markdown("#### 📊 Statistik Preprocessing Dataset")
    avg_raw = df["review_text"].astype(str).apply(lambda x: len(x.split())).mean()
    avg_clean = df["clean_text"].astype(str).apply(lambda x: len(x.split())).mean()
    reduction = (1 - avg_clean / avg_raw) * 100 if avg_raw > 0 else 0
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Rata-rata Token Asli", f"{avg_raw:.1f}")
    s2.metric("Rata-rata Token Bersih", f"{avg_clean:.1f}")
    s3.metric("Reduksi Token", f"{reduction:.1f}%")
    s4.metric("Total Vocab Bersih", f"{len(set(' '.join(df['clean_text'].dropna()).split())):,}")

    df_len = pd.DataFrame(
        {
            "Sebelum": df["review_text"].astype(str).apply(lambda x: len(x.split())),
            "Sesudah": df["clean_text"].astype(str).apply(lambda x: len(x.split())),
        }
    )
    fig_pp1 = go.Figure()
    fig_pp1.add_trace(go.Histogram(x=df_len["Sebelum"], name="Sebelum", opacity=0.7, marker_color="#6c757d", nbinsx=50))
    fig_pp1.add_trace(go.Histogram(x=df_len["Sesudah"], name="Sesudah", opacity=0.7, marker_color="#0d6efd", nbinsx=50))
    fig_pp1.update_layout(
        barmode="overlay",
        title="Distribusi Panjang Teks",
        xaxis_title="Jumlah Kata",
        yaxis_title="Frekuensi",
        height=380,
    )
    st.plotly_chart(fig_pp1, use_container_width=True, key="t7_hist")

    st.markdown("#### 🔤 Top 20 Token per Sentimen")
    sent_sel = st.selectbox("Pilih sentimen:", ["positif", "netral", "negatif"], key="pp_sent")
    all_tok = " ".join(df[df["sentiment"] == sent_sel]["clean_text"].dropna()).split()
    top_tok = pd.DataFrame(Counter(all_tok).most_common(20), columns=["Token", "Frekuensi"])
    fig_pp2 = px.bar(
        top_tok,
        x="Frekuensi",
        y="Token",
        orientation="h",
        color="Frekuensi",
        color_continuous_scale="Blues",
        title=f"Top 20 Token — {sent_sel.capitalize()}",
    )
    fig_pp2.update_layout(height=500, yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig_pp2, use_container_width=True, key="t7_toptoken")


"""Tab visualisasi preprocessing dataset dan demo teks opsional."""

from collections import Counter
from html import escape

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.preprocessing import SLANG_DICT, STOP_WORDS, preprocess_steps
from src.ui import section_header


def render_preprocessing_page(df) -> None:
    section_header("2. Preprocessing Teks")

    st.success(
        f"Preprocessing otomatis telah selesai untuk **{len(df):,} ulasan** "
        "dari dataset yang Anda upload."
    )
    st.info(
        "Hasil `clean_text` di halaman ini dibuat tanpa stemming untuk analisis awal. "
        "Saat training, seluruh ulasan akan diproses ulang sesuai pilihan stemming "
        "dan konfigurasi yang sama digunakan saat prediksi."
    )

    _render_dataset_comparison(df)

    st.divider()
    _render_dataset_review_steps(df)

    st.divider()
    _render_dataset_preprocessing_stats(df)

    st.divider()
    _render_optional_manual_demo()

    st.divider()
    _render_preprocessing_references()


def _render_dataset_comparison(df) -> None:
    st.markdown("#### Perbandingan Teks Asli dan Hasil Preprocessing")
    st.caption(
        "Tabel berikut berasal dari dataset upload. Data ini digunakan untuk analisis "
        "dan menjadi sumber data training."
    )

    comparison = df.reset_index(drop=True).reset_index(names="baris")
    comparison["baris"] = comparison["baris"] + 1
    display_columns = [
        column
        for column in [
            "baris",
            "product_name",
            "rating",
            "sentiment",
            "review_text",
            "clean_text",
        ]
        if column in comparison.columns
    ]
    st.dataframe(
        comparison[display_columns].head(200),
        use_container_width=True,
        hide_index=True,
        column_config={
            "baris": "Baris",
            "product_name": "Produk",
            "rating": "Rating",
            "sentiment": "Sentimen",
            "review_text": st.column_config.TextColumn("Teks Asli", width="large"),
            "clean_text": st.column_config.TextColumn("Hasil Preprocessing", width="large"),
        },
    )
    if len(comparison) > 200:
        st.caption("Tabel menampilkan 200 baris pertama agar dashboard tetap responsif.")


def _render_dataset_review_steps(df) -> None:
    st.markdown("#### Tahapan Preprocessing Salah Satu Ulasan Dataset")
    st.caption(
        "Pilih satu ulasan untuk melihat perubahan teks pada setiap tahap. "
        "Contoh ini berasal dari dataset upload, bukan teks buatan."
    )

    options = list(range(len(df)))
    selected_position = st.selectbox(
        "Pilih ulasan dataset",
        options,
        format_func=lambda position: _review_option_label(df.iloc[position], position),
        key="pp_dataset_review",
    )
    selected = df.iloc[selected_position]

    metadata = [
        {"Item": "Baris dataset", "Nilai": str(selected_position + 1)},
        {"Item": "Produk", "Nilai": str(selected.get("product_name", "-"))},
        {"Item": "Rating", "Nilai": str(selected.get("rating", "-"))},
        {"Item": "Sentimen dari rating", "Nilai": str(selected.get("sentiment", "-"))},
    ]
    st.dataframe(pd.DataFrame(metadata), use_container_width=True, hide_index=True)
    _render_preprocessing_steps(str(selected["review_text"]), use_stemming=False)


def _review_option_label(row: pd.Series, position: int) -> str:
    review = str(row.get("review_text", "")).replace("\n", " ").strip()
    preview = review[:80] + ("..." if len(review) > 80 else "")
    return f"Baris {position + 1} | {row.get('sentiment', '-')} | {preview}"


def _render_optional_manual_demo() -> None:
    st.markdown("#### Demo Preprocessing Teks Sendiri (Opsional)")
    st.caption(
        "Gunakan fitur ini hanya untuk mencoba cara kerja preprocessing pada teks lain."
    )
    try_manual = st.toggle(
        "Saya ingin mencoba preprocessing teks sendiri",
        value=False,
        key="pp_manual_toggle",
    )
    if not try_manual:
        return

    st.warning(
        "Teks percobaan tidak ditambahkan ke dataset, tidak mengubah `clean_text`, "
        "dan tidak digunakan untuk training model."
    )
    demo_text = st.text_area(
        "Masukkan teks percobaan",
        value="Produk ini SANGAT bagus bgt!! Pengiriman cepet banget, recommended deh 👍 harga 50000",
        height=100,
        key="pp_demo",
    )
    use_stem_demo = st.checkbox(
        "Aktifkan stemming untuk demo",
        value=False,
        key="pp_stem",
    )

    if st.button("Jalankan Demo Preprocessing", type="primary", key="pp_run"):
        if not demo_text.strip():
            st.error("Teks percobaan tidak boleh kosong.")
            return
        _render_preprocessing_steps(demo_text, use_stem_demo)


def _render_preprocessing_steps(text: str, use_stemming: bool) -> None:
    steps = preprocess_steps(text, use_stemming=use_stemming)
    step_meta = {
        "original": ("Teks Asli", "#6c757d", "Teks sebelum diproses"),
        "case_folding": ("Case Folding", "#0d6efd", "Semua huruf diubah menjadi lowercase"),
        "cleaning": ("Cleaning", "#6610f2", "URL, mention, angka, tanda baca, dan emoji dihapus"),
        "normalisasi": ("Normalisasi Slang", "#0dcaf0", "Kata slang atau singkatan diubah ke bentuk baku"),
        "tokenisasi": ("Tokenisasi", "#198754", "Teks dipecah menjadi token"),
        "stopword_removal": ("Stopword Removal", "#b8860b", "Kata umum yang tidak informatif dihapus"),
        "stemming": ("Stemming", "#dc3545", "Kata dikembalikan ke bentuk dasar"),
        "hasil_akhir": ("Hasil Akhir", "#20c997", "Teks siap digunakan"),
    }
    for key, (label, color, description) in step_meta.items():
        if key not in steps:
            continue
        value = steps[key]
        display = value if not isinstance(value, list) else (" | ".join(value) if value else "(kosong)")
        token_count = len(value) if isinstance(value, list) else len(str(value).split())
        st.markdown(
            f"""
            <div style="border-left:4px solid {color};background:#f8f9fa;border-radius:8px;
                        padding:12px 16px;margin-bottom:10px;">
                <div style="font-weight:700;color:{color};font-size:0.95rem">{label}</div>
                <div style="font-size:0.78rem;color:#6c757d;margin-bottom:6px">{description}</div>
                <div style="font-family:monospace;font-size:0.9rem;color:#212529;background:white;
                            padding:8px;border-radius:4px;border:1px solid #dee2e6">
                    {escape(str(display))}
                </div>
                <div style="font-size:0.75rem;color:#6c757d;margin-top:4px">
                    Jumlah token: <b>{token_count}</b>
                </div>
            </div>""",
            unsafe_allow_html=True,
        )


def _render_dataset_preprocessing_stats(df) -> None:
    st.markdown("#### Statistik Preprocessing Dataset")
    avg_raw = df["review_text"].astype(str).apply(lambda text: len(text.split())).mean()
    avg_clean = df["clean_text"].astype(str).apply(lambda text: len(text.split())).mean()
    reduction = (1 - avg_clean / avg_raw) * 100 if avg_raw > 0 else 0
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Rata-rata Token Asli", f"{avg_raw:.1f}")
    s2.metric("Rata-rata Token Bersih", f"{avg_clean:.1f}")
    s3.metric("Reduksi Token", f"{reduction:.1f}%")
    s4.metric("Total Vocab Bersih", f"{len(set(' '.join(df['clean_text'].dropna()).split())):,}")

    df_len = pd.DataFrame(
        {
            "Sebelum": df["review_text"].astype(str).apply(lambda text: len(text.split())),
            "Sesudah": df["clean_text"].astype(str).apply(lambda text: len(text.split())),
        }
    )
    fig_pp1 = go.Figure()
    fig_pp1.add_trace(
        go.Histogram(
            x=df_len["Sebelum"],
            name="Sebelum",
            opacity=0.7,
            marker_color="#6c757d",
            nbinsx=50,
        )
    )
    fig_pp1.add_trace(
        go.Histogram(
            x=df_len["Sesudah"],
            name="Sesudah",
            opacity=0.7,
            marker_color="#0d6efd",
            nbinsx=50,
        )
    )
    fig_pp1.update_layout(
        barmode="overlay",
        title="Distribusi Panjang Teks",
        xaxis_title="Jumlah Kata",
        yaxis_title="Frekuensi",
        height=380,
    )
    st.plotly_chart(fig_pp1, use_container_width=True, key="t7_hist")

    st.markdown("#### Top 20 Token per Sentimen")
    sent_sel = st.selectbox(
        "Pilih sentimen",
        ["positif", "netral", "negatif"],
        key="pp_sent",
    )
    all_tokens = " ".join(
        df[df["sentiment"] == sent_sel]["clean_text"].dropna()
    ).split()
    top_tokens = pd.DataFrame(
        Counter(all_tokens).most_common(20),
        columns=["Token", "Frekuensi"],
    )
    if top_tokens.empty:
        st.info(f"Tidak ada token untuk sentimen {sent_sel}.")
        return
    fig_pp2 = px.bar(
        top_tokens,
        x="Frekuensi",
        y="Token",
        orientation="h",
        color="Frekuensi",
        color_continuous_scale="Blues",
        title=f"Top 20 Token - {sent_sel.capitalize()}",
    )
    fig_pp2.update_layout(height=500, yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig_pp2, use_container_width=True, key="t7_toptoken")


def _render_preprocessing_references() -> None:
    st.markdown("#### Referensi Preprocessing")
    with st.expander(f"Kamus Normalisasi Slang ({len(SLANG_DICT)} entri)"):
        st.dataframe(
            pd.DataFrame(list(SLANG_DICT.items()), columns=["Slang", "Bentuk Baku"]),
            use_container_width=True,
            height=300,
        )

    with st.expander(f"Daftar Stopword ({len(STOP_WORDS)} kata)"):
        st.write(", ".join(sorted(STOP_WORDS)))

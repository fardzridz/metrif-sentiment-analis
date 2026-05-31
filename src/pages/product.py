"""Tab analisis produk."""

import plotly.express as px
import streamlit as st

from src.ui import section_header


def render_product_page(df) -> None:
    if "product_name" not in df.columns:
        st.warning("Kolom 'product_name' tidak ditemukan.")
        return

    section_header("📦 Analisis per Produk")
    cf1, cf2 = st.columns(2)
    with cf1:
        top_n_prod = st.slider("Top N produk", 5, 30, 10, key="prod_n")
    with cf2:
        brand_filter = []
        if "brand" in df.columns:
            brand_filter = st.multiselect(
                "Filter Brand", df["brand"].unique().tolist(), default=[], key="prod_brand"
            )
    df_prod = df[df["brand"].isin(brand_filter)].copy() if brand_filter else df.copy()
    top_products = df_prod["product_name"].value_counts().head(top_n_prod).index.tolist()
    df_prod = df_prod[df_prod["product_name"].isin(top_products)]

    ps = df_prod.groupby("product_name").agg(
        total=("sentiment", "count"),
        positif=("sentiment", lambda x: (x == "positif").sum()),
        netral=("sentiment", lambda x: (x == "netral").sum()),
        negatif=("sentiment", lambda x: (x == "negatif").sum()),
        avg_rating=("rating", "mean"),
    ).reset_index()
    ps["skor_sentimen"] = ((ps["positif"] - ps["negatif"]) / ps["total"] * 100).round(1)
    ps = ps.sort_values("skor_sentimen", ascending=False)

    ct, cb_col = st.columns(2)
    with ct:
        fig_p1 = px.bar(
            ps.head(5),
            x="skor_sentimen",
            y="product_name",
            orientation="h",
            color="skor_sentimen",
            color_continuous_scale="Greens",
            title="🏆 Top 5 Produk Terbaik",
            labels={"skor_sentimen": "Skor (%)", "product_name": "Produk"},
        )
        fig_p1.update_layout(height=320, yaxis={"categoryorder": "total ascending"}, margin=dict(t=50, b=10))
        st.plotly_chart(fig_p1, use_container_width=True, key="t3_top5")
    with cb_col:
        fig_p2 = px.bar(
            ps.tail(5),
            x="skor_sentimen",
            y="product_name",
            orientation="h",
            color="skor_sentimen",
            color_continuous_scale="Reds_r",
            title="⚠️ Top 5 Produk Perlu Perhatian",
            labels={"skor_sentimen": "Skor (%)", "product_name": "Produk"},
        )
        fig_p2.update_layout(height=320, yaxis={"categoryorder": "total descending"}, margin=dict(t=50, b=10))
        st.plotly_chart(fig_p2, use_container_width=True, key="t3_bot5")

    fig_p3 = px.bar(
        ps,
        x="product_name",
        y=["positif", "netral", "negatif"],
        barmode="stack",
        title="Komposisi Sentimen per Produk",
        color_discrete_map={"positif": "#2ecc71", "netral": "#f39c12", "negatif": "#e74c3c"},
        labels={"value": "Jumlah", "product_name": "Produk", "variable": "Sentimen"},
    )
    fig_p3.update_layout(height=400, xaxis_tickangle=-35, margin=dict(t=50, b=80))
    st.plotly_chart(fig_p3, use_container_width=True, key="t3_stack")

    section_header("🔎 Detail Produk")
    selected_prod = st.selectbox("Pilih produk:", ps["product_name"].tolist(), key="prod_sel")
    if selected_prod:
        df_sel = df_prod[df_prod["product_name"] == selected_prod]
        s = ps[ps["product_name"] == selected_prod].iloc[0]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Ulasan", f"{int(s['total']):,}")
        m2.metric("Positif", f"{int(s['positif'])} ({s['positif']/s['total']*100:.0f}%)")
        m3.metric("Negatif", f"{int(s['negatif'])} ({s['negatif']/s['total']*100:.0f}%)")
        m4.metric("Avg Rating", f"{s['avg_rating']:.2f} ⭐")
        st.markdown("**Contoh ulasan positif:**")
        for review in df_sel[df_sel["sentiment"] == "positif"]["review_text"].head(3).tolist():
            st.success(f"💬 {review}")
        st.markdown("**Contoh ulasan negatif:**")
        for review in df_sel[df_sel["sentiment"] == "negatif"]["review_text"].head(3).tolist():
            st.error(f"💬 {review}")


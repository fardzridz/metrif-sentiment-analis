"""Tab analisis brand."""

import plotly.express as px
import streamlit as st

from src.ui import section_header


def render_brand_page(df) -> None:
    if "brand" not in df.columns:
        st.warning("Kolom 'brand' tidak ditemukan.")
        return

    section_header("🏷️ Analisis per Brand")
    top_n_brand = st.slider("Tampilkan top N brand", 5, 30, 10, key="brand_n")
    top_brands = df["brand"].value_counts().head(top_n_brand).index.tolist()
    df_brand = df[df["brand"].isin(top_brands)]

    bs = df_brand.groupby("brand").agg(
        total=("sentiment", "count"),
        positif=("sentiment", lambda x: (x == "positif").sum()),
        netral=("sentiment", lambda x: (x == "netral").sum()),
        negatif=("sentiment", lambda x: (x == "negatif").sum()),
        avg_rating=("rating", "mean"),
    ).reset_index()
    bs["skor_sentimen"] = ((bs["positif"] - bs["negatif"]) / bs["total"] * 100).round(1)
    bs["positif_pct"] = (bs["positif"] / bs["total"] * 100).round(1)
    bs = bs.sort_values("skor_sentimen", ascending=False)

    c1, c2 = st.columns(2)
    with c1:
        fig_b1 = px.bar(
            bs,
            x="skor_sentimen",
            y="brand",
            orientation="h",
            color="skor_sentimen",
            color_continuous_scale="RdYlGn",
            title="Skor Sentimen per Brand",
            labels={"skor_sentimen": "Skor (%)", "brand": "Brand"},
            hover_data=["total", "avg_rating"],
        )
        fig_b1.update_layout(height=420, yaxis={"categoryorder": "total ascending"}, margin=dict(t=50, b=20))
        st.plotly_chart(fig_b1, use_container_width=True, key="t2_skor")
    with c2:
        fig_b2 = px.bar(
            bs,
            x="brand",
            y=["positif", "netral", "negatif"],
            barmode="stack",
            title="Komposisi Sentimen per Brand",
            color_discrete_map={"positif": "#2ecc71", "netral": "#f39c12", "negatif": "#e74c3c"},
            labels={"value": "Jumlah", "brand": "Brand", "variable": "Sentimen"},
        )
        fig_b2.update_layout(height=420, xaxis_tickangle=-30, margin=dict(t=50, b=20))
        st.plotly_chart(fig_b2, use_container_width=True, key="t2_stack")

    section_header("📍 Rating vs Skor Sentimen")
    fig_b3 = px.scatter(
        bs,
        x="avg_rating",
        y="skor_sentimen",
        size="total",
        color="skor_sentimen",
        color_continuous_scale="RdYlGn",
        text="brand",
        hover_data=["total"],
        title="Rata-rata Rating vs Skor Sentimen per Brand",
        labels={"avg_rating": "Rata-rata Rating", "skor_sentimen": "Skor Sentimen (%)"},
    )
    fig_b3.update_traces(textposition="top center")
    fig_b3.update_layout(height=450)
    st.plotly_chart(fig_b3, use_container_width=True, key="t2_scatter")

    section_header("📋 Tabel Perbandingan Brand")
    disp = ["brand", "total", "positif", "netral", "negatif", "positif_pct", "avg_rating", "skor_sentimen"]
    st.dataframe(
        bs[disp].rename(
            columns={
                "brand": "Brand",
                "total": "Total",
                "positif": "Positif",
                "netral": "Netral",
                "negatif": "Negatif",
                "positif_pct": "Positif (%)",
                "avg_rating": "Avg Rating",
                "skor_sentimen": "Skor Sentimen (%)",
            }
        ).style.background_gradient(subset=["Skor Sentimen (%)"], cmap="RdYlGn"),
        use_container_width=True,
    )


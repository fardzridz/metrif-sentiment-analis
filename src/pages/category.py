"""Tab analisis kategori."""

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.ui import section_header


def render_category_page(df) -> None:
    if "category" not in df.columns:
        st.warning("Kolom 'category' tidak ditemukan.")
        return

    section_header("🗂️ Analisis per Kategori")
    cs = df.groupby("category").agg(
        total=("sentiment", "count"),
        positif=("sentiment", lambda x: (x == "positif").sum()),
        netral=("sentiment", lambda x: (x == "netral").sum()),
        negatif=("sentiment", lambda x: (x == "negatif").sum()),
        avg_rating=("rating", "mean"),
    ).reset_index()
    cs["skor_sentimen"] = ((cs["positif"] - cs["negatif"]) / cs["total"] * 100).round(1)
    cs["positif_pct"] = (cs["positif"] / cs["total"] * 100).round(1)
    cs = cs.sort_values("total", ascending=False)

    c1, c2 = st.columns(2)
    with c1:
        fig_c1 = px.treemap(
            cs,
            path=["category"],
            values="total",
            color="skor_sentimen",
            color_continuous_scale="RdYlGn",
            title="Treemap Kategori",
            hover_data=["avg_rating", "positif_pct"],
        )
        fig_c1.update_layout(height=420, margin=dict(t=50, b=10))
        st.plotly_chart(fig_c1, use_container_width=True, key="t4_treemap")
    with c2:
        fig_c2 = px.bar(
            cs.head(15),
            x="category",
            y=["positif", "netral", "negatif"],
            barmode="stack",
            title="Komposisi Sentimen per Kategori",
            color_discrete_map={"positif": "#2ecc71", "netral": "#f39c12", "negatif": "#e74c3c"},
            labels={"value": "Jumlah", "category": "Kategori", "variable": "Sentimen"},
        )
        fig_c2.update_layout(height=420, xaxis_tickangle=-30, margin=dict(t=50, b=80))
        st.plotly_chart(fig_c2, use_container_width=True, key="t4_stack")

    section_header("🕸️ Radar Chart Perbandingan Kategori")
    fig_c3 = go.Figure()
    for _, row in cs.head(6).iterrows():
        fig_c3.add_trace(
            go.Scatterpolar(
                r=[row["positif_pct"], row["avg_rating"] * 20, row["skor_sentimen"] + 50],
                theta=["% Positif", "Rating (x20)", "Skor Sentimen (+50)"],
                fill="toself",
                name=row["category"],
            )
        )
    fig_c3.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        title="Radar Perbandingan Kategori",
        height=450,
    )
    st.plotly_chart(fig_c3, use_container_width=True, key="t4_radar")

    section_header("📋 Tabel Kategori")
    st.dataframe(
        cs.rename(
            columns={
                "category": "Kategori",
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


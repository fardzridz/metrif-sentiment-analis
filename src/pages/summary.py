"""Tab ringkasan dashboard."""

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.config import SENTIMENT_COLOR
from src.ui import metric_card, section_header
from src.visualization import make_wordcloud


def render_summary_page(df, show_raw: bool) -> None:
    section_header("📈 Metrik Utama")
    total = len(df)
    pos_pct = (df["sentiment"] == "positif").mean() * 100
    neg_pct = (df["sentiment"] == "negatif").mean() * 100
    net_pct = (df["sentiment"] == "netral").mean() * 100
    avg_rating = df["rating"].mean()
    n_brands = df["brand"].nunique() if "brand" in df.columns else 0

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    for col, val, lbl, color in [
        (c1, f"{total:,}", "Total Ulasan", "#1e3a5f"),
        (c2, f"{pos_pct:.1f}%", "😊 Positif", "#155724"),
        (c3, f"{net_pct:.1f}%", "😐 Netral", "#856404"),
        (c4, f"{neg_pct:.1f}%", "😞 Negatif", "#721c24"),
        (c5, f"{avg_rating:.2f}⭐", "Rata-rata Rating", "#1e3a5f"),
        (c6, f"{n_brands}", "Jumlah Brand", "#1e3a5f"),
    ]:
        metric_card(col, val, lbl, color)

    section_header("📊 Distribusi Sentimen & Rating")
    ca, cb = st.columns(2)
    with ca:
        sc = df["sentiment"].value_counts().reindex(
            ["positif", "netral", "negatif"], fill_value=0
        )
        fig_pie = go.Figure(
            go.Pie(
                labels=sc.index,
                values=sc.values,
                hole=0.5,
                marker_colors=[SENTIMENT_COLOR[s] for s in sc.index],
                textinfo="label+percent",
                hovertemplate="%{label}: %{value:,} ulasan<extra></extra>",
            )
        )
        fig_pie.update_layout(
            title="Distribusi Sentimen", height=350, showlegend=True, margin=dict(t=50, b=20)
        )
        st.plotly_chart(fig_pie, use_container_width=True, key="t1_pie")
    with cb:
        rc = df["rating"].value_counts().sort_index()
        fig_rat = px.bar(
            x=rc.index,
            y=rc.values,
            labels={"x": "Rating", "y": "Jumlah"},
            color=rc.index,
            color_continuous_scale="RdYlGn",
            title="Distribusi Rating",
        )
        fig_rat.update_layout(height=350, showlegend=False, margin=dict(t=50, b=20))
        st.plotly_chart(fig_rat, use_container_width=True, key="t1_rating")

    section_header("☁️ Word Cloud per Sentimen")
    wc1, wc2, wc3 = st.columns(3)
    for col, sent, cmap, title in [
        (wc1, "positif", "Greens", "😊 Positif"),
        (wc2, "netral", "Oranges", "😐 Netral"),
        (wc3, "negatif", "Reds", "😞 Negatif"),
    ]:
        with col:
            st.markdown(f"**{title}**")
            texts = df[df["sentiment"] == sent]["review_text"].dropna().astype(str).tolist()
            fig_wc = make_wordcloud(texts, cmap)
            if fig_wc:
                st.pyplot(fig_wc, use_container_width=True)
                plt.close(fig_wc)
            else:
                st.info("Tidak ada data")

    if show_raw:
        section_header("📋 Data Mentah")
        st.dataframe(df.head(200), use_container_width=True)


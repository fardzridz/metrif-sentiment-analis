"""Tab tren waktu."""

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.config import SENTIMENT_COLOR
from src.ui import section_header


def render_trend_page(df) -> None:
    if "review_date" not in df.columns or df["review_date"].isna().all():
        st.warning("Kolom 'review_date' tidak ditemukan atau tidak bisa diparse.")
        return

    section_header("📅 Tren Sentimen dari Waktu ke Waktu")
    df_time = df.dropna(subset=["review_date"]).copy()
    granularity = st.radio("Granularitas:", ["Bulanan", "Mingguan", "Harian"], horizontal=True, key="time_gran")
    period_code = {"Bulanan": "M", "Mingguan": "W", "Harian": "D"}[granularity]
    df_time["period"] = df_time["review_date"].dt.to_period(period_code).astype(str)

    ts = df_time.groupby(["period", "sentiment"]).size().unstack(fill_value=0).reset_index()
    for col in ["positif", "netral", "negatif"]:
        if col not in ts.columns:
            ts[col] = 0

    fig_t1 = go.Figure()
    for sent, color in SENTIMENT_COLOR.items():
        fig_t1.add_trace(
            go.Scatter(
                x=ts["period"],
                y=ts[sent],
                name=sent.capitalize(),
                line=dict(color=color, width=2),
                mode="lines+markers",
            )
        )
    fig_t1.update_layout(
        title="Tren Jumlah Ulasan per Sentimen",
        xaxis_title="Periode",
        yaxis_title="Jumlah Ulasan",
        height=420,
        hovermode="x unified",
    )
    st.plotly_chart(fig_t1, use_container_width=True, key="t5_tren")

    tr = df_time.groupby("period")["rating"].mean().reset_index()
    fig_t2 = px.line(
        tr,
        x="period",
        y="rating",
        title="Tren Rata-rata Rating",
        labels={"period": "Periode", "rating": "Rata-rata Rating"},
        markers=True,
    )
    fig_t2.add_hline(y=3, line_dash="dash", line_color="orange", annotation_text="Batas Netral (3)")
    fig_t2.update_layout(height=350)
    st.plotly_chart(fig_t2, use_container_width=True, key="t5_rating")

    if df_time["review_date"].dt.hour.nunique() > 1:
        section_header("🗓️ Heatmap Aktivitas Ulasan")
        df_time["day_of_week"] = df_time["review_date"].dt.day_name()
        df_time["hour"] = df_time["review_date"].dt.hour
        hm = df_time.groupby(["day_of_week", "hour"]).size().unstack(fill_value=0)
        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        hm = hm.reindex([day for day in day_order if day in hm.index])
        fig_t3 = px.imshow(
            hm,
            color_continuous_scale="Blues",
            title="Heatmap Aktivitas Ulasan (Hari vs Jam)",
            labels={"x": "Jam", "y": "Hari", "color": "Jumlah"},
        )
        fig_t3.update_layout(height=350)
        st.plotly_chart(fig_t3, use_container_width=True, key="t5_heatmap")


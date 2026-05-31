"""Komponen tampilan umum untuk dashboard Streamlit."""

import streamlit as st


def configure_page() -> None:
    st.set_page_config(
        page_title="SentiPro — Analisis Sentimen Produk",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def inject_custom_css() -> None:
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.metric-card {
    background: linear-gradient(135deg, #1e3a5f 0%, #2d6a9f 100%);
    border-radius: 16px; padding: 20px 24px;
    color: white; text-align: center;
    box-shadow: 0 4px 15px rgba(0,0,0,0.2); margin-bottom: 10px;
}
.metric-card .value { font-size: 2.2rem; font-weight: 700; }
.metric-card .label { font-size: 0.85rem; opacity: 0.85; margin-top: 4px; }
.section-header {
    font-size: 1.3rem; font-weight: 700;
    border-left: 4px solid #2d6a9f;
    padding-left: 12px; margin: 24px 0 16px 0; color: #1e3a5f;
}
[data-testid="stSidebar"] { background: #0f2744; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
    color: white !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
    background: #f8fafc;
    border: 1px dashed #93a4b8;
    border-radius: 10px;
}
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] * {
    color: #102033 !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button {
    background: #1e3a5f;
    color: white !important;
    border: 1px solid #2d6a9f;
}
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] small {
    color: #475569 !important;
}
.stTabs [data-baseweb="tab"] { font-weight: 600; }
#MainMenu, footer { visibility: hidden; }
</style>
""",
        unsafe_allow_html=True,
    )


def section_header(title: str) -> None:
    st.markdown(
        f'<div class="section-header">{title}</div>',
        unsafe_allow_html=True,
    )


def metric_card(column, value: str, label: str, color: str) -> None:
    column.markdown(
        f"""
        <div class="metric-card" style="background:linear-gradient(135deg,{color} 0%,{color}cc 100%)">
            <div class="value">{value}</div><div class="label">{label}</div>
        </div>""",
        unsafe_allow_html=True,
    )

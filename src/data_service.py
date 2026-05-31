"""Load dan preprocessing dataset untuk dashboard."""

import pandas as pd
import streamlit as st


@st.cache_data
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    from src.preprocessing import preprocess as _preprocess

    df = df.copy()
    df.columns = df.columns.str.lower().str.strip().str.replace(" ", "_")
    df = df.dropna(subset=["review_text", "rating"])
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df = df.dropna(subset=["rating"])
    df["rating"] = df["rating"].astype(int)

    def label(rating: int) -> str:
        if rating <= 2:
            return "negatif"
        if rating == 3:
            return "netral"
        return "positif"

    df["sentiment"] = df["rating"].apply(label)
    if "review_date" in df.columns:
        df["review_date"] = pd.to_datetime(df["review_date"], errors="coerce")
    df["review_length"] = df["review_text"].astype(str).apply(lambda x: len(x.split()))
    df["clean_text"] = df["review_text"].astype(str).apply(
        lambda x: _preprocess(x, use_stemming=False)
    )
    return df


def read_uploaded_dataset(uploaded_file) -> pd.DataFrame:
    if uploaded_file.name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    return pd.read_excel(uploaded_file)


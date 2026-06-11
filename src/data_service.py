"""Pembacaan, validasi, dan preprocessing dataset dashboard."""

from __future__ import annotations

from dataclasses import dataclass, field
from io import BytesIO
import math
import re

import pandas as pd
import streamlit as st


REQUIRED_COLUMNS = (
    "product_name",
    "brand",
    "category",
    "rating",
    "review_date",
    "review_text",
)
EXPECTED_SENTIMENTS = ("negatif", "netral", "positif")
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MIN_TRAINING_ROWS = 60
MIN_ROWS_PER_CLASS = 10


@dataclass
class ValidationReport:
    """Hasil validasi yang dapat ditampilkan langsung ke pengguna."""

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, object] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        return not self.errors


class DatasetValidationError(ValueError):
    """Kesalahan dataset yang harus diperbaiki oleh pengguna."""


def normalize_column_name(name: object) -> str:
    normalized = str(name).strip().lower()
    normalized = re.sub(r"[\s\-]+", "_", normalized)
    return re.sub(r"_+", "_", normalized)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    normalized.columns = [normalize_column_name(column) for column in normalized.columns]
    return normalized


def read_uploaded_dataset(uploaded_file) -> pd.DataFrame:
    """Baca file upload setelah format dan ukuran file diverifikasi."""

    name = getattr(uploaded_file, "name", "")
    suffix = name.lower().rsplit(".", 1)[-1] if "." in name else ""
    if suffix not in {"csv", "xlsx"}:
        raise DatasetValidationError("Format file harus CSV atau XLSX.")

    content = uploaded_file.getvalue()
    if not content:
        raise DatasetValidationError("File yang di-upload kosong.")
    if len(content) > MAX_UPLOAD_BYTES:
        raise DatasetValidationError("Ukuran file melebihi batas 20 MB.")

    buffer = BytesIO(content)
    try:
        if suffix == "csv":
            return pd.read_csv(buffer)
        return pd.read_excel(buffer)
    except Exception as exc:
        raise DatasetValidationError(
            "File tidak dapat dibaca. Pastikan file tidak rusak dan format tabelnya benar."
        ) from exc


def validate_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, ValidationReport]:
    """Validasi ketat dataset utama sebelum preprocessing dan analisis."""

    report = ValidationReport()
    if not isinstance(df, pd.DataFrame) or df.empty:
        report.errors.append("Dataset tidak memiliki baris data.")
        return pd.DataFrame(), report

    normalized_names = [normalize_column_name(column) for column in df.columns]
    duplicate_columns = sorted(
        {name for name in normalized_names if normalized_names.count(name) > 1}
    )
    if duplicate_columns:
        report.errors.append(
            "Nama kolom menjadi duplikat setelah normalisasi: "
            + ", ".join(duplicate_columns)
            + "."
        )
        return normalize_columns(df), report

    normalized = normalize_columns(df)
    missing = [column for column in REQUIRED_COLUMNS if column not in normalized.columns]
    if missing:
        report.errors.append(
            "Kolom wajib belum lengkap. Kolom yang hilang: " + ", ".join(missing) + "."
        )
        return normalized, report

    report.stats["rows_uploaded"] = len(normalized)
    report.stats["columns"] = len(normalized.columns)

    text_columns = ["product_name", "brand", "category", "review_text"]
    for column in text_columns:
        values = normalized[column].astype("string")
        invalid = values.isna() | values.str.strip().eq("")
        invalid_count = int(invalid.sum())
        if invalid_count:
            report.errors.append(
                f"Kolom '{column}' memiliki {invalid_count} nilai kosong."
            )

    numeric_rating = pd.to_numeric(normalized["rating"], errors="coerce")
    invalid_rating = numeric_rating.isna()
    non_integer_rating = numeric_rating.notna() & (numeric_rating % 1 != 0)
    outside_rating = numeric_rating.notna() & ~numeric_rating.between(1, 5)
    if invalid_rating.any():
        report.errors.append(
            f"Kolom 'rating' memiliki {int(invalid_rating.sum())} nilai bukan angka."
        )
    if non_integer_rating.any():
        report.errors.append(
            f"Kolom 'rating' memiliki {int(non_integer_rating.sum())} nilai bukan bilangan bulat."
        )
    if outside_rating.any():
        report.errors.append(
            f"Kolom 'rating' memiliki {int(outside_rating.sum())} nilai di luar rentang 1–5."
        )

    parsed_dates = pd.to_datetime(normalized["review_date"], errors="coerce")
    invalid_dates = parsed_dates.isna()
    if invalid_dates.any():
        report.errors.append(
            f"Kolom 'review_date' memiliki {int(invalid_dates.sum())} tanggal kosong/tidak valid."
        )

    duplicate_rows = int(normalized.duplicated().sum())
    if duplicate_rows:
        report.warnings.append(
            f"Ditemukan {duplicate_rows} baris duplikat. Baris tersebut akan dihapus."
        )
    report.stats["duplicate_rows"] = duplicate_rows

    if not invalid_rating.any() and not non_integer_rating.any() and not outside_rating.any():
        sentiments = numeric_rating.astype(int).map(_label_from_rating)
        counts = sentiments.value_counts().reindex(EXPECTED_SENTIMENTS, fill_value=0)
        report.stats["class_counts"] = counts.to_dict()
        if counts.min() == 0:
            missing_classes = counts[counts == 0].index.tolist()
            report.warnings.append(
                "Dataset tidak memiliki kelas sentimen: "
                + ", ".join(missing_classes)
                + ". Analisis masih dapat dilakukan, tetapi training akan diblokir."
            )
        elif counts.min() / counts.max() < 0.25:
            report.warnings.append(
                "Distribusi kelas sangat tidak seimbang. Evaluasi model wajib memperhatikan Macro F1."
            )

    extra_columns = [column for column in normalized.columns if column not in REQUIRED_COLUMNS]
    if extra_columns:
        report.warnings.append(
            "Kolom tambahan akan dipertahankan: " + ", ".join(extra_columns) + "."
        )

    return normalized, report


@st.cache_data
def preprocess_data(df: pd.DataFrame, use_stemming: bool = False) -> pd.DataFrame:
    """Bersihkan dataset valid dan buat label sentimen dari rating."""

    from src.preprocessing import preprocess

    prepared = normalize_columns(df).copy()
    prepared = prepared.drop_duplicates().reset_index(drop=True)
    for column in ["product_name", "brand", "category", "review_text"]:
        prepared[column] = prepared[column].astype(str).str.strip()
    prepared["rating"] = pd.to_numeric(prepared["rating"], errors="raise").astype(int)
    prepared["review_date"] = pd.to_datetime(prepared["review_date"], errors="raise")
    prepared["sentiment"] = prepared["rating"].map(_label_from_rating)
    prepared["review_length"] = prepared["review_text"].str.split().str.len()
    prepared["clean_text"] = prepared["review_text"].apply(
        lambda text: preprocess(text, use_stemming=use_stemming)
    )

    empty_clean = prepared["clean_text"].str.strip().eq("")
    if empty_clean.any():
        raise DatasetValidationError(
            f"{int(empty_clean.sum())} ulasan menjadi kosong setelah preprocessing. "
            "Perbaiki ulasan yang hanya berisi angka, simbol, atau stopword."
        )
    return prepared


def validate_training_data(df: pd.DataFrame, test_size: float) -> ValidationReport:
    """Pastikan dataset cukup dan aman untuk stratified train/test split."""

    report = ValidationReport()
    if "sentiment" not in df.columns or "review_text" not in df.columns:
        report.errors.append("Dataset belum melalui validasi dan preprocessing.")
        return report

    total = len(df)
    counts = df["sentiment"].value_counts().reindex(EXPECTED_SENTIMENTS, fill_value=0)
    report.stats = {"rows": total, "class_counts": counts.to_dict()}

    if total < MIN_TRAINING_ROWS:
        report.errors.append(
            f"Training membutuhkan minimal {MIN_TRAINING_ROWS} baris; dataset hanya memiliki {total}."
        )

    missing_classes = counts[counts == 0].index.tolist()
    if missing_classes:
        report.errors.append(
            "Training tiga kelas membutuhkan negatif, netral, dan positif. Kelas yang tidak ada: "
            + ", ".join(missing_classes)
            + "."
        )

    split_minimum = max(
        MIN_ROWS_PER_CLASS,
        math.ceil(1 / test_size),
        math.ceil(1 / (1 - test_size)),
    )
    small_classes = counts[counts < split_minimum]
    if not small_classes.empty:
        detail = ", ".join(f"{name}={count}" for name, count in small_classes.items())
        report.errors.append(
            f"Setiap kelas membutuhkan minimal {split_minimum} baris untuk pembagian data yang dipilih; "
            f"saat ini {detail}."
        )

    if counts.max() and counts.min() / counts.max() < 0.25:
        report.warnings.append(
            "Distribusi kelas sangat tidak seimbang. Model terbaik akan dipilih menggunakan Macro F1."
        )
    return report


def validate_prediction_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, ValidationReport]:
    """Validasi file batch prediksi yang hanya membutuhkan review_text."""

    report = ValidationReport()
    if not isinstance(df, pd.DataFrame) or df.empty:
        report.errors.append("File batch prediksi tidak memiliki baris data.")
        return pd.DataFrame(), report

    normalized = normalize_columns(df)
    if "review_text" not in normalized.columns:
        report.errors.append("File batch prediksi wajib memiliki kolom 'review_text'.")
        return normalized, report

    reviews = normalized["review_text"].astype("string")
    invalid = reviews.isna() | reviews.str.strip().eq("")
    if invalid.any():
        report.errors.append(
            f"Kolom 'review_text' memiliki {int(invalid.sum())} nilai kosong."
        )
    report.stats["rows"] = len(normalized)
    return normalized, report


def _label_from_rating(rating: int) -> str:
    if rating <= 2:
        return "negatif"
    if rating == 3:
        return "netral"
    return "positif"

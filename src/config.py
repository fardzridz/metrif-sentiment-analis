"""Konfigurasi global dashboard SentiPro."""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

SENTIMENT_COLOR = {
    "positif": "#2ecc71",
    "netral": "#f39c12",
    "negatif": "#e74c3c",
}

SENTIMENT_EMOJI = {
    "positif": "😊",
    "netral": "😐",
    "negatif": "😞",
}

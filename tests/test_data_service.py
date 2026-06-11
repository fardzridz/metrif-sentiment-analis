import unittest

import pandas as pd

from src.data_service import (
    DatasetValidationError,
    preprocess_data,
    read_uploaded_dataset,
    validate_dataset,
    validate_prediction_dataset,
    validate_training_data,
)
from src.model_service import predict_sentiment


def make_dataset(rows_per_class=20):
    rows = []
    rating_text = {
        1: "produk rusak dan sangat mengecewakan",
        3: "produk cukup biasa dan standar",
        5: "produk bagus dan sangat memuaskan",
    }
    for rating, review in rating_text.items():
        for index in range(rows_per_class):
            rows.append(
                {
                    "Product name": f"Produk {index}",
                    "Brand": "Brand A",
                    "Category": "Kategori A",
                    "Rating": rating,
                    "Review date": "2026-01-01",
                    "Review text": f"{review} {index}",
                }
            )
    return pd.DataFrame(rows)


class FakeUpload:
    def __init__(self, name, content):
        self.name = name
        self._content = content

    def getvalue(self):
        return self._content


class DatasetValidationTests(unittest.TestCase):
    def test_upload_rejects_empty_and_unsupported_files(self):
        with self.assertRaises(DatasetValidationError):
            read_uploaded_dataset(FakeUpload("dataset.txt", b"data"))
        with self.assertRaises(DatasetValidationError):
            read_uploaded_dataset(FakeUpload("dataset.csv", b""))

    def test_valid_dataset_passes_and_normalizes_columns(self):
        normalized, report = validate_dataset(make_dataset())

        self.assertTrue(report.is_valid)
        self.assertIn("review_text", normalized.columns)
        self.assertEqual(report.stats["rows_uploaded"], 60)

    def test_missing_required_column_is_rejected(self):
        dataset = make_dataset().drop(columns=["Rating"])

        _, report = validate_dataset(dataset)

        self.assertFalse(report.is_valid)
        self.assertTrue(any("rating" in message for message in report.errors))

    def test_invalid_values_are_rejected(self):
        dataset = make_dataset()
        dataset.loc[0, "Rating"] = 6
        dataset.loc[1, "Review date"] = "bukan tanggal"
        dataset.loc[2, "Review text"] = " "

        _, report = validate_dataset(dataset)

        self.assertFalse(report.is_valid)
        self.assertTrue(any("rentang 1–5" in message for message in report.errors))
        self.assertTrue(any("review_date" in message for message in report.errors))
        self.assertTrue(any("review_text" in message for message in report.errors))

    def test_duplicate_normalized_columns_are_rejected(self):
        dataset = make_dataset()
        dataset["review-text"] = dataset["Review text"]

        _, report = validate_dataset(dataset)

        self.assertFalse(report.is_valid)
        self.assertTrue(any("duplikat" in message for message in report.errors))

    def test_preprocessing_creates_labels_and_removes_duplicates(self):
        dataset = make_dataset()
        dataset = pd.concat([dataset, dataset.iloc[[0]]], ignore_index=True)
        normalized, report = validate_dataset(dataset)

        self.assertTrue(report.is_valid)
        prepared = preprocess_data(normalized, use_stemming=False)

        self.assertEqual(len(prepared), 60)
        self.assertEqual(set(prepared["sentiment"]), {"negatif", "netral", "positif"})
        self.assertFalse(prepared["clean_text"].str.strip().eq("").any())

    def test_training_requires_all_three_classes(self):
        dataset = make_dataset()
        normalized, _ = validate_dataset(dataset)
        prepared = preprocess_data(normalized, use_stemming=False)
        prepared = prepared[prepared["sentiment"] != "netral"]

        report = validate_training_data(prepared, test_size=0.2)

        self.assertFalse(report.is_valid)
        self.assertTrue(any("netral" in message for message in report.errors))

    def test_prediction_batch_requires_nonempty_review_text(self):
        dataset = pd.DataFrame({"Review Text": ["bagus", " "]})

        _, report = validate_prediction_dataset(dataset)

        self.assertFalse(report.is_valid)
        self.assertTrue(any("nilai kosong" in message for message in report.errors))

    def test_prediction_rejects_text_empty_after_preprocessing(self):
        with self.assertRaises(ValueError):
            predict_sentiment(["yang dan untuk"], model=object(), use_stemming=False)


if __name__ == "__main__":
    unittest.main()

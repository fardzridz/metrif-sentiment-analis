import unittest

from streamlit.testing.v1 import AppTest


PREPROCESSING_PAGE_APP = """
import pandas as pd

from src.pages.preprocessing_view import render_preprocessing_page

df = pd.DataFrame(
    {
        "product_name": ["Produk A", "Produk B", "Produk C"],
        "rating": [5, 3, 1],
        "sentiment": ["positif", "netral", "negatif"],
        "review_text": [
            "produk bagus banget",
            "produk cukup biasa",
            "produk rusak parah",
        ],
        "clean_text": [
            "produk bagus",
            "produk biasa",
            "produk rusak parah",
        ],
    }
)

render_preprocessing_page(df)
"""


class PreprocessingViewTests(unittest.TestCase):
    def test_manual_demo_is_hidden_until_user_enables_it(self):
        app = AppTest.from_string(PREPROCESSING_PAGE_APP).run(timeout=30)

        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.text_area), 0)
        self.assertEqual(len(app.toggle), 1)

        app.toggle[0].set_value(True).run(timeout=30)

        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.text_area), 1)
        self.assertEqual(len(app.warning), 1)


if __name__ == "__main__":
    unittest.main()

# Ringkasan Teknis

## Struktur Modul

| Modul | Fungsi Utama |
|---|---|
| `app.py` | Entry point dashboard Streamlit, sidebar, upload dataset, tab navigasi. |
| `src/config.py` | Path dasar project, folder model, warna dan label sentimen. |
| `src/data_service.py` | Membaca dataset upload dan preprocessing data untuk dashboard. |
| `src/model_service.py` | Load model, load informasi model, dan prediksi sentimen. |
| `src/preprocessing.py` | Pipeline cleaning, normalisasi slang, tokenisasi, stopword, dan stemming. |
| `src/train.py` | Training multi-model, evaluasi, dan penyimpanan model terbaik. |
| `src/predict.py` | Prediksi lewat mode command line untuk teks tunggal atau batch file. |
| `src/eda.py` | Exploratory Data Analysis dan penyimpanan grafik laporan. |
| `src/ui.py` | Konfigurasi halaman, CSS, section header, dan metric card. |
| `src/visualization.py` | Helper visualisasi seperti word cloud. |
| `src/pages/` | Komponen halaman per tab dashboard. |

## Format Dataset

| Kolom | Keterangan |
|---|---|
| `Product name` | Nama produk. |
| `Brand` | Merek produk. |
| `Category` | Kategori produk. |
| `Rating` | Nilai rating 1 sampai 5. |
| `Review date` | Tanggal ulasan. |
| `Review text` | Isi teks ulasan. |

Nama kolom dinormalisasi menjadi lowercase dan spasi diganti underscore,
contohnya `Review text` menjadi `review_text`.

## Label Sentimen

| Rating | Label |
|---|---|
| 1-2 | negatif |
| 3 | netral |
| 4-5 | positif |

## Model yang Tersedia

Model yang didefinisikan pada training:

- Logistic Regression
- Complement Naive Bayes
- Linear SVM
- Extra Trees
- XGBoost jika dependency tersedia
- LightGBM jika dependency tersedia
- Voting Ensemble

Metadata model tersimpan saat ini:

| Model | Accuracy | F1-Score |
|---|---:|---:|
| Linear SVM | 80.67% | 73.08% |
| Extra Trees | 79.33% | 70.19% |
| Logistic Regression | 78.67% | 71.16% |
| Voting Ensemble | 78.67% | 72.65% |
| LightGBM | 77.33% | 69.98% |
| Complement Naive Bayes | 72.67% | 71.23% |

Model terbaik saat ini adalah `Linear SVM`, berdasarkan accuracy tertinggi.

## Fitur Dashboard

| Tab | Isi |
|---|---|
| Ringkasan | Metrik utama, distribusi sentimen, distribusi rating, word cloud, data mentah. |
| Brand | Skor sentimen per brand, distribusi sentimen, rating vs skor, tabel ranking. |
| Produk | Top/bottom produk, distribusi sentimen produk, detail ulasan per produk. |
| Kategori | Treemap, stacked bar, radar chart, dan tabel kategori. |
| Tren Waktu | Tren sentimen bulanan/mingguan/harian, rata-rata rating, heatmap aktivitas. |
| Prediksi | Prediksi teks tunggal dan batch upload. |
| Preprocessing | Demo step-by-step preprocessing, kamus slang, stopword, statistik token. |
| Training Model | Training model langsung dari web, evaluasi, confusion matrix, classification report. |

## Artefak Output

| Folder/File | Isi |
|---|---|
| `models/best_model.pkl` | Model terbaik untuk prediksi. |
| `models/all_models.pkl` | Semua model yang berhasil dilatih. |
| `models/model_info.pkl` | Nama model terbaik, hasil evaluasi, dan label kelas. |
| `reports/` | Output grafik EDA dan evaluasi training jika script dijalankan. |


# Ringkasan Teknis

## Prinsip Workflow

```text
Upload → Validasi → Preprocessing → Analisis → Training → Prediksi
```

Tahap berikutnya hanya dapat digunakan setelah tahap sebelumnya valid.
Dashboard analisis memakai label dari rating, sedangkan halaman prediksi memakai
model hasil training pada sesi pengguna.

## Struktur Modul

| Modul | Fungsi Utama |
|---|---|
| `app.py` | Mengatur upload, fingerprint dataset, status workflow, dan navigasi. |
| `src/data_service.py` | Membaca file, validasi dataset, preprocessing, dan validasi training/batch. |
| `src/model_service.py` | Menyimpan model per sesi dan menjalankan prediksi konsisten. |
| `src/preprocessing.py` | Cleaning, normalisasi slang, tokenisasi, stopword, dan stemming. |
| `src/pages/training.py` | Stratified split, training, evaluasi, dan pemilihan model berdasarkan Macro F1. |
| `src/pages/prediction.py` | Prediksi teks tunggal dan batch setelah training berhasil. |
| `src/pages/` lainnya | Visualisasi ringkasan, brand, produk, kategori, dan tren. |

## Validasi Dataset Utama

Seluruh kolom berikut wajib tersedia dan tidak boleh kosong:

| Kolom | Aturan |
|---|---|
| `Product name` | Teks tidak kosong |
| `Brand` | Teks tidak kosong |
| `Category` | Teks tidak kosong |
| `Rating` | Bilangan bulat 1 sampai 5 |
| `Review date` | Tanggal valid |
| `Review text` | Teks tidak kosong |

File dibatasi maksimal 20 MB. Dataset juga ditolak jika kolom menjadi duplikat
setelah normalisasi atau ulasan menjadi kosong setelah preprocessing.

## Validasi Training

- Minimal 60 baris.
- Wajib memiliki kelas negatif, netral, dan positif.
- Setiap kelas wajib memiliki data cukup untuk stratified split.
- Ketidakseimbangan kelas ditampilkan sebagai peringatan.
- Model terbaik dipilih berdasarkan Macro F1.

## Model Dashboard

- Logistic Regression
- Complement Naive Bayes
- Linear SVM
- Extra Trees
- Voting Ensemble

Model dan metadata disimpan pada `st.session_state`. Tidak ada model `.pkl`
bawaan pada workflow dashboard.

## Data Runtime

`data/*`, `models/*`, `reports/`, `__pycache__/`, dan `*.pyc` diabaikan Git.
Dataset/model percobaan tetap dapat disimpan secara lokal, tetapi tidak ikut
GitHub atau deployment.

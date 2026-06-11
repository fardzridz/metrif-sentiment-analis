# SentiPro - Dashboard Analisis Sentimen Produk

SentiPro adalah dashboard Streamlit untuk menganalisis dan melatih model
sentimen dari dataset yang di-upload oleh pengguna. Dataset dan model tidak
disediakan oleh server. Setiap pengguna wajib mengikuti alur aplikasi dari
upload sampai training sebelum fitur prediksi aktif.

## Alur Wajib

```text
Upload Dataset
      ↓
Validasi Struktur dan Nilai
      ↓
Pembersihan dan Preprocessing
      ↓
Pelabelan Sentimen dari Rating
      ↓
Analisis Dataset
      ↓
Validasi Kesiapan Training
      ↓
Pembagian Data Training dan Testing
      ↓
Training dan Evaluasi Beberapa Model
      ↓
Pilih Model Terbaik berdasarkan Macro F1
      ↓
Prediksi Ulasan Baru
```

Prediksi dikunci sampai training berhasil. Model terbaik disimpan dalam
`st.session_state`, sehingga model hanya berlaku untuk sesi pengguna yang
melatihnya dan tidak mengganti model pengguna lain.

## Format Dataset Utama

Dataset harus berupa CSV atau XLSX dengan ukuran maksimal 20 MB dan wajib
memiliki seluruh kolom berikut:

| Kolom | Aturan |
|---|---|
| `Product name` | Tidak boleh kosong |
| `Brand` | Tidak boleh kosong |
| `Category` | Tidak boleh kosong |
| `Rating` | Bilangan bulat 1 sampai 5 |
| `Review date` | Tanggal valid dan tidak kosong |
| `Review text` | Teks ulasan dan tidak boleh kosong |

Nama kolom tidak sensitif terhadap kapital, spasi, atau tanda hubung. Contoh
`Review Text`, `review text`, dan `review-text` dinormalisasi menjadi
`review_text`.

Label sentimen dibuat otomatis dari rating:

| Rating | Sentimen |
|---|---|
| 1-2 | negatif |
| 3 | netral |
| 4-5 | positif |

## Validasi Ketat

Dataset ditolak sebelum dashboard dibuka apabila:

- File kosong, rusak, format tidak didukung, atau lebih besar dari 20 MB.
- Kolom wajib hilang atau menjadi duplikat setelah normalisasi.
- Kolom teks wajib memiliki nilai kosong.
- Rating bukan angka bulat atau berada di luar rentang 1-5.
- Tanggal kosong atau tidak dapat diparse.
- Ulasan menjadi kosong setelah preprocessing.

Training diblokir apabila:

- Dataset memiliki kurang dari 60 baris.
- Salah satu kelas negatif, netral, atau positif tidak tersedia.
- Salah satu kelas memiliki kurang dari jumlah minimum untuk stratified split.

Distribusi kelas yang sangat tidak seimbang ditampilkan sebagai peringatan.
Pemilihan model terbaik menggunakan **Macro F1**, bukan hanya accuracy.

## Analisis dan Prediksi

- Halaman Ringkasan, Brand, Produk, Kategori, dan Tren menggunakan label yang
  dibuat dari rating dataset upload.
- Halaman Preprocessing langsung menampilkan perbandingan teks asli dan
  `clean_text` dari dataset upload. Input teks sendiri hanya demo opsional dan
  tidak digunakan untuk training.
- Halaman Prediksi menggunakan model machine learning hasil training pada sesi
  pengguna.
- File batch prediksi hanya wajib memiliki kolom `Review text`.
- Konfigurasi stemming saat prediksi selalu mengikuti konfigurasi training.

## Model Dashboard

Model yang dapat dilatih dari dashboard:

- Logistic Regression
- Complement Naive Bayes
- Linear SVM
- Extra Trees
- Voting Ensemble

Logistic Regression, Linear SVM, Extra Trees, dan komponen terkait menggunakan
`class_weight="balanced"` untuk membantu menghadapi distribusi kelas tidak
seimbang.

## Data dan Model Runtime

Dataset percobaan dan file model tidak disimpan di GitHub:

```gitignore
data/*
models/*
```

Folder `data/` dan `models/` hanya dipertahankan menggunakan `.gitkeep`.
Dashboard tidak membaca `data/dataset.csv` atau `models/best_model.pkl`.

## Menjalankan Lokal

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Buka `http://localhost:8501`, lalu upload dataset sesuai format wajib.

## Deployment Railway

Start command:

```bash
streamlit run app.py --server.port $PORT --server.address 0.0.0.0
```

Model hanya berada di memori sesi. Model akan hilang saat sesi berakhir,
aplikasi restart, atau Railway melakukan redeploy. Perilaku ini disengaja agar
model satu pengguna tidak digunakan oleh pengguna lain.

## Struktur Utama

```text
sentimen_analis/
├── app.py
├── requirements.txt
├── nixpacks.toml
├── data/
│   └── .gitkeep
├── models/
│   └── .gitkeep
├── src/
│   ├── data_service.py
│   ├── model_service.py
│   ├── preprocessing.py
│   └── pages/
└── tests/
```

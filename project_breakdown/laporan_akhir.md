# Laporan Akhir Project SentiPro

## 1. Judul

SentiPro: Dashboard Analisis Sentimen Ulasan Produk Berbahasa Indonesia
Berbasis Machine Learning dan Streamlit.

## 2. Latar Belakang

Ulasan produk dari pelanggan menyimpan informasi penting tentang kepuasan,
keluhan, kualitas produk, performa brand, dan tren pengalaman pengguna.
Namun, jumlah ulasan yang besar membuat proses analisis manual menjadi tidak
efisien. Karena itu, dibangun aplikasi SentiPro untuk membantu membaca,
membersihkan, mengklasifikasikan, dan memvisualisasikan sentimen ulasan produk.

Project ini berfokus pada analisis sentimen Bahasa Indonesia dengan dashboard
interaktif. Sistem mendukung upload dataset, preprocessing teks, pelabelan
berdasarkan rating, prediksi memakai model machine learning, serta visualisasi
ringkasan sampai detail brand, produk, kategori, dan tren waktu.

## 3. Tujuan

Tujuan project:

1. Membangun dashboard web untuk analisis sentimen ulasan produk.
2. Membuat pipeline preprocessing teks Bahasa Indonesia.
3. Melatih beberapa model machine learning untuk klasifikasi sentimen.
4. Menampilkan insight sentimen melalui visualisasi interaktif.
5. Menyediakan fitur prediksi sentimen untuk teks tunggal dan batch file.
6. Menyiapkan aplikasi agar dapat dihosting sebagai web app di Railway.

## 4. Teknologi yang Digunakan

| Kategori | Teknologi |
|---|---|
| Bahasa pemrograman | Python |
| Web dashboard | Streamlit |
| Data processing | pandas, numpy |
| Machine learning | scikit-learn |
| Preprocessing Bahasa Indonesia | NLTK, PySastrawi |
| Visualisasi | Plotly, Matplotlib, Seaborn, WordCloud |
| Model persistence | joblib |
| Hosting | Railway |

## 5. Dataset

Dataset yang digunakan berupa data ulasan produk dengan format CSV atau Excel.
Kolom utama yang dibutuhkan:

| Kolom | Keterangan |
|---|---|
| Product name | Nama produk. |
| Brand | Merek produk. |
| Category | Kategori produk. |
| Rating | Nilai rating dari 1 sampai 5. |
| Review date | Tanggal ulasan. |
| Review text | Isi ulasan pelanggan. |

Sistem menormalisasi nama kolom menjadi huruf kecil dan mengganti spasi dengan
underscore, misalnya `Review text` menjadi `review_text`.

## 6. Label Sentimen

Label sentimen dibuat dari kolom rating:

| Rating | Sentimen |
|---|---|
| 1-2 | negatif |
| 3 | netral |
| 4-5 | positif |

Metode ini membuat dataset dapat langsung digunakan untuk analisis dan training
tanpa pelabelan manual.

## 7. Preprocessing Teks

Pipeline preprocessing berada di `src/preprocessing.py`. Tahapannya:

1. Case folding untuk mengubah teks menjadi huruf kecil.
2. Cleaning untuk menghapus URL, mention, hashtag, angka, tanda baca, emoji,
   dan spasi berlebih.
3. Normalisasi slang atau singkatan, misalnya kata tidak baku diubah ke bentuk
   yang lebih standar.
4. Tokenisasi untuk memecah teks menjadi kata.
5. Stopword removal untuk menghapus kata umum yang tidak terlalu informatif.
6. Stemming memakai PySastrawi jika opsi stemming diaktifkan.

Hasil akhir preprocessing disimpan sebagai `clean_text` dan digunakan untuk
analisis token, visualisasi, dan training model.

## 8. Arsitektur Sistem

Project disusun modular:

| Bagian | Peran |
|---|---|
| `app.py` | Mengatur alur utama dashboard, upload dataset, sidebar, dan tab. |
| `src/data_service.py` | Membaca dataset upload dan menjalankan preprocessing untuk dashboard. |
| `src/model_service.py` | Memuat model dan melakukan prediksi sentimen. |
| `src/pages/` | Berisi tampilan setiap tab dashboard. |
| `src/train.py` | Melatih beberapa model dan menyimpan model terbaik. |
| `models/` | Menyimpan model dan informasi hasil evaluasi. |

Diagram lengkap tersedia di `project_breakdown/diagram.md`.

## 9. Fitur Dashboard

Fitur utama aplikasi:

| Tab | Fitur |
|---|---|
| Ringkasan | Total ulasan, rating rata-rata, distribusi sentimen, word cloud. |
| Brand | Perbandingan sentimen antar brand dan ranking performa brand. |
| Produk | Analisis produk terbaik/terendah dan contoh ulasan per produk. |
| Kategori | Distribusi sentimen per kategori, treemap, dan radar chart. |
| Tren Waktu | Analisis sentimen berdasarkan tanggal ulasan. |
| Prediksi | Prediksi sentimen teks tunggal atau batch upload. |
| Preprocessing | Visualisasi tahapan preprocessing dan daftar slang/stopword. |
| Training Model | Training model dari dashboard dan evaluasi performa. |

## 10. Training Model

Training dilakukan menggunakan representasi teks TF-IDF dan beberapa algoritma
klasifikasi. Model yang tersedia:

- Logistic Regression
- Complement Naive Bayes
- Linear SVM
- Extra Trees
- XGBoost jika library tersedia
- LightGBM jika library tersedia
- Voting Ensemble

Data dibagi menjadi data training dan testing menggunakan stratified split agar
proporsi label sentimen tetap terjaga.

## 11. Hasil Evaluasi Model

Berdasarkan metadata model yang tersimpan pada `models/model_info.pkl`, hasil
evaluasi saat ini adalah:

| Model | Accuracy | F1-Score |
|---|---:|---:|
| Linear SVM | 80.67% | 73.08% |
| Extra Trees | 79.33% | 70.19% |
| Logistic Regression | 78.67% | 71.16% |
| Voting Ensemble | 78.67% | 72.65% |
| LightGBM | 77.33% | 69.98% |
| Complement Naive Bayes | 72.67% | 71.23% |

Model terbaik saat ini adalah Linear SVM dengan accuracy 80.67%. Model ini
disimpan sebagai `models/best_model.pkl` dan digunakan untuk fitur prediksi.

Catatan evaluasi: pada komentar kode `src/train.py` terdapat target akurasi
lebih dari 90%, tetapi hasil model tersimpan saat ini belum mencapai target
tersebut. Peningkatan dapat dilakukan melalui penambahan data, balancing kelas,
perbaikan kamus normalisasi, tuning parameter, atau penggunaan model NLP yang
lebih kuat.

## 12. Deployment Railway

Aplikasi disiapkan untuk hosting di Railway sebagai Streamlit web app.

Start command yang digunakan:

```bash
streamlit run app.py --server.port $PORT --server.address 0.0.0.0
```

Railway akan menjalankan dependency dari `requirements.txt`. Folder `src/`,
`models/`, dan file `app.py` perlu tersedia di repository agar aplikasi dapat
berjalan dengan fitur dashboard dan prediksi.

URL hosting Railway:

```text
isi-dengan-url-railway
```

## 13. Cara Menjalankan Lokal

Install dependency:

```bash
pip install -r requirements.txt
```

Jalankan dashboard:

```bash
streamlit run app.py
```

Training model lewat command line:

```bash
cd src
python train.py
```

## 14. Kesimpulan

SentiPro berhasil dibangun sebagai dashboard analisis sentimen produk yang
memiliki pipeline preprocessing Bahasa Indonesia, visualisasi interaktif, fitur
prediksi real-time, serta training multi-model. Sistem dapat membantu pengguna
memahami pola sentimen ulasan berdasarkan brand, produk, kategori, dan waktu.

Aplikasi juga sudah disiapkan untuk deployment di Railway sehingga dapat diakses
sebagai web app. Model terbaik yang tersimpan saat ini adalah Linear SVM dengan
accuracy 80.67%, sehingga masih terdapat ruang peningkatan untuk mencapai target
akurasi yang lebih tinggi.

## 15. Saran Pengembangan

Beberapa pengembangan berikut dapat dilakukan:

1. Menambah jumlah dan variasi dataset ulasan.
2. Melakukan balancing data jika distribusi sentimen tidak seimbang.
3. Menambah kamus slang Bahasa Indonesia yang lebih lengkap.
4. Menguji model berbasis transformer Bahasa Indonesia.
5. Menambahkan database agar histori upload dan prediksi dapat disimpan.
6. Mengisi URL Railway production setelah deployment aktif.


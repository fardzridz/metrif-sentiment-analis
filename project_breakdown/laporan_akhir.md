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

Dataset dan model percobaan tidak disediakan oleh server. Setiap pengguna wajib
upload dataset, melewati validasi, dan melakukan training pada sesi masing-masing
sebelum fitur prediksi dapat digunakan.

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
| Model sesi | Streamlit session state |
| Hosting | Railway |

## 5. Dataset

Dataset yang digunakan berupa data ulasan produk dengan format CSV atau XLSX.
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

Dataset akan ditolak apabila file kosong/rusak, ukuran melebihi 20 MB, kolom
wajib tidak lengkap, nilai teks kosong, rating bukan bilangan bulat 1 sampai 5,
atau tanggal tidak valid. Baris duplikat ditampilkan sebagai peringatan dan
dihapus pada tahap pembersihan.

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

Pada halaman Preprocessing, sistem terlebih dahulu menampilkan hasil dari
dataset upload agar pengguna memahami bahwa preprocessing sudah berjalan
otomatis. Fitur input teks sendiri hanya tersedia sebagai demo opsional dan
teksnya tidak ditambahkan ke dataset maupun digunakan untuk training.

## 8. Arsitektur Sistem

Project disusun modular:

| Bagian | Peran |
|---|---|
| `app.py` | Mengatur alur utama dashboard, upload dataset, sidebar, dan tab. |
| `src/data_service.py` | Membaca dataset upload dan menjalankan preprocessing untuk dashboard. |
| `src/model_service.py` | Menyimpan model per sesi dan melakukan prediksi sentimen. |
| `src/pages/` | Berisi tampilan setiap tab dashboard. |
| `src/pages/training.py` | Melatih, mengevaluasi, dan mengaktifkan model terbaik pada sesi. |
| `st.session_state` | Menyimpan model dan informasi evaluasi untuk sesi pengguna aktif. |

Alur aplikasi diwajibkan secara berurutan:

```text
Upload → Validasi → Preprocessing → Analisis → Training → Prediksi
```

Prediksi dikunci sampai training berhasil. Jika pengguna mengganti atau
menghapus dataset upload, model sesi sebelumnya otomatis dihapus.

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
| Preprocessing | Hasil dataset asli, perbandingan `review_text` dan `clean_text`, tahapan salah satu ulasan dataset, serta demo teks manual opsional. |
| Training Model | Training model dari dashboard dan evaluasi performa. |

## 10. Training Model

Training dilakukan menggunakan representasi teks TF-IDF dan beberapa algoritma
klasifikasi. Model yang tersedia:

- Logistic Regression
- Complement Naive Bayes
- Linear SVM
- Extra Trees
- Voting Ensemble

Data dibagi menjadi data training dan testing menggunakan stratified split agar
proporsi label sentimen tetap terjaga.

Sebelum training, sistem memastikan dataset memiliki minimal 60 baris, memiliki
kelas negatif, netral, dan positif, serta setiap kelas memiliki data yang cukup
untuk stratified split. Model terbaik dipilih berdasarkan Macro F1 agar performa
kelas minoritas ikut dipertimbangkan.

## 11. Hasil Evaluasi Model

Hasil evaluasi bergantung pada dataset yang di-upload pengguna dan tidak
ditetapkan sebagai satu angka tetap pada laporan. Setelah training, dashboard
menampilkan accuracy, Macro F1, Weighted F1, confusion matrix, dan
classification report.

Model dengan Macro F1 tertinggi menjadi model aktif untuk prediksi pada sesi
pengguna. Model tidak disimpan sebagai file permanen di server agar model satu
pengguna tidak menggantikan atau digunakan oleh pengguna lain.

## 12. Deployment Railway

Aplikasi disiapkan untuk hosting di Railway sebagai Streamlit web app.

Start command yang digunakan:

```bash
streamlit run app.py --server.port $PORT --server.address 0.0.0.0
```

Railway akan menjalankan dependency dari `requirements.txt`. Folder `src/` dan
file `app.py` perlu tersedia di repository. Dataset dan model percobaan tidak
ikut deployment. Model hasil training berada pada memori sesi dan akan hilang
ketika sesi atau aplikasi berakhir.

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

Setelah dashboard terbuka, pengguna wajib upload dataset valid, membuka tab
Training Model, dan menyelesaikan training sebelum membuka fitur prediksi.

## 14. Kesimpulan

SentiPro berhasil dibangun sebagai dashboard analisis sentimen produk yang
memiliki pipeline preprocessing Bahasa Indonesia, visualisasi interaktif, fitur
prediksi real-time, serta training multi-model. Sistem dapat membantu pengguna
memahami pola sentimen ulasan berdasarkan brand, produk, kategori, dan waktu.

Aplikasi juga sudah disiapkan untuk deployment di Railway sehingga dapat diakses
sebagai web app. Validasi ketat dan penyimpanan model per sesi membantu mencegah
kesalahan format dataset serta mencegah model antar pengguna tercampur.

## 15. Saran Pengembangan

Beberapa pengembangan berikut dapat dilakukan:

1. Menambah jumlah dan variasi dataset ulasan.
2. Melakukan balancing data jika distribusi sentimen tidak seimbang.
3. Menambah kamus slang Bahasa Indonesia yang lebih lengkap.
4. Menguji model berbasis transformer Bahasa Indonesia.
5. Menambahkan database agar histori upload dan prediksi dapat disimpan.
6. Mengisi URL Railway production setelah deployment aktif.

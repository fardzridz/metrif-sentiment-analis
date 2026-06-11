# BAB III
# METODOLOGI DAN PERANCANGAN SISTEM

## 3.1 Alur Penelitian

Project ini menggunakan alur penelitian berbasis pengolahan data teks dan Machine Learning. Secara umum, proses dimulai dari pengumpulan data ulasan produk FemaleDaily, pelabelan sentimen berdasarkan rating, preprocessing teks, ekstraksi fitur menggunakan TF-IDF, pembagian data training dan testing, training beberapa model klasifikasi, evaluasi model, pemilihan model terbaik, dan penyajian hasil melalui dashboard SentiPro.

Alur penelitian dapat dijelaskan sebagai berikut:

1. Mengumpulkan dataset ulasan produk FemaleDaily.
2. Menentukan label sentimen berdasarkan rating.
3. Melakukan preprocessing pada teks ulasan.
4. Mengubah teks hasil preprocessing menjadi fitur numerik menggunakan TF-IDF.
5. Membagi dataset menjadi data training dan data testing.
6. Melatih beberapa algoritma klasifikasi sentimen.
7. Mengevaluasi performa model menggunakan accuracy, Macro F1, dan Weighted F1.
8. Memilih model terbaik berdasarkan Macro F1 dan menyimpannya pada sesi pengguna.
9. Mengaktifkan fitur prediksi setelah training berhasil.

Sebelum preprocessing, sistem memvalidasi format file, kelengkapan kolom,
rating, tanggal, dan nilai kosong. Sebelum training, sistem kembali memvalidasi
jumlah data serta ketersediaan kelas negatif, netral, dan positif.

## 3.2 Sumber Data

Data yang digunakan berasal dari ulasan produk pada platform FemaleDaily. Data tersebut berisi informasi produk, brand, kategori, rating, tanggal ulasan, dan teks ulasan. Dataset disimpan dalam format CSV sehingga dapat dibaca dan diproses oleh sistem.

Kolom utama pada dataset adalah sebagai berikut:

| Kolom | Keterangan |
|---|---|
| Product name | Nama produk |
| Brand | Nama brand atau merek produk |
| Category | Kategori produk |
| Rating | Nilai rating dari pengguna |
| Review date | Tanggal ulasan |
| Review text | Isi ulasan pengguna |

Sebelum diproses, nama kolom dinormalisasi menjadi huruf kecil dan spasi diganti dengan underscore. Contohnya, kolom `Review text` diubah menjadi `review_text`.

## 3.3 Pengumpulan Dataset

Pengumpulan dataset dilakukan dengan bantuan Metrif Scraper. Modul ini digunakan untuk mengambil data ulasan dari halaman produk FemaleDaily secara otomatis. Data yang diperoleh kemudian disimpan dan digunakan sebagai input pada dashboard SentiPro.

Pada laporan ini, Metrif Scraper ditempatkan sebagai modul pendukung karena fokus utama project adalah analisis sentimen. Dataset hasil pengumpulan data digunakan untuk proses sistem cerdas, yaitu preprocessing NLP, ekstraksi fitur, klasifikasi sentimen, dan evaluasi model.

## 3.4 Pelabelan Sentimen

Pelabelan sentimen dilakukan secara otomatis berdasarkan nilai rating. Pendekatan ini digunakan karena rating mencerminkan penilaian pengguna terhadap produk.

Aturan pelabelan yang digunakan adalah:

| Rating | Label Sentimen |
|---|---|
| 1 sampai 2 | Negatif |
| 3 | Netral |
| 4 sampai 5 | Positif |

Dengan aturan tersebut, setiap data ulasan memiliki label yang dapat digunakan untuk proses training model klasifikasi.

## 3.5 Preprocessing Teks

Preprocessing teks dilakukan untuk membersihkan data ulasan sebelum masuk ke tahap ekstraksi fitur. Tahapan preprocessing yang digunakan adalah:

1. Case folding, yaitu mengubah seluruh teks menjadi huruf kecil.
2. Cleaning, yaitu menghapus karakter yang tidak diperlukan seperti URL, angka, tanda baca, emoji, dan simbol.
3. Normalisasi slang, yaitu mengubah kata tidak baku menjadi kata yang lebih standar.
4. Tokenisasi, yaitu memecah teks menjadi token atau kata.
5. Stopword removal, yaitu menghapus kata umum yang tidak terlalu berpengaruh terhadap sentimen.
6. Stemming, yaitu mengubah kata berimbuhan menjadi kata dasar menggunakan pendekatan bahasa Indonesia.

Hasil akhir preprocessing disimpan dalam bentuk teks bersih yang kemudian digunakan untuk visualisasi, training model, dan prediksi sentimen.

## 3.6 Ekstraksi Fitur TF-IDF

Setelah teks dibersihkan, data teks perlu diubah menjadi bentuk numerik agar dapat diproses oleh model Machine Learning. Pada project ini digunakan metode TF-IDF. Metode ini memberikan bobot pada kata berdasarkan frekuensi kemunculannya dalam dokumen dan tingkat keunikannya dalam seluruh dataset.

TF-IDF cocok digunakan untuk klasifikasi teks karena mampu merepresentasikan kata penting dalam ulasan. Fitur hasil TF-IDF kemudian digunakan sebagai input bagi model klasifikasi.

## 3.7 Pembagian Data Training dan Testing

Dataset dibagi menjadi data training dan data testing. Data training digunakan untuk melatih model, sedangkan data testing digunakan untuk mengukur performa model terhadap data yang belum pernah dilihat sebelumnya.

Pembagian data dilakukan menggunakan stratified split untuk mempertahankan
proporsi kelas sentimen antara data training dan testing. Training diblokir jika
jumlah data atau jumlah anggota salah satu kelas tidak mencukupi.

## 3.8 Perancangan Model Machine Learning

Project ini menggunakan beberapa model klasifikasi untuk membandingkan performa dan menentukan model terbaik. Model yang digunakan adalah:

1. Logistic Regression
2. Complement Naive Bayes
3. Linear SVM
4. Extra Trees
5. Voting Ensemble

Setiap model dilatih menggunakan fitur TF-IDF dan label sentimen. Setelah proses
training selesai, model dievaluasi menggunakan accuracy, Macro F1, Weighted F1,
confusion matrix, dan classification report. Model dengan Macro F1 tertinggi
disimpan pada session pengguna dan digunakan untuk fitur prediksi.

## 3.9 Perancangan Dashboard SentiPro

Dashboard SentiPro dirancang untuk membantu pengguna memahami hasil analisis sentimen secara visual dan interaktif. Dashboard dibangun menggunakan Streamlit dengan beberapa halaman utama, yaitu:

| Halaman | Fungsi |
|---|---|
| Ringkasan | Menampilkan metrik utama, distribusi sentimen, rating, word cloud, dan data mentah |
| Brand | Menampilkan analisis sentimen berdasarkan brand |
| Produk | Menampilkan analisis produk terbaik dan terendah berdasarkan sentimen |
| Kategori | Menampilkan distribusi sentimen berdasarkan kategori produk |
| Tren Waktu | Menampilkan perubahan sentimen berdasarkan waktu |
| Prediksi | Melakukan prediksi sentimen dari teks tunggal atau file batch |
| Preprocessing | Menampilkan tahapan preprocessing teks |
| Training Model | Melakukan training model dan menampilkan hasil evaluasi |

Dashboard ini dirancang agar pengguna dapat melihat hasil analisis tanpa harus
menjalankan kode secara manual. Prediksi dikunci sampai pengguna upload dataset
valid dan menyelesaikan training.

## 3.10 Arsitektur Sistem

Arsitektur sistem terdiri dari dua bagian utama, yaitu modul pengumpulan data dan modul analisis sentimen. Modul pengumpulan data berperan menyediakan dataset ulasan. Modul analisis sentimen berperan mengolah dataset, melatih model, melakukan prediksi, dan menampilkan visualisasi.

Alur arsitektur sistem adalah sebagai berikut:

```text
FemaleDaily
    |
    v
Metrif Scraper
    |
    v
Dataset Ulasan
    |
    v
SentiPro Dashboard
    |
    +--> Validasi Dataset
    +--> Preprocessing Teks
    +--> TF-IDF
    +--> Training Model
    +--> Evaluasi Model
    +--> Model Session Pengguna
    +--> Prediksi Sentimen
    +--> Visualisasi Dashboard
```

Dengan arsitektur tersebut, sistem dapat mengubah data ulasan mentah menjadi informasi sentimen yang lebih mudah dianalisis.

# BAB IV
# IMPLEMENTASI DAN PENGUJIAN

## 4.1 Implementasi Dataset

Dataset yang digunakan pada project ini berisi ulasan produk FemaleDaily. Data disimpan dalam format CSV dan digunakan sebagai input utama pada dashboard SentiPro. Dataset memuat informasi nama produk, brand, kategori, rating, tanggal ulasan, dan teks ulasan.

Pada saat dataset dibaca oleh sistem, nama kolom dinormalisasi agar lebih mudah diproses. Contohnya, `Product name` menjadi `product_name` dan `Review text` menjadi `review_text`. Setelah itu, sistem membuat label sentimen berdasarkan nilai rating.

Aturan pelabelan yang digunakan adalah:

| Rating | Label |
|---|---|
| 1 sampai 2 | Negatif |
| 3 | Netral |
| 4 sampai 5 | Positif |

![Gambar 4.1 Potongan kode auto-labeling sentimen](images/gambar_4_1_auto_labeling_sentimen.png)

Gambar 4.1 Potongan kode auto-labeling sentimen berdasarkan rating pada file `src/data_service.py`.

## 4.2 Implementasi Preprocessing

Preprocessing teks diimplementasikan untuk membersihkan ulasan sebelum digunakan dalam proses training dan prediksi. Tahapan preprocessing meliputi case folding, cleaning, normalisasi slang, tokenisasi, stopword removal, dan stemming.

Contoh alur preprocessing adalah sebagai berikut:

```text
Teks ulasan mentah
    -> case folding
    -> cleaning
    -> normalisasi slang
    -> tokenisasi
    -> stopword removal
    -> stemming
    -> clean text
```

Hasil preprocessing digunakan sebagai data teks bersih. Data tersebut menjadi dasar untuk proses visualisasi kata, ekstraksi fitur TF-IDF, dan training model Machine Learning.

![Gambar 4.2 Potongan kode fungsi preprocessing teks](images/gambar_4_2_preprocessing_teks.png)

Gambar 4.2 Potongan kode fungsi preprocessing teks pada file `src/preprocessing.py`.

![Gambar 4.3 Tampilan halaman preprocessing](images/gambar_4_3_tampilan_preprocessing.png)

Gambar 4.3 Tampilan halaman preprocessing pada dashboard SentiPro.

## 4.3 Implementasi TF-IDF

Setelah teks melalui preprocessing, sistem mengubah teks menjadi fitur numerik menggunakan TF-IDF. TF-IDF digunakan karena model Machine Learning tidak dapat memproses teks secara langsung. Dengan TF-IDF, setiap ulasan direpresentasikan sebagai kumpulan nilai numerik berdasarkan bobot kata.

Fitur TF-IDF kemudian digunakan sebagai input untuk model klasifikasi. Label sentimen digunakan sebagai target output pada proses training.

![Gambar 4.4 Potongan kode implementasi TF-IDF](images/gambar_4_4_implementasi_tfidf.png)

Gambar 4.4 Potongan kode implementasi TF-IDF pada file `src/train.py`.

## 4.4 Implementasi Training Multi-Model

Training dilakukan dengan melatih beberapa model Machine Learning untuk membandingkan performanya. Model yang digunakan adalah:

1. Logistic Regression
2. Complement Naive Bayes
3. Linear SVM
4. Extra Trees
5. XGBoost
6. LightGBM
7. Voting Ensemble

Setiap model dilatih menggunakan data training dan diuji menggunakan data testing. Setelah proses evaluasi selesai, model terbaik disimpan ke dalam folder `models/` agar dapat digunakan kembali untuk prediksi sentimen.

File model yang dihasilkan adalah:

| File | Keterangan |
|---|---|
| `best_model.pkl` | Model terbaik yang digunakan untuk prediksi |
| `all_models.pkl` | Seluruh model yang berhasil dilatih |
| `model_info.pkl` | Informasi model, hasil evaluasi, dan metadata |

![Gambar 4.5 Potongan kode daftar model klasifikasi](images/gambar_4_5_daftar_model_klasifikasi.png)

Gambar 4.5 Potongan kode daftar model klasifikasi yang digunakan pada file `src/train.py`.

![Gambar 4.6 Potongan kode proses training dan penyimpanan model](images/gambar_4_6_training_simpan_model.png)

Gambar 4.6 Potongan kode proses training, evaluasi, dan penyimpanan model terbaik pada file `src/train.py`.

## 4.5 Implementasi Dashboard SentiPro

Dashboard SentiPro dibangun menggunakan Streamlit. Dashboard ini berfungsi sebagai antarmuka pengguna untuk membaca dataset, menampilkan hasil analisis, melakukan prediksi sentimen, dan menjalankan training model.

Fitur utama dashboard adalah:

| Tab | Implementasi |
|---|---|
| Ringkasan | Menampilkan total ulasan, rating rata-rata, distribusi sentimen, distribusi rating, word cloud, dan data mentah |
| Brand | Menampilkan analisis performa sentimen berdasarkan brand |
| Produk | Menampilkan produk dengan performa sentimen terbaik dan terendah |
| Kategori | Menampilkan distribusi sentimen berdasarkan kategori produk |
| Tren Waktu | Menampilkan perubahan sentimen berdasarkan tanggal ulasan |
| Prediksi | Menyediakan prediksi sentimen untuk teks tunggal dan batch file |
| Preprocessing | Menampilkan contoh proses preprocessing secara bertahap |
| Training Model | Menjalankan training model dan menampilkan hasil evaluasi |

Dengan dashboard ini, pengguna dapat menganalisis data tanpa harus menjalankan proses melalui command line.

![Gambar 4.7 Potongan kode struktur tab dashboard](images/gambar_4_7_struktur_tab_dashboard.png)

Gambar 4.7 Potongan kode struktur tab dashboard pada file `app.py`.

![Gambar 4.8 Tampilan dashboard ringkasan SentiPro](images/gambar_4_8_tampilan_dashboard_ringkasan.png)

Gambar 4.8 Tampilan halaman ringkasan pada dashboard SentiPro.

## 4.6 Hasil Pengujian Model

Pengujian model dilakukan dengan membandingkan performa beberapa algoritma klasifikasi. Metrik utama yang digunakan adalah accuracy dan F1-score. Accuracy digunakan untuk melihat tingkat prediksi benar secara keseluruhan, sedangkan F1-score digunakan untuk melihat keseimbangan precision dan recall.

Berdasarkan hasil model yang tersimpan, performa model adalah sebagai berikut:

| Model | Accuracy | F1-Score |
|---|---:|---:|
| Linear SVM | 80.67% | 73.08% |
| Extra Trees | 79.33% | 70.19% |
| Logistic Regression | 78.67% | 71.16% |
| Voting Ensemble | 78.67% | 72.65% |
| LightGBM | 77.33% | 69.98% |
| Complement Naive Bayes | 72.67% | 71.23% |

Dari hasil tersebut, Linear SVM menjadi model terbaik karena memperoleh accuracy tertinggi, yaitu 80.67%. Model ini kemudian digunakan sebagai model utama pada fitur prediksi sentimen.

![Gambar 4.9 Tampilan hasil evaluasi model](images/gambar_4_9_hasil_evaluasi_model.png)

Gambar 4.9 Tampilan hasil evaluasi model pada dashboard training SentiPro.

## 4.7 Perbandingan Performa Model

Hasil pengujian menunjukkan bahwa model linier memiliki performa yang baik pada data teks berbasis TF-IDF. Linear SVM memperoleh accuracy tertinggi karena algoritma ini cocok untuk data berdimensi tinggi seperti representasi teks.

Voting Ensemble memiliki F1-score yang cukup baik, tetapi accuracy-nya masih berada di bawah Linear SVM. Complement Naive Bayes menghasilkan performa paling rendah berdasarkan accuracy, tetapi masih relevan karena model ini ringan dan sering digunakan untuk klasifikasi teks.

Perbandingan ini menunjukkan bahwa pemilihan model sangat berpengaruh terhadap hasil klasifikasi sentimen. Oleh karena itu, training beberapa model secara bersamaan membantu sistem menemukan model terbaik berdasarkan data yang digunakan.

## 4.8 Pengujian Prediksi Sentimen

Pengujian prediksi dilakukan dengan memasukkan teks ulasan baru ke dalam dashboard. Sistem akan melakukan preprocessing terhadap teks tersebut, mengubahnya menjadi fitur sesuai pipeline model, kemudian menghasilkan label sentimen.

Contoh skenario pengujian:

| Input Ulasan | Hasil yang Diharapkan |
|---|---|
| Produk ini cocok di kulit saya dan hasilnya bagus | Positif |
| Biasa saja, tidak terlalu buruk tapi tidak terlalu bagus | Netral |
| Produk ini membuat kulit saya iritasi | Negatif |

Fitur prediksi ini menunjukkan bahwa model yang telah dilatih dapat digunakan untuk mengklasifikasikan ulasan baru secara real-time.

![Gambar 4.10 Potongan kode load model dan prediksi sentimen](images/gambar_4_10_load_model_prediksi.png)

Gambar 4.10 Potongan kode load model dan prediksi sentimen pada file `src/model_service.py`.

![Gambar 4.11 Tampilan halaman prediksi sentimen](images/gambar_4_11_tampilan_prediksi_sentimen.png)

Gambar 4.11 Tampilan halaman prediksi sentimen pada dashboard SentiPro.

## 4.9 Analisis Hasil

Berdasarkan hasil implementasi dan pengujian, sistem SentiPro berhasil menerapkan konsep sistem cerdas untuk analisis sentimen. Sistem mampu membaca dataset ulasan, melakukan preprocessing teks, melatih beberapa model Machine Learning, memilih model terbaik, dan menampilkan hasil analisis melalui dashboard interaktif.

Model terbaik yang diperoleh adalah Linear SVM dengan accuracy 80.67%. Hasil ini menunjukkan bahwa kombinasi preprocessing teks, TF-IDF, dan model klasifikasi linier cukup efektif untuk analisis sentimen ulasan produk FemaleDaily.

Namun, performa model masih dapat ditingkatkan. Beberapa faktor yang dapat memengaruhi hasil antara lain jumlah dataset, keseimbangan kelas sentimen, kualitas preprocessing, variasi bahasa informal, dan keterbatasan pelabelan berdasarkan rating.

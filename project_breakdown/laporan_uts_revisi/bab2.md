# BAB II
# LANDASAN TEORI

## 2.1 Sistem Cerdas

Sistem cerdas adalah sistem yang dirancang untuk meniru sebagian kemampuan manusia dalam memahami informasi, mengenali pola, melakukan penalaran, atau mengambil keputusan berdasarkan data. Sistem cerdas umumnya memanfaatkan metode Artificial Intelligence, Machine Learning, atau teknik komputasi lain untuk menghasilkan keluaran yang adaptif dan informatif.

Pada project ini, konsep sistem cerdas diterapkan dalam bentuk analisis sentimen ulasan produk. Sistem menerima data berupa teks ulasan, melakukan pengolahan bahasa alami, mengekstraksi fitur, melatih model klasifikasi, kemudian menghasilkan prediksi sentimen. Dengan demikian, sistem tidak hanya menampilkan data, tetapi juga memberikan keputusan berupa label sentimen positif, netral, atau negatif.

## 2.2 Analisis Sentimen

Analisis sentimen adalah proses untuk mengidentifikasi opini, emosi, atau kecenderungan sikap dalam sebuah teks. Analisis ini banyak digunakan untuk mengetahui persepsi masyarakat terhadap produk, layanan, brand, tokoh, maupun peristiwa tertentu. Dalam konteks ulasan produk, analisis sentimen dapat membantu mengetahui apakah pengguna memberikan tanggapan positif, netral, atau negatif.

Pada project ini, analisis sentimen dilakukan terhadap ulasan produk FemaleDaily. Setiap ulasan diklasifikasikan menjadi tiga kelas, yaitu:

| Rating | Label Sentimen |
|---|---|
| 1 sampai 2 | Negatif |
| 3 | Netral |
| 4 sampai 5 | Positif |

Pelabelan berdasarkan rating digunakan karena rating dapat dianggap sebagai representasi penilaian pengguna terhadap produk.

## 2.3 Natural Language Processing

Natural Language Processing atau NLP adalah bidang dalam Artificial Intelligence yang berfokus pada pemrosesan bahasa alami manusia oleh komputer. NLP memungkinkan komputer untuk membaca, membersihkan, memahami, dan menganalisis teks.

Dalam analisis sentimen, NLP digunakan untuk mengolah teks ulasan yang tidak terstruktur menjadi data yang lebih siap diproses oleh model Machine Learning. Tahapan NLP yang digunakan dalam project ini mencakup pembersihan teks, normalisasi kata, tokenisasi, penghapusan stopword, dan stemming.

## 2.4 Text Preprocessing

Text preprocessing adalah proses awal untuk membersihkan dan menyiapkan teks sebelum digunakan dalam pemodelan. Tahapan ini penting karena teks ulasan pengguna sering mengandung kata tidak baku, simbol, tanda baca, angka, emoji, dan variasi penulisan.

Tahapan preprocessing yang digunakan dalam project ini adalah sebagai berikut:

1. Case folding, yaitu mengubah seluruh huruf menjadi huruf kecil.
2. Cleaning, yaitu menghapus URL, mention, hashtag, angka, tanda baca, emoji, dan karakter yang tidak diperlukan.
3. Normalization, yaitu mengubah kata tidak baku atau slang menjadi bentuk yang lebih standar.
4. Tokenization, yaitu memecah kalimat menjadi kumpulan kata atau token.
5. Stopword removal, yaitu menghapus kata umum yang kurang memiliki makna penting dalam analisis.
6. Stemming, yaitu mengubah kata berimbuhan menjadi bentuk kata dasar.

Tahapan preprocessing membantu model memperoleh representasi teks yang lebih bersih dan konsisten.

## 2.5 TF-IDF

TF-IDF atau Term Frequency-Inverse Document Frequency adalah metode pembobotan kata yang digunakan untuk mengubah teks menjadi nilai numerik. TF-IDF memberikan bobot tinggi pada kata yang sering muncul dalam suatu dokumen, tetapi tidak terlalu sering muncul di seluruh dokumen.

Metode ini terdiri dari dua komponen utama:

1. Term Frequency, yaitu frekuensi kemunculan suatu kata dalam dokumen.
2. Inverse Document Frequency, yaitu ukuran seberapa unik atau penting suatu kata terhadap kumpulan dokumen.

Pada project ini, TF-IDF digunakan untuk mengubah teks ulasan hasil preprocessing menjadi fitur numerik yang dapat diproses oleh algoritma Machine Learning.

## 2.6 Machine Learning untuk Klasifikasi Teks

Machine Learning adalah metode yang memungkinkan komputer belajar dari data tanpa harus diprogram secara eksplisit untuk setiap aturan. Dalam klasifikasi teks, Machine Learning digunakan untuk mempelajari pola dari teks yang sudah memiliki label, kemudian memprediksi label dari teks baru.

Pada project ini, model dilatih menggunakan data ulasan yang telah diberi label sentimen. Fitur input berasal dari hasil ekstraksi TF-IDF, sedangkan output model berupa kelas sentimen positif, netral, atau negatif.

## 2.7 Algoritma Klasifikasi yang Digunakan

Beberapa algoritma Machine Learning digunakan untuk membandingkan performa klasifikasi sentimen.

### 2.7.1 Logistic Regression

Logistic Regression adalah algoritma klasifikasi linier yang sering digunakan untuk klasifikasi teks. Model ini bekerja dengan menghitung probabilitas suatu data masuk ke kelas tertentu.

### 2.7.2 Complement Naive Bayes

Complement Naive Bayes adalah pengembangan dari Naive Bayes yang cocok digunakan pada data teks dan dapat membantu menangani ketidakseimbangan distribusi kelas.

### 2.7.3 Linear Support Vector Machine

Linear SVM adalah algoritma klasifikasi yang mencari hyperplane terbaik untuk memisahkan data antar kelas. Algoritma ini banyak digunakan pada klasifikasi teks karena mampu menangani data berdimensi tinggi.

### 2.7.4 Extra Trees

Extra Trees atau Extremely Randomized Trees adalah algoritma ensemble berbasis pohon keputusan. Model ini menggunakan banyak pohon keputusan dan melakukan pemilihan pemisahan data secara acak untuk meningkatkan generalisasi.

### 2.7.5 XGBoost

XGBoost adalah algoritma gradient boosting yang membangun model secara bertahap untuk memperbaiki kesalahan dari model sebelumnya. Algoritma ini sering digunakan karena performanya baik pada berbagai jenis data.

### 2.7.6 LightGBM

LightGBM adalah algoritma gradient boosting yang dirancang agar lebih cepat dan efisien, terutama pada dataset berukuran besar.

### 2.7.7 Voting Ensemble

Voting Ensemble adalah metode yang menggabungkan beberapa model untuk menghasilkan keputusan akhir. Tujuannya adalah meningkatkan stabilitas dan akurasi prediksi dengan memanfaatkan kekuatan dari beberapa algoritma.

## 2.8 Evaluasi Model Klasifikasi

Evaluasi model dilakukan untuk mengetahui seberapa baik model dalam melakukan klasifikasi sentimen. Beberapa metrik evaluasi yang digunakan adalah:

1. Accuracy, yaitu perbandingan jumlah prediksi benar terhadap seluruh data uji.
2. Precision, yaitu ketepatan model dalam memprediksi suatu kelas.
3. Recall, yaitu kemampuan model menemukan seluruh data yang benar-benar termasuk dalam suatu kelas.
4. F1-score, yaitu rata-rata harmonis antara precision dan recall.
5. Confusion matrix, yaitu tabel yang menunjukkan perbandingan antara label sebenarnya dan hasil prediksi model.

Pada data sentimen yang memiliki lebih dari dua kelas, F1-score penting digunakan karena dapat menunjukkan keseimbangan performa model pada setiap kelas.

## 2.9 Streamlit

Streamlit adalah framework Python yang digunakan untuk membangun dashboard dan aplikasi data secara cepat. Streamlit memungkinkan visualisasi data, input pengguna, serta integrasi model Machine Learning dalam satu aplikasi web.

Pada project ini, Streamlit digunakan untuk membangun dashboard SentiPro. Dashboard tersebut menampilkan ringkasan dataset, distribusi sentimen, analisis brand, analisis produk, analisis kategori, tren waktu, fitur prediksi, visualisasi preprocessing, dan training model.

## 2.10 Web Scraping sebagai Pendukung Dataset

Web scraping adalah proses pengambilan data dari halaman web secara otomatis. Pada project ini, web scraping digunakan sebagai modul pendukung untuk memperoleh dataset ulasan produk dari FemaleDaily. Modul tersebut dinamakan Metrif Scraper.

Meskipun terdapat proses scraping, fokus utama laporan ini bukan pada scraper, melainkan pada sistem cerdas analisis sentimen. Data hasil scraping digunakan sebagai bahan input untuk proses preprocessing, training model, evaluasi, dan visualisasi pada SentiPro.


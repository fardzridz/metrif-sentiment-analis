# BAB I
# PENDAHULUAN

## 1.1 Latar Belakang

Perkembangan teknologi informasi membuat masyarakat semakin mudah memperoleh informasi sebelum membeli suatu produk. Pada bidang kecantikan, calon pembeli sering membaca ulasan pengguna lain untuk mengetahui kualitas produk, kecocokan dengan jenis kulit, kelebihan, kekurangan, serta pengalaman pemakaian secara nyata. Salah satu platform yang banyak digunakan untuk memperoleh ulasan produk kecantikan di Indonesia adalah FemaleDaily.

Ulasan produk pada FemaleDaily memiliki nilai informasi yang besar karena berisi opini langsung dari pengguna. Namun, jumlah ulasan yang banyak membuat proses pembacaan dan penilaian secara manual menjadi tidak efisien. Selain itu, teks ulasan umumnya menggunakan bahasa Indonesia informal, singkatan, slang, campuran bahasa, serta variasi penulisan yang tidak selalu baku. Kondisi tersebut membuat analisis manual menjadi lebih sulit dan berpotensi menghasilkan kesimpulan yang subjektif.

Analisis sentimen dapat digunakan untuk mengelompokkan opini pengguna ke dalam kelas sentimen tertentu, seperti positif, netral, dan negatif. Dengan bantuan Natural Language Processing (NLP) dan Machine Learning, sistem dapat memproses teks ulasan, mengubahnya menjadi fitur numerik, melatih model klasifikasi, serta memprediksi sentimen dari ulasan baru. Pendekatan ini sesuai dengan konsep sistem cerdas karena sistem tidak hanya menyimpan data, tetapi juga melakukan proses analisis dan pengambilan keputusan berdasarkan pola dari data.

Pada project ini dibangun SentiPro, yaitu dashboard analisis sentimen ulasan produk FemaleDaily berbasis Streamlit. Sistem ini menerapkan tahapan preprocessing teks, ekstraksi fitur TF-IDF, training beberapa model Machine Learning, evaluasi performa model, visualisasi hasil analisis, serta prediksi sentimen secara real-time. Dataset ulasan diperoleh melalui modul pendukung bernama Metrif Scraper yang digunakan untuk mengumpulkan review produk FemaleDaily secara otomatis.

Dengan adanya sistem ini, data ulasan yang semula berbentuk teks tidak terstruktur dapat diolah menjadi informasi yang lebih mudah dipahami. Hasil analisis dapat digunakan untuk melihat kecenderungan sentimen pengguna terhadap brand, produk, kategori, dan tren waktu tertentu.

## 1.2 Rumusan Masalah

Berdasarkan latar belakang tersebut, rumusan masalah dalam project ini adalah sebagai berikut:

1. Bagaimana membangun sistem analisis sentimen ulasan produk FemaleDaily berbasis Machine Learning?
2. Bagaimana menerapkan tahapan preprocessing teks untuk menangani ulasan berbahasa Indonesia yang bersifat informal?
3. Bagaimana mengubah teks ulasan menjadi fitur numerik menggunakan metode TF-IDF?
4. Bagaimana membandingkan performa beberapa algoritma Machine Learning dalam klasifikasi sentimen?
5. Bagaimana menyajikan hasil analisis sentimen dalam bentuk dashboard interaktif yang mudah dipahami?

## 1.3 Tujuan

Tujuan dari project ini adalah sebagai berikut:

1. Membangun sistem cerdas untuk analisis sentimen ulasan produk FemaleDaily.
2. Menerapkan pipeline preprocessing teks yang meliputi case folding, cleaning, normalisasi, tokenisasi, stopword removal, dan stemming.
3. Menerapkan ekstraksi fitur teks menggunakan TF-IDF.
4. Melatih dan mengevaluasi beberapa model klasifikasi sentimen, yaitu Logistic Regression, Complement Naive Bayes, Linear SVM, Extra Trees, XGBoost, LightGBM, dan Voting Ensemble.
5. Menentukan model terbaik berdasarkan hasil evaluasi accuracy dan F1-score.
6. Membangun dashboard SentiPro untuk menampilkan visualisasi sentimen dan melakukan prediksi sentimen secara real-time.

## 1.4 Manfaat

Manfaat dari project ini adalah sebagai berikut:

1. Bagi pengguna, sistem dapat membantu memahami kecenderungan opini konsumen terhadap suatu produk atau brand berdasarkan data ulasan.
2. Bagi peneliti, dataset dan hasil analisis dapat digunakan sebagai referensi dalam penelitian analisis sentimen, NLP, dan Machine Learning.
3. Bagi pengembang, project ini menjadi implementasi sistem cerdas yang menggabungkan pengolahan data, preprocessing teks, training model, evaluasi, dan visualisasi dashboard.
4. Bagi pemilik brand atau pelaku bisnis, hasil analisis dapat menjadi bahan pertimbangan untuk memahami kepuasan dan keluhan konsumen.

## 1.5 Batasan Masalah

Agar pembahasan lebih terarah, batasan masalah pada project ini adalah sebagai berikut:

1. Data yang digunakan berupa ulasan produk dari platform FemaleDaily.
2. Analisis difokuskan pada teks ulasan produk kecantikan berbahasa Indonesia.
3. Label sentimen ditentukan berdasarkan rating, yaitu rating 1 sampai 2 sebagai negatif, rating 3 sebagai netral, dan rating 4 sampai 5 sebagai positif.
4. Model yang digunakan adalah model klasifikasi berbasis Machine Learning dengan representasi fitur TF-IDF.
5. Sistem tidak membahas analisis sentimen berbasis deep learning atau transformer.
6. Metrif Scraper hanya digunakan sebagai modul pendukung untuk pengumpulan dataset, sedangkan fokus utama laporan adalah analisis sentimen pada SentiPro.


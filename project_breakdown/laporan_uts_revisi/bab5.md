# BAB V
# PENUTUP

## 5.1 Kesimpulan

Berdasarkan implementasi dan pengujian yang telah dilakukan, dapat disimpulkan bahwa project SentiPro berhasil dibangun sebagai sistem cerdas untuk analisis sentimen ulasan produk FemaleDaily. Sistem mampu mengolah teks ulasan berbahasa Indonesia, melakukan pelabelan sentimen berdasarkan rating, menerapkan preprocessing teks, mengekstraksi fitur menggunakan TF-IDF, melatih beberapa model Machine Learning, serta menyajikan hasil analisis dalam bentuk dashboard interaktif.

Tahapan preprocessing yang digunakan meliputi case folding, cleaning, normalisasi slang, tokenisasi, stopword removal, dan stemming. Tahapan ini membantu membersihkan ulasan yang bersifat informal sehingga teks lebih siap digunakan pada proses training model.

Model Machine Learning yang diuji meliputi Logistic Regression, Complement Naive Bayes, Linear SVM, Extra Trees, XGBoost, LightGBM, dan Voting Ensemble. Berdasarkan hasil evaluasi, model Linear SVM memperoleh performa terbaik dengan accuracy 80.67% dan F1-score 73.08%. Model tersebut digunakan sebagai model utama untuk fitur prediksi sentimen.

Dashboard SentiPro berhasil menyajikan berbagai fitur analisis, seperti ringkasan dataset, distribusi sentimen, analisis brand, analisis produk, analisis kategori, tren waktu, visualisasi preprocessing, training model, dan prediksi sentimen real-time. Dengan adanya dashboard ini, hasil analisis sentimen dapat dipahami lebih mudah oleh pengguna.

Secara keseluruhan, project ini menunjukkan penerapan sistem cerdas dalam bidang analisis sentimen. Data ulasan yang semula berbentuk teks tidak terstruktur dapat diolah menjadi informasi yang lebih terukur dan bermanfaat.

## 5.2 Saran

Beberapa saran pengembangan untuk project ini adalah sebagai berikut:

1. Menambah jumlah dataset agar model dapat mempelajari variasi ulasan yang lebih luas.
2. Melakukan penyeimbangan data jika jumlah sentimen positif, netral, dan negatif tidak merata.
3. Menambah kamus normalisasi slang bahasa Indonesia agar preprocessing lebih akurat.
4. Menguji model berbasis deep learning atau transformer bahasa Indonesia untuk meningkatkan performa klasifikasi.
5. Menambahkan fitur penyimpanan histori prediksi agar pengguna dapat melihat hasil prediksi sebelumnya.
6. Mengembangkan dashboard agar dapat terhubung langsung dengan database sehingga proses analisis menjadi lebih otomatis.
7. Melakukan evaluasi tambahan menggunakan data baru untuk menguji kemampuan generalisasi model.


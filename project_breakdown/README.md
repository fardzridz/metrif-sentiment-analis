# Project Breakdown SentiPro

Folder ini berisi dokumentasi ringkas dan laporan akhir untuk project
SentiPro, yaitu dashboard analisis sentimen ulasan produk berbahasa Indonesia.

## Isi Folder

| File | Isi |
|---|---|
| `laporan_akhir.md` | Laporan akhir lengkap untuk kebutuhan presentasi atau dokumentasi tugas. |
| `diagram.md` | Diagram arsitektur, preprocessing, training, dan deployment dalam format Mermaid. |
| `railway_deployment.md` | Catatan hosting dan langkah deployment ke Railway. |
| `ringkasan_teknis.md` | Breakdown teknis per modul agar struktur kode mudah dijelaskan. |

## Identitas Project

- Nama aplikasi: SentiPro
- Jenis aplikasi: Dashboard analisis sentimen produk
- Bahasa data: Indonesia
- Framework UI: Streamlit
- Machine learning: scikit-learn dengan TF-IDF dan beberapa model klasifikasi
- Hosting: Railway

## Alur Singkat

1. Pengguna mengunggah dataset CSV atau Excel melalui sidebar.
2. Sistem membersihkan dan memproses data ulasan.
3. Sentimen dapat diambil dari rating atau diprediksi memakai model ML.
4. Dashboard menampilkan ringkasan, brand, produk, kategori, tren waktu, prediksi,
   preprocessing, dan training model.
5. Aplikasi dijalankan sebagai web app Streamlit dan disiapkan untuk hosting di Railway.


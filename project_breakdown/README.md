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
2. Sistem memblokir dataset yang gagal validasi struktur atau nilai.
3. Sistem membersihkan teks dan membuat label analisis dari rating.
4. Training hanya aktif jika jumlah data dan tiga kelas memenuhi syarat.
5. Prediksi hanya aktif setelah model sesi berhasil dilatih.
6. Dataset dan model percobaan tidak ikut GitHub atau deployment.

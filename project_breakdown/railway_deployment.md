# Deployment Railway

Dokumen ini menjelaskan catatan hosting project SentiPro di Railway.

## Platform Hosting

- Platform: Railway
- Jenis aplikasi: Streamlit web app
- Entry point: `app.py`
- Dependency utama: `requirements.txt`
- Model dashboard: disimpan sementara pada session pengguna

## Start Command

Gunakan start command berikut di Railway:

```bash
streamlit run app.py --server.port $PORT --server.address 0.0.0.0
```

Alasan:

- Railway menyediakan port aplikasi melalui environment variable `$PORT`.
- Streamlit perlu bind ke `0.0.0.0` agar bisa diakses dari luar container.
- `app.py` adalah entry point dashboard.

## Build Command

Biasanya Railway dapat membaca Python project dari `requirements.txt`.
Jika perlu mengisi manual:

```bash
pip install -r requirements.txt
```

## Konfigurasi Nixpacks

File `nixpacks.toml` hanya menetapkan start command Streamlit. Dashboard tidak
membutuhkan dependency sistem tambahan karena model opsional eksperimen seperti
XGBoost dan LightGBM tidak digunakan pada workflow web.

## Environment Variable

Project ini belum membutuhkan environment variable khusus selain `$PORT` dari
Railway. Jika nanti memakai database, API eksternal, atau penyimpanan cloud,
tambahkan variabel terkait di menu Variables Railway.

## File Penting yang Harus Ikut Ter-deploy

- `app.py`
- `requirements.txt`
- `src/`
- `data/.gitkeep`
- `models/.gitkeep`

## Catatan Railway

1. Pastikan repository sudah tersambung ke Railway.
2. Dataset dan model percobaan tidak perlu ikut deployment.
3. Setiap pengguna wajib upload dataset dan training sebelum prediksi aktif.
4. Model disimpan dalam session dan hilang ketika session atau aplikasi berakhir.
5. Setelah deploy, isi URL production di laporan akhir.

## URL Hosting

- Railway production URL: `isi-dengan-url-railway`

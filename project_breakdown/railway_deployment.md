# Deployment Railway

Dokumen ini menjelaskan catatan hosting project SentiPro di Railway.

## Platform Hosting

- Platform: Railway
- Jenis aplikasi: Streamlit web app
- Entry point: `app.py`
- Dependency utama: `requirements.txt`
- Model tersimpan: `models/best_model.pkl`, `models/model_info.pkl`,
  `models/all_models.pkl`

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

## System Dependency

Project ini memakai LightGBM sebagai salah satu model training. Pada Railway,
LightGBM membutuhkan library sistem Linux `libgomp.so.1`. Library tersebut
disediakan oleh package `libgomp1`.

Dependency sistem sudah ditambahkan melalui file `nixpacks.toml`:

```toml
[phases.setup]
aptPkgs = ["...", "libgomp1"]
```

File `nixpacks.toml` harus berada di root aplikasi yang dideploy oleh Railway.
Jika Railway diarahkan ke folder `sentimen_analis`, maka file tersebut harus
berada di dalam folder `sentimen_analis/`.

## Environment Variable

Project ini belum membutuhkan environment variable khusus selain `$PORT` dari
Railway. Jika nanti memakai database, API eksternal, atau penyimpanan cloud,
tambahkan variabel terkait di menu Variables Railway.

## File Penting yang Harus Ikut Ter-deploy

- `app.py`
- `requirements.txt`
- `src/`
- `models/`
- `data/` jika ingin menyediakan dataset bawaan

## Catatan Railway

1. Pastikan repository sudah tersambung ke Railway.
2. Pastikan model sudah tersedia di folder `models/` jika ingin fitur prediksi
   langsung aktif saat aplikasi dibuka.
3. Jika model belum ada, fitur dashboard tetap bisa membaca dataset dan
   membuat label sentimen dari rating.
4. Setelah deploy, isi URL production di laporan akhir.

## URL Hosting

- Railway production URL: `isi-dengan-url-railway`

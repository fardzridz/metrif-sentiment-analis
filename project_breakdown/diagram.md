# Diagram Project SentiPro

## 1. Workflow Wajib Pengguna

```mermaid
flowchart TD
    A[Upload CSV/Excel] --> B{Validasi File dan Dataset}
    B -->|Tidak valid| C[Tampilkan Error dan Blokir Proses]
    C --> A
    B -->|Valid| D[Pembersihan dan Preprocessing]
    D --> E[Label Sentimen dari Rating]
    E --> F[Dashboard Analisis]
    F --> G{Validasi Kesiapan Training}
    G -->|Tidak siap| H[Tampilkan Syarat yang Belum Terpenuhi]
    G -->|Siap| I[Stratified Train-Test Split]
    I --> J[TF-IDF dan Training Multi-model]
    J --> K[Evaluasi Accuracy, Macro F1, Weighted F1]
    K --> L[Pilih Model dengan Macro F1 Tertinggi]
    L --> M[(Model pada Session Pengguna)]
    M --> N[Prediksi Teks atau Batch]
```

## 2. Arsitektur Aplikasi

```mermaid
flowchart TD
    User[Pengguna] --> App[app.py - Streamlit]
    App --> DataService[src/data_service.py]
    DataService --> Validation[Validasi Struktur, Nilai, dan Training]
    DataService --> Preprocess[src/preprocessing.py]
    App --> Pages[src/pages]
    Pages --> Analysis[Ringkasan, Brand, Produk, Kategori, Tren]
    Pages --> Training[Training dan Evaluasi]
    Training --> Session[(st.session_state)]
    Session --> ModelService[src/model_service.py]
    ModelService --> Prediction[Prediksi Sentimen]
```

Dataset dan model percobaan tidak menjadi input bawaan aplikasi. Model sesi
otomatis dihapus ketika dataset upload diganti atau dihapus.

## 3. Pipeline Preprocessing

```mermaid
flowchart LR
    A[Teks Mentah] --> B[Case Folding]
    B --> C[Cleaning]
    C --> D[Normalisasi Slang]
    D --> E[Tokenisasi]
    E --> F[Stopword Removal]
    F --> G{Stemming Sesuai Training?}
    G -->|Ya| H[Stemming]
    G -->|Tidak| I[Clean Text]
    H --> I
    I --> J[TF-IDF]
```

Konfigurasi stemming yang dipilih saat training selalu digunakan kembali saat
prediksi.

## 4. Deployment Railway

```mermaid
flowchart TD
    Repo[Source Code tanpa Dataset/Model Percobaan] --> Railway[Railway Service]
    Railway --> Install[pip install -r requirements.txt]
    Install --> Start[streamlit run app.py]
    Start --> Web[Public Web App]
    Web --> Session[Session Terpisah per Pengguna]
```

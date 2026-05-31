# Diagram Project SentiPro

Dokumen ini memakai Mermaid agar diagram bisa dibaca langsung di VS Code,
GitHub, atau Markdown viewer yang mendukung Mermaid.

## 1. Arsitektur Aplikasi

```mermaid
flowchart TD
    User[Pengguna] --> Upload[Upload Dataset CSV/Excel]
    Upload --> App[app.py - Streamlit Entry Point]
    App --> DataService[src/data_service.py]
    DataService --> Preprocess[src/preprocessing.py]
    Preprocess --> Dashboard[Dashboard Tabs]

    App --> ModelService[src/model_service.py]
    ModelService --> Models[(models/*.pkl)]
    Models --> Prediction[Prediksi Sentimen]

    Dashboard --> Summary[Ringkasan]
    Dashboard --> Brand[Analisis Brand]
    Dashboard --> Product[Analisis Produk]
    Dashboard --> Category[Analisis Kategori]
    Dashboard --> Trend[Tren Waktu]
    Dashboard --> Predict[Prediksi Real-Time]
    Dashboard --> PreprocessView[Visualisasi Preprocessing]
    Dashboard --> Training[Training Model]
```

## 2. Pipeline Preprocessing Teks

```mermaid
flowchart LR
    A[Teks Ulasan Mentah] --> B[Case Folding]
    B --> C[Cleaning URL, Mention, Hashtag, Angka, Tanda Baca]
    C --> D[Normalisasi Slang]
    D --> E[Tokenisasi]
    E --> F[Stopword Removal]
    F --> G{Stemming Aktif?}
    G -->|Ya| H[Stemming PySastrawi]
    G -->|Tidak| I[Gabung Token]
    H --> I[Clean Text]
    I --> J[Siap untuk TF-IDF atau Analisis]
```

## 3. Alur Training Model

```mermaid
flowchart TD
    A[data/dataset.csv] --> B[Load Dataset]
    B --> C[Normalisasi Kolom]
    C --> D[Label Sentimen dari Rating]
    D --> E[Preprocessing Review Text]
    E --> F[Train-Test Split Stratified]
    F --> G[TF-IDF Feature Extraction]
    G --> H[Training Multi Model]
    H --> I[Evaluasi Accuracy dan F1-Score]
    I --> J[Pilih Model Terbaik]
    J --> K[(models/best_model.pkl)]
    J --> L[(models/all_models.pkl)]
    J --> M[(models/model_info.pkl)]
```

## 4. Alur Prediksi Sentimen

```mermaid
sequenceDiagram
    participant U as Pengguna
    participant UI as Streamlit UI
    participant MS as model_service.py
    participant PP as preprocessing.py
    participant M as best_model.pkl

    U->>UI: Input teks atau upload batch
    UI->>MS: Kirim teks ulasan
    MS->>PP: Preprocess teks
    PP-->>MS: Clean text
    MS->>M: Predict sentimen
    M-->>MS: Label dan probabilitas jika tersedia
    MS-->>UI: Hasil prediksi
    UI-->>U: Tampilkan positif/netral/negatif
```

## 5. Deployment Railway

```mermaid
flowchart TD
    Repo[Repository Project] --> Railway[Railway Service]
    Railway --> Install[pip install -r requirements.txt]
    Install --> Start[streamlit run app.py --server.port $PORT --server.address 0.0.0.0]
    Start --> Web[Public Web App]
    Web --> User[Pengguna Mengakses Dashboard]
```


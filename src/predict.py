"""
predict.py
Script untuk memprediksi sentimen teks baru menggunakan model yang sudah dilatih.
"""

import os
import joblib
import pandas as pd
from preprocessing import preprocess

MODEL_PATH = '../models/best_model.pkl'
INFO_PATH  = '../models/model_info.pkl'


def load_model():
    """Load model dari file."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model tidak ditemukan di {MODEL_PATH}\n"
            "Jalankan train.py terlebih dahulu!"
        )
    model = joblib.load(MODEL_PATH)

    if os.path.exists(INFO_PATH):
        info = joblib.load(INFO_PATH)
        print(f"Model dimuat: {info['model_name']} (akurasi: {info['accuracy']*100:.2f}%)\n")

    return model


def predict_single(text: str, model) -> dict:
    """
    Prediksi sentimen untuk satu teks.

    Return:
        dict dengan 'teks_asli', 'teks_bersih', 'sentimen', 'probabilitas'
    """
    clean = preprocess(text, use_stemming=True)

    # Prediksi label
    label = model.predict([clean])[0]

    # Probabilitas (jika model mendukung)
    result = {
        'teks_asli'  : text,
        'teks_bersih': clean,
        'sentimen'   : label,
    }

    try:
        proba = model.predict_proba([clean])[0]
        classes = model.classes_
        result['probabilitas'] = {cls: round(float(p), 4) for cls, p in zip(classes, proba)}
    except AttributeError:
        # LinearSVC tidak punya predict_proba
        result['probabilitas'] = None

    return result


def predict_batch(filepath: str, model) -> pd.DataFrame:
    """
    Prediksi sentimen untuk seluruh dataset baru (tanpa label).

    Parameter:
        filepath : path ke file CSV/Excel
        model    : model yang sudah diload

    Return:
        DataFrame dengan kolom tambahan 'prediksi_sentimen'
    """
    if filepath.endswith('.csv'):
        df = pd.read_csv(filepath)
    else:
        df = pd.read_excel(filepath)

    df.columns = df.columns.str.lower().str.replace(' ', '_')

    print(f"Memproses {len(df)} baris...")
    df['clean_text'] = df['review_text'].apply(
        lambda x: preprocess(str(x), use_stemming=True)
    )
    df['prediksi_sentimen'] = model.predict(df['clean_text'])

    # Simpan hasil
    output_path = filepath.replace('.csv', '_hasil.csv').replace('.xlsx', '_hasil.csv')
    df.drop(columns=['clean_text']).to_csv(output_path, index=False)
    print(f"Hasil disimpan di: {output_path}")

    return df


def interactive_mode(model):
    """Mode interaktif — ketik teks, langsung dapat prediksi."""
    print("=" * 50)
    print("  MODE INTERAKTIF - Prediksi Sentimen")
    print("  Ketik 'keluar' untuk berhenti")
    print("=" * 50)

    while True:
        text = input("\nMasukkan teks ulasan: ").strip()

        if text.lower() in ('keluar', 'exit', 'quit'):
            print("Sampai jumpa!")
            break

        if not text:
            print("Teks tidak boleh kosong.")
            continue

        result = predict_single(text, model)

        emoji = {'positif': '😊', 'netral': '😐', 'negatif': '😞'}
        print(f"\nSentimen  : {result['sentimen'].upper()} {emoji.get(result['sentimen'], '')}")

        if result['probabilitas']:
            print("Probabilitas:")
            for label, prob in sorted(result['probabilitas'].items(),
                                      key=lambda x: x[1], reverse=True):
                bar = '█' * int(prob * 20)
                print(f"  {label:<10}: {prob:.4f}  {bar}")


if __name__ == "__main__":
    model = load_model()

    print("Pilih mode:")
    print("1. Interaktif (ketik teks manual)")
    print("2. Batch (prediksi dari file CSV/Excel)")
    pilihan = input("Pilihan (1/2): ").strip()

    if pilihan == '1':
        interactive_mode(model)
    elif pilihan == '2':
        path = input("Masukkan path file: ").strip()
        predict_batch(path, model)
    else:
        print("Pilihan tidak valid.")

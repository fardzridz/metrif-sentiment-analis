"""
preprocessing.py
Pipeline preprocessing teks bahasa Indonesia:
  1. Cleaning (URL, mention, hashtag, angka, tanda baca)
  2. Case folding (lowercase)
  3. Normalisasi slang / singkatan
  4. Tokenisasi
  5. Stopword removal
  6. Stemming (PySastrawi)
"""

import re
import string
import pandas as pd
import nltk
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# ── Stemmer ──────────────────────────────────────────────────────────────────
_factory = StemmerFactory()
_stemmer = _factory.create_stemmer()

# ── Stopwords ─────────────────────────────────────────────────────────────────
try:
    STOP_WORDS = set(stopwords.words('indonesian'))
except LookupError:
    try:
        nltk.download('stopwords', quiet=True)
        STOP_WORDS = set(stopwords.words('indonesian'))
    except Exception:
        STOP_WORDS = set()
EXTRA_STOP = {
    'yg','dgn','nya','utk','jg','sy','gw','gue','aja','udah','udh','sdh',
    'sudah','sih','deh','dong','nih','loh','lah','kak','bang','bro','sis',
    'gan','min','tp','tapi','krn','karena','jadi','jd','bgt','banget','emg',
    'emang','memang','klo','kalo','kalau','dr','dari','ke','di','yaa','ya',
    'ga','gak','ngga','tidak','tdk','bisa','ada','ini','itu','juga','dengan',
    'untuk','yang','dan','atau','pada','dari','ke','di','nya','lagi','aja',
    'mau','msh','masih','lg','lagi','sdg','sedang','hrs','harus','blm','belum'
}
STOP_WORDS.update(EXTRA_STOP)

# ── Kamus Normalisasi Slang ───────────────────────────────────────────────────
SLANG_DICT = {
    # Singkatan umum
    'yg':'yang','dgn':'dengan','utk':'untuk','jg':'juga','sy':'saya',
    'gw':'saya','gue':'saya','aq':'saya','ak':'saya','km':'kamu',
    'kmu':'kamu','lo':'kamu','lu':'kamu','mrk':'mereka','kt':'kita',
    # Kata tidak baku
    'bgt':'banget','bngt':'banget','bner':'benar','bnr':'benar',
    'bgs':'bagus','bgus':'bagus','jelek':'jelek','jlek':'jelek',
    'ok':'oke','oke':'oke','okey':'oke','sip':'oke',
    'mantap':'mantap','mantep':'mantap','keren':'keren','krn':'keren',
    'bagus':'bagus','baguss':'bagus','baguus':'bagus',
    'cepet':'cepat','cpet':'cepat','lmbt':'lambat','lambat':'lambat',
    'murah':'murah','mrah':'murah','mahal':'mahal','mhal':'mahal',
    'rusak':'rusak','rsk':'rusak','cacat':'cacat','cct':'cacat',
    'puas':'puas','kecewa':'kecewa','kcw':'kecewa',
    'pengiriman':'pengiriman','kirim':'kirim','krm':'kirim',
    'produk':'produk','prduk':'produk','barang':'barang','brg':'barang',
    'kualitas':'kualitas','kualtas':'kualitas','qlts':'kualitas',
    'harga':'harga','hrg':'harga','diskon':'diskon','dskn':'diskon',
    'recommended':'rekomendasi','rekomen':'rekomendasi','rekomend':'rekomendasi',
    'worth':'sepadan','worth it':'sepadan',
    'original':'asli','ori':'asli','palsu':'palsu','kw':'palsu',
    'fast':'cepat','slow':'lambat','good':'bagus','bad':'jelek',
    'nice':'bagus','great':'bagus','best':'terbaik','worst':'terburuk',
    # Typo umum
    'sangaat':'sangat','sanagt':'sangat','sngt':'sangat',
    'terlaluu':'terlalu','trllu':'terlalu',
    'sekalii':'sekali','skali':'sekali',
    'memuaskan':'memuaskan','memuaskn':'memuaskan',
    'mengecewakan':'mengecewakan','mengecewakn':'mengecewakan',
}


# ── Step-by-step Functions ────────────────────────────────────────────────────

def case_folding(text: str) -> str:
    """Step 1: Ubah semua teks ke huruf kecil."""
    if not isinstance(text, str):
        return ""
    return text.lower().strip()


def clean_text(text: str) -> str:
    """Step 2: Hapus noise — URL, mention, hashtag, angka, tanda baca, emoji."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    # Hapus URL
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    # Hapus mention & hashtag
    text = re.sub(r'[@#]\w+', '', text)
    # Hapus emoji & karakter non-ASCII
    text = text.encode('ascii', 'ignore').decode('ascii')
    # Hapus angka
    text = re.sub(r'\d+', '', text)
    # Hapus tanda baca
    text = re.sub(r'[^\w\s]', ' ', text)
    # Hapus spasi berlebih
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def normalize_slang(text: str) -> str:
    """Step 3: Normalisasi kata slang/singkatan ke bentuk baku."""
    tokens = text.split()
    normalized = [SLANG_DICT.get(t, t) for t in tokens]
    return ' '.join(normalized)


def tokenize(text: str) -> list:
    """Step 4: Tokenisasi — pecah teks menjadi list token (kata)."""
    return text.split()


def remove_stopwords(tokens: list) -> list:
    """Step 5: Hapus stopwords dari list token."""
    return [t for t in tokens if t not in STOP_WORDS and len(t) > 2]


def stem_tokens(tokens: list) -> list:
    """Step 6: Stemming setiap token menggunakan PySastrawi."""
    return [_stemmer.stem(t) for t in tokens]


def preprocess(text: str, use_stemming: bool = True) -> str:
    """
    Pipeline preprocessing lengkap.
    Return: string teks bersih siap untuk feature extraction.
    """
    text = case_folding(text)
    text = clean_text(text)
    text = normalize_slang(text)
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    if use_stemming:
        tokens = stem_tokens(tokens)
    return ' '.join(tokens)


def preprocess_steps(text: str, use_stemming: bool = True) -> dict:
    """
    Jalankan preprocessing step-by-step dan kembalikan setiap tahap.
    Berguna untuk visualisasi di dashboard.
    """
    steps = {}
    steps['original']       = text
    steps['case_folding']   = case_folding(text)
    steps['cleaning']       = clean_text(steps['case_folding'])
    steps['normalisasi']    = normalize_slang(steps['cleaning'])
    tokens                  = tokenize(steps['normalisasi'])
    steps['tokenisasi']     = tokens
    tokens_no_sw            = remove_stopwords(tokens)
    steps['stopword_removal'] = tokens_no_sw
    if use_stemming:
        stemmed             = stem_tokens(tokens_no_sw)
        steps['stemming']   = stemmed
        steps['hasil_akhir'] = ' '.join(stemmed)
    else:
        steps['hasil_akhir'] = ' '.join(tokens_no_sw)
    return steps


def label_from_rating(rating: int) -> str:
    """Konversi rating ke label sentimen."""
    if rating <= 2:   return 'negatif'
    elif rating == 3: return 'netral'
    else:             return 'positif'


def load_and_prepare(filepath: str, use_stemming: bool = True,
                     progress_callback=None) -> pd.DataFrame:
    """
    Load dataset, buat label sentimen, dan jalankan preprocessing.

    progress_callback: fungsi(current, total) untuk update progress bar (opsional).
    """
    if filepath.endswith('.csv'):
        df = pd.read_csv(filepath)
    elif filepath.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(filepath)
    else:
        raise ValueError("Format tidak didukung. Gunakan .csv atau .xlsx")

    # Normalisasi kolom
    df.columns = df.columns.str.lower().str.strip().str.replace(' ', '_')

    # Hapus baris kosong
    df = df.dropna(subset=['review_text', 'rating'])
    df = df[df['review_text'].astype(str).str.strip() != '']
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    df = df.dropna(subset=['rating'])
    df['rating'] = df['rating'].astype(int)

    # Label sentimen
    df['sentiment'] = df['rating'].apply(label_from_rating)

    # Preprocessing teks
    total = len(df)
    clean_texts = []
    for i, text in enumerate(df['review_text'].astype(str)):
        clean_texts.append(preprocess(text, use_stemming=use_stemming))
        if progress_callback and i % 100 == 0:
            progress_callback(i, total)

    df['clean_text'] = clean_texts
    df = df[df['clean_text'].str.strip() != '']

    return df


if __name__ == "__main__":
    contoh = "Produk ini SANGAT bagus bgt!! Pengiriman cepet banget, recommended deh 👍"
    print("=== Demo Preprocessing Pipeline ===\n")
    steps = preprocess_steps(contoh)
    for step, result in steps.items():
        print(f"[{step}]\n  {result}\n")

"""
eda.py
Exploratory Data Analysis (EDA) untuk dataset ulasan produk.
Jalankan ini sebelum training untuk memahami data kamu.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
from preprocessing import load_and_prepare, clean_text

matplotlib.rcParams['font.family'] = 'DejaVu Sans'
os.makedirs('../reports', exist_ok=True)

DATASET_PATH = '../data/dataset.csv'   # Ganti sesuai nama file kamu


def plot_rating_distribution(df):
    """Distribusi rating."""
    plt.figure(figsize=(7, 4))
    colors = ['#e74c3c', '#e67e22', '#f1c40f', '#2ecc71', '#27ae60']
    df['rating'].value_counts().sort_index().plot(
        kind='bar', color=colors, edgecolor='black'
    )
    plt.title('Distribusi Rating')
    plt.xlabel('Rating')
    plt.ylabel('Jumlah Ulasan')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig('../reports/distribusi_rating.png', dpi=150)
    plt.show()


def plot_sentiment_distribution(df):
    """Distribusi sentimen."""
    plt.figure(figsize=(6, 4))
    order = ['positif', 'netral', 'negatif']
    colors = ['#2ecc71', '#f39c12', '#e74c3c']
    counts = df['sentiment'].value_counts().reindex(order, fill_value=0)
    counts.plot(kind='bar', color=colors, edgecolor='black')
    plt.title('Distribusi Sentimen')
    plt.xlabel('Sentimen')
    plt.ylabel('Jumlah')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig('../reports/distribusi_sentimen.png', dpi=150)
    plt.show()


def plot_review_length(df):
    """Distribusi panjang teks ulasan."""
    df['panjang_review'] = df['review_text'].apply(lambda x: len(str(x).split()))

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    df['panjang_review'].hist(bins=50, color='steelblue', edgecolor='black')
    plt.title('Distribusi Panjang Review (kata)')
    plt.xlabel('Jumlah Kata')
    plt.ylabel('Frekuensi')

    plt.subplot(1, 2, 2)
    df.boxplot(column='panjang_review', by='sentiment',
               positions=[0, 1, 2], patch_artist=True)
    plt.title('Panjang Review per Sentimen')
    plt.suptitle('')
    plt.xlabel('Sentimen')
    plt.ylabel('Jumlah Kata')

    plt.tight_layout()
    plt.savefig('../reports/panjang_review.png', dpi=150)
    plt.show()

    print(f"Rata-rata panjang review : {df['panjang_review'].mean():.1f} kata")
    print(f"Median panjang review    : {df['panjang_review'].median():.1f} kata")
    print(f"Review terpendek         : {df['panjang_review'].min()} kata")
    print(f"Review terpanjang        : {df['panjang_review'].max()} kata\n")


def plot_top_words(df, sentiment: str, top_n: int = 20):
    """Kata-kata paling sering muncul per sentimen."""
    texts = df[df['sentiment'] == sentiment]['clean_text'].dropna()
    all_words = ' '.join(texts).split()
    word_freq = Counter(all_words).most_common(top_n)

    words, counts = zip(*word_freq)
    plt.figure(figsize=(10, 5))
    color_map = {'positif': '#2ecc71', 'netral': '#f39c12', 'negatif': '#e74c3c'}
    plt.barh(words[::-1], counts[::-1], color=color_map.get(sentiment, 'steelblue'))
    plt.title(f'Top {top_n} Kata - Sentimen {sentiment.capitalize()}')
    plt.xlabel('Frekuensi')
    plt.tight_layout()
    plt.savefig(f'../reports/top_kata_{sentiment}.png', dpi=150)
    plt.show()


def generate_wordcloud(df, sentiment: str):
    """Word cloud per sentimen."""
    texts = df[df['sentiment'] == sentiment]['clean_text'].dropna()
    text_gabung = ' '.join(texts)

    if not text_gabung.strip():
        print(f"Tidak ada teks untuk sentimen: {sentiment}")
        return

    color_map = {'positif': 'Greens', 'netral': 'Oranges', 'negatif': 'Reds'}
    wc = WordCloud(
        width=800, height=400,
        background_color='white',
        colormap=color_map.get(sentiment, 'Blues'),
        max_words=100
    ).generate(text_gabung)

    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Word Cloud - Sentimen {sentiment.capitalize()}')
    plt.tight_layout()
    plt.savefig(f'../reports/wordcloud_{sentiment}.png', dpi=150)
    plt.show()


def plot_top_brands(df, top_n: int = 10):
    """Brand dengan ulasan terbanyak."""
    if 'brand' not in df.columns:
        return
    plt.figure(figsize=(8, 5))
    df['brand'].value_counts().head(top_n).plot(
        kind='barh', color='steelblue', edgecolor='black'
    )
    plt.title(f'Top {top_n} Brand dengan Ulasan Terbanyak')
    plt.xlabel('Jumlah Ulasan')
    plt.tight_layout()
    plt.savefig('../reports/top_brand.png', dpi=150)
    plt.show()


def plot_sentiment_per_category(df):
    """Distribusi sentimen per kategori produk."""
    if 'category' not in df.columns:
        return
    top_cat = df['category'].value_counts().head(8).index
    df_top = df[df['category'].isin(top_cat)]

    pivot = df_top.groupby(['category', 'sentiment']).size().unstack(fill_value=0)
    pivot = pivot.reindex(columns=['positif', 'netral', 'negatif'], fill_value=0)

    pivot.plot(kind='bar', figsize=(12, 5), color=['#2ecc71', '#f39c12', '#e74c3c'],
               edgecolor='black')
    plt.title('Distribusi Sentimen per Kategori')
    plt.xlabel('Kategori')
    plt.ylabel('Jumlah')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Sentimen')
    plt.tight_layout()
    plt.savefig('../reports/sentimen_per_kategori.png', dpi=150)
    plt.show()


def main():
    print("=" * 50)
    print("  EXPLORATORY DATA ANALYSIS")
    print("=" * 50)

    if not os.path.exists(DATASET_PATH):
        print(f"[ERROR] File tidak ditemukan: {DATASET_PATH}")
        return

    # Load data (tanpa stemming untuk EDA agar lebih cepat)
    df = load_and_prepare(DATASET_PATH, use_stemming=False)

    print("\n--- Info Dataset ---")
    print(df[['product_name', 'brand', 'category', 'rating', 'sentiment']].describe(include='all'))

    # Plot
    plot_rating_distribution(df)
    plot_sentiment_distribution(df)
    plot_review_length(df)
    plot_top_brands(df)
    plot_sentiment_per_category(df)

    # Analisis per sentimen
    for sent in ['positif', 'netral', 'negatif']:
        if sent in df['sentiment'].values:
            plot_top_words(df, sent)
            generate_wordcloud(df, sent)

    print("\nEDA selesai! Semua grafik disimpan di folder '../reports/'")


if __name__ == "__main__":
    main()

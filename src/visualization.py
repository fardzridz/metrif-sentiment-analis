"""Helper visualisasi yang dipakai banyak halaman dashboard."""

import matplotlib.pyplot as plt
from wordcloud import WordCloud


def make_wordcloud(texts, colormap="Blues"):
    text = " ".join(texts)
    if not text.strip():
        return None
    wc = WordCloud(
        width=800,
        height=350,
        background_color="white",
        colormap=colormap,
        max_words=80,
    ).generate(text)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    plt.tight_layout(pad=0)
    return fig


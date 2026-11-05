"""
ADIM 3 — Sözcük Sıklığı & N-gram Analizi
==========================================
Girdi : data/trump_only.txt
Çıktı : output/tables/word_freq.csv
         output/tables/bigrams.csv
         output/tables/trigrams.csv
         output/figures/wordcloud.png
         output/figures/ngrams.png

Çalıştır: python scripts/03_word_freq.py
"""

import re
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from collections import Counter
from wordcloud import WordCloud
import nltk
from nltk.util import ngrams
from nltk.corpus import stopwords

nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
FIG_DIR  = BASE_DIR / "output" / "figures"
TAB_DIR  = BASE_DIR / "output" / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)

# İngilizce stopwords + özel eklemeler
STOP_WORDS = set(stopwords.words("english")) | {
    "would", "said", "well", "know", "think", "going", "got", "say",
    "look", "right", "mean", "yeah", "okay", "like", "just", "lot",
    "thing", "things", "people", "get", "go", "want", "way", "good",
    "come", "made", "make", "really", "much", "also", "something",
    "never", "every", "many", "one", "two", "three", "us", "let",
}

# Konuşma dili normalizasyonu
CONTRACTIONS = {
    r"\bdon't\b": "do not", r"\bdoesn't\b": "does not",
    r"\bcan't\b":  "cannot",  r"\bwon't\b":  "will not",
    r"\bit's\b":   "it is",   r"\bthat's\b": "that is",
    r"\bI'm\b":    "I am",    r"\bthey're\b":"they are",
    r"\bwe're\b":  "we are",  r"\byou're\b": "you are",
    r"\bgonna\b":  "going to",r"\bwanna\b":  "want to",
    r"\bgotta\b":  "got to",  r"\bkinda\b":  "kind of",
}


def normalize(text: str) -> str:
    for pattern, replacement in CONTRACTIONS.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.lower().strip()


def tokenize(text: str, remove_stops: bool = True) -> list[str]:
    tokens = nltk.word_tokenize(normalize(text))
    if remove_stops:
        tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 2]
    return tokens


def get_ngrams(tokens: list[str], n: int) -> Counter:
    return Counter([" ".join(gram) for gram in ngrams(tokens, n)])


def main():
    trump_text = (DATA_DIR / "trump_only.txt").read_text(encoding="utf-8")

    print("Tokenize ediliyor...")
    tokens_clean  = tokenize(trump_text, remove_stops=True)
    tokens_all    = tokenize(trump_text, remove_stops=False)

    print(f"  Toplam token (temiz): {len(tokens_clean):,}")
    print(f"  Benzersiz kelime    : {len(set(tokens_clean)):,}")

    # ── Frekans ───────────────────────────────────────────────────────────────
    freq     = Counter(tokens_clean)
    bigrams  = get_ngrams(tokens_all, 2)
    trigrams = get_ngrams(tokens_all, 3)

    # Bigram/trigram için anlamsız kombinasyonları filtrele
    def filter_ngram(counter, stop_words, min_count=5):
        filtered = {}
        for gram, count in counter.items():
            parts = gram.split()
            if all(p not in stop_words for p in parts) and count >= min_count:
                filtered[gram] = count
        return Counter(filtered)

    bigrams_clean  = filter_ngram(bigrams,  STOP_WORDS, min_count=3)
    trigrams_clean = filter_ngram(trigrams, STOP_WORDS, min_count=2)

    # CSV'ye kaydet
    top_words = pd.DataFrame(freq.most_common(200), columns=["word", "count"])
    top_words["rank"] = range(1, len(top_words) + 1)
    top_words.to_csv(TAB_DIR / "word_freq.csv", index=False)

    top_bi = pd.DataFrame(bigrams_clean.most_common(100), columns=["bigram", "count"])
    top_bi.to_csv(TAB_DIR / "bigrams.csv", index=False)

    top_tri = pd.DataFrame(trigrams_clean.most_common(100), columns=["trigram", "count"])
    top_tri.to_csv(TAB_DIR / "trigrams.csv", index=False)

    print(f"\nTop 20 kelime:")
    for word, count in freq.most_common(20):
        print(f"  {word:<20} {count:>5}")

    print(f"\nTop 10 bigram:")
    for gram, count in bigrams_clean.most_common(10):
        print(f"  {gram:<30} {count:>5}")

    print(f"\nTop 10 trigram:")
    for gram, count in trigrams_clean.most_common(10):
        print(f"  {gram:<40} {count:>5}")

    # ── WordCloud ─────────────────────────────────────────────────────────────
    print("\nWordCloud oluşturuluyor...")
    wc = WordCloud(
        width=1600, height=800,
        background_color="white",
        colormap="Blues",
        max_words=150,
        collocations=False,
    ).generate_from_frequencies(freq)

    fig, ax = plt.subplots(figsize=(16, 8))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title("Trump Interviews — Most Frequent Words (stopwords removed)",
                 fontsize=16, fontweight="bold", pad=20)
    fig.savefig(FIG_DIR / "wordcloud.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"WordCloud kaydedildi: {FIG_DIR / 'wordcloud.png'}")

    # ── N-gram Grafikleri ─────────────────────────────────────────────────────
    print("N-gram grafikleri oluşturuluyor...")
    sns.set_theme(style="whitegrid")
    fig = plt.figure(figsize=(20, 12))
    gs  = gridspec.GridSpec(1, 3, figure=fig)

    # Top 25 kelime
    ax1 = fig.add_subplot(gs[0])
    top25 = pd.DataFrame(freq.most_common(25), columns=["word", "count"])
    sns.barplot(data=top25, y="word", x="count", palette="Blues_d", ax=ax1)
    ax1.set_title("Top 25 Words", fontweight="bold")
    ax1.set_xlabel("Count"); ax1.set_ylabel("")

    # Top 20 bigram
    ax2 = fig.add_subplot(gs[1])
    top20bi = pd.DataFrame(bigrams_clean.most_common(20), columns=["bigram", "count"])
    sns.barplot(data=top20bi, y="bigram", x="count", palette="Oranges_d", ax=ax2)
    ax2.set_title("Top 20 Bigrams", fontweight="bold")
    ax2.set_xlabel("Count"); ax2.set_ylabel("")

    # Top 15 trigram
    ax3 = fig.add_subplot(gs[2])
    top15tri = pd.DataFrame(trigrams_clean.most_common(15), columns=["trigram", "count"])
    sns.barplot(data=top15tri, y="trigram", x="count", palette="Greens_d", ax=ax3)
    ax3.set_title("Top 15 Trigrams", fontweight="bold")
    ax3.set_xlabel("Count"); ax3.set_ylabel("")

    fig.suptitle("Trump Speech — Word & N-gram Frequency Analysis", fontsize=15, fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "ngrams.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"N-gram grafiği kaydedildi: {FIG_DIR / 'ngrams.png'}")


if __name__ == "__main__":
    main()

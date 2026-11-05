"""
ADIM 2 — Corpus Tanımlayıcı İstatistikler
==========================================
Girdi : data/interviews_parsed.json
Çıktı : output/tables/corpus_stats.csv
         output/figures/corpus_overview.png

Çalıştır: python scripts/02_corpus_stats.py
"""

import json
import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import textstat
import nltk
from pathlib import Path
from collections import Counter

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

BASE_DIR    = Path(__file__).parent.parent
DATA_DIR    = BASE_DIR / "data"
FIG_DIR     = BASE_DIR / "output" / "figures"
TAB_DIR     = BASE_DIR / "output" / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)

# ── Yardımcı ─────────────────────────────────────────────────────────────────

def type_token_ratio(text: str) -> float:
    tokens = text.lower().split()
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def avg_sentence_length(text: str) -> float:
    sentences = nltk.sent_tokenize(text)
    if not sentences:
        return 0.0
    lengths = [len(s.split()) for s in sentences]
    return sum(lengths) / len(lengths)


def hapax_ratio(text: str) -> float:
    tokens = text.lower().split()
    freq   = Counter(tokens)
    hapax  = sum(1 for v in freq.values() if v == 1)
    return hapax / len(tokens) if tokens else 0.0


# ── Ana akış ─────────────────────────────────────────────────────────────────

def main():
    interviews = json.loads((DATA_DIR / "interviews_parsed.json").read_text(encoding="utf-8"))

    rows = []
    for iv in interviews:
        trump_text = " ".join(t["text"] for t in iv["trump_turns"])
        if not trump_text.strip():
            continue

        row = {
            "id":                  int(iv["id"]),
            "slug":                iv["slug"],
            "title":               iv["title"][:50],
            "date":                iv["date"],
            "trump_words":         iv["trump_word_count"],
            "interviewer_words":   iv["interviewer_word_count"],
            "total_turns":         iv["total_turns"],
            "trump_turns":         len(iv["trump_turns"]),
            "trump_talk_ratio":    round(
                iv["trump_word_count"] /
                max(iv["trump_word_count"] + iv["interviewer_word_count"], 1) * 100, 1
            ),
            "ttr":                 round(type_token_ratio(trump_text), 4),
            "avg_sent_len":        round(avg_sentence_length(trump_text), 2),
            "hapax_ratio":         round(hapax_ratio(trump_text), 4),
            "flesch_reading_ease": round(textstat.flesch_reading_ease(trump_text), 2),
            "flesch_kincaid_grade":round(textstat.flesch_kincaid_grade(trump_text), 2),
            "gunning_fog":         round(textstat.gunning_fog(trump_text), 2),
        }
        rows.append(row)
        print(f"  [{row['id']:>2}] {row['slug']:<30} TTR={row['ttr']:.3f}  FK={row['flesch_kincaid_grade']}")

    df = pd.DataFrame(rows).sort_values("id")
    df.to_csv(TAB_DIR / "corpus_stats.csv", index=False)
    print(f"\nCSV kaydedildi: {TAB_DIR / 'corpus_stats.csv'}")

    # ── Görselleştirme ────────────────────────────────────────────────────────
    sns.set_theme(style="whitegrid", palette="muted")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Trump Interview Corpus — Descriptive Statistics", fontsize=16, fontweight="bold")

    labels = [s.replace("_", "\n") for s in df["slug"]]

    # 1. Kelime sayısı
    ax = axes[0, 0]
    bars = ax.barh(labels, df["trump_words"], color=sns.color_palette("Blues_d", len(df)))
    ax.set_title("Trump Word Count per Interview")
    ax.set_xlabel("Words")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    # 2. Konuşma oranı
    ax = axes[0, 1]
    ax.barh(labels, df["trump_talk_ratio"], color=sns.color_palette("Oranges_d", len(df)))
    ax.set_title("Trump Talk Ratio (%)")
    ax.set_xlabel("% of total words")
    ax.axvline(50, color="red", linestyle="--", alpha=0.5, label="50%")
    ax.legend()

    # 3. TTR
    ax = axes[0, 2]
    ax.barh(labels, df["ttr"], color=sns.color_palette("Greens_d", len(df)))
    ax.set_title("Type-Token Ratio (Lexical Diversity)")
    ax.set_xlabel("TTR")

    # 4. Ortalama cümle uzunluğu
    ax = axes[1, 0]
    ax.barh(labels, df["avg_sent_len"], color=sns.color_palette("Purples_d", len(df)))
    ax.set_title("Avg. Sentence Length (Words)")
    ax.set_xlabel("Words per sentence")

    # 5. Flesch-Kincaid Grade
    ax = axes[1, 1]
    ax.barh(labels, df["flesch_kincaid_grade"], color=sns.color_palette("Reds_d", len(df)))
    ax.set_title("Flesch-Kincaid Grade Level")
    ax.set_xlabel("Grade level")

    # 6. Hapax ratio
    ax = axes[1, 2]
    ax.barh(labels, df["hapax_ratio"], color=sns.color_palette("YlOrBr_d", len(df)))
    ax.set_title("Hapax Legomena Ratio")
    ax.set_xlabel("Ratio")

    plt.tight_layout()
    fig.savefig(FIG_DIR / "corpus_overview.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Grafik kaydedildi: {FIG_DIR / 'corpus_overview.png'}")

    # Genel özet
    print("\n── Genel Corpus Özeti ──────────────────────────────")
    print(f"Toplam röportaj          : {len(df)}")
    print(f"Trump toplam kelime      : {df['trump_words'].sum():,}")
    print(f"Ortalama TTR             : {df['ttr'].mean():.4f}")
    print(f"Ortalama FK Grade        : {df['flesch_kincaid_grade'].mean():.2f}")
    print(f"Ortalama cümle uzunluğu  : {df['avg_sent_len'].mean():.2f}")
    print(f"Ortalama konuşma oranı   : %{df['trump_talk_ratio'].mean():.1f}")


if __name__ == "__main__":
    main()

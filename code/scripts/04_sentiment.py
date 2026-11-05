"""
ADIM 4 — Duygu Analizi (Sentiment Analysis)
============================================
Girdi : data/interviews_parsed.json
Çıktı : output/tables/sentiment_by_interview.csv
         output/tables/sentiment_by_topic.csv
         output/figures/sentiment_overview.png
         output/figures/sentiment_heatmap.png

Çalıştır: python scripts/04_sentiment.py
"""

import json
import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import nltk
from pathlib import Path
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from tqdm import tqdm

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
FIG_DIR  = BASE_DIR / "output" / "figures"
TAB_DIR  = BASE_DIR / "output" / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)

# Konu anahtar kelimeleri — her cümle bu kelimeler varsa o konuya atanır
TOPIC_KEYWORDS = {
    "economy":    ["economy", "jobs", "trade", "tariff", "tax", "gdp", "stock", "market",
                   "inflation", "business", "deal", "money", "billion", "million"],
    "opponents":  ["biden", "harris", "pelosi", "democrats", "radical", "left",
                   "obama", "clinton", "fake news", "media", "crooked"],
    "covid":      ["covid", "coronavirus", "pandemic", "vaccine", "fauci",
                   "lockdown", "mask", "hydroxychloroquine"],
    "elections":  ["election", "vote", "ballot", "fraud", "rigged", "steal",
                   "supreme court", "november", "campaign"],
    "foreign":    ["china", "russia", "iran", "nato", "ukraine", "north korea",
                   "border", "mexico", "immigration", "military"],
    "self":       ["i won", "i did", "great job", "best president", "nobody",
                   "greatest", "tremendous", "unbelievable", "beautiful"],
}


def classify_topic(text: str) -> str:
    text_lower = text.lower()
    scores = {}
    for topic, keywords in TOPIC_KEYWORDS.items():
        scores[topic] = sum(1 for kw in keywords if kw in text_lower)
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "other"


def analyze_turns(turns: list[dict], analyzer: SentimentIntensityAnalyzer) -> list[dict]:
    results = []
    for turn in turns:
        sentences = nltk.sent_tokenize(turn["text"])
        for sent in sentences:
            if len(sent.split()) < 4:
                continue
            scores = analyzer.polarity_scores(sent)
            results.append({
                "speaker":   turn["speaker"],
                "is_trump":  turn["is_trump"],
                "sentence":  sent,
                "topic":     classify_topic(sent),
                "compound":  scores["compound"],
                "positive":  scores["pos"],
                "negative":  scores["neg"],
                "neutral":   scores["neu"],
                "label":     "positive" if scores["compound"] >= 0.05
                             else "negative" if scores["compound"] <= -0.05
                             else "neutral",
            })
    return results


def main():
    interviews = json.loads((DATA_DIR / "interviews_parsed.json").read_text(encoding="utf-8"))
    analyzer   = SentimentIntensityAnalyzer()

    all_sentences = []
    interview_rows = []

    print("Duygu analizi yapılıyor...")
    for iv in tqdm(interviews):
        trump_turns = iv["trump_turns"]
        if not trump_turns:
            continue

        sents = analyze_turns(trump_turns, analyzer)
        for s in sents:
            s["interview_id"]   = iv["id"]
            s["interview_slug"] = iv["slug"]
        all_sentences.extend(sents)

        trump_sents = [s for s in sents if s["is_trump"]]
        if trump_sents:
            avg_compound = sum(s["compound"] for s in trump_sents) / len(trump_sents)
            pos_ratio    = sum(1 for s in trump_sents if s["label"] == "positive") / len(trump_sents)
            neg_ratio    = sum(1 for s in trump_sents if s["label"] == "negative") / len(trump_sents)
            interview_rows.append({
                "id":           iv["id"],
                "slug":         iv["slug"],
                "date":         iv["date"],
                "avg_compound": round(avg_compound, 4),
                "pos_ratio":    round(pos_ratio, 4),
                "neg_ratio":    round(neg_ratio, 4),
                "n_sentences":  len(trump_sents),
            })

    df_sents = pd.DataFrame(all_sentences)
    df_iv    = pd.DataFrame(interview_rows).sort_values("id")

    # Konu bazlı sentiment
    df_topic = (
        df_sents[df_sents["is_trump"]]
        .groupby("topic")
        .agg(
            avg_compound=("compound", "mean"),
            pos_ratio=("label", lambda x: (x == "positive").mean()),
            neg_ratio=("label", lambda x: (x == "negative").mean()),
            n_sentences=("sentence", "count"),
        )
        .reset_index()
        .sort_values("avg_compound")
    )

    df_iv.to_csv(TAB_DIR / "sentiment_by_interview.csv", index=False)
    df_topic.to_csv(TAB_DIR / "sentiment_by_topic.csv", index=False)
    print(f"\nCSV'ler kaydedildi.")

    print("\nRöportaj bazlı sentiment:")
    print(df_iv[["slug", "avg_compound", "pos_ratio", "neg_ratio"]].to_string(index=False))

    print("\nKonu bazlı sentiment:")
    print(df_topic.to_string(index=False))

    # ── Görselleştirme 1 — Röportaj bazlı ───────────────────────────────────
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle("Trump Speech — Sentiment Analysis (VADER)", fontsize=15, fontweight="bold")

    # Compound score bar
    ax = axes[0]
    colors = ["#e74c3c" if v < 0 else "#2ecc71" for v in df_iv["avg_compound"]]
    slugs  = [s.replace("_", "\n") for s in df_iv["slug"]]
    bars   = ax.barh(slugs, df_iv["avg_compound"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Average Compound Sentiment per Interview")
    ax.set_xlabel("Compound Score (−1 negative → +1 positive)")

    # Stacked pos/neg/neu
    ax = axes[1]
    neu_ratio = 1 - df_iv["pos_ratio"] - df_iv["neg_ratio"]
    ax.barh(slugs, df_iv["pos_ratio"],  color="#2ecc71", label="Positive")
    ax.barh(slugs, neu_ratio,           color="#95a5a6", label="Neutral",
            left=df_iv["pos_ratio"])
    ax.barh(slugs, df_iv["neg_ratio"],  color="#e74c3c", label="Negative",
            left=df_iv["pos_ratio"] + neu_ratio)
    ax.set_title("Sentiment Distribution per Interview")
    ax.set_xlabel("Proportion")
    ax.legend(loc="lower right")

    plt.tight_layout()
    fig.savefig(FIG_DIR / "sentiment_overview.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Grafik kaydedildi: {FIG_DIR / 'sentiment_overview.png'}")

    # ── Görselleştirme 2 — Heatmap (konu × röportaj) ────────────────────────
    trump_sents_df = df_sents[df_sents["is_trump"]].copy()
    pivot = trump_sents_df.pivot_table(
        index="topic", columns="interview_slug",
        values="compound", aggfunc="mean"
    )

    fig, ax = plt.subplots(figsize=(18, 6))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn",
                center=0, linewidths=0.5, ax=ax,
                cbar_kws={"label": "Avg Compound Score"})
    ax.set_title("Sentiment Heatmap — Topic × Interview", fontsize=14, fontweight="bold")
    ax.set_xlabel(""); ax.set_ylabel("Topic")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "sentiment_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Heatmap kaydedildi: {FIG_DIR / 'sentiment_heatmap.png'}")


if __name__ == "__main__":
    main()

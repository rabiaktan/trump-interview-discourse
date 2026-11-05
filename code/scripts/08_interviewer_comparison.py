"""
ADIM 8 — Röportajcıya Göre Farklılık Analizi
=============================================
Girdi : data/interviews_parsed.json
Çıktı : output/tables/interviewer_comparison.csv
         output/figures/interviewer_comparison.png
         output/figures/audience_design.png

Çalıştır: python scripts/08_interviewer_comparison.py
"""

import json
import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import nltk
import textstat
from pathlib import Path
from collections import Counter
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
FIG_DIR  = BASE_DIR / "output" / "figures"
TAB_DIR  = BASE_DIR / "output" / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)

# Röportajcı profilleri (ideolojik/atmosfer etiket)
import sys
sys.path.insert(0, str(Path(__file__).parent))
from corpus_config import META  # noqa: E402
INTERVIEWER_PROFILE = {s: (m["interviewer"], m["profile"]) for s, m in META.items()}

analyzer = SentimentIntensityAnalyzer()

AMPLIFIERS = [
    "very", "hugely", "totally", "absolutely", "completely", "incredibly",
    "tremendously", "strongly", "highly", "extremely", "beautiful",
    "perfect", "great", "amazing", "wonderful", "fantastic", "tremendous",
]

HEDGES = ["maybe", "perhaps", "probably", "possibly", "i think", "i believe",
          "sort of", "kind of", "i guess"]


def type_token_ratio(text: str) -> float:
    tokens = text.lower().split()
    return len(set(tokens)) / len(tokens) if tokens else 0.0


def avg_sentiment(text: str) -> float:
    sentences = nltk.sent_tokenize(text)
    if not sentences:
        return 0.0
    scores = [analyzer.polarity_scores(s)["compound"] for s in sentences]
    return sum(scores) / len(scores)


def interrupt_rate(turns: list[dict]) -> float:
    """Kısa turların oranı (10 kelimeden az = muhtemel kesinti)"""
    trump_turns = [t for t in turns if t["is_trump"]]
    if not trump_turns:
        return 0.0
    short = sum(1 for t in trump_turns if t["word_count"] < 10)
    return short / len(trump_turns)


def rate_per_1k(text: str, words: list[str]) -> float:
    tokens = text.lower().split()
    count  = sum(1 for t in tokens if t in words)
    return round(count / max(len(tokens), 1) * 1000, 2)


def analyze(iv: dict) -> dict:
    slug        = iv["slug"]
    interviewer, profile = INTERVIEWER_PROFILE.get(slug, ("Unknown", "neutral"))
    trump_turns = iv["trump_turns"]
    all_turns   = iv["turns"]

    trump_text  = " ".join(t["text"] for t in trump_turns)
    inter_text  = " ".join(t["text"] for t in all_turns if not t["is_trump"])

    if not trump_text.strip():
        return None

    # Söylem uyumu: Trump'ın interviewer'ın dil özelliklerini yansıtıp yansıtmadığı
    # (basit: sentiment yakınlığı)
    trump_sent  = avg_sentiment(trump_text)
    inter_sent  = avg_sentiment(inter_text) if inter_text else 0.0
    sent_align  = 1 - abs(trump_sent - inter_sent)  # 1'e yakın = yüksek uyum

    return {
        "id":                   iv["id"],
        "slug":                 slug,
        "interviewer":          interviewer,
        "profile":              profile,
        "trump_words":          iv["trump_word_count"],
        "interviewer_words":    iv["interviewer_word_count"],
        "trump_talk_ratio":     round(iv["trump_word_count"] /
                                max(iv["trump_word_count"] + iv["interviewer_word_count"], 1) * 100, 1),
        "avg_trump_sentiment":  round(trump_sent, 4),
        "avg_inter_sentiment":  round(inter_sent, 4),
        "sentiment_alignment":  round(sent_align, 4),
        "ttr":                  round(type_token_ratio(trump_text), 4),
        "fk_grade":             round(textstat.flesch_kincaid_grade(trump_text), 2),
        "avg_sent_len":         round(sum(len(s.split()) for s in nltk.sent_tokenize(trump_text)) /
                                max(len(nltk.sent_tokenize(trump_text)), 1), 2),
        # amplifier/hedge oranları 06_rhetoric.py çıktısından alınır (tek kaynak)
        "amplifier_rate":       RHET.loc[slug, "amplifier_rate"],
        "hedge_rate":           RHET.loc[slug, "hedge_rate"],
        "interrupt_rate":       round(interrupt_rate(all_turns), 4),
        "total_turns":          len(all_turns),
    }


def main():
    global RHET
    RHET = pd.read_csv(TAB_DIR / "rhetoric_features.csv").set_index("slug")
    interviews = json.loads((DATA_DIR / "interviews_parsed.json").read_text(encoding="utf-8"))

    rows = []
    for iv in interviews:
        row = analyze(iv)
        if row:
            rows.append(row)
            print(f"  [{row['id']:>2}] {row['slug']:<28} {row['profile']:<12} "
                  f"talk={row['trump_talk_ratio']:>5}%  sent={row['avg_trump_sentiment']:>6.3f}")

    df = pd.DataFrame(rows).sort_values("id")
    df.to_csv(TAB_DIR / "interviewer_comparison.csv", index=False)
    print(f"\nCSV kaydedildi: {TAB_DIR / 'interviewer_comparison.csv'}")

    # ── Profil bazlı ortalamalar ──────────────────────────────────────────────
    profile_avg = df.groupby("profile").agg(
        avg_trump_words=("trump_words", "mean"),
        avg_talk_ratio=("trump_talk_ratio", "mean"),
        avg_sentiment=("avg_trump_sentiment", "mean"),
        avg_ttr=("ttr", "mean"),
        avg_fk=("fk_grade", "mean"),
        avg_amplifier=("amplifier_rate", "mean"),
        avg_hedge=("hedge_rate", "mean"),
        avg_sent_align=("sentiment_alignment", "mean"),
        n=("slug", "count"),
    ).reset_index()

    print("\n── Profil Bazlı Ortalamalar ────────────────────────")
    print(profile_avg.to_string(index=False))

    # ── Görselleştirme 1 — Temel metrikler × profil ───────────────────────────
    sns.set_theme(style="whitegrid")
    palette = {"supportive": "#2ecc71", "neutral": "#3498db", "adversarial": "#e74c3c"}

    metrics = ["trump_talk_ratio", "avg_trump_sentiment", "ttr",
               "fk_grade", "amplifier_rate", "hedge_rate"]
    titles  = ["Talk Ratio (%)", "Avg Sentiment", "Lexical Diversity (TTR)",
               "FK Grade Level", "Amplifier Rate (/1k)", "Hedge Rate (/1k)"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Trump's Speech Variation by Interviewer Profile",
                 fontsize=15, fontweight="bold")

    for ax, metric, title in zip(axes.flat, metrics, titles):
        sns.boxplot(data=df, x="profile", y=metric,
                    palette=palette, order=["supportive", "neutral", "adversarial"], ax=ax)
        sns.stripplot(data=df, x="profile", y=metric,
                      color="black", size=5, alpha=0.6,
                      order=["supportive", "neutral", "adversarial"], ax=ax)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Interviewer Profile")
        ax.set_ylabel("")
        if "sentiment" in metric:
            ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")

    plt.tight_layout()
    fig.savefig(FIG_DIR / "interviewer_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Grafik kaydedildi: {FIG_DIR / 'interviewer_comparison.png'}")

    # ── Görselleştirme 2 — Radar chart (audience design) ─────────────────────
    profile_order = ["supportive", "neutral", "adversarial"]
    radar_metrics = ["avg_talk_ratio", "avg_sentiment", "avg_ttr",
                     "avg_amplifier", "avg_hedge", "avg_sent_align"]
    radar_labels  = ["Talk Ratio", "Sentiment", "TTR", "Amplifiers", "Hedges", "Sent. Align."]

    # Normalize 0–1
    pa = profile_avg.set_index("profile")[radar_metrics]
    pa_norm = (pa - pa.min()) / (pa.max() - pa.min() + 1e-9)

    N     = len(radar_labels)
    angles= np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})
    colors  = [palette[p] for p in profile_order]

    for profile, color in zip(profile_order, colors):
        if profile not in pa_norm.index:
            continue
        values = pa_norm.loc[profile].tolist() + pa_norm.loc[profile].tolist()[:1]
        ax.plot(angles, values, color=color, linewidth=2, label=profile)
        ax.fill(angles, values, color=color, alpha=0.15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(radar_labels, fontsize=11)
    ax.set_title("Audience Design — Trump's Speech\nby Interviewer Profile",
                 fontsize=13, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    plt.tight_layout()
    fig.savefig(FIG_DIR / "audience_design.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Radar grafik kaydedildi: {FIG_DIR / 'audience_design.png'}")


if __name__ == "__main__":
    main()

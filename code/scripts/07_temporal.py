"""
ADIM 7 — Zamansal Değişim Analizi (2020 → 2025)
=================================================
Girdi : data/interviews_parsed.json
         output/tables/corpus_stats.csv
         output/tables/sentiment_by_interview.csv
         output/tables/rhetoric_features.csv
Çıktı : output/figures/temporal_trends.png
         output/tables/temporal_summary.csv

Çalıştır: python scripts/07_temporal.py
"""

import json
import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import numpy as np
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
FIG_DIR  = BASE_DIR / "output" / "figures"
TAB_DIR  = BASE_DIR / "output" / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Röportaj tarihlerini manuel eşle (dosyada tarih boş olabilir) ─────────────
# Format: slug → "YYYY-MM-DD"
import sys
sys.path.insert(0, str(Path(__file__).parent))
from corpus_config import DATES as MANUAL_DATES, LABEL, period as cfg_period  # noqa: E402


def parse_date(slug: str, raw_date: str) -> datetime | None:
    if slug in MANUAL_DATES:
        return datetime.strptime(MANUAL_DATES[slug], "%Y-%m-%d")
    for fmt in ["%Y-%m-%d", "%B %d, %Y", "%b %d, %Y", "%m/%d/%Y"]:
        try:
            return datetime.strptime(raw_date.strip(), fmt)
        except (ValueError, AttributeError):
            continue
    return None


def main():
    interviews  = json.loads((DATA_DIR / "interviews_parsed.json").read_text(encoding="utf-8"))
    df_stats    = pd.read_csv(TAB_DIR / "corpus_stats.csv")
    df_sent     = pd.read_csv(TAB_DIR / "sentiment_by_interview.csv")
    df_rhet     = pd.read_csv(TAB_DIR / "rhetoric_features.csv")

    # Tarih ekle
    date_map = {}
    for iv in interviews:
        dt = parse_date(iv["slug"], iv.get("date", ""))
        if dt:
            date_map[iv["slug"]] = dt

    for df in [df_stats, df_sent, df_rhet]:
        df["date"] = df["slug"].map(date_map)

    df_stats = df_stats.dropna(subset=["date"]).sort_values("date")
    df_sent  = df_sent.dropna(subset=["date"]).sort_values("date")
    df_rhet  = df_rhet.dropna(subset=["date"]).sort_values("date")

    # Birleşik dataframe
    df_merged = df_stats[["slug", "date", "ttr", "avg_sent_len", "flesch_kincaid_grade"]].merge(
        df_sent[["slug", "avg_compound", "pos_ratio", "neg_ratio"]], on="slug"
    ).merge(
        df_rhet[["slug", "amplifier_rate", "hedge_rate", "self_ref_rate", "direct_ratio"]], on="slug"
    )

    df_merged.to_csv(TAB_DIR / "temporal_summary.csv", index=False)
    print(f"CSV kaydedildi: {TAB_DIR / 'temporal_summary.csv'}")

    # Dönem etiketi
    def period(dt):
        return cfg_period(dt.strftime("%Y-%m-%d"))

    df_merged["period"] = df_merged["date"].apply(period)

    # ── Görselleştirme ────────────────────────────────────────────────────────
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(3, 3, figsize=(20, 16))
    fig.suptitle("Trump Speech — Temporal Change Analysis (2020–2025)",
                 fontsize=16, fontweight="bold")

    METRICS = [
        ("ttr",                 "Lexical Diversity (TTR)",          "Blues"),
        ("avg_sent_len",        "Avg Sentence Length (words)",      "Oranges"),
        ("flesch_kincaid_grade","Flesch-Kincaid Grade Level",       "Greens"),
        ("avg_compound",        "Avg Sentiment (Compound)",         "RdYlGn"),
        ("pos_ratio",           "Positive Sentence Ratio",          "Greens"),
        ("neg_ratio",           "Negative Sentence Ratio",          "Reds"),
        ("amplifier_rate",      "Amplifier Rate (/1000w)",          "Purples"),
        ("hedge_rate",          "Hedge Rate (/1000w)",              "Blues"),
        ("direct_ratio",        "Direct Q&A Response Ratio",        "Oranges"),
    ]

    for ax, (metric, title, cmap) in zip(axes.flat, METRICS):
        # Scatter + trend çizgisi
        x_num = mdates.date2num(df_merged["date"])
        y     = df_merged[metric].values

        sc = ax.scatter(df_merged["date"], y,
                        c=x_num, cmap=cmap, s=100, zorder=5, alpha=0.85)

        # Etiketler
        for _, row in df_merged.iterrows():
            ax.annotate(LABEL.get(row["slug"], row["slug"])[:14],
                        (row["date"], row[metric]),
                        fontsize=6, alpha=0.7, xytext=(4, 2),
                        textcoords="offset points")

        # Trend çizgisi KALDIRILDI (hakem önerisi): 16 noktalık OLS doğrusu
        # istatistiksel bir eğilim izlenimi veriyordu. Testler: 09_revizyon_analizleri.py

        ax.set_title(title, fontweight="bold", fontsize=10)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_major_locator(mdates.YearLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

        # Sentiment sıfır çizgisi
        if "compound" in metric:
            ax.axhline(0, color="black", linewidth=0.8, alpha=0.4)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "temporal_trends.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Grafik kaydedildi: {FIG_DIR / 'temporal_trends.png'}")

    # ── Dönem karşılaştırması ─────────────────────────────────────────────────
    period_order = ["2020", "2021", "2023", "2024", "2025"]

    compare_metrics = ["ttr", "avg_compound", "amplifier_rate", "hedge_rate"]
    df_period = df_merged.groupby("period")[compare_metrics].mean().reindex(
        [p for p in period_order if p in df_merged["period"].values]
    )

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle("Period Comparison — Key Metrics", fontsize=14, fontweight="bold")
    titles = ["Lexical Diversity (TTR)", "Avg Sentiment", "Amplifier Rate", "Hedge Rate"]

    for ax, metric, title in zip(axes, compare_metrics, titles):
        ax.bar(range(len(df_period)), df_period[metric],
               color=sns.color_palette("Set2", len(df_period)))
        ax.set_xticks(range(len(df_period)))
        ax.set_xticklabels([p.replace(" (", "\n(") for p in df_period.index],
                           fontsize=8, rotation=30, ha="right")
        ax.set_title(title, fontweight="bold")
        if "compound" in metric:
            ax.axhline(0, color="red", linewidth=0.8, alpha=0.5)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "period_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Dönem karşılaştırması kaydedildi: {FIG_DIR / 'period_comparison.png'}")

    print("\n── Dönem Ortalamaları ──────────────────────────────")
    print(df_period.round(4).to_string())


if __name__ == "__main__":
    main()

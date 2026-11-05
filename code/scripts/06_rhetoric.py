"""
ADIM 6 — Söylem & Retorik Analizi
====================================
Girdi : data/interviews_parsed.json
Çıktı : output/tables/rhetoric_features.csv
         output/figures/rhetoric_overview.png
         output/figures/amplifiers_heatmap.png

Çalıştır: python scripts/06_rhetoric.py
"""

import json
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from pathlib import Path
from collections import Counter
import sys
sys.path.insert(0, str(Path(__file__).parent))
from corpus_config import LABEL, DATES, PROFILE  # noqa: E402

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)
nltk.download("averaged_perceptron_tagger_eng", quiet=True)

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
FIG_DIR  = BASE_DIR / "output" / "figures"
TAB_DIR  = BASE_DIR / "output" / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)

# ── Retorik sözcük listeleri ──────────────────────────────────────────────────

AMPLIFIERS = [
    "very", "hugely", "totally", "absolutely", "completely", "incredibly",
    "tremendously", "strongly", "highly", "extremely", "massively",
    "unbelievably", "fantastic", "tremendous", "incredible", "beautiful",
    "perfect", "great", "amazing", "wonderful", "spectacular",
]

HEDGES = [
    "maybe", "perhaps", "probably", "possibly", "might", "could",
    "i think", "i believe", "i guess", "sort of", "kind of",
    "i suppose", "apparently", "seemingly", "in my opinion",
]

SELF_REFERENCE = [
    r"\bi\b", r"\bme\b", r"\bmy\b", r"\bmyself\b", r"\bmine\b",
]

REPETITION_PATTERNS = [
    r"\b(\w+)\s+\1\b",                          # kelime tekrarı: "very very"
    r"\b(really|very|so)\s+\1\b",               # amplifier tekrarı
]

FIRST_PERSON_PLURAL = [r"\bwe\b", r"\bour\b", r"\bus\b"]

SUPERLATIVES = [
    "best", "greatest", "worst", "biggest", "largest", "most",
    "least", "highest", "lowest", "strongest", "weakest",
]

NEGATION = [
    "never", "nobody", "nothing", "nowhere", "no one",
    "not", "don't", "doesn't", "didn't", "won't", "can't",
    "shouldn't", "wouldn't", "couldn't",
]


def norm(text: str) -> str:
    """Kıvrık kesme işaretlerini düzleştirir, küçük harfe çevirir."""
    return text.replace("\u2019", "'").replace("\u2018", "'").lower()


def phrase_re(phrase: str):
    """Kelime sınırlı eşleşme: 'very' 'every' içinde sayılmaz, 'very,' sayılır."""
    return re.compile(r"(?<![a-z'])" + r"\s+".join(map(re.escape, phrase.split())) + r"(?![a-z'])")


def count_each(text: str, items: list[str]) -> dict:
    t = norm(text)
    return {w: len(phrase_re(w).findall(t)) for w in items}


def count_pattern_list(text: str, patterns: list[str]) -> int:
    count = 0
    text_lower = norm(text)
    for p in patterns:
        count += len(re.findall(p, text_lower))
    return count


def count_words_in_list(text: str, word_list: list[str]) -> int:
    """DÜZELTİLDİ: önceki sürüm boşlukla bölüyordu; 'very,' ve 'don’t' gibi biçimler kaçıyordu."""
    return sum(count_each(text, word_list).values())


def count_question_response(turns: list[dict]) -> dict:
    """
    Gazeteci sorusuna Trump'ın doğrudan mı yoksa konu değiştirerek mi cevap verdiğini
    basit bir heuristic ile ölçer.
    """
    total_responses = 0
    direct_responses = 0

    for i, turn in enumerate(turns):
        if not turn["is_trump"] and "?" in turn["text"]:
            # Bir sonraki Trump turu
            for j in range(i+1, min(i+3, len(turns))):
                if turns[j]["is_trump"]:
                    response = turns[j]["text"].lower()
                    # Sorudaki anahtar kelimeleri çıkar
                    q_words = set(re.findall(r"\b\w{4,}\b", turn["text"].lower())) - {
                        "what", "when", "where", "which", "that", "this", "your", "about"
                    }
                    resp_words = set(re.findall(r"\b\w+\b", response))
                    # Yanıtta bu kelimelerden en az 2'si (tam kelime olarak) varsa "doğrudan yanıt"
                    overlap = sum(1 for w in q_words if w in resp_words)
                    total_responses += 1
                    if overlap >= 2:
                        direct_responses += 1
                    break

    return {
        "total_q_responses": total_responses,
        "direct_responses":  direct_responses,
        "direct_ratio":      round(direct_responses / max(total_responses, 1), 3),
    }


def analyze_interview(iv: dict) -> dict:
    trump_turns = iv["trump_turns"]
    all_turns   = iv["turns"]
    trump_text  = " ".join(t["text"] for t in trump_turns)
    words       = trump_text.lower().split()
    n_words     = max(len(words), 1)

    # Amplifier oranları (her biri ayrı)
    amp_counts = count_each(trump_text, AMPLIFIERS)

    row = {
        "id":               iv["id"],
        "slug":             iv["slug"],
        "label":            LABEL[iv["slug"]],
        "date":             DATES[iv["slug"]],
        "profile":          PROFILE[iv["slug"]],
        "n_words":          n_words,
        "very_count":       amp_counts["very"],
        "great_count":      amp_counts["great"],
        "amplifier_rate_excl_great": round((sum(amp_counts.values()) - amp_counts["great"]) / n_words * 1000, 2),

        # Amplifiers
        "amplifier_count":  count_words_in_list(trump_text, AMPLIFIERS),
        "amplifier_rate":   round(count_words_in_list(trump_text, AMPLIFIERS) / n_words * 1000, 2),

        # Hedges
        "hedge_count":      count_pattern_list(trump_text, [r"\b" + h.replace(" ", r"\s+") + r"\b" for h in HEDGES]),
        "hedge_rate":       round(count_pattern_list(trump_text, [r"\b" + h.replace(" ", r"\s+") + r"\b" for h in HEDGES]) / n_words * 1000, 2),

        # Self-reference
        "self_ref_count":   count_pattern_list(trump_text, SELF_REFERENCE),
        "self_ref_rate":    round(count_pattern_list(trump_text, SELF_REFERENCE) / n_words * 1000, 2),

        # We/our (solidarity)
        "plural_ref_count": count_pattern_list(trump_text, FIRST_PERSON_PLURAL),
        "plural_ref_rate":  round(count_pattern_list(trump_text, FIRST_PERSON_PLURAL) / n_words * 1000, 2),

        # Superlatives
        "superlative_count":count_words_in_list(trump_text, SUPERLATIVES),
        "superlative_rate": round(count_words_in_list(trump_text, SUPERLATIVES) / n_words * 1000, 2),

        # Negation
        "negation_count":   count_words_in_list(trump_text, NEGATION),
        "negation_rate":    round(count_words_in_list(trump_text, NEGATION) / n_words * 1000, 2),

        # Repetition
        "repetition_count": count_pattern_list(trump_text, REPETITION_PATTERNS),

        # Turn-taking
        "trump_turns":      len(trump_turns),
        "avg_turn_words":   round(sum(t["word_count"] for t in trump_turns) / max(len(trump_turns), 1), 1),

        # Q&A direktlik
        **count_question_response(all_turns),
    }

    # En çok kullanılan 5 amplifier
    top5 = sorted(amp_counts.items(), key=lambda x: -x[1])[:5]
    row["top_amplifiers"] = ", ".join(f"{k}({v})" for k, v in top5 if v > 0)

    return row


def main():
    interviews = json.loads((DATA_DIR / "interviews_parsed.json").read_text(encoding="utf-8"))

    rows = []
    for iv in interviews:
        if not iv["trump_turns"]:
            continue
        row = analyze_interview(iv)
        rows.append(row)
        print(f"  [{row['id']:>2}] {row['slug']:<30} "
              f"amp/1k={row['amplifier_rate']:>5}  "
              f"hedge/1k={row['hedge_rate']:>5}  "
              f"direct_q={row['direct_ratio']:.2f}")

    df = pd.DataFrame(rows).sort_values("date")
    df.to_csv(TAB_DIR / "rhetoric_features.csv", index=False)
    print(f"\nCSV kaydedildi: {TAB_DIR / 'rhetoric_features.csv'}")

    # ── Görselleştirme 1 — Genel retorik özellikler ───────────────────────────
    sns.set_theme(style="whitegrid")
    metrics = ["amplifier_rate", "hedge_rate", "self_ref_rate",
               "superlative_rate", "negation_rate"]
    labels  = ["Amplifiers\n(/1000w)", "Hedges\n(/1000w)", "Self-Ref\n(/1000w)",
               "Superlatives\n(/1000w)", "Negation\n(/1000w)"]
    slugs   = list(df["label"])

    fig, axes = plt.subplots(1, 5, figsize=(22, 9), sharey=True)
    fig.suptitle("Trump Rhetorical Features per Interview", fontsize=15, fontweight="bold")

    for ax, metric, label in zip(axes, metrics, labels):
        ax.barh(slugs, df[metric], color=sns.color_palette("husl", len(df)))
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("Rate")
        if ax != axes[0]:
            ax.tick_params(labelleft=False)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "rhetoric_overview.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Grafik kaydedildi: {FIG_DIR / 'rhetoric_overview.png'}")

    # ── Görselleştirme 2 — Amplifier heatmap (top 15 amplifier × röportaj) ───
    amp_data = {}
    for iv in interviews:
        if not iv["trump_turns"]:
            continue
        trump_text = " ".join(t["text"] for t in iv["trump_turns"])
        n_words    = max(len(trump_text.split()), 1)
        cnt = count_each(trump_text, AMPLIFIERS)
        amp_data[LABEL[iv["slug"]]] = {amp: round(cnt[amp] / n_words * 1000, 2) for amp in AMPLIFIERS}

    amp_df = pd.DataFrame(amp_data).T  # röportaj × amplifier
    # Sadece genel toplamda en yüksek 15'i göster
    top15_amps = amp_df.sum().nlargest(15).index
    amp_df = amp_df[top15_amps]

    fig, ax = plt.subplots(figsize=(18, 8))
    sns.heatmap(amp_df, annot=True, fmt=".1f", cmap="YlOrRd",
                linewidths=0.5, ax=ax,
                cbar_kws={"label": "Rate per 1000 words"})
    ax.set_title("Amplifier Usage Heatmap — Interview × Amplifier Word", fontsize=14, fontweight="bold")
    ax.set_xlabel("Amplifier"); ax.set_ylabel("Interview")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "amplifiers_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Heatmap kaydedildi: {FIG_DIR / 'amplifiers_heatmap.png'}")

    # Genel özet
    print("\n── Retorik Özeti ───────────────────────────────────")
    print(f"Ortalama amplifier oranı  : {df['amplifier_rate'].mean():.2f} /1000 kelime")
    print(f"Ortalama hedge oranı      : {df['hedge_rate'].mean():.2f} /1000 kelime")
    print(f"Ortalama self-ref oranı   : {df['self_ref_rate'].mean():.2f} /1000 kelime")
    print(f"Ortalama soru direktliği  : %{df['direct_ratio'].mean()*100:.1f}")
    print(f"Ortalama tur uzunluğu     : {df['avg_turn_words'].mean():.1f} kelime/tur")


if __name__ == "__main__":
    main()

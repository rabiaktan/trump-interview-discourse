"""
ADIM 9 — HAKEM REVİZYONU İÇİN EK ANALİZLER
===========================================
Girdi : data/interviews_parsed.json   (orijinal 01_preprocess.py çıktısı)
        output/tables/*.csv           (varsa; direct_ratio ve duygu değerleri için)
Çıktı : output/tables/revizyon/*.csv  + output/tables/revizyon/ozet.txt

Çalıştır (trump_nlp klasöründen):
    python scripts/09_revizyon_analizleri.py              # temel analizler (A–F)
    python scripts/09_revizyon_analizleri.py --lda        # + alt-doküman LDA (gensim gerekir)
    python scripts/09_revizyon_analizleri.py --bertopic   # + tüm cümlelerde BERTopic

Gerekli paketler (temel):  pandas numpy scipy textstat nltk vaderSentiment
İsteğe bağlı:              statsmodels (karma etkili model), gensim (LDA), bertopic

Bölümler
  A. Uzunluktan bağımsız sözcük çeşitliliği (MATTR, MTLD, sabit uzunluklu örneklem)
  B. Düzeltilmiş retorik sayımlar (kelime sınırlı regex; "great" hariç amplifier)
  C. Profil karşılaştırması + kesin permütasyon testleri (ana sınıflama + Oval Office ve Meet the Press duyarlılık analizleri)
  D. Tur düzeyinde karma etkili model (statsmodels varsa)
  E. Duygu analizi: konu alanı atamasının döngüsellik testi + rastgele doğrulama örneklemi
  F. Eşdizim ölçüleri (MI, log-likelihood) + uyum (KWIC) satırları
  G. (--lda)      LDA: 250 kelimelik alt dokümanlar, k=4..12 Cv koherans taraması
  H. (--bertopic) BERTopic: tüm Trump cümleleri
"""

import argparse
import itertools
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
TAB_DIR = BASE_DIR / "output" / "tables"
OUT_DIR = TAB_DIR / "revizyon"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 2026
random.seed(SEED)
np.random.seed(SEED)

OZET = []  # ozet.txt satırları


def log(msg=""):
    print(msg)
    OZET.append(str(msg))


# ─────────────────────────────────────────────────────────────────────────────
# Metadata: düzeltilmiş künyeler ve iki profil sınıflaması
# ─────────────────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from corpus_config import DATES, LABEL, PROFILE as PROFILE_ORIG, PROFILE_ALT, SLUGS  # noqa: E402
N_EXPECTED = len(SLUGS)

# ─────────────────────────────────────────────────────────────────────────────
# Sözlükler (06_rhetoric.py ile birebir aynı listeler)
# ─────────────────────────────────────────────────────────────────────────────
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
SELF_REFERENCE = ["i", "me", "my", "myself", "mine"]
SUPERLATIVES = [
    "best", "greatest", "worst", "biggest", "largest", "most",
    "least", "highest", "lowest", "strongest", "weakest",
]
NEGATION = [
    "never", "nobody", "nothing", "nowhere", "no one",
    "not", "don't", "doesn't", "didn't", "won't", "can't",
    "shouldn't", "wouldn't", "couldn't",
]
# 04_sentiment.py ile birebir aynı konu anahtar kelimeleri
TOPIC_KEYWORDS = {
    "economy":   ["economy", "jobs", "trade", "tariff", "tax", "gdp", "stock", "market",
                  "inflation", "business", "deal", "money", "billion", "million"],
    "opponents": ["biden", "harris", "pelosi", "democrats", "radical", "left",
                  "obama", "clinton", "fake news", "media", "crooked"],
    "covid":     ["covid", "coronavirus", "pandemic", "vaccine", "fauci",
                  "lockdown", "mask", "hydroxychloroquine"],
    "elections": ["election", "vote", "ballot", "fraud", "rigged", "steal",
                  "supreme court", "november", "campaign"],
    "foreign":   ["china", "russia", "iran", "nato", "ukraine", "north korea",
                  "border", "mexico", "immigration", "military"],
    "self":      ["i won", "i did", "great job", "best president", "nobody",
                  "greatest", "tremendous", "unbelievable", "beautiful"],
}


# ─────────────────────────────────────────────────────────────────────────────
# Yardımcılar
# ─────────────────────────────────────────────────────────────────────────────
def norm(text):
    """Kıvrık kesme işaretlerini düzleştir, küçük harfe çevir."""
    return text.replace("’", "'").replace("‘", "'").lower()


TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)*")


def clean_tokens(text):
    """Noktalamadan arındırılmış sözcük tokenları ('I' korunur, durak sözcükler korunur)."""
    return TOKEN_RE.findall(norm(text))


def phrase_regex(phrase):
    return re.compile(r"(?<![a-z'])" + r"\s+".join(map(re.escape, phrase.split())) + r"(?![a-z'])")


def count_phrases(text, phrases):
    t = norm(text)
    return {p: len(phrase_regex(p).findall(t)) for p in phrases}


def old_split_count(text, words):
    """06_rhetoric.py'deki yöntem (boşlukla bölme; noktalama yapışık tokenlar kaçar)."""
    toks = text.lower().split()
    return sum(1 for t in toks if t in words)


def old_substring_count(text, word):
    """06_rhetoric.py ısı haritası yöntemi (alt-dize; 'every' içindeki 'very' de sayılır)."""
    return text.lower().count(word)


def get_sent_tokenize():
    try:
        import nltk
        nltk.data.find("tokenizers/punkt")
        return nltk.sent_tokenize, "nltk"
    except Exception:
        try:
            import nltk
            nltk.download("punkt", quiet=True)
            nltk.download("punkt_tab", quiet=True)
            nltk.data.find("tokenizers/punkt")
            return nltk.sent_tokenize, "nltk"
        except Exception:
            print("UYARI: NLTK punkt bulunamadı, basit regex cümle bölücü kullanılıyor.")
            return (lambda s: [x for x in re.split(r"(?<=[.!?])\s+", s) if x.strip()]), "regex"


def mattr(tokens, window):
    n = len(tokens)
    if n < window:
        return float("nan")
    counts = Counter(tokens[:window])
    total = len(counts) / window
    k = 1
    for i in range(window, n):
        out_t, in_t = tokens[i - window], tokens[i]
        counts[out_t] -= 1
        if counts[out_t] == 0:
            del counts[out_t]
        counts[in_t] += 1
        total += len(counts) / window
        k += 1
    return total / k


def _mtld_pass(tokens, threshold=0.72):
    factors, types, count = 0.0, set(), 0
    for t in tokens:
        count += 1
        types.add(t)
        ttr = len(types) / count
        if ttr <= threshold:
            factors += 1
            types, count = set(), 0
    if count > 0:
        ttr = len(types) / count
        factors += (1 - ttr) / (1 - threshold) if ttr != 1 else 0
    return len(tokens) / factors if factors > 0 else float("nan")


def mtld(tokens, threshold=0.72):
    return (_mtld_pass(tokens, threshold) + _mtld_pass(tokens[::-1], threshold)) / 2


def sampled_ttr_hapax(tokens, size=500, n_samples=200, rng=None):
    """Rastgele başlangıçlı, ardışık 'size' tokenlık pencerelerde ortalama TTR ve hapaks oranı."""
    rng = rng or random.Random(SEED)
    n = len(tokens)
    if n < size:
        return float("nan"), float("nan")
    ttrs, hps = [], []
    for _ in range(n_samples):
        s = rng.randint(0, n - size)
        w = tokens[s:s + size]
        c = Counter(w)
        ttrs.append(len(c) / size)
        hps.append(sum(1 for v in c.values() if v == 1) / size)
    return float(np.mean(ttrs)), float(np.mean(hps))


def exact_perm_two_groups(values, mask, max_enum=200000):
    """Grup ortalamaları farkı için kesin (tüm atamalar) permütasyon testi, iki yönlü."""
    values = np.asarray(values, float)
    idx = np.arange(len(values))
    k = int(mask.sum())
    obs = values[mask].mean() - values[~mask].mean()
    combs = list(itertools.combinations(idx, k))
    diffs = []
    for c in combs:
        m = np.zeros(len(values), bool)
        m[list(c)] = True
        diffs.append(values[m].mean() - values[~m].mean())
    diffs = np.array(diffs)
    p = np.mean(np.abs(diffs) >= abs(obs) - 1e-12)
    return obs, p, len(combs)


def exact_perm_k_groups(values, labels):
    """Gruplar arası kareler toplamı için kesin permütasyon testi (grup büyüklükleri sabit)."""
    values = np.asarray(values, float)
    groups = sorted(set(labels))
    sizes = [labels.count(g) for g in groups]
    grand = values.mean()

    def ssb(assign):
        return sum(len(a) * (values[list(a)].mean() - grand) ** 2 for a in assign)

    obs = ssb([[i for i, l in enumerate(labels) if l == g] for g in groups])
    n = len(values)
    count_ge, total = 0, 0
    idx = list(range(n))
    for first in itertools.combinations(idx, sizes[0]):
        rest = [i for i in idx if i not in first]
        for second in itertools.combinations(rest, sizes[1]):
            third = [i for i in rest if i not in second]
            s = ssb([first, second, third])
            total += 1
            if s >= obs - 1e-12:
                count_ge += 1
    return obs, count_ge / total, total


def spearman_perm(x, y, n_perm=20000, rng=None):
    rng = rng or np.random.default_rng(SEED)
    rx = stats.rankdata(np.asarray(x, float))
    ry = stats.rankdata(np.asarray(y, float))
    rx = (rx - rx.mean()) / rx.std()
    ry = (ry - ry.mean()) / ry.std()
    rho = float(np.mean(rx * ry))
    perms = np.array([rng.permutation(ry) for _ in range(n_perm)])
    null = perms @ rx / len(rx)
    return rho, float(np.mean(np.abs(null) >= abs(rho) - 1e-12))


def cliffs_delta(a, b):
    a, b = list(a), list(b)
    gt = sum(1 for x in a for y in b if x > y)
    lt = sum(1 for x in a for y in b if x < y)
    return (gt - lt) / (len(a) * len(b))


# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lda", action="store_true")
    ap.add_argument("--bertopic", action="store_true")
    args = ap.parse_args()

    interviews = json.loads((DATA_DIR / "interviews_parsed.json").read_text(encoding="utf-8"))
    interviews = [iv for iv in interviews if iv["slug"] in PROFILE_ORIG]
    log(f"Röportaj sayısı: {len(interviews)}")
    if len(interviews) != N_EXPECTED:
        log(f"UYARI: {N_EXPECTED} röportaj bekleniyordu. interviews_parsed.json dosyasını kontrol edin.")

    sent_tokenize, sent_method = get_sent_tokenize()
    log(f"Cümle bölücü: {sent_method}")

    # Mevcut tablolar (varsa)
    old_rhet = pd.read_csv(TAB_DIR / "rhetoric_features.csv") if (TAB_DIR / "rhetoric_features.csv").exists() else None
    old_stats = pd.read_csv(TAB_DIR / "corpus_stats.csv") if (TAB_DIR / "corpus_stats.csv").exists() else None
    old_sent = pd.read_csv(TAB_DIR / "sentiment_by_interview.csv") if (TAB_DIR / "sentiment_by_interview.csv").exists() else None

    # FK: kitaptaki değerlerle tutarlılık için önce corpus_stats.csv'den alınır
    fk_lookup = {}
    if old_stats is not None and "flesch_kincaid_grade" in old_stats:
        fk_lookup = dict(zip(old_stats["slug"], old_stats["flesch_kincaid_grade"]))
        log("FK değerleri corpus_stats.csv dosyasından alındı.")

    def fk_value(slug, text):
        if slug in fk_lookup:
            return fk_lookup[slug]
        try:
            import textstat
            return round(textstat.flesch_kincaid_grade(text), 2)
        except Exception:
            return float("nan")

    rows_div, rows_rhet, rows_amp = [], [], []
    trump_texts, trump_turns_by_iv = {}, {}

    # ── A + B: röportaj düzeyi ölçüler ──────────────────────────────────────
    for iv in interviews:
        slug = iv["slug"]
        ttext = " ".join(t["text"] for t in iv["trump_turns"])
        trump_texts[slug] = ttext
        trump_turns_by_iv[slug] = [t["text"] for t in iv["trump_turns"]]
        n_split = len(ttext.split())          # kitaptaki kelime sayısı tanımı
        toks = clean_tokens(ttext)

        # A. sözcük çeşitliliği
        raw_split = ttext.lower().split()
        s_ttr, s_hapax = sampled_ttr_hapax(toks, 500)
        rows_div.append({
            "slug": slug, "interview": LABEL[slug], "date": DATES[slug],
            "trump_words": n_split, "clean_tokens": len(toks),
            "ttr_original_method": round(len(set(raw_split)) / len(raw_split), 4),
            "ttr_clean_tokens": round(len(set(toks)) / len(toks), 4),
            "mattr_100": round(mattr(toks, 100), 4),
            "mattr_500": round(mattr(toks, 500), 4),
            "mtld": round(mtld(toks), 2),
            "ttr_sample500_mean": round(s_ttr, 4),
            "hapax_sample500_mean": round(s_hapax, 4),
        })

        # B. retorik sayımlar
        amp = count_phrases(ttext, AMPLIFIERS)
        hed = count_phrases(ttext, HEDGES)
        neg = count_phrases(ttext, NEGATION)
        sup = count_phrases(ttext, SUPERLATIVES)
        # 06_rhetoric.py ile aynı tanım: kısaltmalı biçimler (I'm, I've, I'll, I'd) de sayılır
        selfr = {w: len(re.findall(r"\b%s\b" % w, norm(ttext))) for w in SELF_REFERENCE}
        amp_total = sum(amp.values())
        per1k = lambda c: round(c / n_split * 1000, 2)
        rows_rhet.append({
            "slug": slug, "interview": LABEL[slug], "n_words": n_split,
            "amp_rate_OLD_split": per1k(old_split_count(ttext, AMPLIFIERS)),
            "amp_rate_NEW": per1k(amp_total),
            "amp_rate_NEW_excl_great": per1k(amp_total - amp["great"]),
            "very_rate_NEW": per1k(amp["very"]),
            "great_rate_NEW": per1k(amp["great"]),
            "very_rate_OLD_substring": per1k(old_substring_count(ttext, "very")),
            "great_rate_OLD_substring": per1k(old_substring_count(ttext, "great")),
            "very_great_share_NEW": round((amp["very"] + amp["great"]) / amp_total, 3) if amp_total else float("nan"),
            "hedge_rate_NEW": per1k(sum(hed.values())),
            "negation_rate_OLD_split": per1k(old_split_count(ttext, NEGATION)),
            "negation_rate_NEW": per1k(sum(neg.values())),
            "superlative_rate_OLD_split": per1k(old_split_count(ttext, SUPERLATIVES)),
            "superlative_rate_NEW": per1k(sum(sup.values())),
            "self_ref_rate_NEW": per1k(sum(selfr.values())),
            "fk_grade": fk_value(slug, ttext),
            "talk_ratio": round(iv["trump_word_count"] / max(iv["trump_word_count"] + iv["interviewer_word_count"], 1) * 100, 1),
        })
        for a in AMPLIFIERS:
            rows_amp.append({"slug": slug, "interview": LABEL[slug], "amplifier": a,
                             "count_NEW": amp[a], "rate_NEW": per1k(amp[a]),
                             "count_OLD_substring": old_substring_count(ttext, a)})

    div = pd.DataFrame(rows_div).sort_values("date")
    rhet = pd.DataFrame(rows_rhet)
    if old_rhet is not None and "direct_ratio" in old_rhet:
        rhet = rhet.merge(old_rhet[["slug", "direct_ratio"]], on="slug", how="left")
    if old_sent is not None:
        sc = [c for c in old_sent.columns if "compound" in c.lower() and "mean" in c.lower()] or \
             [c for c in old_sent.columns if "compound" in c.lower()]
        if sc and "slug" in old_sent:
            rhet = rhet.merge(old_sent[["slug", sc[0]]].rename(columns={sc[0]: "sentiment_compound"}),
                              on="slug", how="left")
    rhet = rhet.merge(div[["slug", "date", "mattr_500", "mtld", "ttr_original_method"]], on="slug")
    rhet = rhet.sort_values("date")

    div.to_csv(OUT_DIR / "A_lexical_diversity.csv", index=False)
    rhet.to_csv(OUT_DIR / "B_rhetoric_corrected.csv", index=False)
    pd.DataFrame(rows_amp).to_csv(OUT_DIR / "B_amplifier_by_word.csv", index=False)

    # A özeti: dönem karşılaştırması ve uzunluk/tarih korelasyonları
    log("\n=== A. SÖZCÜK ÇEŞİTLİLİĞİ ===")
    div["year"] = div["date"].str[:4].astype(int)
    div["date_num"] = pd.to_datetime(div["date"]).map(lambda d: d.year + (d.dayofyear - 1) / 365.25)
    early = div[div.year <= 2021]
    late = div[div.year >= 2023]
    period_rows = []
    for m in ["ttr_original_method", "ttr_clean_tokens", "mattr_100", "mattr_500", "mtld",
              "ttr_sample500_mean", "hapax_sample500_mean"]:
        e, l = early[m].mean(), late[m].mean()
        rho_len, p_len = spearman_perm(np.log(div["clean_tokens"]), div[m])
        rho_date, p_date = spearman_perm(div["date_num"], div[m])
        period_rows.append({"measure": m, "mean_2020_2021": round(e, 4), "mean_2023_2025": round(l, 4),
                            "pct_change": round((l - e) / e * 100, 1),
                            "spearman_rho_vs_log_length": round(rho_len, 3), "p_perm_length": round(p_len, 4),
                            "spearman_rho_vs_date": round(rho_date, 3), "p_perm_date": round(p_date, 4)})
    period = pd.DataFrame(period_rows)
    period.to_csv(OUT_DIR / "A_period_and_correlations.csv", index=False)
    log(period.to_string(index=False))

    # ── C: profil karşılaştırmaları + permütasyon ──────────────────────────
    log("\n=== C. PROFİL KARŞILAŞTIRMASI (kesin permütasyon testleri) ===")
    metrics = ["amp_rate_NEW", "amp_rate_NEW_excl_great", "hedge_rate_NEW", "negation_rate_NEW",
               "superlative_rate_NEW", "self_ref_rate_NEW", "fk_grade", "talk_ratio", "mattr_500"]
    for extra in ["direct_ratio", "sentiment_compound"]:
        if extra in rhet:
            metrics.append(extra)
    means_rows, test_rows = [], []
    # Second sensitivity analysis (book, Section 9.2): Meet the Press coded adversarial (5 adversarial interviews)
    PROFILE_MTP = {s: ("adversarial" if s == "meet_the_press" else p) for s, p in PROFILE_ORIG.items()}
    for scheme_name, scheme in [("ana (Oval Office=adversarial)", PROFILE_ORIG), ("duyarlılık (Oval Office=neutral)", PROFILE_ALT),
                                ("duyarlılık 2 (Meet the Press=adversarial)", PROFILE_MTP)]:
        labs = [scheme[s] for s in rhet["slug"]]
        is_adv = np.array([l == "adversarial" for l in labs])
        for m in metrics:
            vals = rhet[m].values
            row = {"scheme": scheme_name, "metric": m}
            for g in ["supportive", "neutral", "adversarial"]:
                gv = [v for v, l in zip(vals, labs) if l == g]
                row[f"mean_{g}"] = round(float(np.mean(gv)), 3)
                row[f"n_{g}"] = len(gv)
            means_rows.append(row)
            diff, p2, ncomb2 = exact_perm_two_groups(vals, is_adv)
            ssb, pk, ncombk = exact_perm_k_groups(vals, labs)
            test_rows.append({
                "scheme": scheme_name, "metric": m,
                "adv_minus_others": round(diff, 3),
                "cliffs_delta_adv_vs_others": round(cliffs_delta(vals[is_adv], vals[~is_adv]), 3),
                "p_exact_adv_vs_others": round(p2, 4), "n_permutations_2grp": ncomb2,
                "p_exact_3groups": round(pk, 4), "n_permutations_3grp": ncombk,
            })
    means = pd.DataFrame(means_rows)
    tests = pd.DataFrame(test_rows)
    means.to_csv(OUT_DIR / "C_profile_means.csv", index=False)
    tests["bonferroni_alpha"] = round(0.05 / len(metrics), 4)
    tests.to_csv(OUT_DIR / "C_permutation_tests.csv", index=False)
    log(tests.to_string(index=False))

    # Zamansal: ölçü ~ tarih (Spearman, permütasyon)
    temp_rows = []
    dnum = pd.to_datetime(rhet["date"]).map(lambda d: d.year + (d.dayofyear - 1) / 365.25)
    for m in metrics + ["ttr_original_method", "mtld"]:
        if m not in rhet:
            continue
        rho, p = spearman_perm(dnum, rhet[m])
        rho_l, p_l = spearman_perm(np.log(rhet["n_words"]), rhet[m])
        temp_rows.append({"metric": m, "rho_vs_date": round(rho, 3), "p_perm_date": round(p, 4),
                          "rho_vs_log_length": round(rho_l, 3), "p_perm_length": round(p_l, 4)})
    temp = pd.DataFrame(temp_rows)
    temp.to_csv(OUT_DIR / "C_temporal_spearman.csv", index=False)
    log(f"\n--- Zamansal eğilimler ({len(rhet)} röportaj, Spearman + permütasyon) ---")
    log(temp.to_string(index=False))

    # ── D: tur düzeyi karma etkili model ────────────────────────────────────
    log("\n=== D. TUR DÜZEYİ KARMA ETKİLİ MODEL ===")
    turn_rows = []
    for slug, turns in trump_turns_by_iv.items():
        for k, t in enumerate(turns):
            n = len(t.split())
            if n < 20:
                continue
            a = sum(count_phrases(t, AMPLIFIERS).values())
            g = count_phrases(t, ["great"])["great"]
            ng = sum(count_phrases(t, NEGATION).values())
            hd = sum(count_phrases(t, HEDGES).values())
            turn_rows.append({"slug": slug, "turn": k, "n_words": n,
                              "profile_orig": PROFILE_ORIG[slug], "profile_alt": PROFILE_ALT[slug],
                              "amp_per100": a / n * 100, "amp_excl_great_per100": (a - g) / n * 100,
                              "neg_per100": ng / n * 100, "hedge_per100": hd / n * 100})
    turns_df = pd.DataFrame(turn_rows)
    turns_df.to_csv(OUT_DIR / "D_turn_level_data.csv", index=False)
    log(f"Analize giren Trump turu (>=20 kelime): {len(turns_df)}")
    # 1) Küme düzeyi kesin permütasyon: tur düzeyinde havuzlanmış ortalama farkı (adversarial − diğer),
    #    profil etiketleri röportajlar arasında karıştırılır (turların röportaj içi bağımlılığı korunur).
    perm_rows = []
    slugs_t = sorted(turns_df["slug"].unique())
    for scheme in ["profile_orig", "profile_alt"]:
        prof = turns_df.groupby("slug")[scheme].first()
        adv = [s for s in slugs_t if prof[s] == "adversarial"]
        for y in ["amp_per100", "amp_excl_great_per100", "neg_per100", "hedge_per100"]:
            g = turns_df.groupby("slug")[y].agg(["sum", "count"])

            def stat(advset):
                m = g.index.isin(list(advset))
                return g["sum"][m].sum() / g["count"][m].sum() - g["sum"][~m].sum() / g["count"][~m].sum()
            obs = stat(adv)
            null = np.array([stat(c) for c in itertools.combinations(slugs_t, len(adv))])
            perm_rows.append({"scheme": scheme, "outcome": y, "adv_minus_others_per100": round(obs, 3),
                              "p_cluster_perm": round(float(np.mean(np.abs(null) >= abs(obs) - 1e-12)), 4),
                              "n_permutations": len(null)})
    perm = pd.DataFrame(perm_rows)
    perm.to_csv(OUT_DIR / "D_turn_level_cluster_permutation.csv", index=False)
    log("Tur düzeyi küme permütasyon testleri:")
    log(perm.to_string(index=False))

    # 2) Karma etkili model (röportaj rastgele kesişim); sırayla birkaç optimizasyon yöntemi denenir.
    #    Röportaj varyansı sıfır sınırına düşerse model güvenilir değildir ve 'boundary' olarak işaretlenir.
    try:
        import warnings
        import statsmodels.formula.api as smf
        mm_rows = []
        for scheme in ["profile_orig", "profile_alt"]:
            for y in ["amp_per100", "amp_excl_great_per100", "neg_per100", "hedge_per100"]:
                f = f"{y} ~ C({scheme}, Treatment('supportive'))"
                chosen = None
                for meth in ["lbfgs", "bfgs", "cg", "nm"]:
                    with warnings.catch_warnings(record=True) as wlist:
                        warnings.simplefilter("always")
                        try:
                            res = smf.mixedlm(f, turns_df, groups=turns_df["slug"]).fit(reml=True, method=meth)
                        except Exception:
                            continue
                    msgs = " ".join(str(w.message).lower() for w in wlist)
                    boundary = ("boundary" in msgs) or ("singular" in msgs) or float(res.cov_re.iloc[0, 0]) < 1e-6
                    if res.converged and not boundary:
                        chosen = (meth, res, "ok")
                        break
                    if chosen is None:
                        chosen = (meth, res, "boundary/not converged")
                if chosen is None:
                    continue
                meth, res, status = chosen
                ci = res.conf_int()
                for term in res.params.index:
                    if term == "Group Var":
                        continue
                    ok = status == "ok"
                    mm_rows.append({"scheme": scheme, "outcome": y,
                                    "term": term.replace(f"C({scheme}, Treatment('supportive'))", "profile"),
                                    "coef": round(res.params[term], 3),
                                    "ci_low": round(ci.loc[term, 0], 3) if ok else None,
                                    "ci_high": round(ci.loc[term, 1], 3) if ok else None,
                                    "p": round(res.pvalues[term], 4) if ok else None,
                                    "interview_var": round(float(res.cov_re.iloc[0, 0]), 4),
                                    "optimizer": meth, "status": status})
        mm = pd.DataFrame(mm_rows)
        mm.to_csv(OUT_DIR / "D_mixed_models.csv", index=False)
        log("Karma etkili modeller:")
        log(mm.to_string(index=False))
    except ImportError:
        log("statsmodels yüklü değil: 'pip install statsmodels' ile yükleyip tekrar çalıştırın.")

    # ── E: duygu analizi döngüsellik testi + doğrulama örneklemi ───────────
    log("\n=== E. DUYGU ANALİZİ: KONU ATAMA DÖNGÜSELLİK TESTİ ===")
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        an = SentimentIntensityAnalyzer()
        sent_rows = []
        for slug, turns in trump_turns_by_iv.items():
            for k, t in enumerate(turns):
                for s in sent_tokenize(t):
                    if len(s.split()) < 4:
                        continue
                    low = s.lower()
                    scores = {tp: sum(1 for kw in kws if kw in low) for tp, kws in TOPIC_KEYWORDS.items()}
                    best = max(scores, key=scores.get)
                    topic = best if scores[best] > 0 else "other"
                    comp = an.polarity_scores(s)["compound"]
                    masked = s
                    if topic != "other":
                        for kw in TOPIC_KEYWORDS[topic]:
                            masked = re.sub(re.escape(kw), " ", masked, flags=re.I)
                    comp_m = an.polarity_scores(masked)["compound"]
                    has_neg = any(phrase_regex(n).search(norm(s)) for n in NEGATION)
                    sent_rows.append({"slug": slug, "interview": LABEL[slug], "turn": k, "sentence": s,
                                      "topic": topic, "compound": comp, "compound_keywords_masked": comp_m,
                                      "has_negation": has_neg,
                                      "profile_orig": PROFILE_ORIG[slug]})
        sdf = pd.DataFrame(sent_rows)
        dom = sdf.groupby("topic").agg(n=("compound", "size"),
                                       compound_mean=("compound", "mean"),
                                       compound_masked_mean=("compound_keywords_masked", "mean")).round(3)
        dom.to_csv(OUT_DIR / "E_topic_sentiment_masked.csv")
        log(f"Trump cümlesi (>=4 kelime): {len(sdf)}")
        log(dom.to_string())
        if {"self", "opponents"} <= set(dom.index):
            g0 = dom.loc["self", "compound_mean"] - dom.loc["opponents", "compound_mean"]
            g1 = dom.loc["self", "compound_masked_mean"] - dom.loc["opponents", "compound_masked_mean"]
            log(f"Self–opponents farkı: orijinal {g0:.3f} | anahtar kelimeler maskelenince {g1:.3f}")
        # Negasyon x profil: VADER yanlılığı testi için
        neg_tab = sdf.groupby(["profile_orig", "has_negation"])["compound"].agg(["size", "mean"]).round(3)
        neg_tab.to_csv(OUT_DIR / "E_negation_by_profile.csv")
        # Rastgele doğrulama örneklemi (elle etiketlemek için)
        samp = sdf.sample(n=min(150, len(sdf)), random_state=SEED)[
            ["interview", "turn", "sentence", "compound", "topic", "has_negation"]].copy()
        samp["vader_label"] = np.where(samp.compound >= 0.05, "positive",
                                       np.where(samp.compound <= -0.05, "negative", "neutral"))
        samp["manual_label_rater1"] = ""
        samp["manual_label_rater2"] = ""
        samp.to_csv(OUT_DIR / "E_vader_validation_sample_TO_ANNOTATE.csv", index=False, encoding="utf-8-sig")
        sdf.to_csv(OUT_DIR / "E_all_sentences.csv", index=False, encoding="utf-8-sig")
    except ImportError:
        log("vaderSentiment yüklü değil.")

    # ── F: eşdizim ölçüleri + KWIC ─────────────────────────────────────────
    log("\n=== F. EŞDİZİM (MI, LOG-LIKELIHOOD) ===")
    try:
        from nltk.corpus import stopwords
        STOP = set(stopwords.words("english"))
    except Exception:
        STOP = set("""a an the and or but if of to in on at by for with from as is are was were be been being
            it its this that these those i me my we our you your he him his she her they them their what which who
            whom do does did have has had not no so very just will would can could should there here than then
            about into over out up down all any some more most other such only own same too s t don now""".split())
    NODES = ["country", "great", "nobody", "people", "fake", "never", "biden", "china"]
    SPAN = 4
    corpus_tok = []           # (slug, turn, tokens)
    for slug, turns in trump_turns_by_iv.items():
        for k, t in enumerate(turns):
            corpus_tok.append((slug, k, clean_tokens(t)))
    freq = Counter(w for _, _, ts in corpus_tok for w in ts)
    N = sum(freq.values())
    col_rows, kwic_rows = [], []
    rng = random.Random(SEED)
    for node in NODES:
        co = Counter()
        hits = []
        for slug, k, ts in corpus_tok:
            for i, w in enumerate(ts):
                if w == node:
                    hits.append((slug, k, i, ts))
                    for j in range(max(0, i - SPAN), min(len(ts), i + SPAN + 1)):
                        if j != i:
                            co[ts[j]] += 1
        f_node = freq[node]
        win = 2 * SPAN * f_node
        for c, o11 in co.items():
            if o11 < 3 or c in STOP or c == node:
                continue
            f_c = freq[c]
            e11 = win * f_c / N
            mi = math.log2(o11 / e11) if e11 > 0 else float("nan")
            # Dunning G2: 2x2 tablo (pencere içi / dışı)
            o12 = win - o11
            o21 = f_c - o11
            o22 = N - win - o21
            o = [o11, o12, o21, o22]
            r1, r2, c1, c2 = o11 + o12, o21 + o22, o11 + o21, o12 + o22
            e = [r1 * c1 / N, r1 * c2 / N, r2 * c1 / N, r2 * c2 / N]
            g2 = 2 * sum(oi * math.log(oi / ei) for oi, ei in zip(o, e) if oi > 0 and ei > 0)
            col_rows.append({"node": node, "node_freq": f_node, "collocate": c, "joint_freq": o11,
                             "collocate_freq": f_c, "MI": round(mi, 2), "log_likelihood_G2": round(g2, 2)})
        for slug, k, i, ts in rng.sample(hits, min(12, len(hits))):
            kwic_rows.append({"node": node, "interview": LABEL[slug], "turn": k,
                              "left": " ".join(ts[max(0, i - 8):i]), "kw": ts[i],
                              "right": " ".join(ts[i + 1:i + 9])})
    col = pd.DataFrame(col_rows).sort_values(["node", "log_likelihood_G2"], ascending=[True, False])
    top = col.groupby("node").head(15)
    top.to_csv(OUT_DIR / "F_collocates_top15.csv", index=False)
    col.to_csv(OUT_DIR / "F_collocates_all.csv", index=False)
    pd.DataFrame(kwic_rows).to_csv(OUT_DIR / "F_concordance_lines.csv", index=False, encoding="utf-8-sig")
    log(top.groupby("node").head(5).to_string(index=False))

    # ── G: alt-doküman LDA ─────────────────────────────────────────────────
    if args.lda:
        log("\n=== G. LDA (250 kelimelik alt dokümanlar) ===")
        try:
            from gensim import corpora
            from gensim.models import LdaModel, CoherenceModel
            docs, doc_slug = [], []
            extra_stop = {"know", "going", "said", "say", "think", "like", "want", "get", "got",
                          "really", "lot", "thing", "things", "well", "yeah", "okay", "would", "could",
                          "one", "way", "look", "right", "us", "also", "even", "much", "many"}
            for slug, ts_text in trump_texts.items():
                ts = [w for w in clean_tokens(ts_text) if w not in STOP and w not in extra_stop and len(w) > 2]
                for s in range(0, len(ts), 250):
                    seg = ts[s:s + 250]
                    if len(seg) >= 100:
                        docs.append(seg)
                        doc_slug.append(slug)
            log(f"Alt doküman sayısı: {len(docs)}")
            dct = corpora.Dictionary(docs)
            dct.filter_extremes(no_below=5, no_above=0.5)
            bow = [dct.doc2bow(d) for d in docs]
            coh_rows, models = [], {}
            for k in range(4, 13):
                lda = LdaModel(bow, num_topics=k, id2word=dct, random_state=42, passes=20, alpha="auto")
                cv = CoherenceModel(model=lda, texts=docs, dictionary=dct, coherence="c_v").get_coherence()
                coh_rows.append({"k": k, "coherence_cv": round(cv, 4)})
                models[k] = lda
                log(f"  k={k:2d}  Cv={cv:.4f}")
            coh = pd.DataFrame(coh_rows)
            coh.to_csv(OUT_DIR / "G_lda_coherence_by_k.csv", index=False)
            best_k = int(coh.loc[coh.coherence_cv.idxmax(), "k"])
            best = models[best_k]
            log(f"En yüksek Cv: k={best_k}")
            pd.DataFrame([{"topic": t, "top_terms": ", ".join(w for w, _ in best.show_topic(t, 12))}
                          for t in range(best_k)]).to_csv(OUT_DIR / "G_lda_topics_bestk.csv", index=False)
            dist = defaultdict(lambda: np.zeros(best_k))
            for b, s in zip(bow, doc_slug):
                for t, p in best.get_document_topics(b, minimum_probability=0):
                    dist[s][t] += p
            pd.DataFrame({LABEL[s]: v / v.sum() for s, v in dist.items()}).T.round(3) \
                .to_csv(OUT_DIR / "G_lda_interview_topic_share.csv")
        except ImportError:
            log("gensim yüklenemedi (scipy<1.14 gerekli).")

    # ── H: BERTopic tüm cümleler ───────────────────────────────────────────
    if args.bertopic:
        log("\n=== H. BERTopic (tüm Trump cümleleri) ===")
        try:
            from bertopic import BERTopic
            from umap import UMAP
            sents, s_slug = [], []
            for slug, turns in trump_turns_by_iv.items():
                for t in turns:
                    for s in sent_tokenize(t):
                        if len(s.split()) >= 4:
                            sents.append(s)
                            s_slug.append(slug)
            umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=42)
            tm = BERTopic(embedding_model="all-MiniLM-L6-v2", umap_model=umap_model, min_topic_size=10)
            topics, _ = tm.fit_transform(sents)
            info = tm.get_topic_info()
            info.to_csv(OUT_DIR / "H_bertopic_topic_info.csv", index=False)
            pd.crosstab(pd.Series([LABEL[s] for s in s_slug], name="interview"),
                        pd.Series(topics, name="topic")).to_csv(OUT_DIR / "H_bertopic_by_interview.csv")
            log(f"Cümle sayısı: {len(sents)} | konu sayısı (aykırı -1 hariç): {(info.Topic >= 0).sum()} | "
                f"aykırı oranı: {(np.array(topics) == -1).mean():.3f}")
        except ImportError:
            log("bertopic yüklü değil: pip install bertopic")

    # ── Bilgi notları ──────────────────────────────────────────────────────
    log("\n=== NOTLAR ===")
    log(f"Toplam tur (konuşmacı turu, tüm röportajlar): {sum(iv['total_turns'] for iv in interviews)}")
    log(f"Toplam Trump turu: {sum(len(iv['trump_turns']) for iv in interviews)}")
    log("Orijinal TTR: boşlukla bölünmüş, küçük harfli ham tokenlar; durak sözcükler ÇIKARILMADAN, "
        "noktalama yapışık biçimler ayrı tip sayılarak hesaplanmıştır.")
    log("MATTR/MTLD/örneklem ölçüleri: noktalamadan arındırılmış tokenlar, durak sözcükler dahil.")

    (OUT_DIR / "ozet.txt").write_text("\n".join(OZET), encoding="utf-8")
    print(f"\nTüm çıktılar: {OUT_DIR}")
    print("Bu klasörü zip'leyip gönderin.")


if __name__ == "__main__":
    main()

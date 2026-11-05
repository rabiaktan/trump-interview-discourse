"""
ADIM 10 — BERTopic (tüm Trump cümleleri, 16 röportaj)
=====================================================
Hakem 2, madde 5: "BERTopic'i yalnızca ilk 2.000 cümle yerine tüm cümlelere uygulayın."

Girdi : data/interviews_parsed.json  (01_preprocess.py çıktısı)
Çıktı : output/tables/revizyon/H_bertopic_topic_info.csv     konu, cümle sayısı, temsil terimleri
        output/tables/revizyon/H_bertopic_sentences.csv      her cümle: röportaj, tur no, konu
        output/tables/revizyon/H_bertopic_by_interview.csv   röportaj × konu çapraz tablosu
        output/tables/revizyon/H_bertopic_ozet.txt           ayarlar, sürümler, özet

Çalıştır (trump_nlp klasöründen):
    pip install bertopic
    python scripts/10_bertopic.py

Ayarlar: all-MiniLM-L6-v2 gömmeleri; UMAP(n_neighbors=15, n_components=5, min_dist=0,
metric=cosine, random_state=42); HDBSCAN min_cluster_size=10 (BERTopic min_topic_size=10).
Konu TEMSİLİ için İngilizce durak sözcükler ve LDA'daki 27 ek durak sözcük çıkarılır (CountVectorizer); bu yalnızca konuların
etiket terimlerini etkiler, kümelemeyi etkilemez.
"""
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from corpus_config import LABEL, SLUGS  # noqa: E402

DATA = ROOT / "data" / "interviews_parsed.json"
OUT = ROOT / "output" / "tables" / "revizyon"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    import nltk
    for r in ("punkt", "punkt_tab"):
        try:
            nltk.download(r, quiet=True)
        except Exception:
            pass
    from bertopic import BERTopic
    from umap import UMAP
    from hdbscan import HDBSCAN
    from sklearn.feature_extraction.text import CountVectorizer

    ivs = [iv for iv in json.loads(DATA.read_text(encoding="utf-8")) if iv["slug"] in SLUGS]
    rows = []
    for iv in ivs:
        for k, t in enumerate(iv["trump_turns"]):
            for s in nltk.sent_tokenize(t["text"]):
                if len(s.split()) >= 4:
                    rows.append({"slug": iv["slug"], "interview": LABEL[iv["slug"]], "trump_turn": k, "sentence": s})
    df = pd.DataFrame(rows)
    print(f"Röportaj: {len(ivs)} | cümle (>=4 kelime): {len(df)}")

    umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=42)
    hdb = HDBSCAN(min_cluster_size=10, metric="euclidean", cluster_selection_method="eom", prediction_data=True)
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    # LDA ile aynı 27 ek durak sözcük + kesme işaretli kısaltma artıkları
    extra = {"know", "going", "said", "say", "think", "like", "want", "get", "got", "really", "lot", "thing",
             "things", "well", "yeah", "okay", "would", "could", "one", "way", "look", "right", "us", "also",
             "even", "much", "many", "don", "didn", "doesn", "isn", "wasn", "aren", "won", "ve", "ll", "re",
             "let", "gonna"}
    vec = CountVectorizer(stop_words=sorted(ENGLISH_STOP_WORDS | extra), min_df=2, ngram_range=(1, 2),
                          token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b")
    tm = BERTopic(embedding_model="all-MiniLM-L6-v2", umap_model=umap_model, hdbscan_model=hdb,
                  vectorizer_model=vec, min_topic_size=10, top_n_words=10)
    topics, _ = tm.fit_transform(df["sentence"].tolist())
    df["topic"] = topics

    info = tm.get_topic_info()
    info["Representation"] = info["Representation"].apply(lambda xs: ", ".join(xs))
    info["Representative_Docs"] = info["Representative_Docs"].apply(lambda xs: " || ".join(xs) if isinstance(xs, list) else "")
    # her konunun en büyük kaynağı olan röportaj ve payı
    top_src, top_share, n_iv = [], [], []
    for t in info["Topic"]:
        sub = df[df.topic == t]["interview"].value_counts()
        top_src.append(sub.index[0] if len(sub) else "")
        top_share.append(round(sub.iloc[0] / sub.sum(), 3) if len(sub) else np.nan)
        n_iv.append(int((sub > 0).sum()))
    info["largest_source_interview"] = top_src
    info["largest_source_share"] = top_share
    info["n_interviews"] = n_iv
    info.to_csv(OUT / "H_bertopic_topic_info.csv", index=False, encoding="utf-8")
    df.to_csv(OUT / "H_bertopic_sentences.csv", index=False, encoding="utf-8")
    pd.crosstab(df["interview"], df["topic"]).to_csv(OUT / "H_bertopic_by_interview.csv", encoding="utf-8")

    import bertopic, umap, hdbscan, sklearn
    try:
        import sentence_transformers
    except Exception:
        sentence_transformers = type('x', (), {'__version__': '?'})
    n_top = int((info.Topic >= 0).sum())
    out_share = float((df.topic == -1).mean())
    lines = [
        f"Cümle sayısı: {len(df)}",
        f"Konu sayısı (aykırı -1 hariç): {n_top}",
        f"Aykırı (-1) oranı: {out_share:.3f}",
        f"Python {platform.python_version()} | bertopic {bertopic.__version__} | umap-learn {umap.__version__} | "
        f"hdbscan {getattr(hdbscan, '__version__', '?')} | scikit-learn {sklearn.__version__} | "
        f"sentence-transformers {sentence_transformers.__version__} | nltk {nltk.__version__}",
    ]
    (OUT / "H_bertopic_ozet.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(info[["Topic", "Count", "Representation", "largest_source_interview", "largest_source_share"]].head(25).to_string())


if __name__ == "__main__":
    main()

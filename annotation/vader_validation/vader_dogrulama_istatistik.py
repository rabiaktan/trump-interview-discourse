"""VADER doğrulama istatistikleri (kitap, Bölüm 2.4.3).

Girdi : vader_annotated.csv  (150 rastgele cümle; vader_label; manual_label_rater1; manual_label_rater2)
Çıktı : ekrana özet. Etiketleri değiştirirseniz bu betiği yeniden çalıştırın.

    python vader_dogrulama_istatistik.py vader_annotated.csv
"""
import sys
import pandas as pd
from scipy.stats import fisher_exact

ADVERSARIAL = {"Chris Wallace", "Axios/HBO", "60 Minutes Unedited", "Oval Office (ABC)"}
CATS = ["positive", "neutral", "negative"]


def kappa(a, b):
    a, b = pd.Series(list(a)), pd.Series(list(b))
    po = (a == b).mean()
    pe = sum((a == c).mean() * (b == c).mean() for c in CATS)
    return (po - pe) / (1 - pe)


d = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "vader_annotated.csv")
d["adv"] = d.interview.isin(ADVERSARIAL)
print("Örneklem:", len(d))
print("İki tur uyumu: %.1f%%, kappa = %.2f" % ((d.manual_label_rater1 == d.manual_label_rater2).mean() * 100,
                                               kappa(d.manual_label_rater1, d.manual_label_rater2)))
c = d[d.manual_label_rater1 == d.manual_label_rater2].copy()      # referans: iki turun uyuştuğu cümleler
c["gold"] = c.manual_label_rater1
c["err"] = c.vader_label != c.gold
print("Referans cümle:", len(c), "| VADER uyumu: %d (%.1f%%), kappa = %.2f" % ((~c.err).sum(), (1 - c.err.mean()) * 100,
                                                                               kappa(c.vader_label, c.gold)))
print(pd.crosstab(c.gold, c.vader_label))
for name, col in (("Olumsuzlama", "has_negation"), ("Düşmanca röportaj", "adv")):
    g = c.groupby(col).err.agg(["sum", "count"]); g["oran"] = (g["sum"] / g["count"]).round(3)
    print(f"\n{name}:\n{g}\nFisher p = %.4f" % fisher_exact(pd.crosstab(c[col], c.err).values)[1])
n = c[c.has_negation]
t = pd.crosstab(n.adv, n.err).reindex(index=[False, True], columns=[False, True], fill_value=0)
print("\nOlumsuzlamalı cümleler, düşmanca × hata:\n", t, "\nFisher p = %.4f" % fisher_exact(t.values)[1])

# Üçüncü bağımsız tur (sağlamlık denetimi): referans = üç turun da uyuştuğu cümleler
if "manual_label_rater3" in d.columns:
    c3 = c[c.manual_label_rater3 == c.gold]
    print("\nÜçüncü tur referansla uyum: %d / %d" % (len(c3), len(c)))
    print("Üçlü uyum referansı: VADER uyumu %d / %d (%.1f%%)" % ((~c3.err).sum(), len(c3), (1 - c3.err.mean()) * 100))
    g = c3.groupby("has_negation").err.agg(["sum", "count"]); print(g)
    print("Fisher p = %.4f" % fisher_exact(pd.crosstab(c3.has_negation, c3.err).values)[1])

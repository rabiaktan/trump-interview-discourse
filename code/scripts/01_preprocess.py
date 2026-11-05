"""
ADIM 1 — Konuşmacı Ayrımı & Ön İşleme  (DÜZELTİLMİŞ SÜRÜM, Ekim 2026)
======================================================================
Girdi : trump_interviews_nlp_complete.txt
Çıktı : data/interviews_parsed.json, data/trump_only.txt, data/interviewer_only.txt,
        data/speaker_check.csv  (her röportajdaki konuşmacı etiketleri — elle kontrol için)

Önceki sürümdeki iki hata düzeltildi:
  1. Konuşmacı etiketi regex'i, her turun sondaki cümlelerini bir sonraki konuşmacının
     adına katıyordu (ör. "Thank you so much. President Donald J. Trump:"). Bu metin
     analizden düşüyordu. Yeni yöntem: etiket yalnızca büyük harfle başlayan 1–5 sözcükten
     oluşabilir ve röportajda tekrar eden gerçek konuşmacı adına kırpılır.
  2. Röportaj blokları ayırıcı çizgiye göre bölünüyordu; başlığından önce çizgi olmayan
     dört blok (Tucker, Rogan, Oval Office, Will Cain) sessizce atlanıyordu. Yeni yöntem
     'INTERVIEW ID' başlıklarını çapa olarak kullanır ve corpus_config.py'deki listeyi uygular.

Çalıştır: python scripts/01_preprocess.py
"""

import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from corpus_config import SLUGS, META, EXCLUDED  # noqa: E402

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
INPUT_FILE = BASE_DIR / "trump_interviews_nlp_complete.txt"

TS = r"\(\s*\d{1,2}:\d{2}(?::\d{2})?\s*\)"
# Ad sözcüğü: Büyük harfle başlar; iç kısımda kesme/tire olabilir (O'Donnell, Campos-Duffy);
# baş harf (J.) veya unvan kısaltması (Dr., Mr.) olabilir. Noktayla biten sıradan sözcük OLAMAZ.
TOK = r"(?:Speaker\s\d+|Dr\.|Mr\.|Mrs\.|Ms\.|Jr\.|[A-Z]\.|[A-Z][A-Za-z'’]*(?:-[A-Z][A-Za-z'’]*)*)"
# "Speaker 1", "Speaker 2" … : Rev.com'un adsız konuşmacı etiketi (çekim ekibi, spiker) — Trump dışı sayılır
LABEL_RE = re.compile(r"(?<![A-Za-z'’\-])((?:" + TOK + r"\s+){0,4}" + TOK + r")\s*(?::\s*" + TS + r"|" + TS + r"\s*:)")
HEADER_RE = re.compile(r"INTERVIEW ID\s*:\s*(\d+)\s*\nSLUG\s*:\s*(\S+)\s*\nTITLE\s*:\s*(.*)\nDATE\s*:\s*(.*)\nSOURCE URL\s*:\s*(.*)\n")


def read_blocks(raw):
    heads = list(HEADER_RE.finditer(raw))
    blocks = []
    for i, m in enumerate(heads):
        end = heads[i + 1].start() if i + 1 < len(heads) else len(raw)
        seg = raw[m.end():end]
        k = seg.find("─" * 40)                      # başlığı kapatan çizgi
        content = seg[k:] if k != -1 else seg
        content = content.strip("─ \n\r\t")
        content = re.sub(r"─{40,}\s*$", "", content).strip()
        blocks.append({"id": int(m.group(1)), "slug": m.group(2).strip(), "title": m.group(3).strip(),
                       "url": m.group(5).strip(), "content": content})
    return blocks


def is_trump(name):
    return "trump" in name.lower()


FOOTER_RE = re.compile(r"Luckily for you, we deliver\.|Subscribe to our blog today|Copyright Disclaimer Under Title 17|"
                       r"Sign up to get Rev content|Hungry For More\?|"
                       r"Thanks for watching\. Stay updated about breaking news")  # NBC kapanış metni (Meet the Press)


def split_turns(content):
    text = re.sub(r"\s+", " ", content)
    # Rev.com sayfa sonu metnini (abonelik/telif notu) kes: ilk konuşmacı etiketinden SONRA geçen ilk kalıp
    first = LABEL_RE.search(text)
    if first:
        f = FOOTER_RE.search(text, first.end())
        if f:
            text = text[:f.start()]
    cands = list(LABEL_RE.finditer(text))
    suffix_count = Counter()
    for m in cands:
        toks = m.group(1).split()
        for k in range(1, len(toks) + 1):
            suffix_count[" ".join(toks[-k:])] += 1

    def resolve(label):
        toks = label.split()
        base = suffix_count[toks[-1]]
        best = toks[-1]
        for k in range(2, len(toks) + 1):
            s = " ".join(toks[-k:])
            # tekrar eden ve kısa son ekin en az %30'u kadar sık geçen en uzun ad
            if suffix_count[s] >= 2 and suffix_count[s] >= 0.3 * base:
                best = s
        return best

    marks = []
    for m in cands:
        name = resolve(m.group(1))
        start = m.end(1) - len(name)
        marks.append((start, m.end(), name))

    turns = []
    for i, (s, e, name) in enumerate(marks):
        end = marks[i + 1][0] if i + 1 < len(marks) else len(text)
        body = text[e:end]
        body = re.sub(TS, " ", body)
        body = re.sub(r"\[[^\]]*\]", " ", body)           # [crosstalk 00:03:25], [inaudible]
        body = re.sub(r"\s+", " ", body).strip()
        if len(body.split()) >= 1:
            turns.append({"speaker": name, "is_trump": is_trump(name), "text": body,
                          "word_count": len(body.split())})
    return turns


def main():
    if not INPUT_FILE.exists():
        print(f"HATA: {INPUT_FILE} bulunamadı.")
        sys.exit(1)
    raw = INPUT_FILE.read_text(encoding="utf-8", errors="ignore")
    blocks = read_blocks(raw)
    print(f"Dosyada {len(blocks)} röportaj bloğu bulundu.")

    found = {b["slug"] for b in blocks}
    missing = [s for s in SLUGS if s not in found]
    if missing:
        print("HATA: Şu röportajlar dosyada yok:", missing)
        sys.exit(1)

    interviews, check_rows = [], []
    all_trump, all_inter = [], []
    for b in blocks:
        slug = b["slug"]
        if slug not in META:
            print(f"  -- {slug:<24} kapsam dışı ({EXCLUDED.get(slug, 'corpus_config.py listesinde yok')})")
            continue
        turns = split_turns(b["content"])
        tt = [t for t in turns if t["is_trump"]]
        it = [t for t in turns if not t["is_trump"]]
        iv = {
            "id": b["id"], "slug": slug, "title": b["title"], "date": META[slug]["date"], "url": b["url"],
            "turns": turns, "trump_turns": tt, "interviewer_turns": it,
            "trump_word_count": sum(t["word_count"] for t in tt),
            "interviewer_word_count": sum(t["word_count"] for t in it),
            "total_turns": len(turns),
        }
        interviews.append(iv)
        all_trump.append(f"=== {slug} ===\n" + " ".join(t["text"] for t in tt))
        all_inter.append(f"=== {slug} ===\n" + " ".join(t["text"] for t in it))
        spk = Counter(t["speaker"] for t in turns)
        for name, n in spk.most_common():
            check_rows.append({"slug": slug, "speaker": name, "is_trump": is_trump(name), "turns": n,
                               "words": sum(t["word_count"] for t in turns if t["speaker"] == name)})
        print(f"  [{b['id']:>2}] {slug:<24} Trump: {iv['trump_word_count']:>6} kelime  "
              f"Diğer: {iv['interviewer_word_count']:>6}  Tur: {len(turns):>4}  "
              f"Konuşmacılar: {', '.join(f'{k} ({v})' for k, v in spk.most_common(4))}")

    order = {s: i for i, s in enumerate(SLUGS)}
    interviews.sort(key=lambda d: order[d["slug"]])
    (DATA_DIR / "interviews_parsed.json").write_text(json.dumps(interviews, ensure_ascii=False, indent=1), encoding="utf-8")
    (DATA_DIR / "trump_only.txt").write_text("\n\n".join(all_trump), encoding="utf-8")
    (DATA_DIR / "interviewer_only.txt").write_text("\n\n".join(all_inter), encoding="utf-8")
    with open(DATA_DIR / "speaker_check.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["slug", "speaker", "is_trump", "turns", "words"])
        w.writeheader()
        w.writerows(check_rows)

    tw = sum(d["trump_word_count"] for d in interviews)
    ow = sum(d["interviewer_word_count"] for d in interviews)
    print(f"\nRöportaj sayısı        : {len(interviews)}")
    print(f"Trump toplam kelime    : {tw:,}")
    print(f"Diğer konuşmacılar     : {ow:,}")
    print(f"Trump konuşma oranı    : %{tw / (tw + ow) * 100:.1f}")
    print(f"Toplam tur             : {sum(d['total_turns'] for d in interviews):,}")
    print("Konuşmacı listesi      : data/speaker_check.csv (Trump dışı adları kontrol edin)")


if __name__ == "__main__":
    main()

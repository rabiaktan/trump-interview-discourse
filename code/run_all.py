"""
TÜM ANALİZLERİ SIRAYLA ÇALIŞTIR
==================================
Çalıştır: python run_all.py

Bu script adım adım tüm analizleri çalıştırır ve
her adımın çıktısını raporlar.
"""

import subprocess
import sys
import time
from pathlib import Path

SCRIPTS = [
    ("01_preprocess.py",              "Konuşmacı Ayrımı & Ön İşleme"),
    ("02_corpus_stats.py",            "Corpus Tanımlayıcı İstatistikler"),
    ("03_word_freq.py",               "Sözcük Sıklığı & N-gram"),
    ("04_sentiment.py",               "Duygu Analizi"),
    ("06_rhetoric.py",                "Söylem & Retorik Analizi"),
    ("07_temporal.py",                "Zamansal Değişim (2020→2025)"),
    ("08_interviewer_comparison.py",  "Röportajcıya Göre Farklılık"),
    ("09_revizyon_analizleri.py",     "Hakem Revizyonu Ek Analizleri (MATTR, permütasyon, LDA)"),
]

EXTRA_ARGS = {"09_revizyon_analizleri.py": ["--lda"]}

SCRIPTS_DIR = Path(__file__).parent / "scripts"


def run_script(script_name: str, description: str) -> bool:
    print(f"\n{'='*60}")
    print(f"ADIM: {description}")
    print(f"Script: {script_name}")
    print(f"{'='*60}")
    start = time.time()

    result = subprocess.run(
        [sys.executable, str(SCRIPTS_DIR / script_name)] + EXTRA_ARGS.get(script_name, []),
        capture_output=False,
        text=True,
    )

    elapsed = time.time() - start
    if result.returncode == 0:
        print(f"\n✓ Tamamlandı ({elapsed:.1f}s)")
        return True
    else:
        print(f"\n✗ HATA (returncode={result.returncode})")
        return False


def main():
    print("Trump NLP Analiz Pipeline'ı Başlıyor")
    print("=" * 60)
    print("Çalıştırılacak adımlar:")
    for i, (_, desc) in enumerate(SCRIPTS, 1):
        print(f"  {i}. {desc}")

    # Veri dosyasını kontrol et
    data_file = Path(__file__).parent / "trump_interviews_nlp_complete.txt"
    if not data_file.exists():
        print(f"\nHATA: {data_file} bulunamadı!")
        print("trump_interviews_nlp_complete.txt dosyasını bu klasöre kopyala.")
        sys.exit(1)

    results = []
    total_start = time.time()

    for script, description in SCRIPTS:
        success = run_script(script, description)
        results.append((description, success))
        if not success:
            print(f"\n⚠ '{description}' başarısız. Devam ediliyor...")

    total_elapsed = time.time() - total_start

    print(f"\n{'='*60}")
    print("ÖZET RAPOR")
    print(f"{'='*60}")
    for desc, success in results:
        status = "✓" if success else "✗"
        print(f"  {status} {desc}")

    success_count = sum(1 for _, s in results if s)
    print(f"\nToplam: {success_count}/{len(SCRIPTS)} adım başarılı")
    print(f"Toplam süre: {total_elapsed/60:.1f} dakika")
    print(f"\nÇıktılar:")
    print(f"  output/figures/  — tüm grafikler (PNG)")
    print(f"  output/tables/   — tüm tablolar (CSV)")
    print(f"  BERTopic: python scripts/10_bertopic.py (ayrıca çalıştırılır)")


if __name__ == "__main__":
    main()

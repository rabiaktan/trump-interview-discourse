# Supplementary material: Linguistic Patterns in Political Interview Discourse

Analysis code, annotation data, and result tables for the book

> Korkmaz Tan, R., Zengin, B., & Mert Coşkun, O. (2026). *Linguistic patterns in political interview discourse: A corpus-based NLP analysis of Donald Trump's speech (2020–2025)*. Sakarya University Press. https://doi.org/10.59537/saupress.2606

This package contains the material that the book describes as available to readers:

- the analysis scripts (`code/`);
- the speaker-label inventory (`results/speaker_check.csv`);
- the reference labels used to assess the sentiment analysis (`annotation/vader_validation/`);
- the complete collocate tables and a sample of concordance lines with their source interviews (`results/tables/revizyon/F_*.csv`);
- the result tables from which the values reported in the book were taken (`results/tables/`).

The interview transcripts themselves are **not** included, because they are published by Rev.com and are not ours to redistribute. Each transcript is listed with its URL in `corpus_metadata.csv` and in the book's reference list (Rev, 2020a–2025b). Files that would reproduce the full transcript text (the parsed corpus and the sentence-level text files) have been left out for the same reason; sentence-level outputs are given with identifiers and results only.

## Contents

| Path | Content |
|---|---|
| `corpus_metadata.csv` | The 16 corpus interviews and the excluded candidate (The Joe Rogan Experience): outlet, interviewer, broadcast date, format, interviewer profile (main coding and both sensitivity codings), Trump and other-speaker word counts, talk ratio, reference-list key, Rev.com title, page date, and URL. |
| `code/run_all.py` | Runs steps 01–09 in order. |
| `code/scripts/corpus_config.py` | Interview metadata and profile codings; the single source used by all scripts. |
| `code/scripts/01_preprocess.py` | Speaker separation (Section 2.2.1), removal of Rev.com page text and bracketed annotations; writes the speaker-label inventory. |
| `code/scripts/02_corpus_stats.py` | Word counts, talk ratio, raw TTR, average sentence length, Flesch–Kincaid grade, hapax ratio (Chapter 3). |
| `code/scripts/03_word_freq.py` | Word, bigram, and trigram frequencies (Chapter 4). |
| `code/scripts/04_sentiment.py` | Sentence-level VADER scores and keyword-based topic domains (Chapter 5). |
| `code/scripts/06_rhetoric.py` | Rhetorical feature rates and the direct Q&A ratio (Chapter 7). |
| `code/scripts/07_temporal.py`, `08_interviewer_comparison.py` | Period and profile summaries, sentiment alignment (Chapters 8–9). |
| `code/scripts/09_revizyon_analizleri.py` | MATTR, MTLD, and fixed-length samples (A); word-boundary rhetorical counts (B); exact permutation tests, Cliff's delta, Spearman tests, both sensitivity analyses (C); turn-level cluster permutation tests and mixed-effects models (D); keyword masking and the VADER validation sample (E); collocates and concordance lines (F); LDA on 250-token segments, k = 4–12 (G). |
| `code/scripts/10_bertopic.py` | BERTopic on all 8,318 Trump sentences (Section 6.2); run separately. |
| `annotation/vader_validation/` | The 150-sentence validation sample with VADER labels and the three labelling passes, the 140-sentence reference set, and the script that computes the agreement statistics reported in Section 2.4.3. |
| `results/speaker_check.csv` | Speaker-label inventory: every detected speaker label per interview, with turn and word counts. |
| `results/tables/` | Output of steps 01–09 (main tables) and, in `revizyon/`, the revision analyses A–G and the BERTopic results (H). `run_summary_tr.txt` is the console summary of step 09 (in Turkish). |

Code comments and console messages are in Turkish; variable and file names are in English or Turkish.

## Reproducing the analysis

### 1. Obtain the transcripts

The scripts read a single text file, `trump_interviews_nlp_complete.txt`, placed in the `code/` folder. It was assembled from the 17 Rev.com transcript pages listed in `corpus_metadata.csv`, retrieved on 22 April 2026. Each interview is a block that starts with a header of the form

```
INTERVIEW ID   : 01
SLUG           : axios_hbo
TITLE          : Donald Trump Interview Transcript With Jonathan Swan of Axios on HBO
DATE           :
SOURCE URL     : https://www.rev.com/transcripts/donald-trump-interview-transcript-with-axios-on-hbo
────────────────────────────────────────
```

followed by the transcript text as published on the page, with Rev.com's speaker labels and timestamps (for example `Donald Trump ( 00:03 ):` or `Jonathan Swan: ( 00:52 )`). The `SLUG` values must match those in `corpus_config.py`. Rev.com may edit its pages over time, so a transcript retrieved later can differ slightly from the version analysed here.

### 2. Install the software

The values in the book were produced with Python 3.11 and the library versions in `requirements.txt`:

```
pip install -r requirements.txt
```

NLTK 3.8.1 and textstat 0.7.3 give identical values. NLTK 3.10 segments some sentences differently, which changes sentence-level values slightly.

### 3. Run

```
cd code
python run_all.py
```

The first step should report 16 interviews, 97,889 Trump words, and 32,268 words by other speakers. Outputs are written to `code/output/tables/` and `code/data/`.

BERTopic was run separately with the versions in `requirements-bertopic.txt` (it downloads the `all-MiniLM-L6-v2` model on first use):

```
pip install -r requirements-bertopic.txt
python scripts/10_bertopic.py
```

## Where the book's values come from

| Book | File |
|---|---|
| Table 2.2, Table 3.1 | `corpus_stats.csv`, `revizyon/A_lexical_diversity.csv` |
| Section 4.2–4.4, Table 4.1 | `word_freq.csv`, `bigrams.csv`, `trigrams.csv`, `revizyon/F_collocates_*.csv` |
| Table 5.1 | `revizyon/E_topic_sentiment_masked.csv` |
| Section 2.4.3 (VADER validation) | `annotation/vader_validation/` |
| Table 6.1 | `revizyon/G_lda_*.csv` |
| Section 6.2, Table 6.2 | `revizyon/H_bertopic_*` |
| Table 7.1 | `revizyon/B_rhetoric_corrected.csv` |
| Tables 8.1–8.2 | `revizyon/A_*.csv`, `revizyon/B_rhetoric_corrected.csv`, `revizyon/C_temporal_spearman.csv` |
| Tables 9.1–9.2, sensitivity analyses | `revizyon/C_profile_means.csv`, `revizyon/C_permutation_tests.csv`, `interviewer_comparison.csv` |
| Turn-level results (Section 9.2) | `revizyon/D_*.csv` |

In `C_permutation_tests.csv`, the scheme `ana` is the main coding, `duyarlılık` codes Oval Office (ABC) as neutral, and `duyarlılık 2` codes Meet the Press as adversarial. In `D_mixed_models.csv`, models marked `boundary/not converged` were not used in the book.

## VADER validation labels

A simple random sample of 150 sentences was drawn from the 8,318 analysed sentences with a fixed seed (step 09, section E). Each sentence was labelled positive, neutral, or negative in two independent passes by a large language model (Claude, Anthropic) that had no access to the VADER scores; the second pass did not see the first pass's labels. A third independent pass served as a check on the reference. The author reviewed the reference labels sentence by sentence. The labelling instruction was the same in all passes:

- Label the evaluation expressed by the speaker. Praise, satisfaction, optimism, and pride are positive; criticism, accusation, complaint, threat, insult, and describing a bad event as bad are negative.
- Factual, procedural, interrogative, or incomplete sentences, and sentences in which positive and negative are balanced, are neutral.
- Judge each sentence on its own; do not assume irony that is not evident from the wording; take negation into account.
- Do not use any sentiment lexicon or tool.

| File | Content |
|---|---|
| `vader_annotated.csv` | 150 sentences with VADER compound score and label, the three passes (`manual_label_rater1`–`3`), negation flag, topic domain, and interviewer profile. |
| `vader_consensus.csv` | The 140 sentences on which the first two passes agree (the reference set). |
| `vader_rater2.csv`, `vader_rater3.csv` | Labels of the second and third passes, by sample position. |
| `vader_dogrulama_istatistik.py` | Computes the statistics in Section 2.4.3: `python vader_dogrulama_istatistik.py vader_annotated.csv` |

## Licence

Code: MIT Licence (see `LICENSE`). Tables, annotations, and documentation: Creative Commons Attribution 4.0 International (CC BY 4.0). The short transcript excerpts in `F_concordance_lines.csv` and the annotation files are quoted from Rev.com transcripts for research purposes.

## Contact

Rabia Korkmaz Tan, Department of Computer Engineering, Çorlu Faculty of Engineering, Tekirdağ Namık Kemal University, Türkiye. rkorkmaz@nku.edu.tr

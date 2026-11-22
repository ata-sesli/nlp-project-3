# Elif Su Dayı

- Student ID: 2309011032
- GitHub: [elifsudayi](https://github.com/elifsudayi)

My parts of Project 3 (Uzbek speech recognition on USC): text normalization, the shared benchmark evaluation and
table-generation pipeline, and error analysis on the best test system.

## 1. Text normalization (`data/normalize.py`)

- Implemented the shared Uzbek (Latin) normalization policy applied identically to both references and hypotheses:
  - `normalize_text`: lowercasing, standardizing apostrophe variants (`‘ ’ \` ʻ ʼ ´ ʹ ＇`) to ASCII `'`, removing
    punctuation while preserving word-internal apostrophes (`o'`, `g'`, and *tutuq belgisi* such as `o'zbek`, `g'isht`,
    `ma'no`), collapsing whitespace, and keeping digits and Cyrillic characters unchanged under the strict policy.
  - `transliterate_cyrillic_to_latin`, `int_to_uzbek`, `verbalize_numbers`, and `normalize_translit_and_numbers`:
    fixed-rule Uzbek Cyrillic-to-Latin transliteration and Uzbek number verbalization for the secondary evaluation
    column.
  - `has_cyrillic` and `has_digits`: diagnostic flags to count hypotheses containing Cyrillic characters or digits.
- Known edge case: `tok.strip("'")` also strips word-final apostrophes (e.g., `tog'` → `tog`), affecting 34 of 27,042
  test tokens symmetrically in both reference and hypothesis.
- Unit tests: `tests/test_normalize.py` (apostrophe preservation, punctuation/whitespace, strict policy, Cyrillic
  transliteration, Uzbek number verbalization, and secondary normalization).

## 2. Benchmark evaluation & table generation (`benchmark/`)

| File | What I did |
|---|---|
| `benchmark/evaluate.py` | Shared `jiwer`-based scorer reading `benchmark/predictions/<system>__<split>.jsonl` and `data/clean/<split>.csv`; computes normalized WER, normalized CER, orthographic WER, translit+number WER, and Cyrillic/digit counts; writes `benchmark/results/scores.csv`. Fixed metadata extraction to parse nested split-level run logs (`runs.<split>.rtf`, `environment.hardware`, `gpu`, `env.gpu`, `parameters`/`params`) and regenerated `scores.csv` across all 24 dev/test rows. |
| `benchmark/timing.py` | Shared `RTFTimer` context manager to measure processing time and compute Real-Time Factor (`RTF = proc_s / audio_s`). |
| `benchmark/make_tables.py` | Generates Markdown Table A (Baselines) and Table B (Our Contribution) from `scores.csv` and `benchmark/results/confidence_intervals.csv`, adding the `WER 95% CI` column and a footnote for `A1` (`GMM-HMM`) having no neural parameter count. |
| `tests/test_evaluate.py` | Unit tests verifying known WER/CER examples, translit+number scoring, and `RTFTimer`. |

## 3. Error analysis (`analysis/`)

Conducted error analysis on the best test system (`C4a_fastconformer-usc-ft-rnnt`, 11.38% test WER) across all 3,836
test clips (`analysis/error_analysis.py`, `analysis/error_analysis.md`):

- **Erroneous clips:** 1,783 / 3,836 (46.5%)
- **Test OOV rate (vs. train):** 21.53% token-based (5,823 / 27,042) and 45.15% type-based (4,773 / 10,572)

| Error / Substitution Category | Count | Share |
|---|---:|---:|
| Substitution (total) | 2,464 | 80.1% of all errors |
| Deletion | 445 | 14.5% of all errors |
| Insertion | 169 | 5.5% of all errors |
| — Suffix error (common prefix rule) | 1,065 | 43.2% of substitutions |
| — Root / other substitution | 1,334 | 54.1% of substitutions |
| — Apostrophe error (`'`) | 65 | 2.6% of substitutions |

- Generated Top-10 most frequent suffix and apostrophe error tables and a reproducible random sample (`seed=42`) of 100
  erroneous test sentences with error-category tags.

## 4. Documentation & notes

- Prepared the README evaluation metrics and normalization policy section (grounded in HF Audio Course Chapter 5) and
  updated benchmark commands to module invocation (`python -m benchmark.evaluate`, `python -m benchmark.make_tables`).
- **Note on C3 (KenLM + `pyctcdecode`):** Planned word-level KenLM decoding on C2 logits was not completed due to
  build/runtime errors in the KenLM environment; left out of the final tables and noted transparently.
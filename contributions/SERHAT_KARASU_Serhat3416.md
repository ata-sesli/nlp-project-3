# Serhat Karasu — @Serhat3416

## Project
Uzbek Speech Recognition — COE025 NLP Project 3

## Contributions
- Benchmark evaluation and result analysis.
- Implementation and evaluation of MMS-1B and XLS-R Uzbek ASR baselines.
- Comparison of ASR models and recognition systems.
- Documentation and project reporting.
- Testing, validation, and reproducibility checks.

## Technical Work

### A6 — MMS-1B
- Integrated `facebook/mms-1b-all`.
- Activated the Uzbek Latin adapter (`uzb-script_latin`).
- Implemented greedy CTC decoding without a language model.
- Evaluated development and test splits.

### A7 — XLS-R
- Integrated `lucio/xls-r-uzbek-cv8`.
- Implemented greedy CTC decoding without a language model.
- Evaluated development and test splits.

### Benchmark Results

| System | Split | WER (norm) | CER (norm) | RTF |
|---|---|---:|---:|---:|
| A6 MMS-1B | Dev | 31.86% | 7.68% | 0.01202 |
| A6 MMS-1B | Test | 29.38% | 6.60% | 0.01178 |
| A7 XLS-R | Dev | 43.58% | 10.00% | 0.00577 |
| A7 XLS-R | Test | 40.71% | 8.78% | 0.00546 |

### Evaluation and Validation
- Evaluated Word Error Rate (WER) and Character Error Rate (CER).
- Analyzed inference speed using Real-Time Factor (RTF).
- Compared model accuracy, computational speed, and parameter counts.
- Generated prediction JSONL files and benchmark result JSON files.
- Validated all four prediction files against the reference splits: 15,402 predictions, zero missing IDs, zero duplicate IDs, and zero extra IDs.
- Updated `benchmark/evaluate.py` to read per-split runtime metadata and model parameter counts.

## Files
- `baselines/mms/run_mms.py`
- `baselines/mms/README.md`
- `baselines/xlsr/run_xlsr.py`
- `baselines/xlsr/README.md`
- `benchmark/evaluate.py`
- `benchmark/predictions/`
- `benchmark/results/`

# Serhat Karasu — @Serhat3416

## Student Information
- Student Number: 2409111050

## Project
Uzbek Speech Recognition — COE025 NLP Project 3

## Contributions
- Implemented the A6 MMS-1B Uzbek ASR baseline.
- Implemented the A7 XLS-R Uzbek Common Voice 8 ASR baseline.
- Developed model-specific inference scripts for both baselines.
- Prepared documentation for model configurations, adapter usage, and inference.
- Contributed A6 and A7 experimental results and model comparisons to the project report.

## Technical Work

### A6 — MMS-1B
- Integrated `facebook/mms-1b-all`.
- Configured the Uzbek Latin language adapter (`uzb-script_latin`).
- Implemented greedy CTC decoding without a language model.
- Generated development and test predictions.

### A7 — XLS-R Uzbek Common Voice 8
- Integrated `lucio/xls-r-uzbek-cv8`.
- Configured the pretrained XLS-R model for Uzbek ASR.
- Implemented greedy CTC decoding without a language model.
- Generated development and test predictions.

## Experimental Results

| System | Split | WER (norm) | CER (norm) | RTF |
|---|---|---:|---:|---:|
| A6 MMS-1B | Dev | 31.86% | 7.68% | 0.01202 |
| A6 MMS-1B | Test | 29.38% | 6.60% | 0.01178 |
| A7 XLS-R | Dev | 43.58% | 10.00% | 0.00577 |
| A7 XLS-R | Test | 40.71% | 8.78% | 0.00546 |

## Files
- `baselines/mms/run_mms.py`
- `baselines/mms/README.md`
- `baselines/xlsr/run_xlsr.py`
- `baselines/xlsr/README.md`
- `benchmark/predictions/A6_mms-1b__dev.jsonl`
- `benchmark/predictions/A6_mms-1b__test.jsonl`
- `benchmark/predictions/A7_xls-r-uzbek-cv8__dev.jsonl`
- `benchmark/predictions/A7_xls-r-uzbek-cv8__test.jsonl`
- `benchmark/results/A6_mms-1b.json`
- `benchmark/results/A7_xls-r-uzbek-cv8.json`
- `REPORT.md`

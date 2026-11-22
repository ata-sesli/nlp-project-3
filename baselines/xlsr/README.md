# A7 — XLS-R Uzbek Common Voice 8 ASR Baseline

## Model
- Model: `lucio/xls-r-uzbek-cv8`
- Revision: `715a374c1b779d10e094f723c7564785042d866d`
- Decoding: Greedy CTC, no language model
- Precision: float16

The standard `Wav2Vec2Processor` is used without an external language-model decoder.

## Data
- Development: 3,783 clips
- Test: 3,836 clips
- Audio: 16 kHz, cropped using start_s and end_s

## Run
From the repository root:

```bash
python baselines/xlsr/run_xlsr.py --split dev
python baselines/xlsr/run_xlsr.py --split test --allow-test
```

## Outputs
- `benchmark/predictions/A7_xls-r-uzbek-cv8__dev.jsonl`
- `benchmark/predictions/A7_xls-r-uzbek-cv8__test.jsonl`
- `benchmark/results/A7_xls-r-uzbek-cv8.json`

Each prediction line contains `id` and raw decoded `hyp`. Official metrics are computed with `benchmark/evaluate.py`.

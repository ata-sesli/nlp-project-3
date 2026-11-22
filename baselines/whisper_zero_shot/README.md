# A4, A5: Whisper zero-shot (attention encoder-decoder)

| Row | Model | Settings | Training |
|-----|-------|----------|----------|
| A4 | `openai/whisper-small` | `language="uzbek"`, `task="transcribe"`, greedy, fp16 | none |
| A5 | `openai/whisper-large-v3` | same | none |

The exact revision (commit hash) of each model is recorded in
`benchmark/results/<system>.json` when the script runs.

| File | Contents |
|------|----------|
| `transcribe.py` | Transcribes a split, writes `benchmark/predictions/<system>__<split>.jsonl` and the run log `benchmark/results/<system>.json` (model ID, revision, parameters, decoding, RTF, hardware). Also used for C1. |
| `whisper_common.py` | Shared helpers for A4, A5 and C1: reading `data/clean/*.csv`, audio loading with `start_s`/`end_s`, the shared normalizer, output formats, test guard. |

Clips longer than 30 s are cut to their first 30 s by Whisper's feature
extractor; the script prints how many there are. Whisper may write digits or
Cyrillic; we score its output as it is (see the translit + number column).

```bash
pip install -r whisper_finetune/requirements-whisper.txt
python baselines/whisper_zero_shot/transcribe.py --model openai/whisper-small --system A4_whisper-small --split dev
python baselines/whisper_zero_shot/transcribe.py --model openai/whisper-large-v3 --system A5_whisper-large-v3 --split dev
```

`--split test` is refused unless `--after-freeze` is given.

## Dev results (all 3,783 dev clips, normalized text, greedy decoding)

| Row | WER | CER | RTF (A100 40 GB) |
|---|---:|---:|---:|
| A4 | 136.47% | 67.47% | 0.0145 |
| A5 | 96.31% | 34.28% | 0.0244 |

Both models mostly hear the Uzbek words but do not know the Uzbek Latin spelling (q -> k, o' -> o, Turkish
ş/ç), and large-v3 often joins words; WER is above 100% for A4 because a few clips loop on a repeated phrase up to
the 225-token limit. RTF for the final table is measured on a T4 in the test run.

## Test results (all 3,836 test clips, one run after the freeze)

Same settings as dev, run once on a Colab T4 so RTF is comparable with the other GPU systems.

| Row | WER | CER | RTF (T4) |
|---|---:|---:|---:|
| A4 | 119.87% | 53.19% | 0.0379 |
| A5 | 93.67% | 32.70% | 0.0850 |

Both stay far behind the systems trained on our data. A4 is lower than on dev only because fewer test clips
fall into the repetition loop (40 clips, 1.0%, vs 78, 2.1% on dev; a loop = more than 3x the reference length
plus 10 words). Without those clips A4's WER is the same on both splits (110.0% dev, 109.4% test). The RTF values are higher than on dev because
the T4 is slower than the A100, not because the models changed; each split's hardware is in its block of
`benchmark/results/<system>.json` (`runs.<split>.environment`).

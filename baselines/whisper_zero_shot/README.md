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

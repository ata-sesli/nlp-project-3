# A6 — MMS-1B Uzbek ASR Baseline

## Model
- Model: `facebook/mms-1b-all`
- Revision: `3d33597edbdaaba14a8e858e2c8caa76e3cec0cd`
- Language adapter: `uzb-script_latin`
- Decoding: Greedy CTC, no language model
- Precision: float16

The Uzbek Latin adapter is explicitly activated using `processor.tokenizer.set_target_lang('uzb-script_latin')` and `model.load_adapter('uzb-script_latin')`.

## Data
- Development: 3,783 clips
- Test: 3,836 clips
- Audio: 16 kHz, cropped using start_s and end_s

## Run
From the repository root:

```bash
python baselines/mms/run_mms.py --split dev
python baselines/mms/run_mms.py --split test --allow-test
```

## Outputs
- `benchmark/predictions/A6_mms-1b__dev.jsonl`
- `benchmark/predictions/A6_mms-1b__test.jsonl`
- `benchmark/results/A6_mms-1b.json`

Each prediction line contains `id` and raw decoded `hyp`. Official metrics are computed with `benchmark/evaluate.py`.

# common/

Small helpers shared by the scripts in `baselines/scratch/`, `baselines/nvidia_fastconformer/` and `xlsr_ctc/`.

- `asr_data.py`: reads `data/clean/<split>.csv`, loads audio (16 kHz mono, cut to `start_s`/`end_s`), uses `data/normalize.py`, writes `benchmark/predictions/<system>__<split>.jsonl` and `benchmark/results/<system>.json`, measures RTF, blocks test sets unless `--allow-test` is given.
- `make_fake_data.py`: a tiny fake dataset (tones, not speech) to check that the code runs. Never used for results.
- `pin_requirements.py`: rewrites a `requirements.txt` with the exact package versions that ran.

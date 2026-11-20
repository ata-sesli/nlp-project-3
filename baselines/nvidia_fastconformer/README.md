# A8 / A9: NVIDIA Uzbek FastConformer (ready model, zero-shot) and cleaning rule 8

Model: [`nvidia/stt_uz_fastconformer_hybrid_large_pc`](https://huggingface.co/nvidia/stt_uz_fastconformer_hybrid_large_pc), loaded with NVIDIA NeMo. The exact revision (commit hash) is saved in the run logs.
It is one encoder with two decoders, so it gives two baselines:

| Row | Head | Decoding |
|---|---|---|
| A8 | Transducer (RNN-T), the model's default | greedy |
| A9 | CTC (`change_decoding_strategy(decoder_type="ctc")`) | greedy |

The model card says it was trained on Common Voice, UzbekVoice and FLEURS. USC is not in its training data, so the USC test is a fair test for it.

## Files

- `nemo_utils.py`: downloads the `.nemo` file at a fixed revision, switches the head, transcribes clips (cut to `start_s`/`end_s`).
- `run_nvidia.py`: A8 and A9 on `dev` and `test`. Test sets need `--allow-test` (only after the freeze).
- `rule8_flag.py`: cleaning rule 8. Runs the CTC head on every train clip and writes `data/clean/rule8_cer.csv` (CER between model output and human transcript, both normalized) and a random sample of 50 flagged clips (seed 42) in `data/clean/manual_check.csv` for the by-ear check. The human transcript stays the reference; the model only points at suspects.
- `requirements.txt`: NeMo needs its own packages, so this folder has its own pinned file.

## Run (from the repo root)

```bash
pip install -r baselines/nvidia_fastconformer/requirements.txt
python baselines/nvidia_fastconformer/rule8_flag.py --raw-dir data/raw/usc/train      # rule 8, threshold CER > 0.5
python baselines/nvidia_fastconformer/rule8_flag.py --from-cer --threshold 0.7          # new threshold, no model run
python baselines/nvidia_fastconformer/run_nvidia.py --split dev
python baselines/nvidia_fastconformer/run_nvidia.py --split test --allow-test
```

Outputs: `benchmark/predictions/A8_nvidia-fastconformer-rnnt__<split>.jsonl`, `benchmark/predictions/A9_nvidia-fastconformer-ctc__<split>.jsonl`, `benchmark/results/A8_*.json`, `benchmark/results/A9_*.json` (revision, parameters, RTF on the GPU used, package versions). Official WER/CER come from `benchmark/evaluate.py`.

# A2 / A3: Conformer CTC and Conformer RNN-T, trained by us from scratch

Both models share the same encoder code (`model.py`), so the only difference is the output part.

| Part | A2 CTC | A3 RNN-T |
|---|---|---|
| Input | 80 log-mel features, 25 ms window, 10 ms step, mean/variance normalized per clip | same |
| Augmentation | SpecAugment: 2 frequency masks (up to 27), 5 time masks (up to 5% of the clip) | same |
| Encoder | 2 conv layers (4x fewer frames), then `torchaudio.models.Conformer`: 12 layers, size 256, 4 heads, feed-forward 1024, conv kernel 31, dropout 0.1 | same |
| Output | linear layer + CTC loss | LSTM prediction network (320) + joiner (320) + RNN-T loss (`torchaudio.functional.rnnt_loss`), plus a CTC loss on the encoder with weight 0.3 to learn faster |
| Decoding | greedy | greedy transducer decoding (CTC part not used) |
| Parameters | about 20M | about 21M |
| Units | characters of the normalized train transcripts | same |

Training: AdamW (lr 1e-3, betas 0.9/0.98, weight decay 1e-3), 1500 warmup steps, cosine decay to 1% of lr, fp16, gradient clipping 5, batches of up to 400 s (CTC) / 200 s (RNN-T) of padded audio, seed 42. Dev WER after every epoch; the best epoch is kept. The number of epochs is fitted to `--max-hours` after the first epoch.

`model.py` also has a plain PyTorch RNN-T loss, used only if torchaudio has no `rnnt_loss`; `tests/test_asr_models.py` checks that both give the same loss and gradients.

## Run (from the repo root)

```bash
python baselines/scratch/train.py --model ctc  --out-dir runs/a2_ctc
python baselines/scratch/train.py --model rnnt --out-dir runs/a3_rnnt
python baselines/scratch/predict.py --ckpt runs/a2_ctc/best.pt --split dev
python baselines/scratch/predict.py --ckpt runs/a2_ctc/best.pt --split test --allow-test
```

Outputs: `benchmark/predictions/A2_conformer-ctc__<split>.jsonl`, `benchmark/predictions/A3_conformer-rnnt__<split>.jsonl`, `benchmark/results/A2_conformer-ctc.json`, `benchmark/results/A3_conformer-rnnt.json` (all settings, per-epoch loss and dev WER, chosen epoch, RTF, package versions).

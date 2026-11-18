# Data statistics (written by data/stats.py, do not edit by hand)

## Splits after cleaning

| Split | Speakers | Clips | Hours (kept audio) | Hours (whole files) |
|---|---:|---:|---:|---:|
| train | 638 | 27144 | 23.92 | 24.64 |
| dev | 41 | 3783 | 3.95 | 4.00 |
| test | 38 | 3836 | 4.40 | 4.49 |

Speakers never appear in two splits (checked by tests/test_splits.py).

## Dev and test sentences that also appear in train

| Split | Clips | Sentence also in train (word for word, normalized) | Share |
|---|---:|---:|---:|
| dev | 3783 | 152 | 4.0% |
| test | 3836 | 66 | 1.7% |

The clip IDs are in `data/clean/test_in_train.csv`, so scores can also be reported without them.

## Cleaning rules (from data/clean/removed.csv)

| Rule | train removed | train fixed | train flagged | dev removed | dev fixed | dev flagged | test removed | test fixed | test flagged |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| duplicate_audio | 25 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 |
| long_silence | 0 | 3823 | 0 | 0 | 390 | 0 | 0 | 681 | 0 |
| noise_clipping | 1 | 0 | 89 | 0 | 0 | 0 | 0 | 0 | 0 |
| text_audio_mismatch | 40 | 0 | 0 | 0 | 0 | 5 | 0 | 0 | 2 |
| wrong_transcript | 448 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

## Limits used by the rules

```json
{
  "letters_per_second": {
    "low": 3.1,
    "high": 35.95,
    "how": "Tukey far-out fences (3*IQR) on log(letters / trimmed seconds) of train clips"
  },
  "snr_db": {
    "low": 8.37,
    "how": "Tukey outlier fence (Q1 - 1.5*IQR) of train clips; SNR = 95th - 10th percentile frame dB"
  },
  "clipped_share": {
    "high": 0.01,
    "how": "share of samples at int16 full scale"
  },
  "silence": {
    "frame_db_below_peak": 35.0,
    "min_trim_s": 0.5,
    "kept_s": 0.2
  },
  "silent_peak_int16": 64,
  "rule8_flagged_clips": 475
}
```

## By-ear checks (data/clean/manual_check.csv)

| Rule | Result | Sampled | Really wrong | Not wrong |
|---|---|---:|---:|---:|
| 8 wrong transcript | confirmed | 50 | 32 | 18 |
| 9 noise / clipping | rejected | 97 | 2 | 95 |

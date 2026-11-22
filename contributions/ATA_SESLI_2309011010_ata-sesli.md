# Ata Sesli

- Student ID: 2309011010
- GitHub: [ata-sesli](https://github.com/ata-sesli)

My parts of Project 3 (Uzbek speech recognition on USC): the repository setup, the data pipeline from the USC
archive to the raw split data in `data/raw/`, the framework of the cleaning script with the duplicate-audio rule,
the data tests and the root README. The other cleaning rules, `data/stats.py` and the test-in-train count are Enes's
work (see his file).

## 1. Repository setup

- Public repository with the Apache-2.0 `LICENSE` (code only; the USC data keeps its own license).
- `.gitignore` (Python caches, virtual environment, model weights, logits), root `requirements.txt` with exact
  versions, `pytest.ini` (repo root on the import path, tests in `tests/`).
- `README.md`: project summary, the Data section (how the splits are made, why, and what cleaning does), the table
  that explains every root file and folder, and the commands to rebuild the data.
- `data/README.md`: USC source, DOI, both licenses the official sources give (MIT and CC BY 4.0) and the citation.
- Merged the pull requests of the team (#1 and #2, Akif's A1, A4, A5 and C1).

## 2. Train split selection (`data/make_splits.py`)

- USC has no speaker metadata, so I derived the speaker ID from the clip file names. There are three name patterns
  (crowd-sourced clips and two audiobook forms). With this rule, dev and test have exactly the 41 and 38 speakers
  the USC paper reports.
- I found that 26 speakers (10 from dev, 16 from test) also appear in the official train set, although the paper
  calls the splits speaker-disjoint. We keep dev and test complete and leave all their speakers out of our train
  selection, so our three splits share no speaker.
- Hours per speaker in USC train are very uneven; without a limit one speaker would be about 20% of our train set.
  Only speakers with at most 0.25 h are eligible. They are shuffled with seed 42 and whole speakers are added,
  skipping any that would push the total past 25 h. Result: 643 speakers, 27,658 clips, 25.00 h, no speaker above
  about 1% of train. The list is `data/clean/train_speakers.txt`.

## 3. Raw data (`data/get_raw.py`, `data/raw/usc/`)

- `get_raw.py` copies the clips of our splits from the extracted USC archive into
  `data/raw/usc/<split>/<speaker_id>/`: audio as lossless 16 kHz mono FLAC, transcripts as the unchanged original
  files. Files are written once and never rewritten.
- Committed in chunks under 500 MB, without Git LFS: train 27,658 clips (about 1.4 GB), dev 3,783 clips, test
  3,837 clips.

## 4. Cleaning framework and duplicate audio (`data/clean.py`)

- The structure the cleaning rules plug into: a `Rule` is a per-clip check that returns a `Decision` (removed,
  fixed or flagged); each rule names the splits it may remove from, and a removal anywhere else becomes a flag, so
  dev and test lose only clearly broken files. Every decision is logged in `data/clean/removed.csv`, the kept clips
  go to `data/clean/{train,dev,test}.csv`, and nothing is written if a rule removes or flags more than 1% of a split
  until the team has reviewed it (`--allow-over-1pct`). The 30-second train limit is also in this file.
- Rule 1, duplicate audio: clips are compared by a hash of the decoded samples. The plan said to keep one copy, but
  most duplicate pairs have different transcripts, so at least one transcript is wrong and we cannot tell which.
  Such copies are all removed from train and only flagged in dev and test; copies with the same transcript keep the
  first clip ID; a train copy of dev or test audio is removed from train. Result: 25 train clips and 1 test clip
  removed.

## 5. Tests

- `tests/test_make_splits.py` (18 tests): speaker ID rule for every file name pattern, hours per speaker, the
  selection (no overshoot, speaker cap, seed, dev/test speakers excluded).
- `tests/test_get_raw.py` (3 tests): clip selection, lossless FLAC with unchanged transcripts, no rewriting.
- `tests/test_clean.py` (13 tests): loading, every duplicate case, the dev/test protection, fixes, the 1% check,
  the output files, and that the committed CSVs and `removed.csv` together cover every raw clip.
- `tests/test_splits.py` (5 tests): no speaker in two splits (raw data and clean CSVs), raw train speakers match
  `train_speakers.txt`, every audio file has its transcript in the right speaker folder, every CSV path exists and
  its kept interval is inside the clip.

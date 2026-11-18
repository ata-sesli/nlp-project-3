# Uzbek Speech Recognition (COE025 Project 3)

Automatic speech recognition for Uzbek (Latin script). We train and fine-tune
models on 25 hours of the human-verified Uzbek Speech Corpus (USC) and compare
them with every listed baseline that supports Uzbek, on the same test clips and
with one scoring script.

## Data

Source: [Uzbek Speech Corpus (USC), ISSAI](https://huggingface.co/datasets/issai/Uzbek_Speech_Corpus)
([paper](https://arxiv.org/abs/2107.14419)).

| Split | Source | Size |
|-------|--------|------|
| train | Official USC train speakers that are not in dev or test and have at most 0.25 h of audio, shuffled with seed 42; whole speakers added, skipping any that would push the total past 25.0 h | 25.0 h |
| dev   | Official USC dev, all clips | 4.0 h |
| test  | Official USC test, all clips | 4.5 h |

USC has no speaker metadata file. Speaker IDs come from clip file names; see
`data/make_splits.py` for the rule. With it, dev and test have exactly the 41
and 38 speakers the USC paper reports.

The USC paper describes the splits as speaker-disjoint, but 10 dev speakers and
16 test speakers also appear in the official train set. We keep dev and test
complete and unchanged and exclude all dev and test speakers from our train
selection, so our splits share no speaker.

Hours per speaker in USC train are very uneven: the median speaker has under a
minute, the largest has 5.2 h. Without a limit, one speaker would make up about
20% of our train set. Only speakers with at most 0.25 h are eligible, so no
single speaker is more than about 1% of train.

## Repository layout

| Path | Contents |
|------|----------|
| `LICENSE` | Apache-2.0, covers the code only |
| `requirements.txt` | Pinned Python packages for the shared code and tests |
| `pytest.ini` | pytest configuration (repo root on the import path, tests in `tests/`) |
| `data/make_splits.py` | Selects the train speakers and writes `data/clean/train_speakers.txt` |
| `data/clean/train_speakers.txt` | Speaker IDs of our train split, one per line |
| `data/get_raw.py` | Copies the clips of our splits from the extracted USC archive into `data/raw/usc/` |
| `data/raw/usc/<split>/<speaker_id>/` | Our clips as lossless 16 kHz mono FLAC (`<clip_id>.flac`) with the unchanged USC transcript (`<clip_id>.txt`) |
| `tests/` | pytest tests for the data pipeline |

## Reproducing

The USC archive (9.47 GB) is not in this repository; only the clips we use are,
under `data/raw/`. To rebuild them from the original archive:

```bash
pip install -r requirements.txt
curl -L -o ISSAI_USC.tar.gz https://huggingface.co/datasets/issai/Uzbek_Speech_Corpus/resolve/main/ISSAI_USC.tar.gz
shasum -a 256 ISSAI_USC.tar.gz  # 44dc6c70cde073548d975dcdcd3c44ca228ca24ea179905a6cbf5b66ed2fa92d
tar -xzf ISSAI_USC.tar.gz
python data/make_splits.py --usc-dir ISSAI_USC
python data/get_raw.py --usc-dir ISSAI_USC
pytest
```

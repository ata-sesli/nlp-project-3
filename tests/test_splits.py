"""Checks on the committed split data: data/raw/ and, once written, data/clean/."""

import csv
from pathlib import Path

import pytest

from data.make_splits import speaker_id

REPO = Path(__file__).resolve().parent.parent
RAW = REPO / "data" / "raw" / "usc"
CLEAN = REPO / "data" / "clean"
SPLITS = ("train", "dev", "test")


def _raw_speakers(split):
    return {d.name for d in (RAW / split).iterdir() if d.is_dir()}


def _csv(split):
    path = CLEAN / f"{split}.csv"
    if not path.exists():
        pytest.skip(f"{path} not written yet")
    return list(csv.DictReader(open(path, encoding="utf-8")))


def test_no_speaker_in_two_raw_splits():
    speakers = {s: _raw_speakers(s) for s in SPLITS}

    for i, a in enumerate(SPLITS):
        for b in SPLITS[i + 1:]:
            assert not speakers[a] & speakers[b], f"speakers in both {a} and {b}"


def test_raw_train_speakers_match_train_speakers_txt():
    listed = set((CLEAN / "train_speakers.txt").read_text(encoding="utf-8").split())

    assert _raw_speakers("train") == listed


def test_every_raw_clip_has_audio_and_transcript_in_its_speaker_folder():
    for split in SPLITS:
        flacs = {p.with_suffix("") for p in (RAW / split).rglob("*.flac")}
        txts = {p.with_suffix("") for p in (RAW / split).rglob("*.txt")}
        assert flacs == txts, f"{split}: audio without transcript or transcript without audio"
        assert all(speaker_id(p.name) == p.parent.name for p in flacs)


def test_no_speaker_in_two_clean_splits():
    speakers = {s: {r["speaker_id"] for r in _csv(s)} for s in SPLITS}

    for i, a in enumerate(SPLITS):
        for b in SPLITS[i + 1:]:
            assert not speakers[a] & speakers[b], f"speakers in both {a}.csv and {b}.csv"


def test_every_clean_csv_path_exists_and_matches_its_row():
    for split in SPLITS:
        for row in _csv(split):
            path = REPO / row["path"]
            assert path.exists(), row["path"]
            assert path.stem == row["id"]
            assert path.parts[-3:-1] == (split, row["speaker_id"])
            assert 0 <= float(row["start_s"]) < float(row["end_s"]) <= float(row["duration_s"])

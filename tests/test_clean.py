import csv
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from data.clean import (
    CSV_COLUMNS,
    REMOVED_COLUMNS,
    Decision,
    Rule,
    clean,
    load_clips,
    rules_over_limit,
    write_outputs,
)

RATE = 16000
REPO = Path(__file__).resolve().parent.parent


def _clip(root, split, clip_id, seconds=1.0, seed=0, text="salom dunyo"):
    spk = clip_id.split("_")[0]
    d = root / "data" / "raw" / "usc" / split / spk
    d.mkdir(parents=True, exist_ok=True)
    samples = np.random.default_rng(seed).integers(-3000, 3000, int(seconds * RATE), dtype=np.int16)
    sf.write(d / f"{clip_id}.flac", samples, RATE, format="FLAC", subtype="PCM_16")
    (d / f"{clip_id}.txt").write_text(text + "\n", encoding="utf-8")


def _load(root):
    return load_clips(root / "data" / "raw" / "usc", root)


def _by_id(clips):
    return {c.id: c for c in clips}


def test_load_clips_reads_metadata(tmp_path):
    _clip(tmp_path, "dev", "111_1_1_1", seconds=1.5, text="  Salom, dunyo!  ")

    (clip,) = _load(tmp_path)

    assert clip.id == "111_1_1_1"
    assert clip.split == "dev"
    assert clip.speaker_id == "111"
    assert clip.path == "data/raw/usc/dev/111/111_1_1_1.flac"
    assert clip.transcript == "Salom, dunyo!"
    assert clip.duration_s == pytest.approx(1.5)
    assert (clip.start_s, clip.end_s) == (0.0, pytest.approx(1.5))


def test_duplicate_audio_within_split_keeps_first_id(tmp_path):
    _clip(tmp_path, "train", "111_1_1_1", seed=1)
    _clip(tmp_path, "train", "222_1_1_1", seed=1)

    kept, log = clean(_load(tmp_path), rules=[])

    assert [c.id for c in kept] == ["111_1_1_1"]
    ((clip, decision),) = log
    assert (clip.id, decision.action, decision.reason) == ("222_1_1_1", "removed", "duplicate_audio")
    assert "111_1_1_1" in decision.detail


def test_duplicate_audio_with_different_transcripts_removes_all_train_copies(tmp_path):
    _clip(tmp_path, "train", "111_1_1_1", seed=1, text="birinchi gap")
    _clip(tmp_path, "train", "111_1_2_1", seed=1, text="ikkinchi gap")
    _clip(tmp_path, "train", "222_1_1_1", seed=2)

    kept, log = clean(_load(tmp_path), rules=[])

    assert [c.id for c in kept] == ["222_1_1_1"]
    assert sorted((c.id, d.action) for c, d in log) == [
        ("111_1_1_1", "removed"), ("111_1_2_1", "removed")]
    assert all("different transcript" in d.detail for _, d in log)


def test_duplicate_audio_with_different_transcripts_in_test_is_only_flagged(tmp_path):
    _clip(tmp_path, "test", "111_1_1_1", seed=1, text="birinchi gap")
    _clip(tmp_path, "test", "111_1_2_1", seed=1, text="ikkinchi gap")

    kept, log = clean(_load(tmp_path), rules=[])

    assert len(kept) == 2
    assert sorted((c.id, d.action) for c, d in log) == [
        ("111_1_1_1", "flagged"), ("111_1_2_1", "flagged")]


def test_duplicate_audio_across_train_and_dev_removes_train_copy(tmp_path):
    _clip(tmp_path, "train", "111_1_1_1", seed=1)
    _clip(tmp_path, "dev", "222_1_1_1", seed=1)

    kept, log = clean(_load(tmp_path), rules=[])

    assert [(c.split, c.id) for c in kept] == [("dev", "222_1_1_1")]
    assert [(c.id, d.action) for c, d in log] == [("111_1_1_1", "removed")]


def test_duplicate_audio_across_dev_and_test_is_only_flagged(tmp_path):
    _clip(tmp_path, "dev", "111_1_1_1", seed=1)
    _clip(tmp_path, "test", "222_1_1_1", seed=1)

    kept, log = clean(_load(tmp_path), rules=[])

    assert len(kept) == 2
    assert {(c.id, d.action) for c, d in log} == {("111_1_1_1", "flagged"), ("222_1_1_1", "flagged")}


def test_too_long_clips_are_removed_from_train_only(tmp_path):
    _clip(tmp_path, "train", "111_1_1_1", seconds=31, seed=1)
    _clip(tmp_path, "train", "222_1_1_1", seconds=2, seed=2)
    _clip(tmp_path, "dev", "333_1_1_1", seconds=31, seed=3)

    kept, log = clean(_load(tmp_path))

    assert sorted(c.id for c in kept) == ["222_1_1_1", "333_1_1_1"]
    assert [(c.id, d.action, d.reason) for c, d in log] == [("111_1_1_1", "removed", "too_long")]


def test_rule_cannot_remove_from_splits_it_does_not_own(tmp_path):
    _clip(tmp_path, "train", "111_1_1_1", seed=1)
    _clip(tmp_path, "dev", "222_1_1_1", seed=2)
    always = Rule("bad_text", lambda c: Decision("removed", "bad_text", "x"))

    kept, log = clean(_load(tmp_path), rules=[always])

    assert [c.id for c in kept] == ["222_1_1_1"]
    assert {(c.split, d.action) for c, d in log} == {("train", "removed"), ("dev", "flagged")}


def test_fixed_decision_updates_clip_and_records_change(tmp_path):
    _clip(tmp_path, "train", "111_1_1_1", seconds=2, seed=1, text="o`zbek")
    fix = Rule(
        "apostrophe",
        lambda c: Decision("fixed", "apostrophe", transcript=c.transcript.replace("`", "'"),
                           start_s=0.5) if "`" in c.transcript else None,
    )

    kept, log = clean(_load(tmp_path), rules=[fix])

    (clip,) = kept
    assert (clip.transcript, clip.start_s) == ("o'zbek", 0.5)
    ((_, decision),) = log
    assert decision.action == "fixed"
    assert "o`zbek" in decision.detail and "o'zbek" in decision.detail


def test_rule_only_runs_on_its_splits(tmp_path):
    _clip(tmp_path, "dev", "111_1_1_1", seed=1)
    seen = []
    rule = Rule("train_only", lambda c: seen.append(c.id), splits=("train",))

    clean(_load(tmp_path), rules=[rule])

    assert seen == []


def test_rules_over_limit_counts_removed_and_flagged_per_split(tmp_path):
    for i in range(200):
        _clip(tmp_path, "train", f"{100 + i}_1_1_1", seed=i)
    clips = _load(tmp_path)
    log = [(c, Decision("flagged", "noisy")) for c in clips[:3]]
    log += [(c, Decision("fixed", "apostrophe")) for c in clips[:50]]

    assert rules_over_limit(clips, log) == ["noisy: 3 of 200 train clips flagged (1.5%)"]
    assert rules_over_limit(clips, log[:2]) == []


def test_write_outputs(tmp_path):
    _clip(tmp_path, "train", "111_1_1_1", seed=1)
    _clip(tmp_path, "train", "222_1_1_1", seed=1)
    _clip(tmp_path, "test", "333_1_1_1", seed=3)
    kept, log = clean(_load(tmp_path), rules=[])

    write_outputs(kept, log, tmp_path / "out")

    rows = list(csv.DictReader(open(tmp_path / "out" / "train.csv", encoding="utf-8")))
    assert list(rows[0]) == CSV_COLUMNS
    assert [r["id"] for r in rows] == ["111_1_1_1"]
    assert rows[0]["duration_s"] == "1.000"
    dev = list(csv.DictReader(open(tmp_path / "out" / "dev.csv", encoding="utf-8")))
    assert dev == []
    removed = list(csv.DictReader(open(tmp_path / "out" / "removed.csv", encoding="utf-8")))
    assert list(removed[0]) == REMOVED_COLUMNS
    assert [(r["id"], r["split"], r["action"]) for r in removed] == [("222_1_1_1", "train", "removed")]


# Checks on the committed outputs of data/clean.py (skipped until they exist).

CLEAN = REPO / "data" / "clean"


def _read(name):
    path = CLEAN / name
    if not path.exists():
        pytest.skip(f"{path} not written yet")
    return list(csv.DictReader(open(path, encoding="utf-8")))


def test_clean_csvs_and_removed_csv_cover_every_raw_clip():
    kept = {(split, r["id"]) for split in ("train", "dev", "test") for r in _read(f"{split}.csv")}
    removed = {(r["split"], r["id"]) for r in _read("removed.csv") if r["action"] == "removed"}
    raw = {(p.parts[-3], p.stem) for p in (REPO / "data" / "raw" / "usc").rglob("*.flac")}

    assert not kept & removed
    assert kept | removed == raw

import csv

import numpy as np
import pytest
import soundfile as sf

from data import rules as R
from data.clean import clean, load_clips, make_rules

RATE = 16000


def test_lookalike_letters_are_fixed_only_inside_latin_words():
    assert R.fix_lookalikes("mаktab") == "maktab"          # Cyrillic а inside a Latin word
    assert R.fix_lookalikes("салом") == "салом"  # a fully Cyrillic word stays


def test_apostrophe_variants_become_ascii():
    assert R.fix_apostrophes("o‘zbek g’alaba taʼlim ma`no") == "o'zbek g'alaba ta'lim ma'no"


def test_bad_characters():
    assert R.bad_characters("O'zbekiston - go'zal yurt!") == ""
    assert R.bad_characters("2024-yil") == "204"
    assert R.bad_characters("salom дунё") == "дунё"


def test_letter_count():
    assert R.letter_count("o'zbek tili.") == 9


def _tone(seconds, amp=8000, freq=220):
    t = np.arange(int(seconds * RATE)) / RATE
    return (amp * np.sin(2 * np.pi * freq * t)).clip(-32768, 32767).astype(np.int16)


def _write(path, x):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, x, RATE, format="FLAC", subtype="PCM_16")
    return str(path)


def test_audio_features_silence_and_clipping(tmp_path):
    noise = np.random.default_rng(0).integers(-20, 20, RATE, dtype=np.int16)
    x = np.concatenate([noise, _tone(1.0), noise])        # 1 s quiet, 1 s tone, 1 s quiet
    f = R.audio_features(_write(tmp_path / "a.flac", x))
    assert f["ok"] and f["speech_start"] == pytest.approx(1.0, abs=0.05)
    assert f["speech_end"] == pytest.approx(2.0, abs=0.05)
    assert f["clipped"] == 0.0 and f["snr_db"] > 30

    silent = R.audio_features(_write(tmp_path / "s.flac", np.zeros(RATE, dtype=np.int16)))
    assert silent["peak"] < R.SILENT_PEAK

    loud = R.audio_features(_write(tmp_path / "c.flac", _tone(1.0, amp=40000)))
    assert loud["clipped"] > R.MAX_CLIPPED

    bad = tmp_path / "broken.flac"
    bad.write_bytes(b"not audio")
    assert R.audio_features(str(bad))["ok"] is False


def test_check_result():
    rows = [{"rule": "9", "really_wrong": a} for a in ("yes", "yes", "no")]
    assert R.check_result(rows, "9") == ("confirmed", 3, 2, 1)
    assert R.check_result(rows, "8") == ("none", 0, 0, 0)
    assert R.check_result(rows + [{"rule": "9", "really_wrong": ""}], "9")[0] == "pending"
    assert R.check_result([{"rule": "9", "really_wrong": "no"}], "9")[0] == "rejected"


def _clip(root, split, clip_id, x, text):
    spk = clip_id.split("_")[0]
    d = root / "data" / "raw" / "usc" / split / spk
    _write(d / f"{clip_id}.flac", x)
    (d / f"{clip_id}.txt").write_text(text + "\n", encoding="utf-8")


def _speech(seconds, seed):
    rng = np.random.default_rng(seed)
    return (rng.normal(0, 3000, int(seconds * RATE))).clip(-32767, 32767).astype(np.int16)


def test_make_rules_end_to_end(tmp_path):
    for i in range(20):  # normal train clips: 2.6-3.55 s, about 9 letters per second
        _clip(tmp_path, "train", f"{100 + i}_1_1_1", _speech(2.6 + 0.05 * i, i), "salom dunyo bu yerda gap bor ekan")
    quiet = np.zeros(int(1.5 * RATE), dtype=np.int16)
    _clip(tmp_path, "train", "200_1_1_1", np.concatenate([quiet, _speech(3, 50)]), "mаktab o‘quvchi bilan gap bor ekan")
    _clip(tmp_path, "train", "201_1_1_1", _speech(3, 51), "2024 yil")
    _clip(tmp_path, "train", "202_1_1_1", np.zeros(RATE, dtype=np.int16), "salom")
    _clip(tmp_path, "dev", "300_1_1_1", _speech(3, 60), "a")                       # far too few letters
    _clip(tmp_path, "dev", "301_1_1_1", _speech(3, 61), "salom дунё bor ekan")

    clips = load_clips(tmp_path / "data" / "raw" / "usc", tmp_path)
    feats = [R.audio_features(str(tmp_path / c.path)) for c in clips]
    rules, limits = make_rules(clips, feats, tmp_path / "manual_check.csv", tmp_path / "rule8_cer.csv")
    kept, log = clean(clips, rules)
    got = {(c.id, d.reason): d.action for c, d in log}

    assert got[("200_1_1_1", "cyrillic_lookalike")] == "fixed"
    assert got[("200_1_1_1", "apostrophe")] == "fixed"
    assert got[("200_1_1_1", "long_silence")] == "fixed"
    assert got[("201_1_1_1", "non_uzbek_latin")] == "removed"
    assert got[("202_1_1_1", "broken_file")] == "removed"
    assert got[("300_1_1_1", "text_audio_mismatch")] == "flagged"   # dev: never removed by this rule
    assert got[("301_1_1_1", "non_uzbek_latin")] == "flagged"
    fixed = next(c for c in kept if c.id == "200_1_1_1")
    assert fixed.transcript == "maktab o'quvchi bilan gap bor ekan"
    assert fixed.start_s == pytest.approx(1.5 - R.KEEP_SILENCE_S, abs=0.05)
    assert {c.id for c in kept} >= {"300_1_1_1", "301_1_1_1"}
    assert limits["by_ear"]["8"]["result"] == "none"


def test_rule8_needs_the_by_ear_check_before_removing(tmp_path):
    for i in range(10):
        _clip(tmp_path, "train", f"{100 + i}_1_1_1", _speech(2.6 + 0.1 * i, i), "salom dunyo bu yerda gap bor ekan")
    clips = load_clips(tmp_path / "data" / "raw" / "usc", tmp_path)
    feats = [R.audio_features(str(tmp_path / c.path)) for c in clips]
    with open(tmp_path / "rule8_cer.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "speaker_id", "path", "duration_s", "cer", "flagged", "transcript", "model_output"])
        w.writerow(["100_1_1_1", "100", "p", "3", "0.9000", "yes", "t", "m"])
        w.writerow(["101_1_1_1", "101", "p", "3", "0.1000", "no", "t", "m"])

    def run():
        rules, _ = make_rules(clips, feats, tmp_path / "manual_check.csv", tmp_path / "rule8_cer.csv")
        return {(c.id, d.reason): d.action for c, d in clean(clips, rules)[1] if d.reason == "wrong_transcript"}

    assert run() == {("100_1_1_1", "wrong_transcript"): "flagged"}
    R.add_check_sample(tmp_path / "manual_check.csv", "8", [{"id": "100_1_1_1", "path": "p", "transcript": "t"}])
    assert run()[("100_1_1_1", "wrong_transcript")] == "flagged"        # check not filled yet
    rows = R.read_checks(tmp_path / "manual_check.csv")
    rows[0]["really_wrong"] = "yes"
    with open(tmp_path / "manual_check.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=R.CHECK_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    assert run()[("100_1_1_1", "wrong_transcript")] == "removed"


def test_clip_heard_wrong_is_removed_even_if_group_is_kept(tmp_path):
    for i in range(10):
        _clip(tmp_path, "train", f"{100 + i}_1_1_1", _speech(2.6 + 0.1 * i, i), "salom dunyo bu yerda gap bor ekan")
    clips = load_clips(tmp_path / "data" / "raw" / "usc", tmp_path)
    feats = [R.audio_features(str(tmp_path / c.path)) for c in clips]
    with open(tmp_path / "rule8_cer.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "speaker_id", "path", "duration_s", "cer", "flagged", "transcript", "model_output"])
        for i in range(3):
            w.writerow([f"{100 + i}_1_1_1", "s", "p", "3", "0.9000", "yes", "t", "m"])
    with open(tmp_path / "manual_check.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=R.CHECK_COLUMNS)
        w.writeheader()
        for i, a in enumerate(("yes", "no", "no")):
            w.writerow({"rule": "8", "id": f"{100 + i}_1_1_1", "really_wrong": a})
    rules, limits = make_rules(clips, feats, tmp_path / "manual_check.csv", tmp_path / "rule8_cer.csv")
    got = {c.id: d.action for c, d in clean(clips, rules)[1] if d.reason == "wrong_transcript"}
    assert limits["by_ear"]["8"]["result"] == "rejected"
    assert got == {"100_1_1_1": "removed", "101_1_1_1": "flagged", "102_1_1_1": "flagged"}

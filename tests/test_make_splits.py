import wave

import pytest

from data.make_splits import hours_by_speaker, select_speakers, speaker_id, speakers_in


@pytest.mark.parametrize(
    "clip_id, expected",
    [
        ("217014490_1_11634_2", "217014490"),
        ("1459541555_1_28768_1", "1459541555"),
        ("qiyomat1.2_data_qiyomat1.2-5", "qiyomat1.2"),
        ("oy_quyosh_data_oy_quyosh-13", "oy_quyosh"),
        ("12_data_12-3", "12"),
        ("3_A_3_A-64", "3_A"),
        ("24_E_24_E-65", "24_E"),
    ],
)
def test_speaker_id(clip_id, expected):
    assert speaker_id(clip_id) == expected


@pytest.mark.parametrize("clip_id", ["unknown", "123_4", "book_data_other-1"])
def test_speaker_id_rejects_unknown_patterns(clip_id):
    with pytest.raises(ValueError):
        speaker_id(clip_id)


def _write_wav(path, seconds, rate=16000):
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(rate)
        f.writeframes(b"\x00\x00" * int(seconds * rate))


def test_hours_by_speaker_sums_clip_durations(tmp_path):
    _write_wav(tmp_path / "111_1_1_1.wav", 1.8)
    _write_wav(tmp_path / "111_1_2_1.wav", 1.8)
    _write_wav(tmp_path / "book_data_book-1.wav", 3.6)
    (tmp_path / "222_1_1_1.wav").write_bytes(b"")  # empty file counts as 0 s

    hours, n_clips = hours_by_speaker(tmp_path)

    assert hours == pytest.approx({"111": 0.001, "book": 0.001, "222": 0.0})
    assert n_clips == {"111": 2, "book": 1, "222": 1}


HOURS = {f"s{i:02d}": 1.0 + (i % 3) for i in range(20)}  # 1, 2 or 3 h each
CAP = 2.5  # the 3 h speakers are never eligible


def _select(hours=HOURS, seed=42, exclude=frozenset()):
    return select_speakers(hours, target_hours=10.0, max_speaker_hours=CAP,
                           seed=seed, exclude=exclude)


def test_select_speakers_uses_whole_speakers_without_exceeding_target():
    chosen, total = _select()

    assert total == pytest.approx(sum(HOURS[s] for s in chosen))
    assert total <= 10.0
    assert chosen == sorted(chosen)
    assert len(set(chosen)) == len(chosen)


def test_select_speakers_skips_only_speakers_that_would_overshoot():
    chosen, total = _select()

    skipped = [s for s in HOURS if s not in chosen and HOURS[s] <= CAP]
    assert skipped
    assert all(total + HOURS[s] > 10.0 for s in skipped)


def test_select_speakers_respects_per_speaker_cap():
    chosen, _ = _select()

    assert all(HOURS[s] <= CAP for s in chosen)


def test_select_speakers_is_deterministic_and_seed_dependent():
    a = _select()
    b = _select(hours=dict(reversed(list(HOURS.items()))))
    c = _select(seed=7)

    assert a == b
    assert a[0] != c[0]


def test_select_speakers_never_picks_excluded_speakers():
    excluded = {f"s{i:02d}" for i in range(0, 20, 2)}

    chosen, _ = _select(exclude=excluded)

    assert not set(chosen) & excluded


def test_speakers_in_lists_split_speakers(tmp_path):
    for name in ("111_1_1_1", "111_1_2_1", "222_2_1_1", "book_data_book-1"):
        (tmp_path / f"{name}.wav").write_bytes(b"")

    assert speakers_in(tmp_path) == {"111", "222", "book"}


def test_select_speakers_fails_when_eligible_audio_is_below_target():
    with pytest.raises(ValueError):
        _select(hours={"a": 1.0, "b": 2.0, "big": 30.0})

import wave

import numpy as np
import soundfile as sf

from data.get_raw import select_clips, write_clip

RATE = 16000


def _write_clip(split_dir, clip_id, samples, text="salom dunyo o'zbek"):
    split_dir.mkdir(parents=True, exist_ok=True)
    with wave.open(str(split_dir / f"{clip_id}.wav"), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(RATE)
        f.writeframes(np.asarray(samples, dtype="<i2").tobytes())
    (split_dir / f"{clip_id}.txt").write_text(text, encoding="utf-8")
    return split_dir / f"{clip_id}.wav"


def test_select_clips_keeps_selected_train_speakers_and_all_dev_test(tmp_path):
    for split, clip in [
        ("train", "111_1_1_1"), ("train", "111_1_2_1"), ("train", "222_1_1_1"),
        ("train", "book_data_book-1"), ("dev", "333_1_1_1"), ("test", "444_1_1_1"),
    ]:
        _write_clip(tmp_path / split, clip, [0])

    clips = select_clips(tmp_path, train_speakers={"111", "book"})

    assert [(split, wav.stem) for split, wav in clips] == [
        ("train", "111_1_1_1"), ("train", "111_1_2_1"), ("train", "book_data_book-1"),
        ("dev", "333_1_1_1"), ("test", "444_1_1_1"),
    ]


def test_write_clip_is_lossless_and_keeps_transcript_bytes(tmp_path):
    samples = np.random.default_rng(42).integers(-32768, 32767, RATE, dtype=np.int16)
    text = "Bu  o‘zbekcha matn, о kirill harfi bilan.\n"
    wav = _write_clip(tmp_path / "usc" / "dev", "333_1_1_1", samples, text)

    written = write_clip("dev", wav, tmp_path / "raw")

    flac = tmp_path / "raw" / "dev" / "333" / "333_1_1_1.flac"
    assert written
    data, rate = sf.read(flac, dtype="int16")
    assert rate == RATE
    assert np.array_equal(data, samples)
    assert flac.with_suffix(".txt").read_bytes() == text.encode("utf-8")
    assert not list((tmp_path / "raw").rglob("*.part"))


def test_write_clip_never_rewrites_existing_audio(tmp_path):
    wav = _write_clip(tmp_path / "usc" / "dev", "333_1_1_1", [1, 2, 3])
    write_clip("dev", wav, tmp_path / "raw")
    flac = tmp_path / "raw" / "dev" / "333" / "333_1_1_1.flac"
    before = flac.read_bytes()

    _write_clip(tmp_path / "usc" / "dev", "333_1_1_1", [9, 9, 9])
    written = write_clip("dev", wav, tmp_path / "raw")

    assert not written
    assert flac.read_bytes() == before

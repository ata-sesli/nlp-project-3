"""Copy the clips of our splits from the extracted USC archive into data/raw/.

For every selected clip, writes the audio as lossless FLAC and copies the
original transcript byte for byte:

    data/raw/usc/<split>/<speaker_id>/<clip_id>.flac
    data/raw/usc/<split>/<speaker_id>/<clip_id>.txt

Train keeps only the speakers in data/clean/train_speakers.txt (written by
make_splits.py); dev and test keep every clip. All USC clips are 16 kHz mono
16-bit WAV, stored as 16-bit FLAC.

Audio is written once and never rewritten: existing output files are skipped,
and each file is written under a temporary name and renamed when complete, so
an interrupted run can be restarted safely.

Usage:
    python data/get_raw.py --usc-dir /path/to/ISSAI_USC
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data.make_splits import OUT_PATH as TRAIN_SPEAKERS_PATH, speaker_id  # noqa: E402

RAW_DIR = Path(__file__).parent / "raw" / "usc"
SPLITS = ("train", "dev", "test")


def select_clips(usc_dir: Path, train_speakers: set[str]) -> list[tuple[str, Path]]:
    clips = []
    for split in SPLITS:
        for wav in sorted(Path(usc_dir, split).glob("*.wav")):
            if split != "train" or speaker_id(wav.stem) in train_speakers:
                clips.append((split, wav))
    return clips


def _write_atomic(out: Path, write) -> None:
    part = out.with_name(out.name + ".part")
    write(part)
    os.replace(part, out)


def write_clip(split: str, wav: Path, raw_dir: Path = RAW_DIR) -> bool:
    """Write one clip's FLAC and transcript. Returns False if the FLAC exists."""
    out_dir = Path(raw_dir, split, speaker_id(wav.stem))
    flac = out_dir / f"{wav.stem}.flac"
    if flac.exists():
        return False
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_atomic(
        flac.with_suffix(".txt"),
        lambda p: shutil.copyfile(wav.with_suffix(".txt"), p),
    )
    data, rate = sf.read(wav, dtype="int16")
    _write_atomic(
        flac,
        lambda p: sf.write(p, data, rate, format="FLAC", subtype="PCM_16"),
    )
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--usc-dir", type=Path, required=True,
        help="extracted ISSAI_USC directory (contains train/, dev/, test/)",
    )
    args = parser.parse_args()

    train_speakers = set(TRAIN_SPEAKERS_PATH.read_text(encoding="utf-8").split())
    clips = select_clips(args.usc_dir, train_speakers)

    for split in SPLITS:
        todo = [wav for s, wav in clips if s == split]
        written = 0
        for i, wav in enumerate(todo, 1):
            written += write_clip(split, wav)
            if i % 2000 == 0 or i == len(todo):
                print(f"{split}: {i}/{len(todo)} clips ({written} written)", flush=True)
        size = sum(p.stat().st_size for p in (RAW_DIR / split).rglob("*") if p.is_file())
        print(f"{split}: done, {size / 1e6:.0f} MB in {RAW_DIR / split}", flush=True)


if __name__ == "__main__":
    main()

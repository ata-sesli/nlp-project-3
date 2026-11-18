"""Select the train speakers for our 25-hour USC train split.

USC ships no speaker metadata (the official recipe maps each utterance to
itself in utt2spk), so the speaker is taken from the clip file name:

* crowdsourced clips, ``<speaker>_<n>_<n>_<n>``: the first number. With this
  rule dev and test have exactly the 41 and 38 speakers the USC paper reports;
* audiobook clips, ``<book>_data_<book>-<n>``, and the 350 train clips named
  ``<n>_<A|E>_<n>_<A|E>-<k>``: the repeated prefix (book or ``3_A`` group).
  The real narrator is unknown, so one person can count as several "speakers"
  here. These patterns occur only in train.

Unknown file name patterns raise an error instead of being guessed.

Contrary to the USC paper, 26 crowdsourced speakers of the official dev (10)
and test (16) also appear in official train. Dev and test stay complete and
unchanged; every dev/test speaker is excluded from the train candidates.

The remaining train speakers are sorted, shuffled with random.Random(42), and
whole speakers are added in that order, skipping any speaker that would push
the total past the target (25.0 h). To keep train balanced across speakers,
speakers with more than MAX_SPEAKER_HOURS (0.25 h) of train audio are not
eligible: hours per speaker are very uneven (median under a minute, maximum
5.2 h), and without the cap one speaker would be about 20% of train. The chosen
speaker IDs are written to data/clean/train_speakers.txt.

Usage:
    python data/make_splits.py --usc-dir /path/to/ISSAI_USC [--hours 25.0]
"""

import argparse
import random
import re
import wave
from collections import defaultdict
from pathlib import Path

SEED = 42
MAX_SPEAKER_HOURS = 0.25
OUT_PATH = Path(__file__).parent / "clean" / "train_speakers.txt"

CROWD_RE = re.compile(r"^(?P<speaker>\d+)_\d+_\d+_\d+$")
AUDIOBOOK_RE = re.compile(r"^(?P<speaker>.+)_(?:data_)?(?P=speaker)-\d+$")


def speaker_id(clip_id: str) -> str:
    for pattern in (AUDIOBOOK_RE, CROWD_RE):
        match = pattern.match(clip_id)
        if match:
            return match["speaker"]
    raise ValueError(f"unknown USC clip name pattern: {clip_id!r}")


def wav_seconds(path: Path) -> float:
    """Duration of a WAV file; empty or unreadable files count as 0 s
    (clean.py removes them later), as in the official USC recipe."""
    try:
        with wave.open(str(path), "rb") as f:
            return f.getnframes() / f.getframerate()
    except (wave.Error, EOFError):
        return 0.0


def hours_by_speaker(train_dir: Path) -> tuple[dict[str, float], dict[str, int]]:
    hours: dict[str, float] = defaultdict(float)
    n_clips: dict[str, int] = defaultdict(int)
    for path in sorted(Path(train_dir).glob("*.wav")):
        spk = speaker_id(path.stem)
        hours[spk] += wav_seconds(path) / 3600
        n_clips[spk] += 1
    return dict(hours), dict(n_clips)


def speakers_in(split_dir: Path) -> set[str]:
    return {speaker_id(p.stem) for p in Path(split_dir).glob("*.wav")}


def select_speakers(
    hours: dict[str, float],
    target_hours: float,
    max_speaker_hours: float = MAX_SPEAKER_HOURS,
    seed: int = SEED,
    exclude: set[str] = frozenset(),
) -> tuple[list[str], float]:
    speakers = sorted(
        s for s in set(hours) - exclude if hours[s] <= max_speaker_hours
    )
    available = sum(hours[s] for s in speakers)
    if available < target_hours:
        raise ValueError(
            f"only {available:.2f} h from speakers with <= {max_speaker_hours} h, "
            f"target is {target_hours} h"
        )
    random.Random(seed).shuffle(speakers)
    chosen: list[str] = []
    total = 0.0
    for spk in speakers:
        if total + hours[spk] <= target_hours:
            chosen.append(spk)
            total += hours[spk]
    return sorted(chosen), total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--usc-dir", type=Path, required=True,
        help="extracted ISSAI_USC directory (contains train/, dev/, test/)",
    )
    parser.add_argument("--hours", type=float, default=25.0)
    args = parser.parse_args()

    hours, n_clips = hours_by_speaker(args.usc_dir / "train")
    held_out = speakers_in(args.usc_dir / "dev") | speakers_in(args.usc_dir / "test")
    shared = held_out & set(hours)
    chosen, total = select_speakers(hours, args.hours, exclude=held_out)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text("".join(f"{spk}\n" for spk in chosen), encoding="utf-8")

    print(f"official train: {len(hours)} speakers, {sum(n_clips.values())} clips, "
          f"{sum(hours.values()):.2f} h")
    print(f"excluded (also in dev/test): {len(shared)} speakers, "
          f"{sum(n_clips[s] for s in shared)} clips, "
          f"{sum(hours[s] for s in shared):.2f} h")
    over_cap = {s for s in hours if hours[s] > MAX_SPEAKER_HOURS} - shared
    print(f"not eligible (> {MAX_SPEAKER_HOURS} h each): {len(over_cap)} speakers, "
          f"{sum(hours[s] for s in over_cap):.2f} h")
    print(f"selected (seed {SEED}): {len(chosen)} speakers, "
          f"{sum(n_clips[s] for s in chosen)} clips, {total:.2f} h, "
          f"largest speaker {max(hours[s] for s in chosen) / total:.1%} of train")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()

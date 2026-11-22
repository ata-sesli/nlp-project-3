"""Shared helpers for the Whisper systems (A4, A5 zero-shot and C1 fine-tune).

* load_split(): reads data/clean/<split>.csv. Before Gate 1 (no clean CSVs yet)
  it falls back to scanning data/raw/usc/<split>/ and says so loudly; results
  from that fallback are for trying things out only.
* load_audio(): reads one clip as float32 16 kHz mono, cut to [start_s, end_s].
* normalize(): data/normalize.py once it exists; until then a local copy of the
  agreed policy (lowercase, apostrophe variants -> ', punctuation removed but
  word-internal ' kept, spaces collapsed). Only for provisional dev numbers;
  official scores come from benchmark/evaluate.py.
* write_predictions() / write_run_log(): the shared benchmark/ formats.
"""

import csv
import json
import platform
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

SAMPLE_RATE = 16_000
SEED = 42
CLEAN_DIR = ROOT / "data" / "clean"
RAW_DIR = ROOT / "data" / "raw" / "usc"
PRED_DIR = ROOT / "benchmark" / "predictions"
RESULTS_DIR = ROOT / "benchmark" / "results"


# ---------------------------------------------------------------- data

def load_split(split: str) -> list[dict]:
    csv_path = CLEAN_DIR / f"{split}.csv"
    if csv_path.exists():
        with open(csv_path, encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            r["path"] = str(ROOT / r["path"])
            for k in ("duration_s", "start_s", "end_s"):
                r[k] = float(r[k])
        return rows

    warnings.warn(
        f"{csv_path} not found (before Gate 1): scanning data/raw/usc/{split}/ instead. "
        "Use these numbers only for trying things out.", stacklevel=2)
    rows = []
    for flac in sorted(Path(RAW_DIR, split).glob("*/*.flac"), key=lambda p: p.stem):
        dur = sf.info(flac).duration
        rows.append({
            "id": flac.stem,
            "path": str(flac),
            "transcript": flac.with_suffix(".txt").read_text(encoding="utf-8").strip(),
            "speaker_id": flac.parent.name,
            "duration_s": dur, "start_s": 0.0, "end_s": dur,
        })
    if not rows:
        raise FileNotFoundError(f"no clips for split {split!r}: is data/raw/usc/{split}/ checked out?")
    return rows


def load_audio(row: dict) -> np.ndarray:
    start = int(round(row["start_s"] * SAMPLE_RATE))
    stop = int(round(row["end_s"] * SAMPLE_RATE))
    audio, rate = sf.read(row["path"], start=start, stop=stop, dtype="float32", always_2d=False)
    if rate != SAMPLE_RATE:
        raise ValueError(f"{row['path']}: {rate} Hz, expected {SAMPLE_RATE}")
    return audio


def guard_test(split: str, after_freeze: bool) -> None:
    """Test is run once per system, after the freeze (Friday 20 Nov evening)."""
    if split == "test" and not after_freeze:
        sys.exit("refusing to run on test before the freeze; pass --after-freeze once everything is frozen")


# ---------------------------------------------------------------- text

_APOSTROPHES = str.maketrans({c: "'" for c in "‘’`ʻʼ´"})


def _fallback_normalize(text: str) -> str:
    text = text.lower().translate(_APOSTROPHES)
    text = re.sub(r"[^\w\s']", " ", text)   # punctuation -> space
    text = re.sub(r"(?<!\w)'", " ", text)   # ' not after a letter is a quote mark
    text = text.replace("_", " ")
    return " ".join(text.split())


try:  # Elif's shared normalizer, once it is merged
    from data.normalize import normalize  # type: ignore  # noqa: F401
except ImportError:
    normalize = _fallback_normalize


def provisional_scores(refs: list[str], hyps: list[str]) -> dict:
    import jiwer
    pairs = [(normalize(r), normalize(h)) for r, h in zip(refs, hyps)]
    pairs = [(r, h) for r, h in pairs if r]  # jiwer cannot score an empty reference
    r, h = [p[0] for p in pairs], [p[1] for p in pairs]
    return {"wer": jiwer.wer(r, h), "cer": jiwer.cer(r, h), "n_scored": len(pairs)}


# ---------------------------------------------------------------- outputs

def write_predictions(system: str, split: str, ids: list[str], hyps: list[str]) -> Path:
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    out = PRED_DIR / f"{system}__{split}.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for i, h in zip(ids, hyps):
            f.write(json.dumps({"id": i, "hyp": h}, ensure_ascii=False) + "\n")
    return out


def write_run_log(system: str, split: str, info: dict) -> Path:
    """benchmark/results/<system>.json; one file per system, one block per split.

    Each split's block keeps its own hardware and versions under "environment", because
    splits can run on different machines (dev on an A100, test on a T4); the top-level
    copies describe the latest run."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"{system}.json"
    log = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    runs = log.pop("runs", {})
    runs[split] = info.pop("run")
    runs[split]["environment"] = {k: info[k] for k in ("hardware", "python", "torch", "transformers")
                                  if k in info}
    log.update(info)
    log["runs"] = runs
    out.write_text(json.dumps(log, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return out


def hub_revision(model_id: str) -> str | None:
    """Commit hash of a Hugging Face model; None for a local directory."""
    if Path(model_id).exists():
        return None
    from huggingface_hub import HfApi
    return HfApi().model_info(model_id).sha


def environment() -> dict:
    import torch
    import transformers
    return {
        "hardware": torch.cuda.get_device_name(0) if torch.cuda.is_available() else f"CPU ({platform.processor()})",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }

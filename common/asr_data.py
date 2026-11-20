"""Shared helpers for the models in baselines/scratch, baselines/nvidia_fastconformer and xlsr_ctc.

What is here:
- read_split(): reads data/clean/<split>.csv (columns: id, path, transcript, speaker_id,
  duration_s, start_s, end_s).
- load_audio(): reads one clip as 16 kHz mono float32, cut to [start_s, end_s].
- normalize(): the team text normalizer (data/normalize.py). If that file is missing,
  a copy of the plan's rules (B3) is used and a warning is printed.
- write_predictions() / update_run_log(): the file formats of plan section B5.
- set_seed(), env_info(), hf_revision(), RTFTimer.
"""
from __future__ import annotations

import csv
import json
import os
import platform
import random
import re
import sys
import time
import unicodedata
import warnings
from pathlib import Path

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[1]
CLEAN_DIR = REPO_ROOT / "data" / "clean"
# Set ASR_OUTPUT_ROOT for test runs on fake data, so nothing is written into benchmark/.
_OUT_ROOT = Path(os.environ.get("ASR_OUTPUT_ROOT", REPO_ROOT / "benchmark"))
PRED_DIR = _OUT_ROOT / "predictions"
RESULTS_DIR = _OUT_ROOT / "results"
SAMPLE_RATE = 16000
SEED = 42
SPLITS = ("train", "dev", "test")
TEST_SPLITS = ("test",)

# ---------------------------------------------------------------- seeds


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ---------------------------------------------------------------- text

_APOSTROPHES = "‘’`ʻʼ´′"  # ‘ ’ ` ʻ ʼ ´ ′


def _fallback_normalize(text: str) -> str:
    """Plan B3: lowercase, one apostrophe, no punctuation, no number or script conversion."""
    text = unicodedata.normalize("NFC", text).lower()
    for ch in _APOSTROPHES:
        text = text.replace(ch, "'")
    # Any character that is not a letter, digit, space or apostrophe becomes a space.
    text = re.sub(r"[^\w\s']|_", " ", text)
    # Keep ' only inside a word or after o/g (o', g' are letters). Drop the others (quotes).
    text = re.sub(r"(?<![^\W\d_])'|'(?![^\W\d_])(?<![og]')", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _load_team_normalizer():
    sys.path.insert(0, str(REPO_ROOT))
    try:
        from data.normalize import normalize as team_normalize  # Elif's file

        return team_normalize
    except Exception as err:  # file not merged yet, or other function name
        warnings.warn(
            f"data/normalize.py not usable ({err!r}); using the copy of plan B3 in common/asr_data.py. "
            "Final scores are always computed by benchmark/evaluate.py, so this only affects dev WER "
            "printed during training."
        )
        return _fallback_normalize


normalize = _load_team_normalizer()


# ---------------------------------------------------------------- data


def resolve_path(path: str, root: Path = REPO_ROOT) -> Path:
    p = Path(path)
    return p if p.is_absolute() else root / p


def read_split(split: str, clean_dir: str | Path | None = None, limit: int | None = None,
               root: Path = REPO_ROOT) -> list[dict]:
    """Rows of data/clean/<split>.csv with an absolute 'audio' path and float times."""
    csv_path = Path(clean_dir or CLEAN_DIR) / f"{split}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} not found. Has the data step (Ata) been merged?")
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            r["audio"] = str(resolve_path(r["path"], root))
            for key in ("duration_s", "start_s", "end_s"):
                r[key] = float(r[key]) if r.get(key) not in (None, "") else None
            if r["start_s"] is None:
                r["start_s"] = 0.0
            if r["end_s"] is None:
                r["end_s"] = r["duration_s"]
            rows.append(r)
    if limit:
        rows = rows[:limit]
    return rows


def clip_seconds(row: dict) -> float:
    if row.get("end_s") is not None:
        return max(0.0, row["end_s"] - (row.get("start_s") or 0.0))
    return float(row["duration_s"])


def _resample(wav: np.ndarray, sr: int) -> np.ndarray:
    if sr == SAMPLE_RATE:
        return wav
    from math import gcd

    from scipy.signal import resample_poly

    g = gcd(sr, SAMPLE_RATE)
    return resample_poly(wav, SAMPLE_RATE // g, sr // g).astype(np.float32)


def load_audio(row: dict) -> np.ndarray:
    """16 kHz mono float32, cut to [start_s, end_s] (rule 10 silence trim)."""
    path = row["audio"] if "audio" in row else str(resolve_path(row["path"]))
    info = sf.info(path)
    sr = info.samplerate
    start = int(round((row.get("start_s") or 0.0) * sr))
    end = row.get("end_s")
    stop = int(round(end * sr)) if end is not None else None
    wav, sr = sf.read(path, start=start, stop=stop, dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)
    return _resample(wav, sr)


# ---------------------------------------------------------------- outputs


def write_predictions(system: str, testset: str, ids: list[str], hyps: list[str]) -> Path:
    """benchmark/predictions/<system>__<testset>.jsonl, one {"id", "hyp"} per line."""
    assert testset in ("dev", "test"), testset
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    out = PRED_DIR / f"{system}__{testset}.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for i, h in zip(ids, hyps):
            f.write(json.dumps({"id": i, "hyp": h}, ensure_ascii=False) + "\n")
    return out


def update_run_log(system: str, info: dict) -> Path:
    """Merge info into benchmark/results/<system>.json (dev and test runs add to one file)."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"{system}.json"
    data = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    _deep_update(data, info)
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out


def _deep_update(base: dict, new: dict) -> None:
    for k, v in new.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v


def env_info() -> dict:
    """Package versions and GPU, for the run log (send these versions to Ata for requirements)."""
    from importlib import metadata

    info = {"python": platform.python_version()}
    for pkg in ("torch", "torchaudio", "transformers", "nemo_toolkit", "jiwer", "soundfile",
                "numpy", "scipy", "huggingface_hub", "safetensors"):
        try:
            info[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            pass
    try:
        import torch

        info["cuda"] = torch.version.cuda
        info["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except ImportError:
        pass
    return info


def hf_revision(model_id: str, revision: str | None = None) -> str | None:
    """Commit hash of a Hugging Face model (plan: always record the exact revision)."""
    try:
        from huggingface_hub import HfApi

        return HfApi().model_info(model_id, revision=revision).sha
    except Exception as err:
        warnings.warn(f"Could not read the revision of {model_id}: {err!r}")
        return revision


def cer(ref: str, hyp: str) -> float:
    import jiwer

    if not ref:
        return 0.0 if not hyp else 1.0
    return float(jiwer.cer(ref, hyp))


def wer_cer(refs: list[str], hyps: list[str]) -> tuple[float, float]:
    """Corpus WER and CER on normalized text (only for choosing checkpoints on dev)."""
    import jiwer

    pairs = [(normalize(r), normalize(h)) for r, h in zip(refs, hyps)]
    pairs = [(r, h) for r, h in pairs if r]  # jiwer cannot score an empty reference
    r = [p[0] for p in pairs]
    h = [p[1] for p in pairs]
    return float(jiwer.wer(r, h)), float(jiwer.cer(r, h))


class RTFTimer:
    """Real-time factor = model processing time / audio duration (plan B5)."""

    def __init__(self):
        self.proc = 0.0
        self.audio = 0.0
        self._t = None

    def start(self):
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except ImportError:
            pass
        self._t = time.perf_counter()

    def stop(self, audio_seconds: float):
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except ImportError:
            pass
        self.proc += time.perf_counter() - self._t
        self.audio += audio_seconds

    @property
    def rtf(self) -> float:
        return self.proc / self.audio if self.audio else float("nan")


def check_test_allowed(split: str, allow_test: bool) -> None:
    """Plan B4d: test sets are used once, after the freeze."""
    if split in TEST_SPLITS and not allow_test:
        raise SystemExit(
            f"'{split}' is a test set. Plan rule: run it only once, after the freeze "
            "(Saturday 21 November). Add --allow-test when that time comes."
        )

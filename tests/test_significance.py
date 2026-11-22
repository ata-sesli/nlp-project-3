import csv
import json

import numpy as np
import pytest

from benchmark.evaluate import compute_metrics
from benchmark.significance import (
    clip_errors,
    holm,
    main,
    paired_bootstrap,
    speaker_bootstrap,
    speaker_totals,
)

REFS = ["Salom, O‘zbekiston bugun!", "bir ikki uch", "...", "katta rahmat"]
HYPS = ["salom o'zbekiston ertaga", "bir uch", "nimadir", "katta rahmat sizga"]


@pytest.mark.parametrize("unit, key", [("word", "wer_norm"), ("char", "cer_norm")])
def test_summed_clip_errors_equal_evaluate_py(unit, key):
    errors, n = clip_errors(REFS, HYPS, unit)

    # The clip whose normalized reference is empty is skipped, as in evaluate.py.
    assert n[2] == 0 and errors[2] == 0
    assert errors.sum() / n.sum() == pytest.approx(compute_metrics(REFS, HYPS)[key], abs=1e-6)


def test_speaker_totals_sum_clips_per_speaker():
    errors, n = np.array([1, 2, 3]), np.array([4, 5, 6])

    e, t = speaker_totals(errors, n, ["b", "a", "b"], ["a", "b"])

    assert e.tolist() == [2, 4]
    assert t.tolist() == [5, 10]


def _resample(n_speakers, seed=42, n=2000):
    return np.random.default_rng(seed).integers(0, n_speakers, size=(n, n_speakers))


def test_speaker_bootstrap_ci_contains_estimate_and_is_deterministic():
    e = np.array([1, 5, 2, 8, 3, 0, 4, 6])
    t = np.array([10, 20, 10, 30, 15, 10, 20, 25])

    est, low, high = speaker_bootstrap(e, t, _resample(8))

    assert est == pytest.approx(e.sum() / t.sum())
    assert low < est < high
    assert (low, high) == speaker_bootstrap(e, t, _resample(8))[1:]


def test_paired_bootstrap_identical_systems_has_p_one():
    e = np.array([1, 5, 2, 8, 3])
    t = np.array([10, 20, 10, 30, 15])

    diff, low, high, p = paired_bootstrap(e, e, t, _resample(5))

    assert (diff, low, high) == (0.0, 0.0, 0.0)
    assert p == 1.0


def test_paired_bootstrap_clearly_better_system_is_significant():
    t = np.full(30, 20)
    better, worse = np.full(30, 2), np.full(30, 6)

    diff, low, high, p = paired_bootstrap(better, worse, t, _resample(30))

    assert diff == pytest.approx(-0.2)
    assert high < 0
    assert p < 0.001


def test_paired_bootstrap_p_value_is_never_zero():
    t = np.full(30, 20)

    *_, p = paired_bootstrap(np.full(30, 2), np.full(30, 6), t, _resample(30, n=999))

    assert p == pytest.approx(2 / 1000)


def test_holm_adjusts_and_keeps_order():
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert holm([0.6, 0.9]) == [1.0, 1.0]


def _write(path, rows, fields):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def test_main_writes_intervals_and_comparisons(tmp_path):
    clean, preds, out = tmp_path / "clean", tmp_path / "preds", tmp_path / "out"
    clean.mkdir(), preds.mkdir()
    rows = [{"id": f"{s}_{i}", "transcript": "bir ikki uch tort", "speaker_id": str(s)}
            for s in range(10) for i in range(3)]
    _write(clean / "test.csv", rows, ["id", "transcript", "speaker_id"])
    systems = {"A4_whisper-small": "bir ikki", "C1_whisper-small-ft": "bir ikki uch tort",
               "A8_nvidia-fastconformer-rnnt": "bir ikki uch"}
    for system, hyp in systems.items():
        with open(preds / f"{system}__test.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps({"id": r["id"], "hyp": hyp}) + "\n")

    main(["--predictions-dir", str(preds), "--clean-dir", str(clean), "--out-dir", str(out),
          "--resamples", "500"])

    cis = list(csv.DictReader(open(out / "confidence_intervals.csv", encoding="utf-8")))
    assert {(r["system"], r["metric"]) for r in cis} == {(s, m) for s in systems for m in ("wer_norm", "cer_norm")}
    c1 = next(r for r in cis if r["system"] == "C1_whisper-small-ft" and r["metric"] == "wer_norm")
    assert (c1["n_speakers"], float(c1["estimate"]), float(c1["ci_high"])) == ("10", 0.0, 0.0)

    pairs = list(csv.DictReader(open(out / "significance.csv", encoding="utf-8")))
    # Only C1 vs A4 has both prediction files; C4a/A8 has no C4a file.
    assert {(r["system"], r["baseline"]) for r in pairs} == {("C1_whisper-small-ft", "A4_whisper-small")}
    wer = next(r for r in pairs if r["metric"] == "wer_norm")
    assert float(wer["difference"]) == pytest.approx(-0.5)
    assert float(wer["p_holm"]) < 0.01

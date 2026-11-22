"""Confidence intervals and p-values for the scores of benchmark/evaluate.py.

Reads the same files as evaluate.py (benchmark/predictions/<system>__<split>.jsonl
and data/clean/<split>.csv), uses the same normalization and the same WER/CER
definition (total errors / total reference words or characters, clips with an
empty normalized reference skipped), and writes:

    benchmark/results/confidence_intervals.csv  95% CI of every system's WER and CER
    benchmark/results/significance.csv          paired comparisons with p-values

Method: speaker-level bootstrap. Clips of one speaker are not independent, so
the unit that is resampled is the speaker, not the clip (Bisani and Ney, ICASSP
2004, compare both; resampling clips gives intervals that are too narrow). Each
resample draws the split's speakers with replacement (seed 42); WER is
recomputed from the summed errors and words of the drawn speakers. The 95% CI
is the 2.5th to 97.5th percentile of the resampled values.

Paired comparisons use the same resampled speakers for both systems. The
difference is system minus baseline (negative = the system is better). The
two-sided p-value is twice the share of resamples in which the difference has
the other sign, (count + 1) / (resamples + 1) so it is never 0, capped at 1.
Holm's correction is applied over all comparisons of one metric (p_holm).

Usage (from the repo root, after the predictions exist):
    python benchmark/significance.py              # test split, 10,000 resamples
    python benchmark/significance.py --split dev
"""

import argparse
import csv
import sys
from pathlib import Path

import jiwer
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from benchmark.evaluate import load_predictions, load_references  # noqa: E402
from data.normalize import normalize_text  # noqa: E402

SEED = 42
METRICS = {"wer_norm": "word", "cer_norm": "char"}
# (system, what it is compared with), by row label. Pairs whose prediction files
# do not exist yet are skipped.
COMPARISONS = [
    ("C1", "A4"),      # Whisper-small: fine-tuned vs zero-shot
    ("C2long", "C2"),  # XLS-R: trained longer vs first run
    ("C3", "C2"),      # XLS-R: + word LM vs greedy
    ("C4a", "A8"),     # NVIDIA transducer head: fine-tuned vs ready model
    ("C4b", "A9"),     # NVIDIA CTC head: fine-tuned vs ready model
]


def clip_errors(refs: list[str], hyps: list[str], unit: str) -> tuple[np.ndarray, np.ndarray]:
    """Errors (S + D + I) and reference length per clip on normalized text."""
    process = jiwer.process_words if unit == "word" else jiwer.process_characters
    errors, n = np.zeros(len(refs), dtype=np.int64), np.zeros(len(refs), dtype=np.int64)
    for i, (ref, hyp) in enumerate(zip(refs, hyps)):
        ref, hyp = normalize_text(ref), normalize_text(hyp)
        if not ref:
            continue
        out = process(ref, hyp)
        errors[i] = out.substitutions + out.deletions + out.insertions
        n[i] = out.substitutions + out.deletions + out.hits
    return errors, n


def speaker_totals(errors: np.ndarray, n: np.ndarray, clip_speakers: list[str],
                   speakers: list[str]) -> tuple[np.ndarray, np.ndarray]:
    index = {s: i for i, s in enumerate(speakers)}
    rows = np.array([index[s] for s in clip_speakers])
    e, t = np.zeros(len(speakers), dtype=np.int64), np.zeros(len(speakers), dtype=np.int64)
    np.add.at(e, rows, errors)
    np.add.at(t, rows, n)
    return e, t


def _rates(e: np.ndarray, t: np.ndarray, resample: np.ndarray) -> np.ndarray:
    return e[resample].sum(axis=1) / t[resample].sum(axis=1)


def speaker_bootstrap(e: np.ndarray, t: np.ndarray, resample: np.ndarray) -> tuple[float, float, float]:
    """Error rate and its 95% CI; resample is (n_resamples, n_speakers) speaker indices."""
    low, high = np.percentile(_rates(e, t, resample), [2.5, 97.5])
    return float(e.sum() / t.sum()), float(low), float(high)


def paired_bootstrap(e_sys: np.ndarray, e_base: np.ndarray, t: np.ndarray,
                     resample: np.ndarray) -> tuple[float, float, float, float]:
    """Difference system - baseline, its 95% CI and the two-sided p-value."""
    diffs = _rates(e_sys, t, resample) - _rates(e_base, t, resample)
    low, high = np.percentile(diffs, [2.5, 97.5])
    other_side = min(int((diffs <= 0).sum()), int((diffs >= 0).sum()))
    p = min(1.0, 2 * (other_side + 1) / (len(diffs) + 1))
    return float((e_sys.sum() - e_base.sum()) / t.sum()), float(low), float(high), p


def holm(pvalues: list[float]) -> list[float]:
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    adjusted, running = [0.0] * len(pvalues), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvalues) - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted


def _system_for(label: str, systems: list[str]) -> str | None:
    return next((s for s in systems if s.split("_")[0] == label), None)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--predictions-dir", type=Path, default=Path("benchmark/predictions"))
    parser.add_argument("--clean-dir", type=Path, default=Path("data/clean"))
    parser.add_argument("--out-dir", type=Path, default=Path("benchmark/results"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--resamples", type=int, default=10_000)
    args = parser.parse_args(argv)

    csv_path = args.clean_dir / f"{args.split}.csv"
    refs_map = load_references(csv_path)
    ids = list(refs_map)
    with csv_path.open(encoding="utf-8") as f:
        speaker_of = {r["id"]: r["speaker_id"] for r in csv.DictReader(f)}
    clip_speakers = [speaker_of[i] for i in ids]
    speakers = sorted(set(clip_speakers))
    resample = np.random.default_rng(SEED).integers(0, len(speakers), size=(args.resamples, len(speakers)))

    totals = {}  # (system, metric) -> per-speaker errors, per-speaker reference length
    for path in sorted(args.predictions_dir.glob(f"*__{args.split}.jsonl")):
        system = path.stem.rsplit("__", 1)[0]
        preds = load_predictions(path)
        missing = set(ids) - set(preds)
        if missing:
            raise ValueError(f"{path.name} is missing {len(missing)} clip IDs from {args.split}.csv")
        hyps = [preds[i] for i in ids]
        for metric, unit in METRICS.items():
            errors, n = clip_errors([refs_map[i] for i in ids], hyps, unit)
            totals[system, metric] = speaker_totals(errors, n, clip_speakers, speakers)
        print(f"scored {system}", flush=True)

    systems = sorted({s for s, _ in totals})
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.out_dir / "confidence_intervals.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["system", "split", "metric", "n_clips", "n_speakers", "estimate", "ci_low", "ci_high",
                     "resamples"])
        for system in systems:
            for metric in METRICS:
                est, low, high = speaker_bootstrap(*totals[system, metric], resample)
                w.writerow([system, args.split, metric, len(ids), len(speakers),
                            f"{est:.6f}", f"{low:.6f}", f"{high:.6f}", args.resamples])

    pairs = [(_system_for(a, systems), _system_for(b, systems)) for a, b in COMPARISONS]
    pairs = [(a, b) for a, b in pairs if a and b]
    rows = []
    for metric in METRICS:
        results = []
        for system, baseline in pairs:
            (e_sys, t), (e_base, _) = totals[system, metric], totals[baseline, metric]
            results.append((system, baseline, *paired_bootstrap(e_sys, e_base, t, resample)))
        for (system, baseline, diff, low, high, p), p_holm in zip(results, holm([r[-1] for r in results])):
            rows.append([system, baseline, args.split, metric,
                         f"{totals[system, metric][0].sum() / totals[system, metric][1].sum():.6f}",
                         f"{totals[baseline, metric][0].sum() / totals[baseline, metric][1].sum():.6f}",
                         f"{diff:.6f}", f"{low:.6f}", f"{high:.6f}", f"{p:.6f}", f"{p_holm:.6f}"])
    with open(args.out_dir / "significance.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["system", "baseline", "split", "metric", "system_value", "baseline_value", "difference",
                     "ci_low", "ci_high", "p", "p_holm"])
        w.writerows(rows)
    print(f"wrote {args.out_dir}/confidence_intervals.csv ({len(systems)} systems) "
          f"and significance.csv ({len(pairs)} comparisons)")


if __name__ == "__main__":
    main()

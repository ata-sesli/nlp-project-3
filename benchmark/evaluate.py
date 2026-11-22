"""Evaluate ASR predictions against cleaned USC references and write scores.csv."""

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)
from typing import Dict, List, Tuple

import jiwer

from data.normalize import (
    has_cyrillic,
    has_digits,
    normalize_text,
    normalize_translit_and_numbers,
)


def load_references(csv_path: Path) -> Dict[str, str]:
    """Load id -> transcript mapping from data/clean/{dev,test}.csv."""
    refs: Dict[str, str] = {}
    with csv_path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            refs[row["id"]] = row["transcript"]
    return refs


def load_predictions(jsonl_path: Path) -> Dict[str, str]:
    """Load id -> hyp mapping from benchmark/predictions/<sys>__<split>.jsonl."""
    preds: Dict[str, str] = {}
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            preds[item["id"]] = item.get("hyp", "")
    return preds


def compute_metrics(refs: List[str], hyps: List[str]) -> Dict[str, float | int]:
    """Compute WER (norm), CER (norm), WER (orthographic), WER (translit+num), and diagnostics."""
    norm_refs = [normalize_text(r) for r in refs]
    norm_hyps = [normalize_text(h) for h in hyps]

    translit_refs = [normalize_translit_and_numbers(r) for r in refs]
    translit_hyps = [normalize_translit_and_numbers(h) for h in hyps]

    # Drop pairs where normalized reference is empty to avoid jiwer division-by-zero
    valid_norm = [(r, h) for r, h in zip(norm_refs, norm_hyps) if r]
    valid_ortho = [(r.strip(), h.strip()) for r, h in zip(refs, hyps) if r.strip()]
    valid_translit = [(r, h) for r, h in zip(translit_refs, translit_hyps) if r]

    wer_norm = jiwer.wer([r for r, _ in valid_norm], [h for _, h in valid_norm]) if valid_norm else 0.0
    cer_norm = jiwer.cer([r for r, _ in valid_norm], [h for _, h in valid_norm]) if valid_norm else 0.0
    wer_ortho = jiwer.wer([r for r, _ in valid_ortho], [h for _, h in valid_ortho]) if valid_ortho else 0.0
    wer_translit = (
        jiwer.wer([r for r, _ in valid_translit], [h for _, h in valid_translit])
        if valid_translit
        else 0.0
    )

    cyrillic_count = sum(1 for h in hyps if has_cyrillic(h))
    digit_count = sum(1 for h in hyps if has_digits(h))

    return {
        "n_clips": len(refs),
        "wer_norm": round(wer_norm, 6),
        "cer_norm": round(cer_norm, 6),
        "wer_ortho": round(wer_ortho, 6),
        "wer_translit_num": round(wer_translit, 6),
        "hyp_with_cyrillic": cyrillic_count,
        "hyp_with_digits": digit_count,
    }


def evaluate_file(pred_path: Path, clean_dir: Path, results_dir: Path) -> Dict[str, object]:
    """Evaluate a single <system>__<split>.jsonl file."""
    stem = pred_path.stem
    if "__" not in stem:
        raise ValueError(f"Expected filename '<system>__<split>.jsonl', got: {pred_path.name}")
    system, split = stem.rsplit("__", 1)

    ref_path = clean_dir / f"{split}.csv"
    refs_map = load_references(ref_path)
    preds_map = load_predictions(pred_path)

    missing = set(refs_map.keys()) - set(preds_map.keys())
    if missing:
        raise ValueError(f"{pred_path.name} is missing {len(missing)} clip IDs from {split}.csv")

    ordered_ids = list(refs_map.keys())
    refs = [refs_map[cid] for cid in ordered_ids]
    hyps = [preds_map[cid] for cid in ordered_ids]

    metrics = compute_metrics(refs, hyps)

    # Pull optional metadata (RTF, hardware, params) from benchmark/results/<system>.json
    meta_path = results_dir / f"{system}.json"
    rtf, hardware, params = "", "", ""
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        run_meta = meta.get("runs", {}).get(split, {})
        run_env = run_meta.get("environment", {})

        rtf = run_meta.get("rtf", meta.get("rtf", ""))
        hardware = run_env.get("hardware", meta.get("hardware", ""))
        params = meta.get("parameters", meta.get("params", ""))

    return {
        "system": system,
        "split": split,
        **metrics,
        "rtf": rtf,
        "hardware": hardware,
        "params": params,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Score all ASR prediction files.")
    parser.add_argument("--predictions-dir", type=Path, default=Path("benchmark/predictions"))
    parser.add_argument("--clean-dir", type=Path, default=Path("data/clean"))
    parser.add_argument("--results-dir", type=Path, default=Path("benchmark/results"))
    parser.add_argument("--out", type=Path, default=Path("benchmark/results/scores.csv"))
    args = parser.parse_args()

    pred_files = sorted(args.predictions_dir.glob("*__*.jsonl"))
    rows = [evaluate_file(p, args.clean_dir, args.results_dir) for p in pred_files]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "system", "split", "n_clips",
        "wer_norm", "cer_norm", "wer_ortho", "wer_translit_num",
        "hyp_with_cyrillic", "hyp_with_digits",
        "rtf", "hardware", "params",
    ]
    with args.out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
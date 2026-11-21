"""A1: pick the LM weight on dev, write predictions and the run log.

On dev, every LM weight in --lmwt and word insertion penalty in --wip is tried
on the decoding lattices; the pair with the lowest provisional dev WER (shared
normalizer, references from data/clean/dev.csv) is saved to
exp/<model>/decode_dev/best_params.json. Test uses only that saved pair.

Writes benchmark/predictions/A1_gmm-hmm__<split>.jsonl and
benchmark/results/A1_gmm-hmm.json. RTF = wall-clock time of MFCC + CMVN +
decoding (exp/timing_<split>.txt, --nj parallel CPU jobs) / audio time.

Usage (called by run.sh stage 9):
    python baselines/gmm_hmm/score.py --work ~/gmm_hmm_work --model tri3b --split dev
"""

import argparse
import json
import platform
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "baselines" / "whisper_zero_shot"))
import whisper_common as wc  # noqa: E402

SYSTEM = "A1_gmm-hmm"
STAGES = {"mono": "monophone", "tri1": "triphone + deltas", "tri2b": "triphone LDA+MLLT",
          "tri3b": "triphone LDA+MLLT+SAT (fMLLR)"}


def best_paths(work: Path, decode: Path, words: Path, lmwt: int, wip: float) -> dict[str, str]:
    cmd = (f"lattice-scale --inv-acoustic-scale={lmwt} 'ark:gunzip -c {decode}/lat.*.gz|' ark:- | "
           f"lattice-add-penalty --word-ins-penalty={wip} ark:- ark:- | "
           f"lattice-best-path --word-symbol-table={words} ark:- ark,t:- 2>/dev/null | "
           f"{work}/utils/int2sym.pl -f 2- {words}")
    out = subprocess.run(["bash", "-c", cmd], check=True, capture_output=True, text=True, encoding="utf-8").stdout
    hyps = {}
    for line in out.splitlines():
        utt, *ws = line.split()
        hyps[utt] = " ".join(w for w in ws if w != "<unk>")
    return hyps


def cpu_name() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown CPU"


def gaussians(work: Path, model: str) -> int | None:
    out = subprocess.run(["gmm-info", str(work / "exp" / model / "final.mdl")],
                         capture_output=True, text=True).stdout
    m = re.search(r"number of gaussians (\d+)", out)
    return int(m.group(1)) if m else None


def kaldi_version() -> str | None:
    meta = sorted(Path(sys.prefix, "conda-meta").glob("kaldi-*.json"))
    return json.loads(meta[-1].read_text())["version"] if meta else None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--model", default="tri3b", choices=list(STAGES))
    p.add_argument("--split", default="dev", choices=["dev", "test"])
    p.add_argument("--after-freeze", action="store_true")
    p.add_argument("--nj", type=int, default=12)
    p.add_argument("--lmwt", type=int, nargs="+", default=list(range(7, 18)))
    p.add_argument("--wip", type=float, nargs="+", default=[0.0, 0.5, 1.0])
    args = p.parse_args()
    wc.guard_test(args.split, args.after_freeze)

    exp = args.work / "exp" / args.model
    decode = exp / f"decode_{args.split}"
    words = exp / "graph" / "words.txt"
    uttmap = dict(line.split("\t") for line in
                  (args.work / "data" / f"uttmap_{args.split}.tsv").read_text(encoding="utf-8").splitlines())
    rows = wc.load_split(args.split)
    refs = [r["transcript"] for r in rows]

    def hyps_for(lmwt, wip):
        by_clip = {uttmap[u]: h for u, h in best_paths(args.work, decode, words, lmwt, wip).items()}
        return [by_clip.get(r["id"], "") for r in rows]

    params_file = exp / "decode_dev" / "best_params.json"
    if args.split == "dev":
        grid = []
        for lmwt in args.lmwt:
            for wip in args.wip:
                s = wc.provisional_scores(refs, hyps_for(lmwt, wip))
                grid.append({"lmwt": lmwt, "wip": wip, "wer": s["wer"], "cer": s["cer"]})
                print(f"LMWT {lmwt:2d}  WIP {wip:.1f}  WER {s['wer']:.2%}  CER {s['cer']:.2%}")
        best = min(grid, key=lambda g: g["wer"])
        params_file.write_text(json.dumps({"best": best, "grid": grid}, indent=2), encoding="utf-8")
    else:
        saved = json.loads(params_file.read_text(encoding="utf-8"))
        best = saved["best"]
        # report the grid that was searched on dev, not this run's defaults
        args.lmwt = sorted({g["lmwt"] for g in saved["grid"]})
        args.wip = sorted({g["wip"] for g in saved["grid"]})
    hyps = hyps_for(best["lmwt"], best["wip"])

    pred = wc.write_predictions(SYSTEM, args.split, [r["id"] for r in rows], hyps)
    scores = wc.provisional_scores(refs, hyps)
    audio_s = sum(r["end_s"] - r["start_s"] for r in rows)
    timing = args.work / "exp" / f"timing_{args.split}.txt"
    wall_s = sum(float(x) for x in timing.read_text().split()) if timing.exists() else None
    # written by stage 2; missing when only the saved models were restored (eval-only test run)
    lm_file = args.work / "data" / "local" / "lm_source.txt"
    lm_source = (lm_file.read_text(encoding="utf-8").strip() if lm_file.exists() else
                 "LM: baselines/gmm_hmm/lm_arpa.py (interpolated Kneser-Ney 3-gram, train transcripts)")

    log = wc.write_run_log(SYSTEM, args.split, {
        "system": SYSTEM,
        "model_id": f"Kaldi GMM-HMM {args.model}: {STAGES[args.model]}, trained by us",
        "revision": None,
        "parameters": None,
        "gaussians": gaussians(args.work, args.model),
        "toolkit": {"kaldi_conda_forge": kaldi_version(),
                    "kaldi_scripts_commit": subprocess.run(
                        ["git", "-C", str(Path.home() / "kaldi-src"), "rev-parse", "HEAD"],
                        capture_output=True, text=True).stdout.strip() or None},
        "features": "13 MFCC + per-speaker CMVN (+deltas / LDA+MLLT / fMLLR by stage)",
        "lexicon": "graphemes; o', g', sh, ch, ng and ' are one unit each",
        "lm": lm_source,
        "decoding": {"lmwt": best["lmwt"], "word_ins_penalty": best["wip"],
                     "selected_on": "dev", "grid_lmwt": args.lmwt, "grid_wip": args.wip},
        "seed": wc.SEED,
        "hardware": f"CPU: {cpu_name()}, {args.nj} parallel jobs",
        "python": platform.python_version(),
        "run": {
            "clips": len(rows),
            "decoded": sum(bool(h) for h in hyps),
            "audio_hours": round(audio_s / 3600, 3),
            "rtf": round(wall_s / audio_s, 5) if wall_s else None,
            "data_source": "data/clean CSV" if (wc.CLEAN_DIR / f"{args.split}.csv").exists()
                           else "data/raw scan (pre-Gate-1)",
            "provisional_scores": scores,
            "predictions": pred.relative_to(wc.ROOT).as_posix(),
        },
    })
    print(f"{args.split}: LMWT {best['lmwt']} WIP {best['wip']}  WER {scores['wer']:.2%}  CER {scores['cer']:.2%}")
    print(f"wrote {pred}\nwrote {log}")


if __name__ == "__main__":
    main()

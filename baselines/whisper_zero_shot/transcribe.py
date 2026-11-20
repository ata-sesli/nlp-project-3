"""Transcribe a split with a Whisper model and write the shared benchmark files.

Used for A4 (openai/whisper-small), A5 (openai/whisper-large-v3) zero-shot and,
with a local or uploaded checkpoint, for C1 (our whisper-small fine-tune).
Decoding: language="uzbek", task="transcribe", greedy unless --num-beams is set
(any decoding setting is chosen on dev only).

Writes
    benchmark/predictions/<system>__<split>.jsonl   {"id", "hyp"} per clip
    benchmark/results/<system>.json                 model ID, revision, params,
                                                    decoding, RTF, hardware
and prints a provisional WER/CER (official scores: benchmark/evaluate.py).

RTF = (feature extraction + generation time) / audio time, measured on the
GPU named in the run log. Audio loading from disk is not counted.

Usage:
    python baselines/whisper_zero_shot/transcribe.py --model openai/whisper-small --system A4_whisper-small
    python baselines/whisper_zero_shot/transcribe.py --model openai/whisper-large-v3 --system A5_whisper-large-v3
    python baselines/whisper_zero_shot/transcribe.py --model <dir or hub id> --system C1_whisper-small-ft --revision <hash>
    ... --split test --after-freeze     # once, after the freeze
"""

import argparse
import sys
import time
from pathlib import Path

import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

sys.path.insert(0, str(Path(__file__).resolve().parent))
import whisper_common as wc  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", required=True, help="Hugging Face ID or local checkpoint directory")
    p.add_argument("--system", required=True, help="row code + name, e.g. A4_whisper-small")
    p.add_argument("--split", default="dev", choices=["dev", "test", "train"])
    p.add_argument("--after-freeze", action="store_true", help="required for --split test")
    p.add_argument("--revision", help="commit hash to record (default: looked up on the Hub)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-beams", type=int, default=1)
    p.add_argument("--max-new-tokens", type=int, default=225)
    p.add_argument("--limit", type=int, help="first N clips only (quick try, not for results)")
    args = p.parse_args()
    wc.guard_test(args.split, args.after_freeze)
    torch.manual_seed(wc.SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    revision = args.revision or wc.hub_revision(args.model)
    processor = WhisperProcessor.from_pretrained(args.model, revision=revision)
    model = WhisperForConditionalGeneration.from_pretrained(
        args.model, revision=revision, dtype=dtype).to(device).eval()

    rows = wc.load_split(args.split)
    if args.limit:
        rows = rows[: args.limit]
    n_long = sum(r["end_s"] - r["start_s"] > 30 for r in rows)
    if n_long:
        print(f"note: {n_long} clips are longer than 30 s; Whisper sees only their first 30 s")

    # Sort by length so each batch pads little; restore the CSV order at the end.
    order = sorted(range(len(rows)), key=lambda i: rows[i]["end_s"] - rows[i]["start_s"])
    hyps: list[str] = [""] * len(rows)
    compute_s = 0.0
    audio_s = 0.0
    for b in range(0, len(order), args.batch_size):
        idx = order[b: b + args.batch_size]
        audios = [wc.load_audio(rows[i]) for i in idx]
        audio_s += sum(len(a) for a in audios) / wc.SAMPLE_RATE
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        feats = processor.feature_extractor(
            audios, sampling_rate=wc.SAMPLE_RATE, return_tensors="pt", return_attention_mask=True)
        with torch.inference_mode():
            out = model.generate(
                feats.input_features.to(device, dtype),
                attention_mask=feats.attention_mask.to(device),
                language="uzbek", task="transcribe",
                num_beams=args.num_beams, max_new_tokens=args.max_new_tokens,
            )
        texts = processor.batch_decode(out, skip_special_tokens=True)
        if device == "cuda":
            torch.cuda.synchronize()
        compute_s += time.perf_counter() - t0
        for i, t in zip(idx, texts):
            hyps[i] = t.strip()
        done = min(b + args.batch_size, len(order))
        if done % (args.batch_size * 20) < args.batch_size or done == len(order):
            print(f"{done}/{len(order)} clips", flush=True)

    ids = [r["id"] for r in rows]
    pred_path = wc.write_predictions(args.system, args.split, ids, hyps)
    scores = wc.provisional_scores([r["transcript"] for r in rows], hyps)
    used_clean_csv = (wc.CLEAN_DIR / f"{args.split}.csv").exists()

    log_path = wc.write_run_log(args.system, args.split, {
        "system": args.system,
        "model_id": args.model,
        "revision": revision,
        "parameters": sum(p.numel() for p in model.parameters()),
        "decoding": {"language": "uzbek", "task": "transcribe", "num_beams": args.num_beams,
                     "max_new_tokens": args.max_new_tokens, "dtype": str(dtype).replace("torch.", ""),
                     "batch_size": args.batch_size},
        "seed": wc.SEED,
        **wc.environment(),
        "run": {
            "clips": len(rows),
            "audio_hours": round(audio_s / 3600, 3),
            "rtf": round(compute_s / audio_s, 5),
            "data_source": "data/clean CSV" if used_clean_csv else "data/raw scan (pre-Gate-1)",
            "limit": args.limit,
            "provisional_scores": scores,
            "predictions": pred_path.relative_to(wc.ROOT).as_posix(),
        },
    })
    print(f"wrote {pred_path}\nwrote {log_path}")
    print(f"provisional {args.split}: WER {scores['wer']:.2%}  CER {scores['cer']:.2%}  "
          f"RTF {compute_s / audio_s:.4f} ({wc.environment()['hardware']})")


if __name__ == "__main__":
    main()

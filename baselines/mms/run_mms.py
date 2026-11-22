import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import json
import time
from pathlib import Path

import torch
from huggingface_hub import HfApi
from transformers import AutoProcessor, Wav2Vec2ForCTC

from common.asr_data import read_split, load_audio, env_info, check_test_allowed


MODEL_ID = "facebook/mms-1b-all"
TARGET_LANG = "uzb-script_latin"
SYSTEM = "A6_mms-1b"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["dev", "test"], required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--allow-test", action="store_true")
    args = parser.parse_args()

    check_test_allowed(args.split, args.allow_test)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU bulunamadı.")

    print("GPU:", torch.cuda.get_device_name(0))
    print("Processor yükleniyor...")

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    processor.tokenizer.set_target_lang(TARGET_LANG)

    print("MMS modeli yükleniyor...")
    model = Wav2Vec2ForCTC.from_pretrained(
        MODEL_ID,
        ignore_mismatched_sizes=True,
        torch_dtype=torch.float16,
    )

    model.to("cuda:0")

    print("Uzbek Latin adapter yükleniyor...")
    model.load_adapter(TARGET_LANG)
    model.to("cuda:0")
    model.eval()

    revision = HfApi().model_info(MODEL_ID).sha
    parameters = sum(p.numel() for p in model.parameters())

    rows = read_split(args.split, limit=args.limit)
    if not rows:
        raise RuntimeError("Veri kümesi boş.")

    predictions = []
    process_seconds = 0.0
    audio_seconds = 0.0

    for i, row in enumerate(rows, 1):
        audio = load_audio(row)
        duration = len(audio) / 16000

        inputs = processor(
            audio,
            sampling_rate=16000,
            return_tensors="pt",
        )
        inputs["input_values"] = inputs["input_values"].to(
            device=model.device,
            dtype=next(model.parameters()).dtype,
        )

        if "attention_mask" in inputs:
            inputs["attention_mask"] = inputs["attention_mask"].to(model.device)

        torch.cuda.synchronize()
        start = time.perf_counter()

        with torch.inference_mode():
            logits = model(**inputs).logits

        torch.cuda.synchronize()
        process_seconds += time.perf_counter() - start
        audio_seconds += duration

        hyp = processor.batch_decode(
            torch.argmax(logits, dim=-1)
        )[0]

        predictions.append({"id": row["id"], "hyp": hyp})

        if i % 100 == 0 or i == len(rows):
            print(f"{i}/{len(rows)} tamamlandı")

    expected = {"dev": 3783, "test": 3836}[args.split]
    if args.limit is None and len(predictions) != expected:
        raise RuntimeError(
            f"Beklenen {expected} klip, üretilen {len(predictions)}."
        )

    pred_path = Path(
        f"benchmark/predictions/{SYSTEM}__{args.split}.jsonl"
    )
    pred_path.parent.mkdir(parents=True, exist_ok=True)

    with pred_path.open("w", encoding="utf-8") as f:
        for item in predictions:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    environment = env_info()
    hardware = torch.cuda.get_device_name(0)

    run = {
        "clips": len(predictions),
        "audio_hours": round(audio_seconds / 3600, 3),
        "rtf": process_seconds / audio_seconds,
        "data_source": "data/clean CSV",
        "limit": args.limit,
        "predictions": str(pred_path),
        "environment": {
            "hardware": hardware,
            "python": environment.get("python"),
            "torch": environment.get("torch"),
            "transformers": environment.get("transformers"),
        },
    }

    result_path = Path(f"benchmark/results/{SYSTEM}.json")
    result_path.parent.mkdir(parents=True, exist_ok=True)

    if result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
    else:
        result = {
            "system": SYSTEM,
            "model_id": MODEL_ID,
            "revision": revision,
            "parameters": parameters,
            "decoding": {
                "method": "greedy CTC",
                "language": TARGET_LANG,
                "lm": None,
                "dtype": "float16",
            },
            "seed": 42,
            "hardware": hardware,
            "runs": {},
        }

    result["revision"] = revision
    result["parameters"] = parameters
    result["hardware"] = hardware
    result["runs"][args.split] = run

    result_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nTahmin dosyası:", pred_path)
    print("Sonuç dosyası:", result_path)
    print("Klip:", len(predictions))
    print("RTF:", round(run["rtf"], 5))
    print("GPU:", hardware)


if __name__ == "__main__":
    main()

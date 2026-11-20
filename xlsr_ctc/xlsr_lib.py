"""Shared code for C2 (XLS-R 300M + CTC): vocabulary, data loading, batching, greedy decoding."""
from __future__ import annotations

import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.asr_data import SAMPLE_RATE, RTFTimer, clip_seconds, load_audio, normalize  # noqa: E402

UNK, PAD, SEP = "[UNK]", "[PAD]", "|"
# XLSR_SYSTEM lets a second training stage write its own result files (e.g. C2long) next to C2.
SYSTEM = os.environ.get("XLSR_SYSTEM", "C2_xlsr-300m-ctc")
BASE_MODEL = "facebook/wav2vec2-xls-r-300m"


# ---------------------------------------------------------------- text and vocabulary


def target_text(transcript: str) -> str:
    """Training target = the team-normalized transcript (lowercase, one apostrophe, no punctuation)."""
    return normalize(transcript)


def build_vocab(texts: list[str]) -> dict[str, int]:
    """Character vocabulary from train transcripts; '|' is the word separator, [PAD] is the CTC blank."""
    chars = sorted(set("".join(texts)) - {" "})
    vocab = {c: i for i, c in enumerate(chars)}
    vocab[SEP] = len(vocab)
    vocab[UNK] = len(vocab)
    vocab[PAD] = len(vocab)
    return vocab


def make_processor(vocab_path: Path):
    from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor

    tok = Wav2Vec2CTCTokenizer(str(vocab_path), unk_token=UNK, pad_token=PAD, word_delimiter_token=SEP,
                               bos_token=None, eos_token=None)
    fe = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=SAMPLE_RATE, padding_value=0.0,
                                  do_normalize=True, return_attention_mask=True)
    return Wav2Vec2Processor(feature_extractor=fe, tokenizer=tok)


def ctc_labels(processor) -> list[str]:
    """Token list in id order, in the form pyctcdecode expects ('' = blank, ' ' = word separator)."""
    vocab = processor.tokenizer.get_vocab()
    labels = [""] * len(vocab)
    for tok, i in vocab.items():
        labels[i] = {PAD: "", SEP: " ", UNK: "⁇"}.get(tok, tok)
    return labels


# ---------------------------------------------------------------- data


class ClipDataset(torch.utils.data.Dataset):
    def __init__(self, rows: list[dict], label_ids: list[list[int]] | None = None):
        self.rows = rows
        self.label_ids = label_ids

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        wav = load_audio(self.rows[i])
        labels = self.label_ids[i] if self.label_ids is not None else None
        return i, wav, labels


class Collator:
    def __init__(self, processor):
        self.fe = processor.feature_extractor

    def __call__(self, items):
        idx = [it[0] for it in items]
        feats = self.fe([it[1] for it in items], sampling_rate=SAMPLE_RATE, padding=True,
                        return_attention_mask=True, return_tensors="pt")
        batch = {"idx": idx, "input_values": feats["input_values"], "attention_mask": feats["attention_mask"]}
        if items[0][2] is not None:
            longest = max(len(it[2]) for it in items)
            labels = torch.full((len(items), longest), -100, dtype=torch.long)
            for k, it in enumerate(items):
                labels[k, :len(it[2])] = torch.tensor(it[2], dtype=torch.long)
            batch["labels"] = labels
        return batch


def length_grouped_batches(lengths: list[float], batch_size: int, seed: int, mega: int = 50) -> list[list[int]]:
    """Shuffle, then sort inside groups of `mega` batches so a batch holds clips of similar length
    (less padding). Same seed -> same batches, so a resumed run continues the same epoch."""
    rng = random.Random(seed)
    idx = list(range(len(lengths)))
    rng.shuffle(idx)
    batches = []
    for s in range(0, len(idx), batch_size * mega):
        group = sorted(idx[s:s + batch_size * mega], key=lambda i: -lengths[i])
        batches += [group[k:k + batch_size] for k in range(0, len(group), batch_size)]
    rng.shuffle(batches)
    return batches


def sorted_batches(lengths: list[float], batch_size: int) -> list[list[int]]:
    """For inference: longest first, no shuffling."""
    order = sorted(range(len(lengths)), key=lambda i: -lengths[i])
    return [order[k:k + batch_size] for k in range(0, len(order), batch_size)]


# ---------------------------------------------------------------- decoding


@torch.no_grad()
def transcribe(model, processor, rows: list[dict], batch_size: int = 16, device: str = "cuda",
               fp16: bool = True, num_workers: int = 2, timer: RTFTimer | None = None,
               logits_dir: Path | None = None) -> list[str]:
    """Greedy CTC decoding. Optionally saves log-probabilities (float16, frames x vocab) per clip
    as <logits_dir>/<id>.npy for the KenLM decoder (C3)."""
    model.eval()
    if logits_dir is not None:
        logits_dir.mkdir(parents=True, exist_ok=True)
    lengths = [clip_seconds(r) for r in rows]
    loader = torch.utils.data.DataLoader(ClipDataset(rows), batch_sampler=sorted_batches(lengths, batch_size),
                                         collate_fn=Collator(processor), num_workers=num_workers)
    hyps: list[str] = [""] * len(rows)
    pad_id = processor.tokenizer.pad_token_id
    use_amp = fp16 and device.startswith("cuda")
    for batch in loader:
        x = batch["input_values"].to(device)
        mask = batch["attention_mask"].to(device)
        if timer:
            timer.start()
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            logits = model(x, attention_mask=mask).logits
        frames = model._get_feat_extract_output_lengths(mask.sum(-1)).tolist()
        ids = logits.argmax(-1)
        for k, n in enumerate(frames):
            ids[k, n:] = pad_id
        texts = processor.tokenizer.batch_decode(ids.cpu().tolist())
        if timer:
            timer.stop(sum(lengths[i] for i in batch["idx"]))
        for k, i in enumerate(batch["idx"]):
            hyps[i] = texts[k].strip()
        if logits_dir is not None:
            logp = torch.log_softmax(logits.float(), dim=-1).cpu().numpy()
            for k, i in enumerate(batch["idx"]):
                np.save(logits_dir / f"{rows[i]['id']}.npy", logp[k, :frames[k]].astype(np.float16))
    return hyps


def lr_lambda_factory(state: dict):
    """Linear warmup, then linear decay to 0 at state['total_steps'] (read at every call,
    so the total can be changed after the speed is measured)."""

    def f(step: int) -> float:
        warm, total = state["warmup_steps"], max(state["total_steps"], 1)
        if step < warm:
            return (step + 1) / warm
        return max(0.0, (total - step) / max(1, total - warm))

    return f


def save_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def ceil_div(a: int, b: int) -> int:
    return math.ceil(a / b)

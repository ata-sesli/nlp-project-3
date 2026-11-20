"""Load the NVIDIA Uzbek FastConformer (A8/A9) at an exact revision and transcribe clips.

Model card: https://huggingface.co/nvidia/stt_uz_fastconformer_hybrid_large_pc
It is a hybrid model: one encoder with a transducer (RNN-T) head and a CTC head.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.asr_data import SAMPLE_RATE, RTFTimer, clip_seconds, hf_revision, load_audio  # noqa: E402

MODEL_ID = "nvidia/stt_uz_fastconformer_hybrid_large_pc"


def load_model(revision: str | None = None, device: str | None = None):
    """Download the .nemo file at a fixed commit and restore it. Returns (model, commit_hash)."""
    import torch
    from huggingface_hub import hf_hub_download, list_repo_files
    from nemo.collections.asr.models import ASRModel

    sha = hf_revision(MODEL_ID, revision)
    nemo_files = [f for f in list_repo_files(MODEL_ID, revision=sha) if f.endswith(".nemo")]
    if not nemo_files:
        raise RuntimeError(f"No .nemo file in {MODEL_ID}@{sha}")
    path = hf_hub_download(MODEL_ID, nemo_files[0], revision=sha)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = ASRModel.restore_from(path, map_location=torch.device(device))
    model.eval()
    return model, sha


def set_head(model, head: str) -> None:
    """head = 'rnnt' (A8, the default head) or 'ctc' (A9)."""
    assert head in ("rnnt", "ctc"), head
    if head == "ctc":
        # 'greedy_batch' gives the same output as 'greedy', only faster (NeMo's own advice).
        try:
            from omegaconf import open_dict

            cfg = model.cfg.aux_ctc.decoding
            with open_dict(cfg):
                cfg.strategy = "greedy_batch"
            model.change_decoding_strategy(decoding_cfg=cfg, decoder_type="ctc")
            return
        except Exception as err:
            print(f"greedy_batch not available ({err!r}); using the default CTC decoding")
    model.change_decoding_strategy(decoder_type=head)


def _to_text(item) -> str:
    if isinstance(item, str):
        return item
    if hasattr(item, "text"):
        return item.text if isinstance(item.text, str) else str(item.text)
    if isinstance(item, (list, tuple)) and item:
        return _to_text(item[0])
    return ""


def _texts(output) -> list[str]:
    # Older NeMo returns (best_hyps, all_hyps) for transducers; newer returns a list.
    if isinstance(output, tuple):
        output = output[0]
    return [_to_text(x) for x in output]


def transcribe_rows(model, rows: list[dict], batch_size: int = 32, chunk: int = 512,
                    timer: RTFTimer | None = None, progress: bool = True) -> list[str]:
    """Transcribe clips (cut to start_s/end_s, 16 kHz mono). Returns one text per row, same order.

    Clips are written as temporary 16 kHz WAV files in chunks, so this works with every NeMo
    version. Only model.transcribe() is timed for the RTF.
    """
    import torch

    hyps: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        for start in range(0, len(rows), chunk):
            part = rows[start:start + chunk]
            paths = []
            for k, row in enumerate(part):
                p = Path(tmp) / f"{start + k}.wav"
                sf.write(p, load_audio(row), SAMPLE_RATE, subtype="PCM_16")
                paths.append(str(p))
            if timer:
                timer.start()
            with torch.inference_mode():
                out = model.transcribe(paths, batch_size=batch_size, verbose=False)
            if timer:
                timer.stop(sum(clip_seconds(r) for r in part))
            texts = _texts(out)
            if len(texts) != len(part):
                raise RuntimeError(f"NeMo returned {len(texts)} texts for {len(part)} clips")
            hyps.extend(texts)
            for p in paths:
                Path(p).unlink()
            if progress:
                print(f"  {min(start + chunk, len(rows))}/{len(rows)} clips", flush=True)
    return hyps


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())

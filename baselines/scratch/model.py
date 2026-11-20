"""A2 (CTC) and A3 (RNN-T) trained from scratch. Both use the same encoder code:

    16 kHz audio -> 80-dim log-mel (10 ms hop) -> per-clip mean/variance normalization
    -> SpecAugment (training only) -> 2 conv layers (4x time subsampling, 40 ms frames)
    -> torchaudio.models.Conformer -> head

A2 head: linear layer + CTC loss, greedy decoding.
A3 head: LSTM prediction network + joiner + RNN-T loss (torchaudio.functional.rnnt_loss,
         or the pure-PyTorch version below if torchaudio does not have it), greedy decoding.
         An auxiliary CTC loss on the encoder (weight --aux-ctc) helps it learn faster; decoding
         uses only the transducer.
"""
from __future__ import annotations

import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

BLANK = 0
HOP = 160  # 10 ms at 16 kHz


# ---------------------------------------------------------------- front end


class LogMel(nn.Module):
    def __init__(self, n_mels: int = 80):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=400, win_length=400,
                                                        hop_length=HOP, n_mels=n_mels)

    def forward(self, wav: torch.Tensor, wav_lens: torch.Tensor):
        with torch.autocast(device_type=wav.device.type, enabled=False):
            x = (self.mel(wav.float()) + 1e-6).log()  # (B, n_mels, T)
            lens = torch.clamp(wav_lens // HOP + 1, max=x.size(-1))
            mask = (torch.arange(x.size(-1), device=x.device)[None] < lens[:, None]).unsqueeze(1)
            n = lens[:, None, None].float()
            mean = (x * mask).sum(-1, keepdim=True) / n
            var = (((x - mean) * mask) ** 2).sum(-1, keepdim=True) / n
            x = (x - mean) / (var.sqrt() + 1e-5) * mask
        return x.transpose(1, 2), lens  # (B, T, n_mels)


def spec_augment(x: torch.Tensor, lens: torch.Tensor, freq_masks: int = 2, freq_width: int = 27,
                 time_masks: int = 5, time_ratio: float = 0.05) -> torch.Tensor:
    """SpecAugment without time warping. Time mask width is at most 5% of the clip length."""
    x = x.clone()
    n_mels = x.size(2)
    for b in range(x.size(0)):
        T = int(lens[b])
        for _ in range(freq_masks):
            w = random.randint(0, freq_width)
            f0 = random.randint(0, max(0, n_mels - w))
            x[b, :, f0:f0 + w] = 0
        max_w = max(1, int(T * time_ratio))
        for _ in range(time_masks):
            w = random.randint(0, max_w)
            t0 = random.randint(0, max(0, T - w))
            x[b, t0:t0 + w, :] = 0
    return x


class ConvSubsampling(nn.Module):
    """Two 3x3 conv layers with stride 2: 4x fewer frames (10 ms -> 40 ms)."""

    def __init__(self, n_mels: int, d_model: int):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(1, d_model, 3, 2), nn.ReLU(), nn.Conv2d(d_model, d_model, 3, 2), nn.ReLU())
        f = ((n_mels - 3) // 2 + 1 - 3) // 2 + 1
        self.out = nn.Linear(d_model * f, d_model)

    def forward(self, x: torch.Tensor, lens: torch.Tensor):
        x = self.conv(x.unsqueeze(1))  # (B, C, T', F')
        b, c, t, f = x.shape
        x = self.out(x.transpose(1, 2).reshape(b, t, c * f))
        lens = ((lens - 3) // 2 + 1 - 3) // 2 + 1
        return x, torch.clamp(lens, min=1, max=t)


class Encoder(nn.Module):
    def __init__(self, n_mels=80, d_model=256, layers=12, heads=4, ffn=1024, kernel=31, dropout=0.1):
        super().__init__()
        self.frontend = LogMel(n_mels)
        self.subsample = ConvSubsampling(n_mels, d_model)
        self.dropout = nn.Dropout(dropout)
        self.conformer = torchaudio.models.Conformer(input_dim=d_model, num_heads=heads, ffn_dim=ffn,
                                                     num_layers=layers, depthwise_conv_kernel_size=kernel,
                                                     dropout=dropout)

    def forward(self, wav, wav_lens, augment: bool = False):
        x, lens = self.frontend(wav, wav_lens)
        if augment:
            x = spec_augment(x, lens)
        x, lens = self.subsample(x, lens)
        x, lens = self.conformer(self.dropout(x), lens)
        return x, lens


def ctc_loss(logits, lens, targets, target_lens):
    logp = logits.float().log_softmax(-1).transpose(0, 1)  # (T, B, V)
    flat = torch.cat([targets[b, :target_lens[b]] for b in range(targets.size(0))])
    return F.ctc_loss(logp, flat, lens, target_lens, blank=BLANK, zero_infinity=True)


def ctc_greedy(logits, lens) -> list[list[int]]:
    best = logits.argmax(-1).cpu()
    out = []
    for b in range(best.size(0)):
        seq = torch.unique_consecutive(best[b, :int(lens[b])]).tolist()
        out.append([t for t in seq if t != BLANK])
    return out


# ---------------------------------------------------------------- A2: CTC


class CTCModel(nn.Module):
    def __init__(self, vocab_size: int, **enc):
        super().__init__()
        self.encoder = Encoder(**enc)
        self.head = nn.Linear(enc.get("d_model", 256), vocab_size)

    def loss(self, wav, wav_lens, targets, target_lens):
        enc, lens = self.encoder(wav, wav_lens, augment=self.training)
        return ctc_loss(self.head(enc), lens, targets, target_lens)

    @torch.no_grad()
    def decode(self, wav, wav_lens) -> list[list[int]]:
        enc, lens = self.encoder(wav, wav_lens)
        return ctc_greedy(self.head(enc), lens)


# ---------------------------------------------------------------- A3: RNN-T


def rnnt_loss_torch(logits, targets, logit_lens, target_lens, blank: int = BLANK):
    """RNN-T loss (Graves 2012) in plain PyTorch; same result as torchaudio.functional.rnnt_loss
    with reduction='mean'. Loops over time; each step is vectorized over batch and labels."""
    logp = logits.float().log_softmax(-1)
    B, T, U1, _ = logp.shape
    blank_lp = logp[..., blank]  # (B, T, U+1)
    if U1 > 1:
        idx = targets.long()[:, None, :, None].expand(B, T, U1 - 1, 1)
        emit_lp = logp[:, :, :-1, :].gather(-1, idx).squeeze(-1)  # (B, T, U)
        cum = torch.cat([logp.new_zeros(B, T, 1), emit_lp.cumsum(-1)], dim=-1)  # (B, T, U+1)
    else:
        cum = logp.new_zeros(B, T, 1)
    # alpha[t, u] = logaddexp(alpha[t-1, u] + blank[t-1, u], alpha[t, u-1] + emit[t, u-1])
    #             = cum[t, u] + logcumsumexp_k<=u (alpha[t-1, k] + blank[t-1, k] - cum[t, k])
    start = torch.full((B, U1), float("-inf"), device=logp.device)
    start[:, 0] = 0.0
    alpha = cum[:, 0] + torch.logcumsumexp(start - cum[:, 0], dim=-1)
    alphas = [alpha]
    for t in range(1, T):
        prev = alpha + blank_lp[:, t - 1]
        alpha = cum[:, t] + torch.logcumsumexp(prev - cum[:, t], dim=-1)
        alphas.append(alpha)
    alphas = torch.stack(alphas, dim=1)  # (B, T, U+1)
    b = torch.arange(B, device=logp.device)
    t_last = logit_lens.long() - 1
    u_last = target_lens.long()
    ll = alphas[b, t_last, u_last] + blank_lp[b, t_last, u_last]
    return -ll.mean()


def rnnt_loss(logits, targets, logit_lens, target_lens):
    fn = getattr(torchaudio.functional, "rnnt_loss", None)
    if fn is not None:
        try:
            return fn(logits.float(), targets.int(), logit_lens.int(), target_lens.int(), blank=BLANK,
                      reduction="mean")
        except (RuntimeError, NotImplementedError):  # torchaudio built without the RNN-T op
            pass
    return rnnt_loss_torch(logits, targets, logit_lens, target_lens)


class RNNTModel(nn.Module):
    def __init__(self, vocab_size: int, pred_dim: int = 320, joint_dim: int = 320, aux_ctc: float = 0.3, **enc):
        super().__init__()
        d = enc.get("d_model", 256)
        self.encoder = Encoder(**enc)
        self.embed = nn.Embedding(vocab_size, pred_dim)  # id 0 (blank) is also the start symbol
        self.lstm = nn.LSTM(pred_dim, pred_dim, num_layers=1, batch_first=True)
        self.enc_proj = nn.Linear(d, joint_dim)
        self.pred_proj = nn.Linear(pred_dim, joint_dim)
        self.joint_out = nn.Linear(joint_dim, vocab_size)
        self.aux_ctc = aux_ctc
        self.ctc_head = nn.Linear(d, vocab_size) if aux_ctc > 0 else None

    def predictor(self, tokens, state=None):
        out, state = self.lstm(self.embed(tokens), state)
        return self.pred_proj(out), state

    def loss(self, wav, wav_lens, targets, target_lens):
        enc, lens = self.encoder(wav, wav_lens, augment=self.training)
        U = int(target_lens.max())
        targets = targets[:, :U]
        pred_in = torch.cat([targets.new_zeros(targets.size(0), 1), targets], dim=1)
        pred, _ = self.predictor(pred_in)  # (B, U+1, J)
        logits = self.joint_out(torch.tanh(self.enc_proj(enc).unsqueeze(2) + pred.unsqueeze(1)))
        loss = rnnt_loss(logits, targets, lens, target_lens)
        if self.ctc_head is not None:
            loss = (1 - self.aux_ctc) * loss + self.aux_ctc * ctc_loss(self.ctc_head(enc), lens, targets, target_lens)
        return loss

    @torch.no_grad()
    def decode(self, wav, wav_lens, max_symbols: int = 5) -> list[list[int]]:
        enc, lens = self.encoder(wav, wav_lens)
        encp = self.enc_proj(enc)
        B, T, _ = encp.shape
        tokens = torch.zeros(B, 1, dtype=torch.long, device=enc.device)
        pred, state = self.predictor(tokens)
        pred = pred[:, 0]
        hyps: list[list[int]] = [[] for _ in range(B)]
        for t in range(T):
            active = t < lens
            for _ in range(max_symbols):
                k = self.joint_out(torch.tanh(encp[:, t] + pred)).argmax(-1)
                emit = (k != BLANK) & active
                if not bool(emit.any()):
                    break
                new_pred, new_state = self.predictor(k[:, None], state)
                m = emit[:, None]
                pred = torch.where(m, new_pred[:, 0], pred)
                state = tuple(torch.where(m[None], n, o) for n, o in zip(new_state, state))
                for b in emit.nonzero().flatten().tolist():
                    hyps[b].append(int(k[b]))
                active = emit
        return hyps


def build_model(kind: str, vocab_size: int, cfg: dict) -> nn.Module:
    enc = {k: cfg[k] for k in ("d_model", "layers", "heads", "ffn", "kernel", "dropout")}
    if kind == "ctc":
        return CTCModel(vocab_size, **enc)
    return RNNTModel(vocab_size, pred_dim=cfg["pred_dim"], joint_dim=cfg["joint_dim"], aux_ctc=cfg["aux_ctc"], **enc)

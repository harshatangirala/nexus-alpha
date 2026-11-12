"""Transformer encoders for filing text, shared by the builder, the notebook (GPU) and inference (E28, D-053).

Encoder: ChronoBERT v1 with a 2010-12-31 text cutoff (manelalab/chrono-bert-v1-20101231, ModernBERT-base, MIT). It is
chronologically consistent: it was pretrained only on text dated before 2011, so it has never read a filing from the
inner (2016), validation (2017-18), calibration (2019) or test (2020-22) years. SEC-BERT (pretrained on 10-K filings to
2019) and FinBERT would have seen text from the evaluation years or its era; they are not used for filings.

Text: the SEC study's event text (S09: the filing's main sections, first 12,000 characters). Paragraphs that are safe
harbour boilerplate are dropped, whitespace is collapsed, and the first MAX_TOKENS tokens are encoded. The document
vector is the attention-masked mean of the last hidden layer (768-d). Weights are fixed: no fine-tuning.
"""
from __future__ import annotations

import hashlib
import re
import time
from pathlib import Path

import numpy as np

MODEL = "manelalab/chrono-bert-v1-20101231"
REVISION = "1520cf239e7922584ce7762af9e5040e8edd9374"
MAX_TOKENS = 256
_BOILER = re.compile(r"forward[- ]looking statements?|safe harbou?r|private securities litigation reform act"
                     r"|undertakes? no obligation to (publicly )?update", re.I)
_WS = re.compile(r"\s+")


def clean_filing_text(text: str | None, max_chars: int = 4000) -> str:
    """Drop safe-harbour paragraphs, collapse whitespace, keep the first max_chars characters (more than MAX_TOKENS
    tokens need)."""
    if not text:
        return ""
    paras = [p for p in re.split(r"\n\s*\n", text) if p.strip() and not _BOILER.search(p)]
    return _WS.sub(" ", " ".join(paras)).strip()[:max_chars]


def load(device: str = "cpu", revision: str = REVISION):
    import torch
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL, revision=revision)
    mdl = AutoModel.from_pretrained(MODEL, revision=revision).eval().to(device)
    if device == "cuda":
        mdl = mdl.to(torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
    return tok, mdl


def encode(texts: list[str], tok, mdl, batch: int = 64, max_tokens: int = MAX_TOKENS, ckpt_dir: Path | None = None,
           log=print, every: int = 20) -> np.ndarray:
    """Mean-pooled last-layer vectors (float16, input order). Length-sorted batches; checkpoints every `every` batches
    keyed to the exact text list, so a stopped run resumes where it ended."""
    import torch
    device = next(mdl.parameters()).device
    order = np.argsort([len(t) for t in texts], kind="stable")
    st = [texts[j] for j in order]
    tag = hashlib.sha256(("\x00".join(texts) + f"|{max_tokens}|{MODEL}|{REVISION}").encode()).hexdigest()[:16]
    done, start = [], 0
    if ckpt_dir is not None:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        for p in sorted(ckpt_dir.glob(f"{tag}_*.npy"), key=lambda p: int(p.stem.split("_")[1])):
            done.append(np.load(p))
        start = sum(len(x) for x in done)
        if start:
            log(f"resuming at {start}/{len(st)} from {len(done)} checkpoints")
    pend, t0 = [], time.time()
    with torch.inference_mode():
        for i in range(start, len(st), batch):
            b = tok(st[i:i + batch], padding=True, truncation=True, max_length=max_tokens, return_tensors="pt").to(device)
            hs = mdl(**b).last_hidden_state.float()
            m = b["attention_mask"].unsqueeze(-1).float()
            pend.append(((hs * m).sum(1) / m.sum(1).clamp(min=1)).cpu().numpy().astype(np.float16))
            k = i + len(st[i:i + batch])
            if ((i - start) // batch + 1) % every == 0 or k >= len(st):
                E_ = np.concatenate(pend)
                if ckpt_dir is not None:
                    np.save(ckpt_dir / f"{tag}_{k:07d}.npy", E_)
                done.append(E_)
                pend = []
                rate = (k - start) / max(time.time() - t0, 1e-9)
                log(f"{k}/{len(st)} encoded ({rate:.1f}/s, about {(len(st) - k) / max(rate, 1e-9) / 60:.0f} min left)")
    E = np.concatenate(done) if done else np.zeros((0, mdl.config.hidden_size), np.float16)
    inv = np.empty_like(order)
    inv[order] = np.arange(len(order))
    return E[inv]

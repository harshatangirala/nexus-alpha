"""Kronos weights: download from the Hugging Face Hub at pinned revisions, hash-check, load on CPU or GPU.

Weights are MIT-licensed safetensors files (no pickle). They live in the git-ignored artifacts/kronos/<name>/ and are
never committed. `manifest.json` there records each file's SHA-256; the frozen record of the forward-test model pins
the same hashes, so a silently changed file is refused.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .. import paths

# name -> (Hugging Face repo, pinned commit). Revisions read from the Hub API on 2026-11-07.
REPOS = {
    "Kronos-mini": ("NeoQuasar/Kronos-mini", "f4e68697d9d5aed55cef5c96aabc3376bcad9f81"),
    "Kronos-small": ("NeoQuasar/Kronos-small", "901c26c1332695a2a8f243eb2f37243a37bea320"),
    "Kronos-base": ("NeoQuasar/Kronos-base", "2b554741eca47781b64468546e77fef3e85130e6"),
    "Kronos-Tokenizer-base": ("NeoQuasar/Kronos-Tokenizer-base", "0e0117387f39004a9016484a186a908917e22426"),
    "Kronos-Tokenizer-2k": ("NeoQuasar/Kronos-Tokenizer-2k", "26966d0035065a0cae0ebad7af8ece35bc1fb51c"),
}
TOKENIZER_FOR = {"Kronos-mini": "Kronos-Tokenizer-2k", "Kronos-small": "Kronos-Tokenizer-base",
                 "Kronos-base": "Kronos-Tokenizer-base"}
MAX_CONTEXT = {"Kronos-mini": 2048, "Kronos-small": 512, "Kronos-base": 512}
FILES = ("config.json", "model.safetensors")


def root() -> Path:
    return paths.ARTIFACTS / "kronos"


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def manifest() -> dict:
    p = root() / "manifest.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def download(name: str) -> Path:
    """Download one model or tokenizer at its pinned revision (public; no token needed). Returns its folder."""
    from huggingface_hub import HfApi, snapshot_download
    repo, rev = REPOS[name]
    if rev is None:
        rev = manifest().get(name, {}).get("revision") or HfApi().model_info(repo).sha
    d = root() / name
    snapshot_download(repo_id=repo, revision=rev, local_dir=str(d), allow_patterns=list(FILES))
    m = manifest()
    m[name] = {"repo": repo, "revision": rev, "files": {f: _sha(d / f) for f in FILES}}
    root().mkdir(parents=True, exist_ok=True)
    (root() / "manifest.json").write_text(json.dumps(m, indent=1), encoding="utf-8")
    return d


def verify(name: str, expected: dict | None = None) -> Path:
    """The local files must match the manifest (or `expected` = {file: sha256}, e.g. from a frozen record)."""
    d = root() / name
    want = expected or manifest().get(name, {}).get("files")
    if not want:
        raise FileNotFoundError(f"{name}: not downloaded (run scripts/kronos/fetch_weights.py)")
    for f, h in want.items():
        if not (d / f).exists() or _sha(d / f) != h:
            raise RuntimeError(f"{name}/{f}: missing or SHA-256 mismatch")
    return d


def load(name: str, device: str = "cpu", expected: dict | None = None):
    """(model, tokenizer, max_context) in eval mode, hash-checked."""
    from .vendor import Kronos, KronosTokenizer
    tok_name = TOKENIZER_FOR[name]
    model = Kronos.from_pretrained(str(verify(name, (expected or {}).get(name))))
    tok = KronosTokenizer.from_pretrained(str(verify(tok_name, (expected or {}).get(tok_name))))
    return model.to(device).eval(), tok.to(device).eval(), MAX_CONTEXT[name]

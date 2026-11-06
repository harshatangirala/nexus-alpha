"""Download Kronos weights at pinned revisions into artifacts/kronos/ and record their SHA-256.

    python scripts/kronos/fetch_weights.py [Kronos-mini Kronos-small Kronos-base]

Each model brings its tokenizer. Public Hugging Face repositories (MIT); no token is needed or read.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from nexus_alpha.kronos import weights as W  # noqa: E402


def main(names: list[str]) -> None:
    todo = []
    for n in names:
        todo += [n, W.TOKENIZER_FOR[n]]
    for n in dict.fromkeys(todo):
        d = W.download(n)
        m = W.manifest()[n]
        print(f"{n}: {m['repo']}@{m['revision'][:12]} -> {d}")
        for f, h in m["files"].items():
            print(f"   {f}: sha256 {h[:16]}..., {(d / f).stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main(sys.argv[1:] or ["Kronos-mini", "Kronos-small"])

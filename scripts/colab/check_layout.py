"""Check that the data the v3 stages need is where the NEXUS_* variables point (this PC or Colab).

    python scripts/colab/check_layout.py [--group fundamentals_build experiments]

Exit code 0 when everything is present, 1 otherwise (and lists what is missing).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(HERE))

import layout  # noqa: E402
from nexus_alpha import paths  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", nargs="*", default=None)
    a = ap.parse_args()
    ok, missing = layout.check(REPO, paths, tuple(a.group) if a.group else None)
    for line in ok:
        print("ok     ", line)
    for line in missing:
        print("MISSING", line)
    print(f"\n{len(ok)} present, {len(missing)} missing")
    sys.exit(1 if missing else 0)


if __name__ == "__main__":
    main()

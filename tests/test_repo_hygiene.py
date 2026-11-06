"""No stray control characters in tracked text files.

Audit 4 / 2026-11-07: patches written through shell heredocs turned "\\v" and "\\a" inside Windows paths into a vertical
tab (run_daily.cmd) and a bell character (docs/COLAB.md). Tabs, CR and LF are the only control bytes allowed."""
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TEXT = {".py", ".md", ".cmd", ".bat", ".ps1", ".yaml", ".yml", ".toml", ".txt", ".ipynb", ".json"}


def test_no_control_characters_in_tracked_text_files():
    files = subprocess.run(["git", "ls-files"], cwd=REPO, capture_output=True, text=True, check=True).stdout.split()
    bad = []
    for f in files:
        p = REPO / f
        if p.suffix.lower() not in TEXT or not p.exists() or f.startswith("shadow/"):
            continue
        b = p.read_bytes()
        c = sorted({x for x in b if x < 32 and x not in (9, 10, 13)})
        if c:
            bad.append((f, c))
    assert not bad, bad

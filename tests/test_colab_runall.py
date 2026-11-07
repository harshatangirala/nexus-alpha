"""The one-pass Colab path (notebooks/NEXUS_Alpha_Colab_RunAll.ipynb -> scripts/colab/reproduce.py) must stay runnable.

Added 2026-11-07 (A-52) after an independent review found that "Run all" on the orchestrator notebook stops at the first
v1-pipeline cell when only the minimal fundamentals data set is staged."""
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NB = REPO / "notebooks" / "NEXUS_Alpha_Colab_RunAll.ipynb"


def _pins(path: Path) -> dict:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "==" in line and not line.lstrip().startswith("#"):
            k, v = line.strip().split("==")
            out[k.lower()] = v
    return out


def test_runall_notebook_is_clean_and_references_existing_files():
    nb = json.loads(NB.read_text(encoding="utf-8"))
    src = "\n".join("".join(c["source"]) for c in nb["cells"])
    assert all(not c.get("outputs") for c in nb["cells"] if c["cell_type"] == "code"), "commit the notebook without outputs"
    refs = set(re.findall(r"(scripts/[\w/]+\.py|requirements-[\w-]+\.txt)", src))
    assert "scripts/colab/reproduce.py" in refs
    for r in refs:
        assert (REPO / r).exists(), f"the run-all notebook references a missing file: {r}"
    assert "NEXUS_HOLDOUT" not in src and "userdata" not in src, "the run-all path needs no secret and never the holdout"


def test_repro_pins_match_the_full_colab_pins():
    small, full = _pins(REPO / "requirements-colab-repro.txt"), _pins(REPO / "requirements-colab.txt")
    for k, v in small.items():
        assert full.get(k) == v, f"{k}: {v} in requirements-colab-repro.txt but {full.get(k)} in requirements-colab.txt"
    assert "torch" not in small, "the reproduction path must not reinstall torch on Colab"


def test_summary_scrubs_tokens_and_emails():
    sys.path.insert(0, str(REPO / "scripts" / "colab"))
    import reproduce
    s = reproduce.scrub("UA Jane jane.doe@example.com hf_" + "x" * 30 + " ghp_" + "y" * 30 + " fine")
    assert "@" not in s and "hf_" not in s and "ghp_" not in s and s.endswith("fine")


def test_reproduce_copies_what_the_layout_check_needs():
    sys.path.insert(0, str(REPO / "scripts" / "colab"))
    import reproduce
    for d in ("nexus_artifacts/fundamentals", "nexus_artifacts/samples", "nexus_artifacts/bundles",
              "us_stocks_ohlcv/final/1day", "us_stocks_ohlcv/work", "sec_inputs_staging/full"):
        assert d in reproduce.SYNC_DIRS, d

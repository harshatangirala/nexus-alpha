"""Fast checks of the one-command audit (audit 7) that also belong in every test run: the sealed scorer is unchanged and
the committed notebook equals a fresh build of its builder (a builder edit without a rebuild would ship a stale
notebook)."""
import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _run_all():
    spec = importlib.util.spec_from_file_location("run_all", REPO / "scripts" / "verify" / "run_all.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_sealed_scorer_is_unchanged():
    ok, note = _run_all().c_scorer()
    assert ok, note


def test_committed_notebook_equals_a_fresh_build():
    ok, note = _run_all().c_notebook()
    assert ok, note

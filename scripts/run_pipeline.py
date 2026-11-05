"""Stage orchestrator used by the Colab notebook and locally.

    python scripts/run_pipeline.py --profile smoke_test --stages all
    python scripts/run_pipeline.py --profile research --stages fusion,backtest --i-accept-the-cost

Each stage is skipped when its output already exists (re-runs resume after a Colab disconnect). Profiles heavier than
`small` print a resource estimate and refuse to start without --i-accept-the-cost. Secrets are read from the
environment (HF_TOKEN via Colab Secrets) and never printed.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nexus_alpha import paths  # noqa: E402
from nexus_alpha.training.hardware import detect  # noqa: E402

A = paths.ARTIFACTS
STAGES = [
    ("audit", ["scripts/audit/p1_ohlcv_coverage.py"], ROOT / "data_manifests" / "ohlcv_coverage_summary.json"),
    ("embeddings", ["scripts/data/fetch_triframe_embeddings.py"], A / "cv_embeddings" / "s500" / "manifest.json"),
    ("samples", ["scripts/data/build_samples.py"], A / "samples" / "s500_frame.parquet"),
    ("news_index", ["scripts/data/build_news_index.py"], A / "news" / "news_index.parquet"),
    ("news_text", ["scripts/data/fetch_news_text.py"], None),
    ("text_features", ["scripts/data/build_text_features.py"], A / "features" / "sec_features.parquet"),
    ("rule_features", ["scripts/data/build_rule_features.py"], A / "features" / "rule_features.parquet"),
    ("structured", ["scripts/experiments/e01_structured_baselines.py"], None),
    ("cv_branch", ["scripts/experiments/e02_cv_branch.py"], None),
    ("fusion", ["scripts/experiments/e03_fusion.py"], None),
    ("walk_forward", ["scripts/experiments/e05_walk_forward.py"], None),
    ("rag", ["scripts/rag/build_kb.py", "scripts/rag/eval_rag.py"], ROOT / "reports" / "rag" / "rag_eval.json"),
]
ESTIMATES = {   # measured on the laptop CPU (Ryzen 5 5500U); a Colab A100 is ~5-20x faster for the neural stages
    "smoke_test": "minutes; no network beyond cached models",
    "debug": "~20 min CPU",
    "small": "~1 h CPU",
    "research": "~4-5 h CPU for all stages (fusion ~1.5 h, rules ~15 min, text features ~10 min); 10 GB artifacts",
    "full": "GPU required: re-render + re-embed 979 stocks x 3 views x backbones (~2-4 h A100), ~40 GB Drive",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="smoke_test", choices=list(ESTIMATES))
    ap.add_argument("--stages", default="all")
    ap.add_argument("--i-accept-the-cost", action="store_true")
    a = ap.parse_args()
    hw = detect()
    print(f"hardware: {hw.get('device')} {hw.get('gpu') or ''} vram {hw.get('vram_gb', '-')} GB; disk free {hw.get('disk_free_gb')} GB")
    print(f"profile {a.profile}: estimated cost {ESTIMATES[a.profile]}")
    if a.profile in ("research", "full") and not a.i_accept_the_cost:
        print("Refusing to start a heavy profile without --i-accept-the-cost.")
        return 2
    if a.profile == "smoke_test":
        cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(ROOT / "tests")]
        print("smoke_test: running the automated test suite (contracts, leakage, schema, trading, fusion).")
        return subprocess.call(cmd, cwd=ROOT)
    want = None if a.stages == "all" else set(a.stages.split(","))
    for name, scripts, done in STAGES:
        if want and name not in want:
            continue
        if done is not None and Path(done).exists():
            print(f"[skip] {name}: output exists ({done})")
            continue
        for s in scripts:
            print(f"[run] {name}: {s}", flush=True)
            rc = subprocess.call([sys.executable, "-I", str(ROOT / s)], cwd=ROOT)
            if rc != 0:
                print(f"[fail] {name} exited with {rc}; fix and re-run (completed stages are skipped)")
                return rc
    return 0


if __name__ == "__main__":
    sys.exit(main())

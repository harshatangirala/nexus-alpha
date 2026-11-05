"""Phase 0 source inventory (read-only).

Hashes every file NEXUS Alpha may read from the two source projects, the raw data store, the books and the
rulebooks, and compares the TriFrame workspace copy with the git copy. It never writes inside a source tree:
the only outputs are `data_manifests/source_inventory.json` and `data_manifests/triframe_copy_comparison.json`
in this repository.

Large binary trees that already carry their own hash manifests (chart caches, 1-minute bars, SEC raw
downloads) are recorded by file count, total size and the manifest's root hash instead of being re-hashed.

Run: python -I scripts/audit/p0_source_inventory.py
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "data_manifests"

DL = Path(r"C:\Users\harsh\Downloads")
SOURCES = {
    "triframe_workspace_repo": DL / "TriFrame_Workspace" / "repo",
    "triframe_workspace_rulebooks": DL / "TriFrame_Workspace" / "rulebooks_and_specs",
    "triframe_git_repo": Path(r"C:\Users\harsh\triframe-cv"),
    "sec_repo": DL / "sec-filings-price-impact",
}
# Trees recorded by count/size only (too large to re-hash; they carry their own manifests).
SUMMARY_ONLY = {
    "ohlcv_final_1day": DL / "us_stocks_ohlcv" / "final" / "1day",
    "ohlcv_final_1hour": DL / "us_stocks_ohlcv" / "final" / "1hour",
    "ohlcv_final_1week": DL / "us_stocks_ohlcv" / "final" / "1week",
    "ohlcv_final_meta": DL / "us_stocks_ohlcv" / "final" / "meta",
    "ohlcv_features_visible": DL / "us_stocks_ohlcv" / "features_visible",
    "ohlcv_cache_2010_2026": DL / "us_stocks_ohlcv" / "cache",
    "ohlcv_cache_2001_2009": DL / "us_stocks_ohlcv" / "cache_2001_2009",
    "ohlcv_cache_patch_r11": DL / "us_stocks_ohlcv" / "cache_patch_r11",
    "ohlcv_candlestick": DL / "us_stocks_ohlcv" / "candlestick",
    "ohlcv_1min": DL / "us_stocks_ohlcv" / "1min",
    "sec_mini_run": DL / "sec_report_mini" / "full",
    "sec_mini_inputs": DL / "sec_report_mini" / "inputs",
    "sec_mini_gpu_results": DL / "sec_results_mini" / "full",
    "sec_full500_cpu_part": DL / "sec_inputs_staging" / "full",
    "sec_raw_hot": DL / "sec_inputs_staging" / "_hot",
    "sec_export_bundle": DL / "SEC_Project_Export",
    "sec_report_desktop": Path(r"C:\Users\harsh\OneDrive\Desktop\SEC_Report_150_Stocks"),
}
MANIFESTS = {
    "ohlcv_final": DL / "us_stocks_ohlcv" / "final" / "MANIFEST.json",
    "ohlcv_cache": DL / "us_stocks_ohlcv" / "cache" / "MANIFEST_cache.json",
    "ohlcv_cache_2001_2009": DL / "us_stocks_ohlcv" / "cache_2001_2009" / "MANIFEST_cache_2001_2009.json",
    "ohlcv_features_visible": DL / "us_stocks_ohlcv" / "features_visible" / "MANIFEST_features_visible.json",
}
SINGLE_FILES = {
    "book_complete_guide_to_trading": DL / "The-Complete-Guide-to-Trading.pdf",
    "book_tsam": DL / "TSaM.pdf",
    "book_nison_candlesticks": DL / "Japanese Candlestick Charting Techniques, 2nd Edition, Steve Nison_text.pdf",
    "book_ta_workbook": DL / "TA_wrkbk.pdf",
    "book_identifying_chart_patterns": DL / "Idenitfying-Chart-Patterns.pdf",
    "book_fundamentals_of_investments_wk2": DL / "B.Com(Hons)_IIIyearVIsem_FundamentalsofInvestments_Week2_DrKanuJain.pdf",
    "book_essentials_quant_finance": DL / "Essentials Of Quantitative Finance_1.pdf",
    "book_dobelman_williams_qfa": DL / "Dobelman, John A._Williams, Edward E - Quantitative financial analytics_ the path to investment profits-World Scientific (2018).pdf",
    "rulebook_entry_150": DL / "01_ENTRY_RULEBOOK_150.md",
    "rulebook_exit_120": DL / "02_EXIT_RULEBOOK_120.md",
    "spec_mtf_decision_engine": DL / "03_MULTI_TIMEFRAME_DECISION_ENGINE.md",
    "spec_nn_training": DL / "04_NEURAL_NETWORK_TRAINING_SPEC.md",
    "spec_data_contract": DL / "05_DATA_CONTRACT_AND_FEATURE_DICTIONARY.md",
    "spec_labeling_backtesting": DL / "06_LABELING_BACKTESTING_AND_STATISTICAL_VALIDATION.md",
    "spec_colab_guide": DL / "11_COLAB_AND_TRAINING_EXECUTION_GUIDE.md",
    "spec_verification_handoff": DL / "12_VERIFICATION_AND_HANDOFF.md",
    "spec_readme_1": DL / "README (1).md",
    "master_prompt": DL / "TriFrame_Fusion_Claude_Code_Execution_Prompt.md",
    "zip_triframe_workspace": DL / "TriFrame_Workspace.zip",
    "zip_sec_complete_study": DL / "sec_filings_price_impact_COMPLETE_STUDY.zip",
    "zip_sec_inputs_for_colab": Path(r"C:\Users\harsh\OneDrive\Desktop\sec_inputs_for_colab.zip"),
    "ohlcv_market_calendar": DL / "us_stocks_ohlcv" / "market_calendar.csv",
    "ohlcv_universe": DL / "us_stocks_ohlcv" / "universe.csv",
    "ohlcv_excluded_tickers": DL / "us_stocks_ohlcv" / "excluded_tickers.csv",
}
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", ".venv", "node_modules", ".ipynb_checkpoints"}


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def walk(root: Path):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in filenames:
            yield Path(dirpath) / fn


def hash_tree(root: Path) -> dict:
    files = {}
    for p in sorted(walk(root)):
        rel = p.relative_to(root).as_posix()
        st = p.stat()
        files[rel] = {"bytes": st.st_size, "sha256": sha256(p),
                      "mtime_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(st.st_mtime))}
    root_hash = hashlib.sha256("\n".join(f"{k}:{v['sha256']}" for k, v in sorted(files.items())).encode()).hexdigest()
    return {"path": str(root), "exists": True, "n_files": len(files),
            "total_bytes": sum(v["bytes"] for v in files.values()), "root_sha256": root_hash, "files": files}


def summarise_tree(root: Path) -> dict:
    if not root.exists():
        return {"path": str(root), "exists": False}
    n, total, newest = 0, 0, 0.0
    for p in walk(root):
        st = p.stat()
        n += 1
        total += st.st_size
        newest = max(newest, st.st_mtime)
    return {"path": str(root), "exists": True, "n_files": n, "total_bytes": total,
            "newest_mtime_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(newest)) if n else None}


def manifest_root(path: Path) -> dict:
    if not path.exists():
        return {"path": str(path), "exists": False}
    try:
        with open(path, encoding="utf-8") as f:
            m = json.load(f)
    except Exception as e:  # recorded, not fatal
        return {"path": str(path), "exists": True, "error": repr(e)}
    keys = {k: m[k] for k in ("root", "root_sha256", "created", "frozen_at", "n_files") if k in m}
    return {"path": str(path), "exists": True, "file_sha256": sha256(path), "declared": keys}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    inv = {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "python": sys.version,
           "hashed_trees": {}, "summarised_trees": {}, "source_manifests": {}, "single_files": {}}
    for k, p in SOURCES.items():
        inv["hashed_trees"][k] = hash_tree(p) if p.exists() else {"path": str(p), "exists": False}
        print(f"hashed {k}: {inv['hashed_trees'][k].get('n_files')} files", flush=True)
    for k, p in SUMMARY_ONLY.items():
        inv["summarised_trees"][k] = summarise_tree(p)
        print(f"summarised {k}: {inv['summarised_trees'][k].get('n_files')} files", flush=True)
    for k, p in MANIFESTS.items():
        inv["source_manifests"][k] = manifest_root(p)
    for k, p in SINGLE_FILES.items():
        if p.exists():
            st = p.stat()
            inv["single_files"][k] = {"path": str(p), "exists": True, "bytes": st.st_size, "sha256": sha256(p),
                                      "mtime_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(st.st_mtime))}
        else:
            inv["single_files"][k] = {"path": str(p), "exists": False}
    inv["elapsed_s"] = round(time.time() - t0, 1)
    with open(OUT / "source_inventory.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(inv, f, indent=1)

    # Workspace copy vs git copy of TriFrame.
    a = inv["hashed_trees"]["triframe_workspace_repo"]["files"]
    b = inv["hashed_trees"]["triframe_git_repo"]["files"]
    only_ws = sorted(set(a) - set(b))
    only_git = sorted(set(b) - set(a))
    differ = sorted(k for k in set(a) & set(b) if a[k]["sha256"] != b[k]["sha256"])
    cmp = {"workspace": str(SOURCES["triframe_workspace_repo"]), "git_copy": str(SOURCES["triframe_git_repo"]),
           "n_workspace": len(a), "n_git_copy": len(b), "identical_common_files": len(set(a) & set(b)) - len(differ),
           "only_in_workspace": only_ws, "only_in_git_copy": only_git, "content_differs": differ}
    with open(OUT / "triframe_copy_comparison.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(cmp, f, indent=1)
    print(json.dumps({k: (v if not isinstance(v, list) else v[:25]) for k, v in cmp.items()}, indent=1))
    print(f"done in {inv['elapsed_s']} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())

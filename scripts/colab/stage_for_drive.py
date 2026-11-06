"""Copy the minimal data set for the v3 stages (fundamentals build, audit, E22, E23, v3_fund refit) into the local Google
Drive folder, laid out as the Colab notebook expects. About 1.8 GB.

    python scripts/colab/stage_for_drive.py              # dry run: lists what would be copied
    python scripts/colab/stage_for_drive.py --go         # copy (files already there with the same size are skipped)
    python scripts/colab/stage_for_drive.py --go --dest "G:/My Drive/antigravity_workspace"

Layout on Drive (DRIVE_ROOT = the destination):
    nexus_artifacts/fundamentals/{raw,snapshots,panel.parquet,cik_map.parquet,sic_sec.parquet,splits.parquet}
    nexus_artifacts/samples/v2_frame.parquet
    nexus_artifacts/bundles/<v1.1 bundle>/, <v3_fund bundle>/
    us_stocks_ohlcv/final/1day/*.parquet, us_stocks_ohlcv/work/{yf,verify_fresh}/*.splits.parquet, market_calendar.csv
    sec_inputs_staging/full/{S02,S03,S06,S07}/

Never copied: local_sec.env, local_env.cmd, any *.env or token file, the sealed folder, the forward-test logs. The SEC
User-Agent and the Hugging Face token go in Colab Secrets. The repository itself comes from GitHub (public).
Read-only on the source projects.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(HERE))

import layout  # noqa: E402
from nexus_alpha import paths  # noqa: E402

DEFAULT_DEST = Path("G:/My Drive/antigravity_workspace")
BLOCKED = (".env", ".cmd", ".key", ".pem", ".token")


def plan(dest: Path) -> list[tuple[Path, Path]]:
    art, ohlcv, sec = paths.ARTIFACTS, paths.OHLCV_ROOT, paths.SEC_FULL500_RUN
    pairs: list[tuple[Path, Path]] = []

    def add(src_dir: Path, dst_dir: Path, pattern: str = "**/*") -> None:
        for f in sorted(src_dir.glob(pattern)):
            if f.is_file() and not f.name.endswith(BLOCKED):
                pairs.append((f, dst_dir / f.relative_to(src_dir)))

    fund = art / "fundamentals"
    for name in ("raw", "snapshots"):                       # snapshots_pre_audit3 is an obsolete cache: not copied
        add(fund / name, dest / "nexus_artifacts" / "fundamentals" / name)
    for name in ("panel.parquet", "cik_map.parquet", "sic_sec.parquet", "splits.parquet"):
        if (fund / name).exists():
            pairs.append((fund / name, dest / "nexus_artifacts" / "fundamentals" / name))
    pairs.append((art / "samples" / "v2_frame.parquet", dest / "nexus_artifacts" / "samples" / "v2_frame.parquet"))
    for pat in ("*_e08_final_model_5c4327", "*_v3_fund_refit_*"):          # v1.1 (feature list) and the frozen v3_fund
        for b in sorted((art / "bundles").glob(pat)):
            add(b, dest / "nexus_artifacts" / "bundles" / b.name)
    add(ohlcv / "final" / "1day", dest / "us_stocks_ohlcv" / "final" / "1day", "*.parquet")
    add(ohlcv / "work" / "yf", dest / "us_stocks_ohlcv" / "work" / "yf", "*.splits.parquet")
    add(ohlcv / "work" / "verify_fresh", dest / "us_stocks_ohlcv" / "work" / "verify_fresh", "*.splits.parquet")
    for name in ("market_calendar.csv", "universe.csv"):
        if (ohlcv / name).exists():
            pairs.append((ohlcv / name, dest / "us_stocks_ohlcv" / name))
    for s in layout.SEC_STAGES:
        add(sec / s, dest / "sec_inputs_staging" / "full" / s)
    return pairs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--go", action="store_true")
    ap.add_argument("--dest", default=str(DEFAULT_DEST))
    a = ap.parse_args()
    dest = Path(a.dest)
    if not dest.exists():
        sys.exit(f"destination {dest} does not exist (is Google Drive for desktop running?)")
    pairs = plan(dest)
    todo = [(s, d) for s, d in pairs if not (d.exists() and d.stat().st_size == s.stat().st_size)]
    gb = sum(s.stat().st_size for s, _ in todo) / 1e9
    print(f"{len(pairs)} files in the set, {len(todo)} to copy ({gb:.2f} GB) to {dest}")
    if not a.go:
        for s, d in todo[:8]:
            print("  would copy", s, "->", d)
        print("dry run: add --go to copy")
        return
    done = 0
    for s, d in todo:
        d.parent.mkdir(parents=True, exist_ok=True)
        tmp = d.with_name(d.name + ".part")
        shutil.copyfile(s, tmp)
        tmp.replace(d)
        done += 1
        if done % 200 == 0:
            print(f"  copied {done}/{len(todo)}", flush=True)
    print(f"copied {done} files ({gb:.2f} GB). Google Drive for desktop now uploads them; wait until it shows 'synced'.")


if __name__ == "__main__":
    main()

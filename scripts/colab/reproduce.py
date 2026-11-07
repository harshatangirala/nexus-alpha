"""One-command reproduction of the v3 fundamentals work on Colab (or any machine): the notebook's only heavy cell.

    python scripts/colab/reproduce.py --drive-root /content/drive/MyDrive/antigravity_workspace

Steps, in order (a failed step is recorded and the summary is still written):
  env report -> Drive check -> copy Drive to local disk -> layout check -> tests -> ratio build -> audit (7 hard
  checks) -> E22 -> E23 -> compare with the committed, pre-registered records -> verify the frozen v3_fund bundle.

Writes one small summary (COLAB_RESULTS.md and COLAB_RESULTS.json) to <drive-root>/colab_results/<UTC stamp>/ and prints it.
The summary holds versions, hardware, timings, pass/fail counts and the metric differences; it holds no secrets, no data
and no article text. A re-run is a reproduction only: it never touches shadow/, registry/ or the sealed period.

    --no-sync            use the data where it is (no copy to local disk)
    --skip-experiments   skip the ratio build, E22, E23 and the comparison (quick check of everything else)
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[2]
PKGS = ["numpy", "pandas", "pyarrow", "scipy", "scikit-learn", "lightgbm", "pydantic", "pytest", "torch"]
SYNC_DIRS = ["nexus_artifacts/fundamentals", "nexus_artifacts/samples", "nexus_artifacts/bundles",
             "us_stocks_ohlcv/final/1day", "us_stocks_ohlcv/work", "sec_inputs_staging/full"]
DRIVE_NEEDS = SYNC_DIRS + ["us_stocks_ohlcv/market_calendar.csv"]
KEY = re.compile(r"(Error|FAILED|MISMATCH|Traceback|MISSING)")
SECRET_PATTERNS = re.compile(r"(hf_[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}|[\w.+-]+@[\w-]+\.[\w.]+)")


def scrub(text: str) -> str:
    """Summaries are meant to be pasted elsewhere: drop anything shaped like a token or an email address."""
    return SECRET_PATTERNS.sub("[removed]", text)


def versions() -> dict:
    out = {}
    for p in PKGS:
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            out[p] = "not installed"
    return out


def hardware() -> dict:
    hw = {"python": sys.version.split()[0], "platform": platform.platform(), "cpus": os.cpu_count()}
    try:
        hw["ram_gb"] = round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9, 1)
    except (AttributeError, ValueError, OSError):
        hw["ram_gb"] = None
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=20)
        hw["gpu"] = r.stdout.strip() if r.returncode == 0 and r.stdout.strip() else "none"
    except (OSError, subprocess.SubprocessError):
        hw["gpu"] = "none"
    return hw


class Runner:
    def __init__(self, env: dict, logdir: Path):
        self.env, self.logdir, self.steps = env, logdir, []
        logdir.mkdir(parents=True, exist_ok=True)

    def run(self, name: str, cmd: list[str], fatal: bool = False) -> dict:
        t0 = time.time()
        print(f"\n=== {name}: {' '.join(cmd[:6])}{' ...' if len(cmd) > 6 else ''}", flush=True)
        log = self.logdir / (re.sub(r"\W+", "_", name) + ".log")
        with open(log, "w", encoding="utf-8", errors="replace") as fh:
            p = subprocess.Popen(cmd, cwd=REPO, env=self.env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True, encoding="utf-8", errors="replace")
            for line in p.stdout:
                fh.write(line)
                print(line, end="", flush=True)
            p.wait()
        text = log.read_text(encoding="utf-8", errors="replace")
        step = {"step": name, "ok": p.returncode == 0, "seconds": round(time.time() - t0, 1), "exit": p.returncode,
                "tail": scrub("\n".join(text.strip().splitlines()[-12:])), "fatal": fatal,
                "errors": scrub("\n".join([ln for ln in text.splitlines() if KEY.search(ln)][:15]))}
        self.steps.append(step)
        print(f"--- {name}: {'ok' if step['ok'] else 'FAILED'} ({step['seconds']} s)", flush=True)
        return step


def sync(drive: Path, local: Path) -> tuple[bool, str]:
    for sub in SYNC_DIRS:
        src, dst = drive / sub, local / sub
        dst.mkdir(parents=True, exist_ok=True)
        if shutil.which("rsync"):
            r = subprocess.run(["rsync", "-a", f"{src}/", f"{dst}/"], capture_output=True, text=True)
            if r.returncode:
                return False, f"rsync failed for {sub}: {r.stderr.strip()[-300:]}"
        else:
            shutil.copytree(src, dst, dirs_exist_ok=True)
    for f in ("market_calendar.csv", "universe.csv"):
        s = drive / "us_stocks_ohlcv" / f
        if s.exists():
            shutil.copy(s, local / "us_stocks_ohlcv" / f)
    return True, "copied"


def parse_tests(tail: str) -> str:
    m = re.findall(r"(\d+ (?:passed|failed|error)[^\n]*)", tail)
    return m[-1] if m else "no summary line"


def write_summary(out_dir: Path, summary: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "COLAB_RESULTS.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    L = ["# NEXUS Alpha: Colab reproduction result", "", f"Run: {summary['started_utc']}  |  commit {summary['commit']}  |  verdict: **{summary['verdict']}**", ""]
    hw = summary["hardware"]
    L += ["## Machine", f"Python {hw['python']}, {hw['cpus']} CPUs, RAM {hw['ram_gb']} GB, GPU: {hw['gpu']}",
          "Packages: " + ", ".join(f"{k} {v}" for k, v in summary["versions"].items()), ""]
    L += ["## Steps", "| step | result | seconds |", "|---|---|---|"]
    for s in summary["steps"]:
        L.append(f"| {s['step']} | {'ok' if s['ok'] else 'FAILED'} | {s['seconds']} |")
    L.append("")
    for key, title in (("tests", "Tests"), ("compare", "Reproduction vs committed E22/E23"), ("frozen", "Frozen v3_fund")):
        if summary.get(key):
            L += [f"## {title}", "```", summary[key], "```", ""]
    failed = [s for s in summary["steps"] if not s["ok"]]
    for s in failed:
        L += [f"## Failure detail: {s['step']}", "```", s.get("errors") or "", "(last lines)", s["tail"], "```", ""]
    L += ["Note: a re-run is a reproduction only. Nothing here replaces a committed record or touches the sealed period."]
    (out_dir / "COLAB_RESULTS.md").write_text(scrub("\n".join(L)) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive-root", required=True)
    ap.add_argument("--work", default="/content/nexus_work")
    ap.add_argument("--no-sync", action="store_true")
    ap.add_argument("--skip-experiments", action="store_true")
    a = ap.parse_args()
    drive, work = Path(a.drive_root), Path(a.work)
    started = datetime.now(timezone.utc)
    out_dir = drive / "colab_results" / started.strftime("%Y%m%dT%H%M%SZ")
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True,
                                text=True).stdout.strip()
    except OSError:
        commit = ""
    summary = {"started_utc": started.isoformat(timespec="seconds"), "commit": commit or "unknown",
               "hardware": hardware(), "versions": versions(),
               "steps": [], "verdict": "INCOMPLETE"}
    print(json.dumps({k: summary[k] for k in ("hardware", "versions")}, indent=1), flush=True)

    # 0. E22/E23 load the v1.1 bundle's feature list through the inference module, which imports torch (preinstalled on
    #    Colab; this script never reinstalls it)
    if summary["versions"]["torch"] == "not installed" and not a.skip_experiments:
        summary["steps"].append({"step": "environment", "ok": False, "seconds": 0, "exit": 1, "fatal": True,
                                 "tail": "torch is not installed; E22/E23 need it. On Colab it is preinstalled (use a standard "
                                         "runtime); elsewhere: pip install torch, then run again."})
        summary["verdict"] = "BLOCKED: torch missing"
        write_summary(out_dir, summary)
        print((out_dir / "COLAB_RESULTS.md").read_text(encoding="utf-8"))
        return 1

    # 1. everything must be on Drive before anything is copied or run
    missing = [d for d in DRIVE_NEEDS if not (drive / d).exists()]
    if missing:
        summary["steps"].append({"step": "drive check", "ok": False, "seconds": 0, "exit": 1, "fatal": True,
                                 "tail": "Not on Drive yet: " + ", ".join(missing) +
                                         ". Run scripts/colab/stage_for_drive.py --go on the PC and wait until Google Drive "
                                         "for desktop shows it as synced; then run this cell again."})
        summary["verdict"] = "BLOCKED: data not on Drive"
        write_summary(out_dir, summary)
        print((out_dir / "COLAB_RESULTS.md").read_text(encoding="utf-8"))
        return 1

    # 2. copy to local disk (the Drive mount is slow for thousands of small files)
    data = drive
    if not a.no_sync:
        t0 = time.time()
        ok, msg = sync(drive, work)
        summary["steps"].append({"step": "copy Drive to local disk", "ok": ok, "seconds": round(time.time() - t0, 1),
                                 "exit": 0 if ok else 1, "tail": msg, "fatal": True})
        if not ok:
            summary["verdict"] = "FAILED: copy"
            write_summary(out_dir, summary)
            print((out_dir / "COLAB_RESULTS.md").read_text(encoding="utf-8"))
            return 1
        data = work
    env = dict(os.environ)
    env.update({"NEXUS_ARTIFACTS": str(data / "nexus_artifacts"), "NEXUS_OHLCV_ROOT": str(data / "us_stocks_ohlcv"),
                "NEXUS_SEC_FULL500_RUN": str(data / "sec_inputs_staging" / "full"), "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONPATH": str(REPO / "src") + os.pathsep + env.get("PYTHONPATH", "")})
    env.pop("NEXUS_HOLDOUT_CONFIG", None)           # the sealed period is never reachable from here
    env.pop("NEXUS_HOLDOUT_UNLOCK", None)
    r = Runner(env, work / "logs")
    r.steps = summary["steps"]
    py = sys.executable

    if not r.run("layout check", [py, "scripts/colab/check_layout.py"], fatal=True)["ok"]:
        summary["verdict"] = "FAILED: layout"
    else:
        t = r.run("tests", [py, "-m", "pytest", "-p", "no:warnings", "-p", "no:cacheprovider",
                            "tests/test_fundamentals.py", "tests/test_shadow_runner.py", "tests/test_repo_hygiene.py"])
        summary["tests"] = parse_tests(t["tail"])
        if not a.skip_experiments:
            if r.run("ratio build", [py, "scripts/data/build_fundamentals.py", "--workers", str(os.cpu_count() or 2)],
                     fatal=True)["ok"]:
                r.run("audit (7 hard checks)", [py, "scripts/verify/audit_fundamentals.py"])
                r.run("E22", [py, "scripts/experiments/e22_fundamentals.py"])
                r.run("E23", [py, "scripts/experiments/e23_industry_fundamentals.py"])
                c = r.run("compare with committed", [py, "scripts/colab/compare_with_committed.py",
                                                    "e22_fundamentals", "e23_industry_fundamentals"])
                summary["compare"] = "\n".join(l for l in c["tail"].splitlines() if "REPRODUCED" in l or "MISMATCH" in l
                                              or "keep decisions" in l or "committed" in l)
        else:
            r.run("audit (7 hard checks)", [py, "scripts/verify/audit_fundamentals.py"])
        f = r.run("frozen v3_fund verify", [py, "-c",
                  "import sys; sys.path.insert(0,'src'); from nexus_alpha import shadow as SH; "
                  "b, rec = SH.verify_frozen('v3_fund'); print('v3_fund frozen record verified:', rec['bundle_id'])"])
        summary["frozen"] = f["tail"].splitlines()[-1] if f["tail"] else ""
        bad = [s["step"] for s in summary["steps"] if not s["ok"]]
        cmp_ = summary.get("compare", "")
        if bad:
            summary["verdict"] = "FAILED: " + ", ".join(bad)
        elif a.skip_experiments:
            summary["verdict"] = "CHECKS PASSED (quick mode: E22/E23 not re-run, nothing reproduced)"
        elif "REPRODUCED" in cmp_ and "MISMATCH" not in cmp_:
            summary["verdict"] = "REPRODUCED"
        else:
            summary["verdict"] = "NOT REPRODUCED: comparison"
    summary["finished_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    write_summary(out_dir, summary)
    print("\n" + "=" * 70 + f"\nSummary written to {out_dir}\n" + "=" * 70)
    print((out_dir / "COLAB_RESULTS.md").read_text(encoding="utf-8"))
    return 0 if summary["verdict"].startswith(("REPRODUCED", "CHECKS PASSED")) else 1


if __name__ == "__main__":
    sys.exit(main())

"""One-command audit of the whole project (audit 7). Read-only: it never writes a forecast, a bundle, a freeze record,
the sealed scorer or a forward-test log, and it never reads the sealed 2023+ period.

Checks, each PASS or FAIL:
 1. test suite (pytest);
 2. every frozen bundle matches its freeze record (SHA-256 of every file);
 3. every forward-test hash chain is intact and every logged file unchanged;
 4. the sealed scorer's combined hash equals the value recorded in audits 3-4 (`32c7b3b7...`);
 5. no model reads calendar time (`time_proxy_scan.py`);
 6. tracked files hold no API key, token or private key;
 7. `requirements.lock` pins every installed distribution at its installed version (local environment only);
 8. the committed end-to-end notebook equals a fresh build of `make_end_to_end_notebook.py`;
 9. README's test count equals the number pytest collects;
10. (--full) the E28 audit of the pinned run (`audit_e28.py --run`), about 5 minutes.

    .venv/Scripts/python -I scripts/verify/run_all.py [--full]        # exit 1 if any check fails
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
PY = sys.executable
SCORER_SHA256 = "32c7b3b73d3420bc8e8623f8e2fd4a29ba944568a93668c99b5d0ab87a7d9486"   # recorded before any sealed read
FROZEN = tuple(sorted(p.stem for p in (REPO / "registry" / "frozen_models").glob("*.json")))   # every frozen record (audit 12)
SECRET = r"AIza[0-9A-Za-z_-]{30}|sk-[A-Za-z0-9]{20}|hf_[A-Za-z0-9]{30}|ghp_[A-Za-z0-9]{30}|-----BEGIN [A-Z ]*PRIVATE KEY"


def run(args: list[str], timeout: int = 3600) -> tuple[int, str]:
    p = subprocess.run(args, cwd=REPO, capture_output=True, text=True, timeout=timeout, encoding="utf-8", errors="replace")
    return p.returncode, (p.stdout + p.stderr)


def c_pytest():
    code, out = run([PY, "-m", "pytest", "-p", "no:cacheprovider", "-q"])
    fails = re.findall(r"^(?:FAILED|ERROR) (\S+)", out, re.M)
    return code == 0, "all tests pass" if code == 0 else f"exit {code}; {fails[:5]}"


def c_frozen():
    from nexus_alpha import shadow as SH
    bad = []
    for v in FROZEN:
        try:
            SH.verify_frozen(v)
        except Exception as e:                                   # noqa: BLE001
            bad.append(f"{v}: {e}")
    return not bad, f"{len(FROZEN)} frozen models match their records" if not bad else "; ".join(bad)


def c_chains():
    from nexus_alpha import shadow as SH
    logs = sorted({*FROZEN, *(p.parent.name for p in SH.SHADOW_ROOT.glob("*/chain.csv"))})    # incl. v1.1_news (E29b)
    probs = {v: SH.verify_chain(v) for v in logs}
    bad = {v: p for v, p in probs.items() if p}
    n = sum(len(SH.chain_rows(v)) for v in logs)
    return not bad, f"{n} chained files intact" if not bad else json.dumps(bad)[:300]


def c_scorer():
    files = sorted([*(REPO / "scripts" / "sealed").glob("*.py"), *(REPO / "src" / "nexus_alpha" / "sealed").glob("*.py")])
    h = hashlib.sha256()
    for f in files:
        h.update(f.relative_to(REPO).as_posix().encode() + b"\0" + f.read_bytes() + b"\0")
    ok = h.hexdigest() == SCORER_SHA256
    return ok, f"{len(files)} scorer files, hash {h.hexdigest()[:16]}" + ("" if ok else " DIFFERS from the record")


def c_time():
    code, out = run([PY, "-I", "scripts/verify/time_proxy_scan.py"])
    return code == 0, out.strip().splitlines()[-1] if out.strip() else f"exit {code}"


def c_secrets():
    code, out = run(["git", "grep", "-n", "-I", "-E", SECRET, "--", "."])
    hits = [ln for ln in out.splitlines() if ln.strip()]
    return not hits, "no keys or tokens in tracked files" if not hits else f"{len(hits)} hits: {hits[:2]}"


def c_lock():
    from importlib.metadata import distributions
    env = {d.metadata["Name"].lower().replace("_", "-"): d.version for d in distributions()}
    env.pop("nexus-alpha", None)
    lock = {}
    for line in (REPO / "requirements.lock").read_text(encoding="utf-8").splitlines():
        m = re.match(r"^([A-Za-z0-9_.\-]+)==(\S+)", line.strip())
        if m:
            lock[m.group(1).lower().replace("_", "-")] = m.group(2)
    missing = sorted(set(env) - set(lock))
    differ = sorted(k for k in set(env) & set(lock) if env[k] != lock[k])
    ok = not missing and not differ
    return ok, f"{len(lock)} pins match the environment" if ok else f"missing {missing[:6]}; differ {differ[:6]}"


def c_notebook():
    spec = importlib.util.spec_from_file_location("mk_nb", REPO / "scripts" / "colab" / "make_end_to_end_notebook.py")
    mk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mk)
    fresh = json.dumps(mk.build(), indent=1, ensure_ascii=False) + "\n"
    committed = mk.OUT.read_bytes().decode("utf-8").replace("\r\n", "\n")
    ok = fresh == committed
    return ok, "committed notebook = fresh build" if ok else "the committed notebook is stale: rebuild it"


def c_readme():
    code, out = run([PY, "-m", "pytest", "-p", "no:cacheprovider", "--co", "-q"])
    n = sum(int(m) for m in re.findall(r": (\d+)$", out, re.M))
    m = re.search(r"\| `tests/` \| (\d+) tests", (REPO / "README.md").read_text(encoding="utf-8"))
    claimed = int(m.group(1)) if m else None
    return claimed == n, f"README {claimed}, collected {n}"


def c_e28():
    from nexus_alpha import language_view as LV
    code, out = run([PY, "-I", "scripts/verify/audit_e28.py", "--run", LV.E28_RUN])
    last = [ln for ln in out.splitlines() if "checks passed" in ln or ln.startswith("[FAIL]")]
    return code == 0, "; ".join(last[-3:]) or f"exit {code}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="also re-run the E28 audit (about 5 minutes)")
    a = ap.parse_args()
    checks = [("test suite", c_pytest), ("frozen models", c_frozen), ("forward-test chains", c_chains),
              ("sealed scorer", c_scorer), ("no calendar time in inputs", c_time), ("secrets", c_secrets),
              ("dependency lock", c_lock), ("notebook build", c_notebook), ("README test count", c_readme)]
    if a.full:
        checks.append(("E28 audit", c_e28))
    failed = 0
    for name, fn in checks:
        t0 = time.time()
        try:
            ok, note = fn()
        except Exception as e:                                   # noqa: BLE001
            ok, note = False, f"{type(e).__name__}: {e}"
        failed += not ok
        print(f"[{'PASS' if ok else 'FAIL'}] {name:<28} {note}  ({time.time() - t0:.0f} s)", flush=True)
    print(f"{len(checks) - failed}/{len(checks)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

"""Nightly option-implied volatility snapshot for the E26 forward benchmark (pre-registered in docs/V3_RESEARCH.md).

    python scripts/shadow/iv_snapshot.py [--dry-run] [--tickers ...]

For the as-of session (the last closed one), fetches each forward-test stock's option chain from CBOE's delayed quotes
through OpenBB (scripts/openbb/options_chains.py in artifacts/envs/openbb) and computes the E26 measures
(nexus_alpha.options). Run after the forecasts and before 09:00 New York on the next session.

Data is kept private (CBOE's terms): artifacts/iv_snapshots/<asof>_chains.parquet (contracts) and <asof>.parquet (one
row per stock: IV30, front ATM IV, put/call open interest). The public record is shadow/iv_snapshot/<asof>.json, which
holds counts, times and both files' SHA-256, hash-chained in shadow/iv_snapshot/chain.csv. First write wins. Non-fatal
in the nightly run: a failure here never touches the forecasts.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from nexus_alpha import envs, options as O, paths, shadow as SH  # noqa: E402

VERSION = "iv_snapshot"
MIN_STOCKS = 300            # fewer stocks with an IV than this is logged, but flagged


def universe() -> list[str]:
    from nexus_alpha.cv import store as ST
    return sorted(ST.sample_index().ticker.unique())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--tickers", nargs="*")
    a = ap.parse_args()
    py = envs.env_python("openbb")
    if py is None:
        print("iv snapshot: OpenBB environment not installed (artifacts/envs/openbb); skipped")
        return 0
    asof = str(SH.expected_asof().date())
    pub = SH.SHADOW_ROOT / VERSION / f"{asof}.json"
    if pub.exists() and not a.dry_run:
        print(f"iv snapshot {asof}: already logged; nothing rewritten")
        return 0
    tickers = a.tickers or universe()
    priv = paths.ARTIFACTS / "iv_snapshots"
    priv.mkdir(parents=True, exist_ok=True)
    tag = "dryrun_" if a.dry_run else ""
    chains_f = priv / f"{tag}{asof}_chains.parquet"
    tfile = priv / f"{tag}{asof}_tickers.txt"
    tfile.write_text("\n".join(tickers), encoding="utf-8")
    t0 = SH.now_utc()
    r = subprocess.run([str(py), "-I", str(REPO / "scripts" / "openbb" / "options_chains.py"), "--tickers-file",
                        str(tfile), "--out", str(chains_f)], capture_output=True, text=True, timeout=3600,
                       env=envs.child_env())
    print(r.stdout.strip()[-300:])
    if r.returncode != 0 or not chains_f.exists():
        print(f"iv snapshot {asof}: chain fetch failed (exit {r.returncode}): {r.stderr.strip()[-300:]}")
        return 1
    errs = json.loads(Path(str(chains_f) + ".errors.json").read_text(encoding="utf-8"))
    chains = pd.read_parquet(chains_f)
    rows = []
    for t, g in chains.groupby("ticker"):
        s = O.summarize(g)
        rows.append({"asof_session": asof, "ticker": t, **s})
    summ = pd.DataFrame(rows)
    summ_f = priv / f"{tag}{asof}.parquet"
    summ.to_parquet(summ_f, index=False)
    run = SH.now_utc()
    dec = SH.decision_utc_for(pd.Timestamp(asof))
    n_iv = int(summ.iv30.notna().sum()) if len(summ) else 0
    rec = {"asof_session": asof, "snapshot_started_utc": t0.isoformat(timespec="seconds"),
           "snapshot_utc": run.isoformat(timespec="seconds"), "decision_utc": dec.isoformat(), "late": bool(run > dec),
           "tickers_requested": len(tickers), "tickers_with_chain": int(chains.ticker.nunique()),
           "tickers_with_iv30": n_iv, "fallback_parser": errs.get("fallback_parser", []),
           "chains_sha256": SH.sha256_file(chains_f), "summary_sha256": SH.sha256_file(summ_f),
           "source": "CBOE delayed quotes via OpenBB 5.0.0 (openbb_cboe); data kept private",
           "definitions": "nexus_alpha.options (E26 pre-registration)"}
    msg = (f"iv snapshot {asof}: {rec['tickers_with_chain']} of {len(tickers)} stocks with a chain, {n_iv} with IV30, "
           f"late {rec['late']}")
    if a.dry_run:
        print("[dry-run]", msg)
        return 0
    pub.parent.mkdir(parents=True, exist_ok=True)
    pub.write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
    SH.append_chain(VERSION, {"run_utc": run.isoformat(timespec="seconds"), "asof_session": asof,
                              "file": pub.name, "file_sha256": SH.sha256_file(pub), "n_rows": n_iv,
                              "n_failed": len(tickers) - rec["tickers_with_chain"], "late_rows": int(rec["late"])})
    probs = SH.verify_chain(VERSION)
    print(msg + ("" if n_iv >= MIN_STOCKS else f" (below {MIN_STOCKS}: flagged)") +
          f"; chain {'intact' if not probs else probs}", flush=True)
    os._exit(0 if not probs else 1)


if __name__ == "__main__":
    sys.exit(main())

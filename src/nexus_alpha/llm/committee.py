"""The LLM committee: TradingAgents v0.7.1 (Apache-2.0) as a live, on-demand second opinion (D-046, D-050; E27).

* Runs in its own environment (artifacts/envs/tradingagents) through scripts/llm/tradingagents_runner.py, on the
  owner's Gemini key. Only the key's value travels, through the child process environment, never through a file.
* Never a model feature, never blended into a probability, never an input to the risk engine. Its language model has
  read about the outcomes of 2003-2022, so it is never scored on history (D-020). It is only scored forward (E27).
* Every run is logged twice:
  * privately, the full reports in artifacts/llm_committee/runs/ (LLM text built from news and filings);
  * publicly, the rating with metadata and the reports' SHA-256 in shadow/committee/runs/, hash-chained in
    shadow/committee/chain.csv.
  A run after the decision time (09:00 New York on the next session) is flagged late and never scored.
* Quota: the free Gemini tier allows 20 requests per model per day, the same pool as the Trade Desk's Gemini read.
  A run is refused when the estimated calls exceed what is left today. Calls actually made are added to the shared
  usage counter.
"""
from __future__ import annotations

import json
import os
import subprocess

import pandas as pd

from .. import envs, paths, shadow as SH
from . import gemini as GM

RUNNER = paths.REPO_ROOT / "scripts" / "llm" / "tradingagents_runner.py"
VERSION = "committee"
DAILY_LIMIT = 20
ANALYSTS = ("market", "fundamentals", "news", "social")
DIRECTION = {"Buy": 2, "Overweight": 2, "Hold": 1, "Underweight": 0, "Sell": 0}     # E27 mapping; REVIEW = failure


def estimate_calls(n_analysts: int) -> int:
    """About 4 calls per analyst (tool rounds) plus 8 for the researchers, managers, trader and risk debate."""
    return 8 + 4 * max(1, n_analysts)


def available() -> tuple[bool, str]:
    if envs.env_python("tradingagents") is None:
        return False, "TradingAgents is not installed (artifacts/envs/tradingagents)."
    if not GM.available():
        return False, "Gemini is not configured (GEMINI_API_KEY)."
    return True, ""


def model() -> str:
    return os.environ.get("NEXUS_COMMITTEE_MODEL") or GM.pick_model()


def quota_left(m: str) -> int:
    return DAILY_LIMIT - int(GM.usage_today().get(m, 0))


def history() -> pd.DataFrame:
    rows = [json.loads(p.read_text(encoding="utf-8"))
            for p in sorted((SH.SHADOW_ROOT / VERSION / "runs").glob("*.json"))]
    return pd.DataFrame(rows)


TRANSIENT = ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED", "DEADLINE", "504")
MAX_MODELS = 3


def _attempt(py, work, run_id: str, ticker: str, asof, analysts: list[str], m: str, extra: dict, timeout_s: int) -> dict:
    out = work / "runs" / f"{run_id}_{m}.json"
    req = work / "requests" / f"{run_id}_{m}.json"
    req.write_text(json.dumps({"ticker": ticker, "trade_date": str(asof.date()), "analysts": analysts,
                               "quick_model": m, "deep_model": m, "work_dir": str(work / "work"), "out": str(out),
                               "llm_max_retries": 3}), encoding="utf-8")
    try:
        subprocess.run([str(py), "-I", str(RUNNER), str(req)], capture_output=True, text=True, timeout=timeout_s,
                       env=envs.child_env(extra), cwd=str(work / "work"))
    except subprocess.TimeoutExpired:
        pass
    res = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {"error": "no output (timeout?)"}
    res["_out"] = out
    for _ in range(int(res.get("llm_calls", 0))):           # keep the shared free-tier counter truthful
        GM._count(m)
    return res


def run(ticker: str, analysts: list[str], timeout_s: int = 1800, quick_model: str | None = None) -> dict:
    """One committee run. A transient API failure early in the run (Google 503 "high demand", 429, deadline) falls
    over to the next flash model the key can use, as the Trade Desk's Gemini read does, up to MAX_MODELS models."""
    ok, why = available()
    if not ok:
        raise RuntimeError(why)
    analysts = [a for a in analysts if a in ANALYSTS] or ["market"]
    need = estimate_calls(len(analysts))
    models = [quick_model] if quick_model else [model()] + [x for x in GM.candidate_models() if x != model()]
    models = [x for x in models if quota_left(x) >= need][:MAX_MODELS]
    if not models:
        m0 = quick_model or model()
        raise RuntimeError(f"about {need} Gemini calls needed, {quota_left(m0)} left today on {m0} (free tier: "
                           f"{DAILY_LIMIT} per day, reset at midnight Pacific). Try fewer analysts or tomorrow.")
    from .. import secrets
    secrets.load()
    asof = SH.expected_asof()
    decision = SH.decision_utc_for(asof)
    start = SH.now_utc()
    run_id = f"{start:%Y%m%dT%H%M%SZ}_{ticker}"
    work = paths.ARTIFACTS / "llm_committee"
    for d in ("requests", "runs", "work"):
        (work / d).mkdir(parents=True, exist_ok=True)
    extra = {"GOOGLE_API_KEY": GM.api_key() or "", "TRADINGAGENTS_NO_UPDATE_CHECK": "1"}
    if os.environ.get("SEC_USER_AGENT"):
        extra["SEC_EDGAR_USER_AGENT"] = os.environ["SEC_USER_AGENT"]
    if os.environ.get("FRED_API_KEY"):
        extra["FRED_API_KEY"] = os.environ["FRED_API_KEY"]
    py = envs.env_python("tradingagents")
    tried = []
    for m in models:
        res = _attempt(py, work, run_id, ticker, asof, analysts, m, extra, timeout_s)
        tried.append(f"{m}: {res.get('error', 'ok')[:120]}")
        if "error" not in res:
            break
        if not (any(t in res["error"] for t in TRANSIENT) and int(res.get("llm_calls", 0)) <= 3):
            break                                           # not transient, or failed mid-run: do not spend more
    if "error" in res:
        raise RuntimeError(f"TradingAgents failed ({len(tried)} model(s) tried): " + " | ".join(tried))
    out = res.pop("_out")
    end = SH.now_utc()
    rec = {"run_id": run_id, "ticker": ticker, "asof_session": str(asof.date()),
           "run_started_utc": start.isoformat(timespec="seconds"), "run_utc": end.isoformat(timespec="seconds"),
           "decision_utc": decision.isoformat(), "late": bool(end > decision), "rating": res["rating"],
           "direction_class": DIRECTION.get(res["rating"]), "analysts": analysts, "quick_model": m, "deep_model": m,
           "llm_calls": res["llm_calls"], "seconds": res["seconds"], "reports_sha256": SH.sha256_file(out),
           "tradingagents_version": res.get("tradingagents_version")}
    pub = SH.SHADOW_ROOT / VERSION / "runs" / f"{run_id}.json"
    pub.parent.mkdir(parents=True, exist_ok=True)
    pub.write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
    SH.append_chain(VERSION, {"run_utc": rec["run_utc"], "asof_session": rec["asof_session"],
                              "file": f"runs/{pub.name}", "file_sha256": SH.sha256_file(pub), "n_rows": 1,
                              "n_failed": int(rec["direction_class"] is None), "late_rows": int(rec["late"])})
    return {**rec, "reports": res.get("reports", {}), "chain_ok": not SH.verify_chain(VERSION)}

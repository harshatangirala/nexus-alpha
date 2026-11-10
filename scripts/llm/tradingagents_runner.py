"""One TradingAgents committee run (runs in artifacts/envs/tradingagents, never in the project env). E27, D-050.

    <tradingagents env python> -I scripts/llm/tradingagents_runner.py <request.json>

Request keys: ticker, trade_date (the as-of session), analysts, quick_model, deep_model, work_dir, out, and optionally
max_debate_rounds and max_risk_rounds (default 1). Keys come only from the environment (GOOGLE_API_KEY,
SEC_EDGAR_USER_AGENT), never from the request file.

Writes <out> as JSON: rating (Buy, Overweight, Hold, Underweight, Sell, or REVIEW), llm_calls (counted by a callback on
every chat-model start), seconds, every text field of the final state, and the package version. On failure it writes
{"error": ...} and exits 1. TradingAgents' result folders, cache and memory log all go under work_dir, never under the
user's home folder.
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path


def main() -> int:
    req = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    out = Path(req["out"])
    os.environ.setdefault("TRADINGAGENTS_NO_UPDATE_CHECK", "1")
    t0 = time.time()
    calls = {"n": 0}
    try:
        from importlib import metadata

        from langchain_core.callbacks import BaseCallbackHandler
        from tradingagents.default_config import DEFAULT_CONFIG
        from tradingagents.graph.trading_graph import TradingAgentsGraph

        class Counter(BaseCallbackHandler):
            def on_chat_model_start(self, *a, **k):
                calls["n"] += 1

            def on_llm_start(self, *a, **k):
                calls["n"] += 1

        work = Path(req["work_dir"])
        cfg = DEFAULT_CONFIG.copy()
        cfg.update({"llm_provider": "google", "quick_think_llm": req["quick_model"], "deep_think_llm": req["deep_model"],
                    "max_debate_rounds": int(req.get("max_debate_rounds", 1)),
                    "max_risk_discuss_rounds": int(req.get("max_risk_rounds", 1)),
                    "results_dir": str(work / "logs"), "data_cache_dir": str(work / "cache"),
                    "memory_log_path": str(work / "memory" / "trading_memory.md"),
                    "llm_max_retries": int(req.get("llm_max_retries", 6)),
                    "output_language": "English"})
        ta = TradingAgentsGraph(selected_analysts=tuple(req["analysts"]), config=cfg, callbacks=[Counter()])
        state, rating = ta.propagate(req["ticker"], req["trade_date"])
        reports = {}
        for k, v in dict(state).items():
            if isinstance(v, str) and v.strip():
                reports[k] = v
            elif isinstance(v, dict):
                for kk, vv in v.items():
                    if isinstance(vv, str) and vv.strip():
                        reports[f"{k}.{kk}"] = vv
        res = {"rating": str(rating), "llm_calls": calls["n"], "seconds": round(time.time() - t0, 1),
               "reports": reports, "tradingagents_version": metadata.version("tradingagents"),
               "quick_model": req["quick_model"], "deep_model": req["deep_model"], "analysts": list(req["analysts"])}
        out.write_text(json.dumps(res, indent=1), encoding="utf-8")
        return 0
    except Exception as e:  # noqa: BLE001  (reported to the caller as data)
        out.write_text(json.dumps({"error": f"{type(e).__name__}: {e}"[:2000], "llm_calls": calls["n"],
                                   "trace": traceback.format_exc()[-3000:], "seconds": round(time.time() - t0, 1)}),
                       encoding="utf-8")
        return 1


if __name__ == "__main__":
    sys.exit(main())

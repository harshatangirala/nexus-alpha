"""E17: accuracy of Gemini's legend reading on TriFrame's 66 real TradingView screenshots (22 stocks x 1W/1D/1H).

Reference: the hand-recorded legend values in us_stocks_ohlcv/screenshots/legend_readings.csv. One Gemini call per stock
(three images). Reports per-field absolute and relative errors and the share read exactly (to the printed cent).
This validates only the screenshot check of the Trade Desk (D-019); no forecast is evaluated, no sealed data is read.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from nexus_alpha import paths  # noqa: E402
from nexus_alpha.llm import gemini as G  # noqa: E402

SHOTS = paths.OHLCV_ROOT / "screenshots"
SLOT = {"1W": "W", "1D": "D", "1H": "H"}


def main() -> None:
    ref = pd.read_csv(SHOTS / "legend_readings.csv")
    rows, model = [], None
    cache = paths.artifact("llm_reads", "e17_cache", "x").parent       # one JSON per stock: resumable, no re-spend
    failed = []
    for t in sorted(ref.ticker.unique()):
        imgs = [(SLOT[tf], (SHOTS / "images" / f"{t}_{tf}.png").read_bytes(), "image/png")
                for tf in ("1W", "1D", "1H") if (SHOTS / "images" / f"{t}_{tf}.png").exists()]
        if not imgs:
            continue
        cf = cache / f"{t}.json"
        if cf.exists():
            out = json.loads(cf.read_text(encoding="utf-8"))
        else:
            try:
                out = G.read_legends(imgs)
            except Exception as e:
                failed.append({"ticker": t, "error": str(e)[:160]})
                print(t, "FAILED", str(e)[:120], flush=True)
                time.sleep(30)
                continue
            cf.write_text(json.dumps(out), encoding="utf-8")
            time.sleep(12)                                # free-tier requests-per-minute
        model = out.get("_model")
        got = {c.get("slot"): c for c in out.get("charts", [])}
        for _, r in ref[ref.ticker == t].iterrows():
            g = got.get(SLOT[r.tf], {})
            for f in ("open", "high", "low", "close"):
                v = g.get(f"legend_{f}")
                rows.append({"ticker": t, "tf": r.tf, "field": f, "reference": float(r[f]),
                             "gemini": None if v is None else float(v), "model": out.get("_model"),
                             "ticker_read": g.get("ticker_shown"), "tf_read": g.get("timeframe_shown")})
        print(t, "done", flush=True)
    d = pd.DataFrame(rows)
    d["read"] = d.gemini.notna()
    d["abs_err"] = (d.gemini - d.reference).abs()
    d["exact_cent"] = d.abs_err <= 0.005 + 1e-9
    d["rel_err_pct"] = d.abs_err / d.reference * 100
    by_tf = d.groupby("tf").agg(values=("field", "size"), read_share=("read", "mean"), exact_share=("exact_cent", "mean"),
                                max_rel_err_pct=("rel_err_pct", "max"), median_rel_err_pct=("rel_err_pct", "median"))
    out = {"model": model, "models_used": d.model.value_counts().to_dict(), "failed_stocks": failed, "screenshots": int(d.groupby(["ticker", "tf"]).ngroups), "values": int(len(d)),
           "read_share": float(d.read.mean()), "exact_share_of_all": float(d.exact_cent.mean()),
           "within_0p3pct_share_of_all": float((d.rel_err_pct <= 0.3).mean()),
           "ticker_correct_share_of_read_screens": float(np.mean([str(tr).upper().find(str(tk).upper()) >= 0 for tr, tk in
                                                         zip(d[d.read].ticker_read.fillna(""), d[d.read].ticker)]))
           if d.read.any() else None,
           "by_timeframe": json.loads(by_tf.to_json(orient="index"))}
    (REPO / "reports" / "verification").mkdir(parents=True, exist_ok=True)
    d.to_csv(REPO / "reports" / "verification" / "gemini_legend_reader.csv", index=False)
    (REPO / "reports" / "verification" / "gemini_legend_reader.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

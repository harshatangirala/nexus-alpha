"""The one-time joint sealed evaluation (docs/SEALED_EVALUATION_PLAN.md), its configuration, and its rehearsal.

    python scripts/sealed/run_sealed_eval.py --write-config            # writes governance/sealed_eval_config.json
    python scripts/sealed/run_sealed_eval.py --rehearsal 2022          # the same pipeline on development year 2022
    python scripts/sealed/run_sealed_eval.py --who "Owner Name"        # THE evaluation: owner only, once

The evaluation run needs, in the owner's own terminal:
    NEXUS_HOLDOUT_CONFIG = <path to governance/sealed_eval_config.json>
    NEXUS_HOLDOUT_UNLOCK = <its SHA-256>        (printed by --write-config; code never creates or guesses it)
    NEXUS_ENV_FILES      = <owner key files> (FRED key for post-seal ALFRED vintages)
It refuses to start unless the git tree is clean and the plan, scorer and frozen records match the configuration.
Steps: open the gate (unlock, hash-chained log) -> fetch post-seal ETF minute bars, Yahoo series and ALFRED vintages ->
samples (every 10th own bar after each stock's last development as-of) -> bars, labels, features, chart embeddings (CPU)
-> news and SEC features (frozen TF-IDF) -> v1.1, v2 and the E19 market read -> endpoints and frozen backtests ->
reports/sealed/. Expected CPU time: about 3-5 hours (embedding about 2.4 h at the measured 16.5 images/s).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "experiments"))
sys.path.insert(0, str(REPO / "scripts" / "data"))
warnings.filterwarnings("ignore")

from nexus_alpha import paths, seal, shadow as SH  # noqa: E402

CONFIG = REPO / "governance" / "sealed_eval_config.json"
PLAN = REPO / "docs" / "SEALED_EVALUATION_PLAN.md"
SCORER_FILES = sorted([*(REPO / "scripts" / "sealed").glob("*.py"), *(REPO / "src" / "nexus_alpha" / "sealed").glob("*.py")])
FROZEN = ("v1.1", "v2", "market_e19")


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def scorer_sha() -> str:
    h = hashlib.sha256()
    for f in SCORER_FILES:
        h.update(f.relative_to(REPO).as_posix().encode() + b"\0" + f.read_bytes() + b"\0")
    return h.hexdigest()


def git(*args) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True).stdout.strip()


def config_payload() -> dict:
    return {"purpose": "NEXUS Alpha joint sealed evaluation (D-018), opened once", "code_commit": git("rev-parse", "HEAD"),
            "plan_sha256": sha(PLAN), "scorer_sha256": scorer_sha(),
            "scorer_files": [f.relative_to(REPO).as_posix() for f in SCORER_FILES],
            "frozen_records_sha256": {v: sha(SH.FROZEN_DIR / f"{v}.json") for v in FROZEN},
            "outputs": "reports/sealed/"}


def write_config() -> None:
    if git("status", "--porcelain", "--untracked-files=no"):
        sys.exit("commit every change first: the configuration pins the committed code")
    CONFIG.parent.mkdir(parents=True, exist_ok=True)
    CONFIG.write_text(json.dumps(config_payload(), indent=1), encoding="utf-8")
    print(f"wrote {CONFIG}\nits SHA-256 (the owner's NEXUS_HOLDOUT_UNLOCK): {sha(CONFIG)}")


def preflight() -> None:
    if git("status", "--porcelain", "--untracked-files=no"):
        sys.exit("refusing: the git tree has uncommitted changes")
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    now = config_payload()
    for k in ("plan_sha256", "scorer_sha256", "frozen_records_sha256"):
        if cfg[k] != now[k]:
            sys.exit(f"refusing: {k} differs from the frozen configuration")
    for v in FROZEN:
        SH.verify_frozen(v)


# ------------------------------------------------------------------------------------------------ steps
def refresh_sealed_sources(log) -> None:
    """Post-seal ETF minute bars -> ETF bars (artifacts/etf_sealed), Yahoo market series, ALFRED vintages."""
    import build_etf_bars as BEB
    import fetch_etf_minute as FEM
    from nexus_alpha.macro import fred as FR, market as MK
    FEM.run("2023-01", pd.Timestamp.now().strftime("%Y-%m"), include_sealed=True)
    BEB.build(include_sealed=True)
    MK.build_full_store()
    FR.build_store_full()
    log("sealed sources refreshed")


def samples_for(mode: str, year: int | None, log) -> pd.DataFrame:
    from nexus_alpha.cv import store as ST
    from nexus_alpha.sealed import frame as SF
    if mode == "rehearsal":
        dev = pd.read_parquet(paths.ARTIFACTS / "samples" / "s500_frame.parquet", columns=["sample_id", "ticker", "asof_date"])
        return dev[dev.asof_date.dt.year == year].reset_index(drop=True)
    idx = ST.sample_index()
    last = idx.groupby("ticker").asof_date.max().to_dict()
    s = SF.continue_schedule(last, str(seal.SEAL_DATE.date()))
    log(f"sealed samples: {len(s):,} ({s.ticker.nunique()} stocks, {s.asof_date.min().date()}..{s.asof_date.max().date()})")
    return s


def market_read(start: pd.Timestamp, end: pd.Timestamp, beyond_seal: bool) -> pd.DataFrame:
    from nexus_alpha.data import etf_bars as EB
    from nexus_alpha.indicators.market import market_context
    from nexus_alpha.macro import market_model as MM
    from nexus_alpha.macro.context import period_date_features
    from nexus_alpha.targets_v2 import make_labels_v2
    mm = MM.MarketModel(SH.verify_frozen("market_e19")[0])
    f = period_date_features(beyond_seal).join(market_context()[MM.UNIVERSE_COLS])
    f = f[(f.index >= start) & (f.index <= end)]
    lab = make_labels_v2(EB.read_daily("SPY")).set_index("date")
    rows = []
    for d, r in f.iterrows():
        p = mm.predict(r)["20D"]
        rows.append({"asof_date": d, "p_down": p["p_down"], "p_side": p["p_side"], "p_up": p["p_up"],
                     "missing_share": p["missing_share"], "y_20D": lab.y_20D.get(d, np.nan), "z_20D": lab.z_20D.get(d, np.nan)})
    return pd.DataFrame(rows)


def backtests(full: pd.DataFrame, events: pd.DataFrame, r1: dict, r2: dict, start, end, log) -> dict:
    import e04_backtest as E04
    from nexus_alpha.data import bars as B
    from nexus_alpha.targets import HORIZONS
    from nexus_alpha.targets_v2 import HORIZONS_V2
    from nexus_alpha.trading.backtest import performance
    from nexus_alpha.trading.costs import CostModel
    E04.BLOCK_DATES["sealed"] = (str(start.date()), str(end.date()))
    ev = full[["sample_id"]].merge(events, on="sample_id", how="left")
    daily = {t: B.read_daily(t) for t in ev.ticker.dropna().unique()}
    out = {}
    specs = [("v1.1", r1["probs"], HORIZONS, json.loads((REPO / "reports" / "backtest" / "e04_backtest_v1.1.json").read_text()),
              json.loads((SH.verify_frozen("v1.1")[0] / "bundle_meta.json").read_text())["direction_skill"], ("5D", "20D")),
             ("v2", r2["probs"], HORIZONS_V2, json.loads((REPO / "reports" / "backtest" / "e15_backtest_v2.json").read_text()),
              json.loads((SH.verify_frozen("v2")[0] / "bundle_meta.json").read_text())["direction_skill"],
              ("1D", "5D", "10D", "20D", "3M"))]
    for name, P, hz_map, frozen_run, skill, hz in specs:
        E04.HORIZONS = hz_map
        frozen = frozen_run["frozen_thresholds"]
        trials = frozen_run.get("n_trials_2019") or frozen_run.get("n_trials_valid")
        for h in hz:
            for pol in ("rule_only", "model_atr", "nexus", "nexus_gated"):
                th = frozen.get(f"{pol}|{h}", 0.10)
                for sc in ("base", "stressed", "severe"):
                    ret, tr, kept = E04.run_config(E04.orders_for(ev, P, pol, h, th, skill), daily, "sealed",
                                                   CostModel.from_scenario(sc))
                    out[f"{name}|{pol}|{h}|theta={th}|{sc}"] = performance(ret, kept if len(kept) else None, n_trials=trials)
        log(f"backtests {name} done")
    E04.HORIZONS = HORIZONS
    from nexus_alpha.data.calendar import sessions
    from nexus_alpha.indicators.market import market_context
    s = sessions()
    cal = s[(s >= start) & (s <= end)]
    bench = {"ew_survivors": performance(market_context().mkt_ret1.reindex(cal).fillna(0.0))}
    try:
        import yfinance as yf
        spy = yf.Ticker("SPY").history(start=str((start - pd.Timedelta(days=10)).date()),
                                       end=str((end + pd.Timedelta(days=3)).date()), auto_adjust=True)["Close"]
        spy.index = pd.DatetimeIndex(spy.index).tz_localize(None).normalize()
        bench["spy_total_return"] = performance(spy.pct_change().reindex(cal).fillna(0.0))
    except Exception as e:
        bench["spy_total_return"] = {"error": repr(e)[:200]}
    return {"policies": out, "benchmarks": bench, "window": [str(start.date()), str(end.date())]}


def descriptive(full: pd.DataFrame, r1: dict, r2: dict) -> dict:
    """Per-year, news-covered vs not, and SPY regime (above/below its 200-day average): log loss and AUROC."""
    from nexus_alpha.data import etf_bars as EB
    from nexus_alpha.evaluation import ladder as L
    spy = EB.read_daily("SPY").set_index("date").close
    above = (spy > spy.rolling(200).mean()).reindex(full.asof_date).fillna(False).to_numpy().astype(bool)
    splits = {f"year_{y}": (full.asof_date.dt.year == y).to_numpy() for y in sorted(full.asof_date.dt.year.unique())}
    splits |= {"news_present": (full.news_present > 0).to_numpy(), "news_absent": (full.news_present <= 0).to_numpy(),
               "spy_above_200d": above.astype(bool), "spy_below_200d": ~above.astype(bool)}
    out = {}
    for name, r in (("v1.1", r1), ("v2", r2)):
        for sname, m in splits.items():
            for j, h in enumerate(r["horizons"]):
                if h not in ("1D", "5D", "20D", "1M", "3M", "1Y"):
                    continue
                y = full[f"y_{h}"].to_numpy()
                k = m & ~np.isnan(y)
                if k.sum() < 50:
                    continue
                yy = y[k].astype(int)
                out.setdefault(name, {}).setdefault(sname, {})[h] = {
                    "n": int(k.sum()), "log_loss": float(L.ll_rows(yy, r["probs"][k, j]).mean()),
                    "auroc_ud": L.auroc_ud(yy, r["probs"][k, j])}
    return out


def evaluate(mode: str, year: int | None, who: str | None) -> None:
    from nexus_alpha.inference.pipeline import Bundle
    from nexus_alpha.inference.pipeline_v2 import NexusPredictorV2
    from nexus_alpha.sealed import apply as AP, assemble as AS, frame as SF, metrics as M
    from nexus_alpha.targets import HORIZONS
    from nexus_alpha.targets_v2 import HORIZONS_V2
    import e13_sequence as E13
    t0 = time.time()
    tag = f"rehearsal_{year}" if mode == "rehearsal" else "sealed"
    outdir = (REPO / "reports" / "sealed") if mode == "sealed" else (REPO / "reports" / "verification" / f"sealed_dry_run_{year}")
    outdir.mkdir(parents=True, exist_ok=True)
    logf = open(outdir / "run.log", "a", encoding="utf-8")

    def log(m):
        line = f"[{time.strftime('%H:%M:%S')}] {m}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    beyond = mode == "sealed"
    if beyond:
        preflight()
        h = seal.open_for_evaluation(who)
        log(f"gate opened (configuration {h[:16]}...)")
        refresh_sealed_sources(log)
    art = paths.ARTIFACTS / "sealed" / tag
    if mode == "rehearsal" and (art / "frame.parquet").exists():
        fr = pd.read_parquet(art / "frame.parquet")
        emb = {v: np.load(art / f"emb_{v}.npy") for v in ("W", "D", "H")}
        if (art / "events.parquet").exists():
            events = pd.read_parquet(art / "events.parquet")
        else:   # development rows: their rule events are the stored ones (rule parity is part of the rehearsal)
            events = pd.read_parquet(paths.ARTIFACTS / "features" / "rule_events.parquet")
            events = events[events.sample_id.isin(set(fr.sample_id))]
        log(f"reusing the rehearsal build ({len(fr):,} rows)")
    else:
        s = samples_for(mode, year, log)
        fr, emb, events = SF.build(s, log=log)
        art.mkdir(parents=True, exist_ok=True)
        fr.to_parquet(art / "frame.parquet", index=False)
        events.to_parquet(art / "events.parquet", index=False)
        for v, e in emb.items():
            np.save(art / f"emb_{v}.npy", e)
    full, info = AS.assemble(fr, beyond_seal=beyond, fetch_news=beyond, log=log)
    b1 = Bundle(SH.verify_frozen("v1.1")[0])
    r1 = AP.v11(full, emb, b1)
    P2 = NexusPredictorV2(SH.verify_frozen("v2")[0])
    r2 = AP.v2(full, emb, E13.build_X(full[["ticker", "asof_date"]].reset_index(drop=True)), P2)
    log("models scored")
    np.save(art / "probs_v11.npy", r1["probs"].astype(np.float32))
    np.save(art / "probs_v2.npy", r2["probs"].astype(np.float32))
    rec1 = json.loads((SH.FROZEN_DIR / "v1.1.json").read_text())
    rec2 = json.loads((SH.FROZEN_DIR / "v2.json").read_text())
    lab = full[["asof_date"] + [c for c in full.columns if c.startswith(("y_", "z_"))]]
    ev1 = M.evaluate("v1.1", r1["probs"], r1["horizons"], dict(HORIZONS), lab, rec1["train_class_prior"], M.GROUPS_V1, r1["quantiles"])
    ev2 = M.evaluate("v2", r2["probs"], r2["horizons"], dict(HORIZONS_V2), lab, rec2["train_class_prior"], M.GROUPS_V2, r2["quantiles"])
    cmp_ = M.compare("v1.1", r1["probs"], r1["horizons"], "v2", r2["probs"], r2["horizons"], lab,
                     ["1D", "2D", "5D", "10D", "20D"], dict(HORIZONS_V2))
    start = full.asof_date.min()
    end = full.asof_date.max()
    mr = market_read(start, end, beyond)
    recm = json.loads((SH.FROZEN_DIR / "market_e19.json").read_text())
    mlab = mr[["asof_date", "y_20D", "z_20D"]]
    evm = M.evaluate("market_e19", mr[["p_down", "p_side", "p_up"]].to_numpy()[:, None, :], ["20D"], {"20D": 20}, mlab,
                     recm["train_class_prior"], {"swing": ["20D"]})
    log("endpoints computed")
    from nexus_alpha.data import etf_bars as EB
    last_bar = EB.read_daily("SPY").date.max()                     # never past the data the gate allows
    bt = backtests(full, events, r1, r2, start, min(end + pd.Timedelta(days=45), last_bar), log)
    desc = descriptive(full, r1, r2)
    report = {"mode": mode, "year": year, "who": who, "elapsed_s": None, "assembly": info,
              "period": [str(start.date()), str(end.date())], "rows": len(full), "stocks": int(full.ticker.nunique()),
              "primary": {"v1.1": ev1["groups_primary"], "v2": ev2["groups_primary"]},
              "by_horizon": {"v1.1": ev1["by_horizon"], "v2": ev2["by_horizon"]}, "v2_vs_v11": cmp_,
              "market_e19": evm, "backtests": bt, "descriptive": desc,
              "scorer_sha256": scorer_sha(), "plan_sha256": sha(PLAN), "code_commit": git("rev-parse", "HEAD")}
    report["elapsed_s"] = round(time.time() - t0, 1)
    (outdir / "sealed_evaluation.json").write_text(json.dumps(report, indent=1, default=float), encoding="utf-8")
    mr.to_csv(outdir / "market_read_daily.csv", index=False, float_format="%.6g")
    log(f"report written to {outdir} ({report['elapsed_s']:.0f}s)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-config", action="store_true")
    ap.add_argument("--rehearsal", type=int)
    ap.add_argument("--who")
    a = ap.parse_args()
    if a.write_config:
        write_config()
    elif a.rehearsal:
        evaluate("rehearsal", a.rehearsal, None)
    elif a.who:
        evaluate("sealed", None, a.who)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()

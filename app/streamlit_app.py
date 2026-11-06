"""NEXUS Alpha research dashboard (Streamlit). Research and paper-trading decision support only.

Run from the repository root:  .venv/Scripts/streamlit run app/streamlit_app.py
Everything shown comes from the real inference bundle (artifacts/bundles/<run_id>) or from saved experiment artifacts.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nexus_alpha import paths  # noqa: E402
from nexus_alpha.trading.plan import RiskConfig  # noqa: E402

st.set_page_config(page_title="NEXUS Alpha", layout="wide")
HZ = ["1D", "2D", "5D", "10D", "20D", "1M"]


@st.cache_resource(show_spinner="Loading the inference bundle and knowledge base...")
def load_predictor():
    from nexus_alpha import shadow as SH
    from nexus_alpha.inference.pipeline import NexusPredictor
    from nexus_alpha.rag.explain import load_kb
    kb = load_kb()
    bdir, _ = SH.verify_frozen("v1.1")          # the frozen model, hash-verified (not "the newest bundle")
    return NexusPredictor(bdir, kb=kb), kb


@st.cache_data
def universe():
    from nexus_alpha.cv import store as ST
    return sorted(ST.sample_index().ticker.unique())


def latest_run(prefix: str) -> dict | None:
    runs = sorted((ROOT / "experiments").glob(f"*_{prefix}*"))
    for r in reversed(runs):
        man = json.loads((r / "manifest.json").read_text(encoding="utf-8"))
        if man.get("status") == "COMPLETED" and (r / "metrics.json").exists():
            return {"id": r.name, "manifest": man, "metrics": json.loads((r / "metrics.json").read_text(encoding="utf-8"))}
    return None


def chart_png(img: np.ndarray) -> np.ndarray:
    a = (1 - img.astype(np.float32)) * 255
    return np.kron(a, np.ones((2, 2))).astype(np.uint8)


# ------------------------------------------------------------------------------------------------ sidebar
st.sidebar.title("NEXUS Alpha")
st.sidebar.caption("Research and paper-trading decision support. Not investment advice. No orders are ever placed.")
try:
    predictor, kb = load_predictor()
    bundle_ok = True
except Exception as e:
    bundle_ok, predictor, kb = False, None, None
    st.sidebar.error(f"No inference bundle: {e}")
mode = st.sidebar.radio("Data mode", ["historical", "live"], help="historical: sealed bars to 2022-12-30; live: Yahoo Finance bars")
tick = st.sidebar.selectbox("Ticker (s500 universe)", universe(), index=universe().index("AAPL") if "AAPL" in universe() else 0)
asof = None
if mode == "historical":
    asof = st.sidebar.date_input("As-of session", value=pd.Timestamp("2021-06-15"), min_value=pd.Timestamp("2004-01-02"),
                                 max_value=pd.Timestamp("2022-12-29"))
st.sidebar.subheader("Risk settings (paper)")
capital = st.sidebar.number_input("Account capital", 1000.0, 1e9, 100_000.0, step=1000.0)
risk_pct = st.sidebar.slider("Risk per trade (%)", 0.1, 2.0, 0.5, 0.1) / 100
max_pos = st.sidebar.slider("Max position (% of capital)", 1, 50, 10) / 100
allow_short = st.sidebar.checkbox("Short selling allowed", True)
cost_sc = st.sidebar.selectbox("Cost scenario", ["base", "stressed", "severe"])
occl = st.sidebar.checkbox("Chart occlusion attribution (slower)", False)

st.sidebar.subheader("News articles (1-7, optional)")
n_art = st.sidebar.number_input("Number of articles", 0, 7, 0)
articles = []
for i in range(int(n_art)):
    with st.sidebar.expander(f"Article {i + 1}"):
        h = st.text_input("Headline", key=f"h{i}")
        t = st.text_area("Text", key=f"t{i}", height=80)
        when = st.text_input("Published (UTC, e.g. 2021-06-14 21:30)", key=f"w{i}")
        src = st.text_input("Source", key=f"s{i}")
        if h or t:
            articles.append({"headline": h, "text": t, "published_utc": when, "source": src})
shots = st.sidebar.file_uploader("Chart screenshots (weekly, daily, hourly; shown only)", type=["png", "jpg"],
                                 accept_multiple_files=True)
go = st.sidebar.button("Run analysis", type="primary", disabled=not bundle_ok)

if go:
    risk = RiskConfig(account_capital=capital, risk_pct_per_trade=risk_pct, max_position_pct=max_pos,
                      allow_short=allow_short, cost_scenario=cost_sc)
    with st.spinner("Rendering charts, embedding, scoring..."):
        try:
            res = predictor.predict(tick, asof=str(asof) if asof else None, mode=mode, articles=articles, risk=risk,
                                    explain=True, occlusion=occl)
            st.session_state["res"] = res
            from nexus_alpha.cv import render as R
            if mode == "historical":
                sb = R.stock_bars_for_render(tick)
                st.session_state["trip"] = R.triplet_at(sb, sb.k_of(pd.Timestamp(asof)))
            else:
                st.session_state["trip"] = None
        except Exception as e:
            st.error(f"Analysis failed: {type(e).__name__}: {e}")

res = st.session_state.get("res")
all_tabs = st.tabs(["Trade Desk", "A Forecast", "B Charts", "C News", "D SEC", "E Indicators", "F Explain",
                    "G Trade planner", "H Model diagnostics", "I Backtesting", "J Knowledge base", "K Configuration",
                    "L Audit"])
with all_tabs[0]:
    import trade_desk_tab  # noqa: E402  (app/ is on sys.path when Streamlit runs this script)
    trade_desk_tab.render(predictor if bundle_ok else None, universe(),
                          dict(account_capital=capital, risk_pct_per_trade=risk_pct, max_position_pct=max_pos,
                               allow_short=allow_short, cost_scenario=cost_sc))
tabs = all_tabs[1:]

with tabs[0]:
    st.header("Forecast overview")
    if res is None:
        st.info("Choose a ticker and press Run analysis.")
    else:
        i = res.instrument
        c = st.columns(4)
        c[0].metric("Symbol", i.symbol)
        c[1].metric("As-of session", i.asof_session)
        c[2].metric("Reference close", f"{i.reference_price:,.2f}")
        c[3].metric("Data quality", i.data_quality)
        st.caption(f"Decision timestamp {i.forecast_timestamp} (09:00 New York, next session). Bundle {res.model.bundle_id}; "
                   f"calibration {res.model.calibration_status}; trained through {res.model.trained_through}.")
        df = pd.DataFrame({h: {"UP": f.up_probability, "SIDEWAYS": f.sideways_probability, "DOWN": f.down_probability,
                               "predicted": f.predicted_direction, "uncertainty (normalised entropy)": f.uncertainty,
                               "direction skill validated": f.direction_skill_validated,
                               "median return": f.expected_return, "return q10": f.return_q10, "return q90": f.return_q90}
                           for h, f in res.forecasts.items()}).T
        st.dataframe(df, width="stretch")
        st.caption("Return range: quantile heads of the fusion seeds (10%, 50%, 90% of the horizon's simple return), shown only "
                   "where the 10-90% band covered 80% ± 8 points on 2017-2018 validation. A range, not a direction call.")
        st.bar_chart(df[["UP", "SIDEWAYS", "DOWN"]].astype(float))
        st.subheader("Decision (separate risk engine)")
        st.dataframe(pd.DataFrame({h: {"decision": p.decision, "first reason": (p.reasons or [""])[0]}
                                   for h, p in res.trade_plans.items()}).T, width="stretch")
        for lim in res.limitations:
            st.warning(lim)

with tabs[1]:
    st.header("Tri-frame chart intelligence")
    trip = st.session_state.get("trip")
    if trip is not None:
        cols = st.columns(3)
        for col, v, lab in zip(cols, ("W", "D", "H"), ("Weekly (104 bars)", "Daily (126 bars)", "Hourly (140 bars)")):
            col.image(chart_png(trip[v]), caption=f"{lab}: model input (renderer r1, binary OHLC bars)")
    elif res is not None:
        st.info("In live mode the model renders the charts from Yahoo bars; they are not cached for display.")
    if shots:
        st.subheader("Your screenshots (displayed only)")
        st.caption("Screenshots are never parsed for price levels; the model uses structured OHLCV (spec 13.4).")
        for s in shots:
            st.image(s, width="stretch")
    if res is not None:
        st.json(res.evidence.get("charts", {}))
        occ = res.evidence.get("chart_occlusion")
        if occ:
            st.subheader("Occlusion attribution of the 5D score (diagnostic, not proof of a pattern)")
            st.dataframe(pd.DataFrame({v: d["attribution"] for v, d in occ.items()}), width="stretch")

with tabs[2]:
    st.header("News intelligence")
    if res is not None:
        news = res.evidence.get("news") or []
        if not news:
            st.info("No articles supplied. In historical training data, missing news is a mask, never neutral news.")
        else:
            st.dataframe(pd.DataFrame(news), width="stretch")
            st.caption("User-supplied articles are inference-time inputs; the model was trained on the 2003-2015 corpus.")

with tabs[3]:
    st.header("SEC filing intelligence")
    if res is not None:
        sec = res.evidence.get("sec_filings") or []
        if sec:
            st.dataframe(pd.DataFrame(sec), width="stretch")
        missing = [m for m in res.confluence.get("missing_information", []) if "SEC" in m]
        for m in missing:
            st.warning(m)

with tabs[4]:
    st.header("Indicators and market context")
    if res is not None:
        st.subheader(f"Daily indicators as of {res.evidence['indicators']['asof_session']} (deterministic calculations)")
        st.json(res.evidence["indicators"])
        st.subheader("Model inputs (scale-free features)")
        st.dataframe(pd.Series(res.evidence.get("structured_features", {}), name="value"), width="stretch")

with tabs[5]:
    st.header("Explain my prediction")
    if res is not None:
        cf = res.confluence
        st.write(cf["summary"])
        c1, c2, c3 = st.columns(3)
        c1.subheader("Supporting")
        c1.write(cf["supporting_signals"] or ["none"])
        c2.subheader("Conflicting")
        c2.write(cf["conflicting_signals"] or ["none"])
        c3.subheader("Missing")
        c3.write(cf["missing_information"] or ["none"])
        diag = res.evidence.get("modality_diagnostics", {})
        if diag:
            st.subheader("Leave-one-modality-out diagnostic: change in P(UP) - P(DOWN) by horizon")
            st.dataframe(pd.DataFrame({k: v["delta_up_minus_down"] for k, v in diag["by_modality"].items()},
                                      index=diag["horizons"]), width="stretch")
            st.caption("Attributions describe this model's sensitivity, not causes.")
        st.subheader("Rule events")
        st.json(res.evidence.get("rules", {}))
        st.subheader("RAG citations (explanation only; RAG never changes a probability)")
        st.dataframe(pd.DataFrame(res.evidence.get("rag_sources", [])), width="stretch")

with tabs[6]:
    st.header("Trade planner (paper)")
    if res is not None:
        h = st.selectbox("Horizon", HZ, index=2)
        p = res.trade_plans[h]
        st.subheader(p.decision)
        st.write(p.reasons)
        if p.decision in ("LONG", "SHORT"):
            st.dataframe(pd.Series({k: v for k, v in p.model_dump().items() if v is not None and k not in ("reasons",)},
                                   dtype=object), width="stretch")
        elif p.setup_family:
            st.caption(f"A {p.setup_family} setup exists, but a gate failed (see reasons). Levels are not shown for "
                       "rejected plans to avoid implying a trade.")

with tabs[7]:
    st.header("Model diagnostics and experiments")
    r = latest_run("e03_fusion")
    if r:
        m = r["metrics"]
        st.caption(f"Run {r['id']}: {r['manifest'].get('summary', '')}")
        rows = []
        for name, v in m.items():
            if isinstance(v, dict) and "valid" in v and "test_raw" in v:
                rows.append({"model": name, "modalities": ",".join(v["modalities"]), "valid mean log loss": v["valid"]["mean_log_loss"],
                             "test mean log loss": v["test_raw"]["mean_log_loss"]})
        st.dataframe(pd.DataFrame(rows).sort_values("valid mean log loss"), width="stretch")
        if "hypotheses" in m:
            st.subheader("Registered hypotheses (positive = improvement)")
            hy = m["hypotheses"]
            st.dataframe(pd.DataFrame({k: {"estimate": v["estimate"], "ci_low": v["ci95"][0], "ci_high": v["ci95"][1],
                                           "p_holm": hy["holm"][k]["p_holm"]} for k, v in hy["tests"].items()}).T)
        if "missing_modality_stress" in m:
            st.subheader("Missing-modality stress test (test block)")
            st.dataframe(pd.DataFrame(m["missing_modality_stress"]).T)
    st.subheader("Data coverage")
    cov = ROOT / "data_manifests" / "s500_frame_summary.json"
    if cov.exists():
        st.json(json.loads(cov.read_text(encoding="utf-8")))
    log = ROOT / "docs" / "EXPERIMENT_LOG.md"
    if log.exists():
        with st.expander("Experiment log"):
            st.markdown(log.read_text(encoding="utf-8"))

with tabs[8]:
    st.header("Backtesting")
    versions = [v for v in ("v1.1", "v1") if (ROOT / "reports" / "backtest" / ("e04_backtest.json" if v == "v1" else f"e04_backtest_{v}.json")).exists()]
    ver = st.radio("Backtest version", versions, horizontal=True,
                   help="v1.1: audited engine (prior-bar targets, corrected P&L, kill-switch, >= 30-trade threshold rule) with the "
                        "deployed E08 model. v1: original engine with the E03 model, kept for comparison.") if versions else None
    bt = ROOT / "reports" / "backtest" / ("e04_backtest.json" if ver == "v1" else f"e04_backtest_{ver}.json")
    if ver and bt.exists():
        b = json.loads(bt.read_text(encoding="utf-8"))
        eq = ROOT / "reports" / "backtest" / f"e04_daily_returns_{ver}.csv"
        if eq.exists():
            d = pd.read_csv(eq, index_col=0, parse_dates=True)
            st.subheader("Equity curves, test 2020-2022 (base costs)")
            st.line_chart((1 + d).cumprod())
        st.caption("Entry at the next open; gap-through stops fill at the open; stop-first on ambiguous bars; "
                   "costs 10/15/20 bps round trip plus 0.5%/yr short borrow. Benchmark: equal-weight survivors.")
        st.dataframe(pd.DataFrame(b["test_final"]).T[["trades", "sharpe", "sharpe_ci95_month_blocks", "cagr", "max_drawdown",
                                                       "win_rate", "profit_factor", "expectancy_R", "psr_vs_0"]]
                     if b["test_final"] else pd.DataFrame(), width="stretch")
        st.json({"frozen_thresholds": b["frozen_thresholds"], "pbo": b["pbo"], "benchmark_test": b["benchmark_ew_survivors"]["test"]})
    else:
        st.info("Backtest not run yet (scripts/experiments/e04_backtest.py).")

with tabs[9]:
    st.header("RAG knowledge base")
    man = paths.ARTIFACTS / "rag" / "manifest.json"
    if man.exists():
        st.json(json.loads(man.read_text(encoding="utf-8")))
    ev = ROOT / "reports" / "rag" / "rag_eval.json"
    if ev.exists():
        e = json.loads(ev.read_text(encoding="utf-8"))
        st.subheader("Retrieval evaluation")
        st.dataframe(pd.DataFrame(e["book"]).T, width="stretch")
        st.dataframe(pd.DataFrame(e["rule"]).T, width="stretch")
    if kb is not None:
        q = st.text_input("Search the knowledge base")
        meth = st.selectbox("Method", ["bm25", "hybrid_rrf", "dense", "rule_lookup", "hybrid_rerank"])
        if q:
            r = kb.search(meth, q, k=8)
            r["snippet"] = r.text.str.slice(0, 200)
            st.dataframe(r[["doc_id", "page", "rule_id", "section", "score", "snippet"]], width="stretch")
            st.caption("Snippets are limited to 200 characters; the books are private research material.")

with tabs[10]:
    st.header("Experiment configuration")
    st.caption("View and export a validated configuration. Heavy training runs in Colab or a separate process, never here.")
    cfgs = sorted((ROOT / "configs").glob("*.yaml"))
    sel = st.selectbox("Config file", [c.name for c in cfgs]) if cfgs else None
    if sel:
        import yaml
        txt = st.text_area("YAML", (ROOT / "configs" / sel).read_text(encoding="utf-8"), height=300)
        try:
            yaml.safe_load(txt)
            st.success("Valid YAML")
            st.download_button("Export", txt, file_name=sel)
        except yaml.YAMLError as e:
            st.error(f"Invalid YAML: {e}")

with tabs[11]:
    st.header("Audit: curves, diagnostics and evaluation figures")
    st.caption("Generated from saved artifacts by scripts/report/audit_report.py; findings and SWOT in docs/CHIEF_AUDIT_REPORT.md.")
    figs = sorted((ROOT / "reports" / "figures").glob("*.png"))
    if not figs:
        st.info("Run scripts/report/audit_report.py to build the figures.")
    for f in figs:
        st.subheader(f.stem)
        st.image(str(f), width="stretch")

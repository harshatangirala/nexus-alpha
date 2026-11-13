"""Trade Desk tab: trade type + 3 screenshots + 1-10 texts -> model forecast, risk plan, market/macro context, LLM read.

Rendered by app/streamlit_app.py. What is validated and what is not is stated on the page itself:
* model probabilities and return ranges: the frozen model (v1.1 until v2 is frozen), with its skill gate and backtest;
* market/macro panel: descriptive context (percentiles vs 2003-2022); v1.1 does not use these inputs;
* Gemini read: live LLM analysis, unvalidated, shown beside the model and logged for forward scoring;
* screenshots: checked against real bars through their legend values; the model never reads screenshot pixels.
"""
from __future__ import annotations

import datetime as dt

import pandas as pd
import streamlit as st

import skin as SK

from nexus_alpha import text_time as TX
from nexus_alpha import trade_desk as TD
from nexus_alpha.llm import gemini as GM
from nexus_alpha.trading.plan import RiskConfig

KINDS = ["company news", "macro release / commentary", "SEC filing (8-K, 10-K, 10-Q)", "other"]


@st.cache_resource(show_spinner="Loading the v2 bundle...")
def _v2():
    from nexus_alpha.inference.pipeline_v2 import NexusPredictorV2, latest_v2_bundle
    b = latest_v2_bundle()
    return NexusPredictorV2(b) if b else None


def _v2_frozen(bundle_id: str) -> bool:
    import json
    from nexus_alpha import shadow as SH
    p = SH.FROZEN_DIR / "v2.json"
    return p.exists() and json.loads(p.read_text(encoding="utf-8")).get("bundle_id") == bundle_id


def _v2_section(tick, mode, asof, trade, swing, lt, company_news, res, risk):
    """v2 forecast for the exact requested horizon, with the same deterministic risk engine."""
    from nexus_alpha.targets_v2 import HORIZONS_V2
    from nexus_alpha.trading import skill_evidence as SE
    from nexus_alpha.trading.plan import plan_trade
    try:
        v2 = _v2()
    except Exception as e:
        st.info(f"v2 bundle could not be loaded: {type(e).__name__}: {e}")
        return
    if v2 is None:
        st.caption("v2 (all swing lengths, 3M/6M/1Y) appears here once its bundle is built (E14).")
        return
    SK.section("2b. v2 model (exact horizon)")
    status = "FROZEN" if _v2_frozen(v2.meta["bundle_id"]) else "NOT YET FROZEN (research build; not in the forward test)"
    choice = TD.choose_horizon(trade, swing_days=swing, long_term=lt, available=HORIZONS_V2)
    with st.spinner("Scoring v2..."):
        try:
            r = v2.predict(tick, asof=str(asof) if mode == "historical" else None, mode=mode, articles=company_news)
        except Exception as e:
            st.warning(f"v2 failed: {type(e).__name__}: {e}")
            return
    h = choice.served_by
    f = r["forecasts"][h]
    c = st.columns(4)
    c[0].metric(f"v2 {h}: lean", "UP" if f["p_up"] > f["p_down"] else "DOWN", f"{f['p_up'] - f['p_down']:+.3f} (up - down)")
    c[1].metric("P(up / sideways / down)", f"{f['p_up']:.2f} / {f['p_side']:.2f} / {f['p_down']:.2f}")
    td = v2.meta.get("test_diagnostic", {}).get(h, {})
    # A-31: 2017-18 holds fewer than 4 independent 6M windows and about one 1Y window, so no skill claim is possible there
    rep_v2 = SE.replicated(SE.model_of_bundle(r.get("bundle_id")), h)       # None: never checked out of sample
    skill_txt = ("not validatable (too few independent windows in 2017-18)" if h in ("6M", "1Y")
                 else "NOT validated" if not f["direction_skill_validated"]
                 else "validated on 2017-18 (not checked out of sample)" if rep_v2 is None
                 else "validated and replicated" if rep_v2 else "validated on 2017-18, NOT replicated on 2020-22")
    c[2].metric("Direction skill", skill_txt,
                f"2020-22 AUROC {td['auroc_ud']:.2f}" if td.get("auroc_ud") is not None else None, delta_color="off")
    c[3].metric("Return range q10..q90", "-" if f["ret_q10"] is None else f"{f['ret_q10']:+.1%} .. {f['ret_q90']:+.1%}")
    st.caption(f"Bundle {r['bundle_id']} ({status}); as-of {r['asof_session']}; share of model inputs missing "
               f"{r['features_missing_share']:.0%}. On 2020-2022 (a reused diagnostic) v2's probabilities beat the base "
               "rate at 1D-20D but its up-vs-down ranking was close to chance (AUROC 0.42-0.52): it estimates how much a "
               "stock is likely to move better than which way.")
    ev = res.evidence.get("rules", {})
    ind = res.evidence.get("indicators", {})
    if ind.get("atr14") and ev:
        # D-057: gate 3 also requires the direction skill to have replicated on 2020-2022 (it did not for v2)
        gate_open, gate_reason = SE.direction_gate(SE.model_of_bundle(r.get("bundle_id")), h,
                                                   bool(f["direction_skill_validated"]))
        tp = plan_trade(h, HORIZONS_V2[h], {"up": f["p_up"], "down": f["p_down"], "side": f["p_side"]}, calibrated=True,
                        skill_validated=gate_open, levels_available=True,
                        weekly={"regime": ev.get("weekly_regime"), "permission": ev.get("weekly_permission")},
                        setups=ev.get("daily_setups", []), trigger_by_side=ev.get("hourly_triggers", {}),
                        atr=float(ind["atr14"]), close=float(ind["close"]), risk=risk, data_gate_reasons=[],
                        skill_reason=gate_reason)
        st.write(f"Risk engine on v2 probabilities: **{tp.decision}**")
        st.json({k: v for k, v in tp.model_dump().items() if v not in (None, [], {})})
    with st.expander("All v2 horizons"):
        st.dataframe(pd.DataFrame(r["forecasts"]).T, width="stretch")


def _e19_note() -> str:
    """One line on whether market/macro information has validated skill for the market's direction (E19)."""
    import json
    from pathlib import Path
    runs = sorted(Path(__file__).resolve().parents[1].joinpath("experiments").glob("*_e19_market_model_*"), reverse=True)
    for r in runs:
        try:
            man = json.loads((r / "manifest.json").read_text(encoding="utf-8"))
            if man.get("status") != "COMPLETED":
                continue
            m = json.loads((r / "metrics.json").read_text(encoding="utf-8"))
        except Exception:
            continue
        kept = [k for k, v in m["keep_rule_decisions"].items() if v and m["placebo_shows_no_gain"].get(k.split("|")[0])]
        if kept:
            return f"E19 (market-direction model): validated gain for {', '.join(kept)} on 2017-2018 (see the report)."
        return ("E19 (market-direction model): these market and macro readings showed no validated skill for predicting "
                "SPY's direction on 2017-2018, so they are context, not a signal.")
    return "E19 (market-direction model) has not been run yet: these readings are context, not a validated signal."


@st.cache_resource(show_spinner=False)
def _market_model():
    from nexus_alpha import shadow as SH
    from nexus_alpha.macro import market_model as MM
    try:
        bdir, _ = SH.verify_frozen("market_e19")
        return MM.MarketModel(bdir)
    except Exception:
        return None


@st.cache_data(show_spinner="Loading development market features...")
def _dev_market_rows(cols: tuple):
    from nexus_alpha import paths
    fr = pd.read_parquet(paths.ARTIFACTS / "samples" / "v2_frame.parquet", columns=["asof_date", *cols])
    return fr.groupby("asof_date").first()


def _latest_logged_read():
    from nexus_alpha import shadow as SH
    files = sorted((SH.SHADOW_ROOT / "market_e19" / "forecasts").glob("*.csv"))
    return pd.read_csv(files[-1]) if files else None


def _market_read_section(mode, asof):
    """E19 SPY 20D direction lean (D-031): the only market/macro read with validated direction skill."""
    mm = _market_model()
    if mm is None:
        return
    ev = mm.meta["evidence"]["20D"]
    st.markdown("**SPY 20-session direction lean (E19 market model, frozen `market_e19`)**")
    if mode == "live":
        r = _latest_logged_read()
        if r is None or r.empty:
            st.caption("No logged read yet (the daily task logs one every evening).")
            return
        row = r.iloc[0]
        p_up, p_down, p_side, when = float(row.p_up), float(row.p_down), float(row.p_side), row.asof_session
        src = f"logged in the forward test for session {when} ({row.data_quality})"
    else:
        cols = tuple(mm.models["20D"]["cols"])
        dev = _dev_market_rows(cols)
        d = dev.loc[:pd.Timestamp(asof)]
        if d.empty:
            st.caption("No development market features at this date.")
            return
        when = d.index[-1]
        p = mm.predict(d.iloc[-1])["20D"]
        p_up, p_down, p_side = p["p_up"], p["p_down"], p["p_side"]
        insample = when <= pd.Timestamp("2018-12-31")
        src = f"session {when.date()}; " + ("IN-SAMPLE (the deployed refit used 2003-2018)" if insample
                                            else "out of sample for the 2003-2018 refit")
    c = st.columns(3)
    c[0].metric("SPY 20D lean", "UP" if p_up > p_down else "DOWN", f"{p_up - p_down:+.3f} (up - down)")
    c[1].metric("P(up / sideways / down)", f"{p_up:.2f} / {p_side:.2f} / {p_down:.2f}")
    c[2].metric("Direction AUROC (2017-18 | 2020-22)", f"{ev['valid_auroc_ud']:.2f} | {ev['test_auroc_ud']:.2f}")
    st.caption(f"{src}. Validation interval {ev['valid_auroc_ci95'][0]:.2f}-{ev['valid_auroc_ci95'][1]:.2f}; "
               f"2020-2022 interval {ev['test_auroc_ci95'][0]:.2f}-{ev['test_auroc_ci95'][1]:.2f}. Its probabilities "
               f"were no better than the base rate in 2020-2022 (log loss {ev['test_ll']:.3f} vs {ev['test_ll_prior']:.3f}), "
               "so read it as a lean, not as calibrated odds. Market and macro inputs only; it does not use this stock.")


@st.cache_data(ttl=3600, show_spinner="Fetching live market context...")
def _live_ctx():
    from nexus_alpha.macro.context import live_date_features
    return live_date_features()


@st.cache_data(show_spinner=False)
def _dev_ctx():
    from nexus_alpha.macro.context import dev_date_features
    try:
        return dev_date_features()
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def _gmd(asof: str):
    from nexus_alpha.macro import gmd as G
    try:
        return G.asof_features(pd.DatetimeIndex([asof])).iloc[0]
    except Exception:
        return None


# feature suffix -> (FRED series short name used for the data-age column, label)
FRED_LABELS = {
    "cpi_yoy": ("cpi", "CPI inflation, % y/y"), "core_cpi_yoy": ("core_cpi", "Core CPI, % y/y"),
    "core_pce_yoy": ("core_pce", "Core PCE, % y/y"),
    "payrolls_chg3m_avg": ("payrolls", "Payrolls, avg monthly change over 3 months (thousands)"),
    "unrate_level": ("unrate", "Unemployment rate, %"),
    "unrate_sahm": ("unrate", "Sahm-rule gap, pp (>= 0.5 has marked US recessions)"),
    "claims_log4w_vs_52w": ("claims", "Jobless claims, 4-week vs 52-week average (log)"),
    "indpro_yoy": ("indpro", "Industrial production, % y/y"), "retail_yoy": ("retail", "Retail sales, % y/y"),
    "housing_starts_yoy": ("housing_starts", "Housing starts, % y/y"),
    "umich_sentiment_level": ("umich_sentiment", "UMich consumer sentiment"),
    "real_gdp_qoq_saar": ("real_gdp", "Real GDP, % q/q annualised"), "m2_yoy": ("m2", "M2 money stock, % y/y"),
    "nfci_level": ("nfci", "Chicago Fed NFCI (> 0 = tighter than average)"),
    "fedfunds_level": ("fedfunds", "Fed funds rate, %")}


@st.cache_data(ttl=6 * 3600, show_spinner="Fetching FRED releases...")
def _fred(asof: str, live: bool):
    """US macro releases known at the as-of date: exact ALFRED vintages for historical dates, the latest release live."""
    from nexus_alpha.macro import fred as F
    try:
        if live:
            if not F.have_key():
                return None
            s = F.live_features(pd.Timestamp(asof))
        else:
            s = F.asof_features(pd.DatetimeIndex([asof])).iloc[0]
    except Exception:
        return None
    rows = []
    for k, (series, lab) in FRED_LABELS.items():
        v, age = s.get(f"fred_{k}"), s.get(f"fred_{series}_obs_age_days")
        rows.append({"release": lab, "value": None if v is None or pd.isna(v) else round(float(v), 3),
                     "data age (days)": None if age is None or pd.isna(age) else int(age)})
    return pd.DataFrame(rows).set_index("release")


def _bars_for_check(ticker: str, mode: str, asof):
    """Daily, weekly and hourly bars to check screenshot legends against."""
    from nexus_alpha.cv.render import rth_hourly
    from nexus_alpha.data import bars as B
    from nexus_alpha.data.calendar import calendar
    if mode == "historical":
        d = B.read_daily(ticker)
        d = d[d.date <= pd.Timestamp(asof)]
        early = set(calendar().date[calendar().early_close.astype(bool)])
        h = rth_hourly(B.read_hourly(ticker), early)
    else:
        from nexus_alpha.inference.pipeline import fetch_live
        d, h = fetch_live(ticker)
    w = B.weekly_from_daily(d).rename(columns={"week_start": "date"})
    h = h.rename(columns={"timestamp": "date"})
    return {"W": w, "D": d, "H": h}


def _headline_fetcher(ticker: str) -> None:
    """Recent headlines (Yahoo Finance RSS) and, if SEC_USER_AGENT is set, 8-K filings; ticked items join the texts."""
    from nexus_alpha import trade_tools as TT
    with st.expander("Fetch recent headlines and filings for the selected ticker (adds them as texts)"):
        c = st.columns(2)
        if c[0].button(f"Fetch Yahoo Finance headlines ({ticker})", key="desk_fetch_news"):
            try:
                items = [x | {"kind": "company news", "ticker": ticker} for x in TT.yahoo_headlines(ticker, 10)]
            except Exception as e:
                items = []
                st.warning(f"Headline feed unavailable: {type(e).__name__}: {e}")
            st.session_state["desk_fetched"] = items
        if c[1].button(f"Fetch SEC 8-K filings ({ticker})", key="desk_fetch_sec"):
            try:
                items = [x | {"kind": "SEC filing (8-K, 10-K, 10-Q)", "ticker": ticker} for x in TT.sec_filings(ticker)]
                st.session_state["desk_fetched"] = st.session_state.get("desk_fetched", []) + items
            except Exception as e:
                st.info(str(e))
        for i, x in enumerate(st.session_state.get("desk_fetched", [])):
            st.checkbox(f"[{x['published_utc']} UTC] {x['headline'][:140]} ({x['source']})", key=f"desk_pick_{i}")
        st.caption("Only items published before the decision time count for the model; Gemini sees every ticked item.")


@st.cache_data(ttl=3600, show_spinner="Computing market-regime rules...")
def _regime_rows(mode: str, asof: str):
    from nexus_alpha.rules.market_regime import regime_features
    if mode == "historical":
        from nexus_alpha.data import etf_bars as EB
        spy, qqq = EB.read_daily("SPY"), EB.read_daily("QQQ")
    else:
        import yfinance as yf

        def yd(sym):
            d = yf.Ticker(sym).history(period="2y", auto_adjust=False, actions=False)
            di = pd.DatetimeIndex(d.index).tz_localize(None).normalize()
            return pd.DataFrame({"date": di, "open": d.Open.values, "high": d.High.values, "low": d.Low.values,
                                 "close": d.Close.values, "volume": d.Volume.values}).dropna()
        spy, qqq = yd("SPY"), yd("QQQ")
        spy, qqq = spy.tail(400), qqq.tail(400)
    R = regime_features(spy, qqq)
    if mode == "historical":
        R = R.loc[:pd.Timestamp(asof)]
    return R.iloc[-1], str(R.index[-1].date())


def _e20_note() -> str:
    import json
    from pathlib import Path
    for r in sorted(Path(__file__).resolve().parents[1].joinpath("experiments").glob("*_e20_regime_rules_*"), reverse=True):
        try:
            m = json.loads((r / "metrics.json").read_text(encoding="utf-8"))
        except Exception:
            continue
        add = any(v for k, v in m["keep_rule_decisions"].items() if "M2R_vs_M2" in k)
        return ("E20: these rules " + ("ADD" if add else "do NOT add") + " validated skill to the SPY market read; on "
                "their own they showed a small 3-month gain on 2017-18 that did not replicate in 2020-22. Descriptive "
                "context, not a signal.")
    return "E20 has not been run: descriptive context only."


def _regime_section(mode, asof) -> None:
    st.markdown("**Market-regime rules (O'Neil/IBD, from tradermonty/claude-trading-skills; E20)**")
    try:
        r, day = _regime_rows(mode, str(pd.Timestamp(asof).date()) if mode == "historical" else str(pd.Timestamp.now().date()))
    except Exception as e:
        st.info(f"Regime rules unavailable: {type(e).__name__}: {e}")
        return
    risk = {0: "NORMAL", 1: "CAUTION", 2: "HIGH", 3: "SEVERE"}
    state = next((c.replace("rr_ftd_", "") for c in r.index
                  if c.startswith("rr_ftd_") and c != "rr_ftd_dual" and r[c] == 1), "?")
    c = st.columns(4)
    c[0].metric("SPY distribution days (25 sessions)", int(r.rr_spy_dd_d25), risk[int(r.rr_spy_dd_risk)], delta_color="off")
    c[1].metric("QQQ distribution days (25 sessions)", int(r.rr_qqq_dd_d25), risk[int(r.rr_qqq_dd_risk)], delta_color="off")
    c[2].metric("Follow-through state", state.replace("_", " ").capitalize().replace("Ftd", "FTD").replace("ftd", "FTD"))
    c[3].metric("Power trend (SPY / QQQ)",
                f"{'yes' if r.rr_spy_power_trend else 'no'} / {'yes' if r.rr_qqq_power_trend else 'no'}")
    st.caption(f"Session {day}. " + _e20_note())


JOURNAL = "journal/paper_trades.csv"


def _discipline_section(plan, ticker: str) -> None:
    """Pre-trade discipline gate + drawdown circuit breaker + sizing (tools adapted from MIT skills; D-034)."""
    from nexus_alpha import paths, trade_tools as TT
    SK.section("7. Trade discipline (paper trading; nothing is ever sent to a broker)")
    jp = paths.ARTIFACTS / JOURNAL
    journal = pd.read_csv(jp) if jp.exists() else pd.DataFrame(columns=["exit_utc", "ticker", "pnl"])
    acct = st.number_input("Paper account size ($)", 1_000.0, 1e9, 100_000.0, step=1_000.0, key="desk_acct")
    br = TT.circuit_breaker(journal, acct)
    st.write(f"Drawdown circuit breaker: **{br['recommendation']}** " + "; ".join(br.get("reasons", [])))
    planned = float(plan.planned_loss_cash) if plan is not None and plan.planned_loss_cash else None
    c = st.columns(3)
    a1 = c[0].checkbox("Entry is in my written plan", key="desk_g1")
    a2 = c[1].checkbox("Stop is predefined", value=bool(plan is not None and plan.stop), key="desk_g2")
    a3 = c[2].checkbox("Size is within plan", key="desk_g3")
    c = st.columns(2)
    pr = c[0].number_input("Planned risk ($)", 0.0, 1e9, planned or 0.0, key="desk_pr")
    ar = c[1].number_input("Actual risk of the order ($)", 0.0, 1e9, planned or 0.0, key="desk_ar")
    pnl_col = pd.to_numeric(journal["pnl"], errors="coerce") if "pnl" in journal else pd.Series(dtype=float)
    losses = journal[pnl_col < 0]
    last_loss_h = None
    if len(losses):
        last_loss_h = (pd.Timestamp.now(tz="UTC") - pd.to_datetime(losses.exit_utc, utc=True).max()).total_seconds() / 3600
    g = TT.discipline_gate(TT.Checklist(a1, a2, a3, pr, ar), br, recent_loss_hours=last_loss_h)
    st.write(f"Pre-trade gate: **{g['decision']}**")
    for x in g["blocked_by"] + g["review"]:
        st.caption("- " + x)
    if plan is not None and plan.entry_reference and plan.stop:
        ps = TT.position_size(pr or 0.0, plan.entry_reference, plan.stop)
        st.caption(f"Fixed-risk size at ${pr:,.0f}: {ps['shares']} shares "
                   f"(risk per share {ps.get('risk_per_share', 0):.2f}).")
    with st.expander("Record a closed paper trade (feeds the circuit breaker)"):
        c = st.columns(3)
        pnl = c[0].number_input("Realised P&L ($)", -1e9, 1e9, 0.0, key="desk_pnl")
        when = c[1].text_input("Exit time (UTC, e.g. 2026-11-06 20:55)", key="desk_exit")
        if c[2].button("Record", key="desk_rec"):
            jp.parent.mkdir(parents=True, exist_ok=True)
            row = pd.DataFrame([{"exit_utc": when or pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M"),
                                 "ticker": ticker, "pnl": pnl}])
            pd.concat([journal, row], ignore_index=True).to_csv(jp, index=False)
            st.success("Recorded (local, git-ignored journal).")


@st.cache_data(ttl=6 * 3600, show_spinner="Loading fundamentals (SEC XBRL)...")
def _fund_data(ticker: str, mode: str, asof: str):
    from nexus_alpha.fundamentals import service as FS
    try:
        return FS.historical(ticker, asof) if mode == "historical" else FS.live(ticker)
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def _e22_note() -> str:
    import json
    from pathlib import Path
    for r in sorted(Path(__file__).resolve().parents[1].joinpath("experiments").glob("*_e22_fundamentals_*"), reverse=True):
        try:
            m = json.loads((r / "metrics.json").read_text(encoding="utf-8"))
        except Exception:
            continue
        return ("E22: fundamentals " + ("ADD validated skill to the stock model" if m.get("kept") else
                "did NOT add validated skill to the stock model") + " on 2017-18 (see docs/V3_RESEARCH.md). "
                "Shown here as analysis, not as a model input.")
    return "E22 has not been run yet: fundamentals are analysis here, not a model input."


def _fundamentals_section(tick, mode, asof) -> None:
    import numpy as np
    from nexus_alpha.fundamentals import service as FS
    SK.section("4c. Fundamental analysis (SEC XBRL, point in time)")
    d = _fund_data(tick, mode, str(pd.Timestamp(asof).date()) if mode == "historical" else str(pd.Timestamp.now().date()))
    if not d or d.get("error"):
        st.info(f"Fundamentals unavailable for {tick}: {(d or {}).get('error', 'no XBRL facts')}")
        return
    s, r = d["snapshot"], d["ratios"]
    g = lambda k: float(s.get(k)) if s.get(k) is not None and pd.notna(s.get(k)) else np.nan
    debt = g("lt_debt_total") if np.isfinite(g("lt_debt_total")) else np.nansum([g("lt_debt_noncurrent"), g("lt_debt_current")])
    debt += 0 if not np.isfinite(g("st_borrowings")) else g("st_borrowings")
    gp = g("ttm_gross_profit") if np.isfinite(g("ttm_gross_profit")) else g("ttm_revenue") - g("ttm_cogs")
    gp_p = g("ttm_gross_profit_prev") if np.isfinite(g("ttm_gross_profit_prev")) else g("ttm_revenue_prev") - g("ttm_cogs_prev")
    rows = [("Income", "Revenue", g("ttm_revenue"), g("ttm_revenue_prev")), ("Income", "Gross profit", gp, gp_p),
            ("Income", "Operating income (EBIT)", g("ttm_operating_income"), g("ttm_operating_income_prev")),
            ("Income", "EBITDA", g("ttm_operating_income") + g("ttm_da"), np.nan),
            ("Income", "Net income", g("ttm_net_income"), g("ttm_net_income_prev")),
            ("Income", "Diluted EPS", g("ttm_eps_diluted"), g("ttm_eps_diluted_prev")),
            ("Balance", "Total assets", g("assets"), g("assets_prev")), ("Balance", "Current assets", g("assets_current"), g("assets_current_prev")),
            ("Balance", "Total liabilities", g("liabilities"), g("liabilities_prev")),
            ("Balance", "Current liabilities", g("liabilities_current"), g("liabilities_current_prev")),
            ("Balance", "Shareholders' equity", g("equity"), g("equity_prev")), ("Balance", "Cash", g("cash"), g("cash_prev")),
            ("Balance", "Total debt", debt, np.nan),
            ("Cash flow", "Operating cash flow", g("ttm_cfo"), g("ttm_cfo_prev")), ("Cash flow", "Capital expenditure", g("ttm_capex"), g("ttm_capex_prev")),
            ("Cash flow", "Free cash flow", g("ttm_cfo") - np.nan_to_num(g("ttm_capex")), g("ttm_cfo_prev") - np.nan_to_num(g("ttm_capex_prev"))),
            ("Cash flow", "Dividends paid", g("ttm_dividends"), g("ttm_dividends_prev")), ("Cash flow", "Buybacks", g("ttm_buybacks"), g("ttm_buybacks_prev"))]
    t = pd.DataFrame(rows, columns=["statement", "item", "latest (TTM / latest)", "a year earlier"])
    t["change"] = (t["latest (TTM / latest)"] / t["a year earlier"] - 1).where(t["a year earlier"].abs() > 0)
    st.caption(f"{d['source']}. Statements as known at the decision time; flows are trailing twelve months; latest period "
               f"end {pd.Timestamp(s.get('period_end')).date() if pd.notna(s.get('period_end')) else '?'}.")
    st.dataframe(t.style.format({"latest (TTM / latest)": lambda v: "-" if pd.isna(v) else (f"{v:,.2f}" if abs(v) < 1000 else f"{v / 1e6:,.0f} M"),
                                 "a year earlier": lambda v: "-" if pd.isna(v) else (f"{v:,.2f}" if abs(v) < 1000 else f"{v / 1e6:,.0f} M"),
                                 "change": lambda v: "-" if pd.isna(v) else f"{v:+.1%}"}), width="stretch", hide_index=True)
    if r is None:
        st.info("No price-based ratios for this date.")
        return
    if pd.notna(r.get("foreign_filer")) and r.get("foreign_filer") == 1:
        st.info("Foreign private issuer (20-F / 40-F): statements are annual and often in another currency, and the "
                "cover-page count is ordinary shares rather than ADS, so ratios are not computed (audit 3).")
    elif pd.notna(r.get("shares_basis_ok")) and r.get("shares_basis_ok") == 0:
        st.info("Share basis not reliable for this listing (another share class on the same filer, or the cover-page "
                "count disagrees with diluted shares): price-based ratios are not computed (audit 3).")
    c = st.columns(6)
    for col, (lab, k) in zip(c, (("P/E", "pe"), ("EV/EBITDA", "ev_ebitda"), ("P/S", "ps"), ("P/B", "pb"), ("P/FCF", "p_fcf"),
                                 ("Market cap ($bn)", "market_cap"))):
        v = r.get(k)
        col.metric(lab, "-" if v is None or pd.isna(v) else (f"{v / 1e9:,.1f}" if k == "market_cap" else f"{v:,.1f}"))
    groups = {"Valuation": ["earnings_yield", "sales_yield", "book_to_price", "ebitda_to_ev", "fcf_yield", "dividend_yield",
                            "shareholder_yield", "dcf_to_price"],
              "Profitability": ["gross_margin", "operating_margin", "net_margin", "ebitda_margin", "roe", "roa", "roic"],
              "Growth": ["revenue_growth", "operating_income_growth", "eps_growth", "asset_growth", "share_issuance"],
              "Leverage and liquidity": ["debt_to_equity", "net_debt_to_ebitda", "interest_coverage", "current_ratio", "cash_to_assets"],
              "Efficiency": ["asset_turnover", "receivables_to_sales"],
              "Quality and events": ["accruals", "piotroski_f", "altman_z", "sue"]}
    pct_like = {"earnings_yield", "sales_yield", "ebitda_to_ev", "fcf_yield", "dividend_yield", "shareholder_yield", "gross_margin",
                "operating_margin", "net_margin", "ebitda_margin", "roe", "roa", "roic", "revenue_growth", "operating_income_growth",
                "eps_growth", "asset_growth", "share_issuance", "cash_to_assets", "accruals", "receivables_to_sales"}
    rr = []
    for grp, ks in groups.items():
        for k in ks:
            v = r.get(k)
            p_ = r.get(f"fr_{k}")
            rr.append({"group": grp, "ratio": k.replace("_", " "),
                       "value": "-" if v is None or pd.isna(v) else (f"{v:+.1%}" if k in pct_like else f"{v:,.2f}"),
                       "universe percentile": "-" if p_ is None or pd.isna(p_) else f"{100 * p_:.0f}",
                       "industry percentile": "-" if r.get(f"fi_{k}") is None or pd.isna(r.get(f"fi_{k}"))
                       else f"{100 * r.get(f'fi_{k}'):.0f}"})
    with st.expander("All 31 ratios (the stock's percentile in the universe and within its SIC industry that day)",
                     expanded=False):
        st.dataframe(pd.DataFrame(rr), width="stretch", hide_index=True)
    pf, az = r.get("piotroski_f"), r.get("altman_z")
    zone = "-" if az is None or pd.isna(az) else ("safe (> 2.99)" if az > 2.99 else "grey (1.81-2.99)" if az >= 1.81 else "distress (< 1.81)")
    c = st.columns(3)
    c[0].metric("Piotroski F-score (0-9)", "-" if pf is None or pd.isna(pf) else f"{int(pf)}", "8-9 strong, 0-2 weak", delta_color="off")
    c[1].metric("Altman Z", "-" if az is None or pd.isna(az) else f"{az:.2f}", zone, delta_color="off")
    c[2].metric("Days since last 10-Q/10-K", "-" if pd.isna(r.get("days_since_filing")) else int(r.get("days_since_filing")))
    with st.expander("Discounted cash flow (adjust the assumptions)"):
        fcf = g("ttm_cfo") - np.nan_to_num(g("ttm_capex"))
        rg = float(r.get("revenue_growth")) if pd.notna(r.get("revenue_growth")) else 0.0
        cc = st.columns(4)
        gr = cc[0].slider("FCF growth, years 1-5", -0.10, 0.30, float(np.clip(rg, -0.05, 0.15)), 0.01, key="dcf_g")
        rr_ = cc[1].slider("Discount rate", 0.05, 0.15, 0.09, 0.005, key="dcf_r")
        gt = cc[2].slider("Terminal growth", 0.0, 0.04, 0.025, 0.005, key="dcf_gt")
        yrs = cc[3].slider("Explicit years", 3, 10, 5, key="dcf_n")
        price = float(r.get("price_as_traded")) if pd.notna(r.get("price_as_traded")) else np.nan
        shares = float(r.get("market_cap")) / price if np.isfinite(price) and price > 0 else np.nan
        v = FS.dcf_per_share(fcf, gr, rr_, gt, yrs, debt, g("cash"), shares)
        if np.isfinite(v):
            st.write(f"Intrinsic value per share **{v:,.2f}** vs price **{price:,.2f}**: margin of safety **{v / price - 1:+.0%}**.")
        else:
            st.write("DCF not defined (missing or negative free cash flow, or discount rate <= terminal growth).")
        st.caption("A DCF is only as good as its assumptions; it is shown as analysis and is not validated as a signal. "
                   "The model's fixed-assumption version is the dcf_to_price ratio above.")
    st.caption(_e22_note())


@st.cache_resource(show_spinner="Loading Kronos-small (about 25M parameters, CPU)...")
def _kronos_model():
    from nexus_alpha.kronos import service as KS
    return KS.model()


@st.cache_data(ttl=6 * 3600, show_spinner="Sampling 32 Kronos paths...")
def _kronos_run(ticker: str, mode: str, asof: str):
    from nexus_alpha.kronos import service as KS
    try:
        if mode == "historical":
            from nexus_alpha.data import bars as B
            d = B.read_daily(ticker)
            d = d[d.date <= pd.Timestamp(asof)]
        else:
            from nexus_alpha.inference.pipeline import fetch_live
            d = fetch_live(ticker)[0]
        _kronos_model()
        return KS.forecast(ticker, d)
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def _kronos_status() -> str:
    import json
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    runs = sorted(root.joinpath("experiments").glob("*_e24_kronos_zero_shot_*"))
    if not runs:
        return "E24 (the contaminated 2017-18 check) has not been scored yet."
    m = json.loads((runs[0] / "metrics.json").read_text(encoding="utf-8"))
    frozen = root.joinpath("registry", "frozen_models", "kronos_zs.json").exists()
    return ("E24: the necessary condition was " + ("met" if m.get("necessary_condition_met") else "NOT met") +
            " on 2017-18 (inside Kronos's training data, so this is not evidence of skill). " +
            ("Kronos is in the nightly forward test as kronos_zs; it is shown as a signal only if it passes the "
             "pre-registered forward criterion." if frozen else "Kronos is not in the forward test."))


def _kronos_section(tick, mode, asof) -> None:
    import numpy as np
    SK.section("4d. Kronos foundation-model paths (zero-shot, analysis only)")
    st.caption("Kronos (MIT, AAAI 2026) samples future daily candles from the last 400 bars. Its pre-training data "
               "includes US daily stock bars from 2000 to June 2024, so before July 2024 any output can reflect memory "
               "of what happened. It is never an input of the NEXUS models. " + _kronos_status())
    key = f"kronos_{tick}_{mode}_{asof}"
    if not st.button("Run Kronos for this stock", key="kronos_go"):
        if key not in st.session_state:
            return
    k = st.session_state.get(key) or _kronos_run(tick, mode, str(pd.Timestamp(asof).date()) if mode == "historical"
                                                 else str(pd.Timestamp.now().date()))
    st.session_state[key] = k
    if not k or k.get("error"):
        st.info(f"Kronos unavailable for {tick}: {(k or {}).get('error', 'fewer than 400 daily bars')}")
        return
    if k["in_pretraining"]:
        st.warning(f"As-of {k['asof'].date()} is inside Kronos's pre-training period (to June 2024): treat this as "
                   "an in-sample illustration, not a forecast.")
    import plotly.graph_objects as go
    closes = k["paths"][:, :, 3]
    q10, q50, q90 = (np.quantile(closes, q, axis=0) for q in (0.1, 0.5, 0.9))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=k["history"].date, y=k["history"].close, name="close", line=dict(color="#334155")))
    for j in range(min(8, closes.shape[0])):
        fig.add_trace(go.Scatter(x=k["dates"], y=closes[j], line=dict(width=0.6, color="rgba(37,99,235,0.25)"),
                                 showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=k["dates"], y=q90, line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=k["dates"], y=q10, fill="tonexty", fillcolor="rgba(37,99,235,0.15)", line=dict(width=0),
                             name="10-90% of 32 paths"))
    fig.add_trace(go.Scatter(x=k["dates"], y=q50, name="median path", line=dict(color="#2563eb")))
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h"))
    st.plotly_chart(fig, width="stretch")
    rows = [{"horizon": h, "P(down)": f"{o['p'][0]:.0%}", "P(sideways)": f"{o['p'][1]:.0%}", "P(up)": f"{o['p'][2]:.0%}",
             "mean return": f"{o['ret_mean']:+.2%}", "10%-90% return": f"{o['ret_q10']:+.1%} .. {o['ret_q90']:+.1%}"}
            for h, o in k["probs"].items()]
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    st.caption("Classes by the NEXUS label rule on each path (entry at the next open; sideways within 0.5 sigma sqrt(h), "
               "at least 10 bp). 32 paths give probabilities in steps of about 3%; they are smoothed by 0.5 per class.")


def render(predictor, tickers: list[str], risk_defaults: dict) -> None:
    SK.banner("Trade Desk", "Pick a trade, get a plan", "Real bars, your texts and screenshots, a deterministic risk engine. Paper only.")
    with SK.fold("About this view", "info"):
        st.caption("Pick the trade, add up to three chart screenshots (weekly, daily, hourly) and 1-10 texts. The model uses "
                   "real bars for the confirmed ticker; screenshots are checked against those bars, not read as inputs.")
    if predictor is None:
        st.error("No inference bundle loaded.")
        return
    _headline_fetcher(st.session_state.get("desk_tick", "AAPL"))
    n_txt = st.number_input("How many texts (news, macro, filings)?", 0, 10, 1, key="desk_ntxt")
    with st.form("desk"):
        c = st.columns(3)
        tick = c[0].selectbox("Ticker", tickers, index=tickers.index("AAPL") if "AAPL" in tickers else 0, key="desk_tick")
        mode = c[1].radio("Data", ["live", "historical"], horizontal=True, key="desk_mode")
        asof = c[2].date_input("As-of session (historical only)", value=pd.Timestamp("2021-06-15"),
                               min_value=pd.Timestamp("2004-01-02"), max_value=pd.Timestamp("2022-12-29"), key="desk_asof")
        c = st.columns(3)
        trade = c[0].radio("Trade type", ["Intraday", "Swing", "Long-term"], horizontal=True, key="desk_trade")
        swing = c[1].slider("Swing: trading days in the trade", 2, 20, 5, key="desk_swing")
        lt = c[2].select_slider("Long-term horizon", options=list(TD.LONG_TERM), value="4 weeks", key="desk_lt")
        SK.section("Chart screenshots")
        slots = {}
        cs = st.columns(3)
        for col, (k, lab) in zip(cs, (("W", "Weekly (1W)"), ("D", "Daily (1D)"), ("H", "Hourly (1H)"))):
            slots[k] = col.file_uploader(lab, type=["png", "jpg", "jpeg"], key=f"desk_shot_{k}")
        manual = {}
        with st.expander("Legend values typed by you (used when Gemini is not configured)"):
            for k in ("W", "D", "H"):
                cc = st.columns(4)
                manual[k] = {f: cc[i].number_input(f"{k} {f}", value=0.0, format="%.4f", key=f"desk_{k}_{f}")
                             for i, f in enumerate(("open", "high", "low", "close"))}
        SK.section("Texts")
        texts = []
        for i in range(int(n_txt)):
            with st.expander(f"Text {i + 1}", expanded=i == 0):
                kind = st.selectbox("Type", KINDS, key=f"desk_kind{i}")
                head = st.text_input("Headline / title", key=f"desk_head{i}")
                body = st.text_area("Text", key=f"desk_body{i}", height=100)
                cc = st.columns([1.1, 0.9, 0.7, 1.3])
                day = cc[0].date_input("Published date", value=dt.date.today(), key=f"desk_day{i}")
                clock = cc[1].time_input("time", value=dt.time(9, 0), key=f"desk_clock{i}", step=60)
                known = cc[2].checkbox("time known", value=True, key=f"desk_known{i}")
                tz = cc[3].selectbox("time zone", list(TX.TIMEZONES), key=f"desk_tz{i}")
                src = st.text_input("Source", key=f"desk_src{i}")
                if head or body:
                    when = TX.to_utc(day, clock if known else None, tz)
                    texts.append({"kind": kind, "headline": head, "text": body, "published_utc": when.isoformat(),
                                  "time_known": known, "source": src})
        use_llm = st.checkbox("Ask Gemini for a read of the texts and screenshots", value=GM.available(),
                              disabled=not GM.available(),
                              help=None if GM.available() else "Set GEMINI_API_KEY in your environment to enable.")
        go = st.form_submit_button("Analyse trade", type="primary")
    if not go:
        return

    picked = [x for i, x in enumerate(st.session_state.get("desk_fetched", []))
              if st.session_state.get(f"desk_pick_{i}") and x.get("ticker") == tick]
    texts = (texts + [{"kind": x["kind"], "headline": x["headline"], "text": x["text"], "time_known": True,
                       "published_utc": x["published_utc"], "source": x["source"]} for x in picked])[:10]
    choice = TD.choose_horizon(trade, swing_days=swing, long_term=lt)
    SK.section("1. What the model is asked")
    st.write(f"**{choice.label}**. Model horizon: **{choice.served_by or 'none'}** - {choice.note}.")
    risk = RiskConfig(**risk_defaults)
    company_news = [t for t in texts if t["kind"] == "company news"]
    with st.spinner("Scoring the frozen model..."):
        res = predictor.predict(tick, asof=str(asof) if mode == "historical" else None, mode=mode,
                                articles=company_news, risk=risk, explain=False)
    h = choice.served_by
    bars_all = None
    SK.section("2. Model forecast")
    if h is None:
        st.warning("No frozen model covers this horizon yet. Showing all available horizons for context only.")
    SK.prob_rings({k: (f.down_probability, f.sideways_probability, f.up_probability)
                   for k, f in res.forecasts.items()})
    SK.pills([("DOWN", "#FF6B6B"), ("SIDEWAYS", SK.SIDE), ("UP", "#3DDC97")] + ([(f"asked: {h}", "#3FD0E0")] if h else []))
    rows = []
    for k, f in res.forecasts.items():
        rows.append({"horizon": k, "P(up)": f.up_probability, "P(sideways)": f.sideways_probability,
                     "P(down)": f.down_probability, "median return": f.expected_return, "q10": f.return_q10,
                     "q90": f.return_q90, "direction skill validated": f.direction_skill_validated,
                     "replicated 2020-22": f.direction_skill_replicated, "selected": "<--" if k == h else ""})
    st.dataframe(pd.DataFrame(rows).style.format({c: "{:.3f}" for c in ("P(up)", "P(sideways)", "P(down)")}
                                                 | {c: "{:+.2%}" for c in ("median return", "q10", "q90")}, na_rep="-"),
                 width="stretch")
    st.caption(f"As-of session {res.instrument.asof_session}; reference close {res.instrument.reference_price:.2f} "
               f"({res.instrument.reference_price_source}); data quality: {res.instrument.data_quality}. "
               "Only company-news texts enter the model's news branch; macro and filing texts are analysed below.")
    if h is not None:
        f = res.forecasts[h]
        p = res.trade_plans[h]
        lean = f.up_probability - f.down_probability
        skill = ("NOT validated" if not f.direction_skill_validated
                 else "validated (not checked out of sample)" if f.direction_skill_replicated is None
                 else "validated + replicated" if f.direction_skill_replicated else "validated, NOT replicated")
        SK.tiles([{"label": "Lean (up - down)", "value": "UP" if lean > 0 else "DOWN", "sub": f"{lean:+.3f}",
                   "color": "#3DDC97" if lean > 0 else "#FF6B6B"},
                  {"label": "Direction skill here", "value": skill.split(" (")[0], "sub": skill,
                   "color": "#3DDC97" if skill == "validated + replicated" else "#F2B544" if f.direction_skill_validated else "#FF6B6B"},
                  {"label": "Risk engine", "value": p.decision, "color": {"LONG": "#3DDC97", "SHORT": "#FF6B6B"}.get(p.decision, "#8098AE")},
                  {"label": "Return range q10..q90",
                   "value": "-" if f.return_q10 is None else f"{f.return_q10:+.1%} .. {f.return_q90:+.1%}", "color": "#9B8CFF"}])
        SK.section("3. Trade plan (deterministic risk engine): levels and the reasons for them")
        import plan_view as PV
        try:
            bars_all = _bars_for_check(tick, mode, asof)
        except Exception as e:  # noqa: BLE001  (the chart is optional; the plan below is the record)
            st.info(f"Price chart unavailable: {type(e).__name__}: {e}")
        PV.explainer(res, h, risk, bars_all["D"] if bars_all else None, key="desk")
        st.caption("Shaded box: the model's 10-90% return range for this horizon, where its coverage was validated. "
                   "Solid levels: an approved plan. Dashed levels marked CONDITIONAL: the rule setup's own geometry "
                   "when the engine said NO_TRADE, with the gates it failed.")
        with st.expander("Full plan record", expanded=p.decision in ("LONG", "SHORT")):
            st.json({k: v for k, v in p.model_dump().items() if v not in (None, [], {})})
    for lim in res.limitations:
        st.caption(lim)

    _v2_section(tick, mode, asof, trade, swing, lt, company_news, res, risk)

    SK.section("4. Market and macro context (descriptive; not inputs of the frozen v1.1 model)")
    try:
        dev = _dev_ctx()
        if mode == "live":
            latest = _live_ctx().dropna(how="all").iloc[-1]
        else:
            latest = dev.loc[:pd.Timestamp(asof)].iloc[-1] if dev is not None else None
        if latest is not None:
            st.dataframe(TD.context_table(latest, dev if dev is not None else pd.DataFrame(), TD.CONTEXT_LABELS),
                         width="stretch")
    except Exception as e:
        st.info(f"Market context unavailable: {type(e).__name__}: {e}")
    g = _gmd(str(pd.Timestamp(asof).date()) if mode == "historical" else str(pd.Timestamp.now().date()))
    if g is not None:
        st.caption(f"Global Macro Database (2026_09), latest year usable at this date: {int(g['gmd_data_year'])}")
        st.dataframe(g.drop("gmd_data_year").rename(lambda s: s.replace("gmd_", "")).to_frame("value").T,
                     width="stretch")
    fz = _fred(str(pd.Timestamp(asof).date()) if mode == "historical" else str(pd.Timestamp.now().date()), mode == "live")
    if fz is not None:
        st.caption("US releases from FRED/ALFRED as known at this date (historical: exact real-time vintages; live: the "
                   "latest release). Data age = days since the observation period the latest value refers to.")
        st.dataframe(fz, width="stretch")
    st.caption(_e19_note())
    _market_read_section(mode, asof)
    _regime_section(mode, asof)
    _fundamentals_section(tick, mode, asof)
    _kronos_section(tick, mode, asof)

    SK.section("5. Screenshots checked against real bars")
    images = [(k, f.getvalue(), f.type or "image/png") for k, f in slots.items() if f is not None]
    llm = None
    if use_llm and GM.available() and (texts or images):
        with st.spinner("Asking Gemini..."):
            try:
                llm = GM.analyze(tick, choice.label, res.instrument.forecast_timestamp, texts, images)
            except Exception as e:
                st.warning(f"Gemini call failed: {type(e).__name__}: {e}")
    if images:
        bars = bars_all if bars_all is not None else _bars_for_check(tick, mode, asof)
        cols = st.columns(len(images))
        for col, (k, data, _) in zip(cols, images):
            col.image(data, caption=f"{k} screenshot", width="stretch")
            legend = manual.get(k, {})
            src = "typed"
            if llm:
                r = next((c for c in llm.get("charts", []) if c.get("slot") == k), None)
                if r and any(r.get(f"legend_{x}") for x in ("open", "high", "low", "close")):
                    legend = {x: r.get(f"legend_{x}") for x in ("open", "high", "low", "close")}
                    src = "read by Gemini"
            legend = {x: v for x, v in legend.items() if v}
            chk = TD.check_legend(bars[k], legend, tolerance_pct=0.5 if k == "H" else 0.3)
            col.write(f"Legend ({src}): {legend or 'none'}")
            col.write(chk)
    else:
        st.info("No screenshots supplied.")

    SK.section("6. Gemini read (live LLM, unvalidated, logged for forward scoring)")
    if GM.available():
        st.caption(f"Gemini free tier: 20 requests per model per day (one request per analysis); used today: "
                   f"{GM.usage_today() or 'none'}")
    if llm:
        o = llm.get("overall", {})
        st.write(f"Model used: `{llm.get('_model')}`. Overall lean: **{o.get('direction_lean')}** "
                 f"(confidence {o.get('confidence')}). {o.get('summary', '')}")
        if h is not None:
            mlean = "bullish" if res.forecasts[h].up_probability > res.forecasts[h].down_probability else "bearish"
            st.write("Agreement with the model's lean: " + ("agree" if o.get("direction_lean") == mlean else
                                                           f"**differs** (model {mlean})"))
        st.dataframe(pd.DataFrame(llm.get("texts", [])), width="stretch")
    elif not GM.available():
        st.info("Gemini is not configured (GEMINI_API_KEY not set). The model, risk plan and context above do not need it.")
    else:
        st.info("No texts or screenshots to analyse.")
    news = res.evidence.get("news") or []
    if news:
        st.caption("Company-news texts as used by the model (availability check and Loughran-McDonald tone):")
        st.dataframe(pd.DataFrame(news), width="stretch")
    _discipline_section(res.trade_plans.get(h) if h is not None else None, tick)

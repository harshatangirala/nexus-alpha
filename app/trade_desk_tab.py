"""Trade Desk tab: trade type + 3 screenshots + 1-10 texts -> model forecast, risk plan, market/macro context, LLM read.

Rendered by app/streamlit_app.py. What is validated and what is not is stated on the page itself:
* model probabilities and return ranges: the frozen model (v1.1 until v2 is frozen), with its skill gate and backtest;
* market/macro panel: descriptive context (percentiles vs 2003-2022); v1.1 does not use these inputs;
* Gemini read: live LLM analysis, unvalidated, shown beside the model and logged for forward scoring;
* screenshots: checked against real bars through their legend values; the model never reads screenshot pixels.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

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
    from nexus_alpha.trading.plan import plan_trade
    try:
        v2 = _v2()
    except Exception as e:
        st.info(f"v2 bundle could not be loaded: {type(e).__name__}: {e}")
        return
    if v2 is None:
        st.caption("v2 (all swing lengths, 3M/6M/1Y) appears here once its bundle is built (E14).")
        return
    st.subheader("2b. v2 model (exact horizon)")
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
    c[2].metric("Direction skill (validation)", "validated" if f["direction_skill_validated"] else "NOT validated")
    c[3].metric("Return range q10..q90", "-" if f["ret_q10"] is None else f"{f['ret_q10']:+.1%} .. {f['ret_q90']:+.1%}")
    st.caption(f"Bundle {r['bundle_id']} ({status}); as-of {r['asof_session']}; share of model inputs missing "
               f"{r['features_missing_share']:.0%}.")
    ev = res.evidence.get("rules", {})
    ind = res.evidence.get("indicators", {})
    if ind.get("atr14") and ev:
        tp = plan_trade(h, HORIZONS_V2[h], {"up": f["p_up"], "down": f["p_down"], "side": f["p_side"]}, calibrated=True,
                        skill_validated=f["direction_skill_validated"], levels_available=True,
                        weekly={"regime": ev.get("weekly_regime"), "permission": ev.get("weekly_permission")},
                        setups=ev.get("daily_setups", []), trigger_by_side=ev.get("hourly_triggers", {}),
                        atr=float(ind["atr14"]), close=float(ind["close"]), risk=risk, data_gate_reasons=[])
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


def render(predictor, tickers: list[str], risk_defaults: dict) -> None:
    st.header("Trade Desk")
    st.caption("Pick the trade, add up to three chart screenshots (weekly, daily, hourly) and 1-10 texts. The model uses "
               "real bars for the confirmed ticker; screenshots are checked against those bars, not read as inputs.")
    if predictor is None:
        st.error("No inference bundle loaded.")
        return
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
        st.subheader("Chart screenshots")
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
        st.subheader("Texts")
        texts = []
        for i in range(int(n_txt)):
            with st.expander(f"Text {i + 1}", expanded=i == 0):
                kind = st.selectbox("Type", KINDS, key=f"desk_kind{i}")
                head = st.text_input("Headline / title", key=f"desk_head{i}")
                body = st.text_area("Text", key=f"desk_body{i}", height=100)
                when = st.text_input("Published (UTC, e.g. 2026-11-05 21:30)", key=f"desk_when{i}")
                src = st.text_input("Source", key=f"desk_src{i}")
                if head or body:
                    texts.append({"kind": kind, "headline": head, "text": body, "published_utc": when, "source": src})
        use_llm = st.checkbox("Ask Gemini for a read of the texts and screenshots", value=GM.available(),
                              disabled=not GM.available(),
                              help=None if GM.available() else "Set GEMINI_API_KEY in your environment to enable.")
        go = st.form_submit_button("Analyse trade", type="primary")
    if not go:
        return

    choice = TD.choose_horizon(trade, swing_days=swing, long_term=lt)
    st.subheader("1. What the model is asked")
    st.write(f"**{choice.label}**. Model horizon: **{choice.served_by or 'none'}** - {choice.note}.")
    risk = RiskConfig(**risk_defaults)
    company_news = [t for t in texts if t["kind"] == "company news"]
    with st.spinner("Scoring the frozen model..."):
        res = predictor.predict(tick, asof=str(asof) if mode == "historical" else None, mode=mode,
                                articles=company_news, risk=risk, explain=False)
    h = choice.served_by
    st.subheader("2. Model forecast")
    if h is None:
        st.warning("No frozen model covers this horizon yet. Showing all available horizons for context only.")
    rows = []
    for k, f in res.forecasts.items():
        rows.append({"horizon": k, "P(up)": f.up_probability, "P(sideways)": f.sideways_probability,
                     "P(down)": f.down_probability, "median return": f.expected_return, "q10": f.return_q10,
                     "q90": f.return_q90, "direction skill validated": f.direction_skill_validated,
                     "selected": "<--" if k == h else ""})
    st.dataframe(pd.DataFrame(rows).style.format({c: "{:.3f}" for c in ("P(up)", "P(sideways)", "P(down)")}
                                                 | {c: "{:+.2%}" for c in ("median return", "q10", "q90")}, na_rep="-"),
                 width="stretch")
    st.caption(f"As-of session {res.instrument.asof_session}; reference close {res.instrument.reference_price:.2f} "
               f"({res.instrument.reference_price_source}); data quality: {res.instrument.data_quality}. "
               "Only company-news texts enter the model's news branch; macro and filing texts are analysed below.")
    if h is not None:
        f = res.forecasts[h]
        p = res.trade_plans[h]
        c = st.columns(4)
        c[0].metric("Lean", "UP" if f.up_probability > f.down_probability else "DOWN",
                    f"{(f.up_probability - f.down_probability):+.3f} (up - down)")
        c[1].metric("Direction skill at this horizon", "validated" if f.direction_skill_validated else "NOT validated")
        c[2].metric("Risk engine decision", p.decision)
        c[3].metric("Return range q10..q90", "-" if f.return_q10 is None else f"{f.return_q10:+.1%} .. {f.return_q90:+.1%}")
        st.subheader("3. Trade plan (deterministic risk engine)")
        st.json({k: v for k, v in p.model_dump().items() if v not in (None, [], {})})
    for lim in res.limitations:
        st.caption(lim)

    _v2_section(tick, mode, asof, trade, swing, lt, company_news, res, risk)

    st.subheader("4. Market and macro context (descriptive; not inputs of the frozen v1.1 model)")
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

    st.subheader("5. Screenshots checked against real bars")
    images = [(k, f.getvalue(), f.type or "image/png") for k, f in slots.items() if f is not None]
    llm = None
    if use_llm and GM.available() and (texts or images):
        with st.spinner("Asking Gemini..."):
            try:
                llm = GM.analyze(tick, choice.label, res.instrument.forecast_timestamp, texts, images)
            except Exception as e:
                st.warning(f"Gemini call failed: {type(e).__name__}: {e}")
    if images:
        bars = _bars_for_check(tick, mode, asof)
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

    st.subheader("6. Gemini read (live LLM, unvalidated, logged for forward scoring)")
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

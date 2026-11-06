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

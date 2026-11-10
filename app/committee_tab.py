"""Tab M: the LLM committee (TradingAgents v0.7.1; D-050, E27). Live, on demand, logged; never changes a forecast."""
from __future__ import annotations

import json

import pandas as pd
import streamlit as st

import ui as UI
from nexus_alpha import shadow as SH
from nexus_alpha.llm import committee as C

ORDER = [("market_report", "Market analyst"), ("fundamentals_report", "Fundamentals analyst"),
         ("news_report", "News analyst"), ("sentiment_report", "Sentiment analyst"),
         ("investment_debate_state.bull_history", "Bull researcher"), ("investment_debate_state.bear_history", "Bear researcher"),
         ("investment_debate_state.judge_decision", "Research manager"), ("trader_investment_plan", "Trader"),
         ("risk_debate_state.judge_decision", "Risk team (portfolio manager's review)"),
         ("final_trade_decision", "Portfolio manager: final decision")]
COLOUR = {"Buy": UI.GOOD, "Overweight": UI.GOOD, "Hold": UI.MUTED, "Underweight": UI.BAD, "Sell": UI.BAD}


@st.cache_data(ttl=3600, show_spinner=False)
def _model() -> str:
    return C.model()


def _v11_lean(ticker: str, asof: str) -> float | None:
    f = SH.SHADOW_ROOT / "v1.1" / "forecasts" / f"{asof}.csv"
    if not f.exists():
        return None
    v = pd.read_csv(f)
    v = v[(v.ticker == ticker) & (v.horizon == "5D")]
    return float(v.p_up.iloc[0] - v.p_down.iloc[0]) if len(v) else None


def render(tickers: list[str]) -> None:
    st.header("LLM committee (TradingAgents)")
    st.caption("A multi-agent language-model committee: analysts, a bull/bear debate, a trader, a risk team and a "
               "portfolio manager, ending in a five-tier rating. Live and on demand only, on your Gemini key. It never "
               "changes a forecast or the risk engine, and it is never scored on history: its language model has read "
               "about those years (D-020). Every run is logged publicly and scored forward (E27).")
    ok, why = C.available()
    if not ok:
        st.info(why)
        return
    try:
        m = _model()           # lists the key's models (network); the tab must still open offline
    except Exception as e:  # noqa: BLE001
        st.info(f"The Gemini model list is unavailable ({type(e).__name__}); the committee cannot run now.")
        return
    c = st.columns([2, 3, 1.2])
    tick = c[0].selectbox("Ticker", tickers, index=tickers.index("AAPL") if "AAPL" in tickers else 0, key="cm_tick")
    analysts = c[1].multiselect("Analysts", list(C.ANALYSTS), default=["market"], key="cm_analysts",
                                help="Each analyst adds about 4 model calls.")
    left, need = C.quota_left(m), C.estimate_calls(len(analysts))
    c[2].metric("Gemini calls left today", left)
    st.caption(f"Model `{m}`. This run needs about {need} calls; the free tier allows {C.DAILY_LIMIT} a day per model, "
               "shared with the Trade Desk's Gemini read (reset at midnight Pacific, 13:30 IST).")
    asof = SH.expected_asof()
    late = SH.now_utc() > SH.decision_utc_for(asof)
    if late:
        st.warning(f"The decision time for session {asof.date()} (09:00 New York) has passed: a run now is logged as "
                   "late and never scored. On-time runs happen between the US close and the next 09:00 New York.")
    if st.button("Run the committee", type="primary", disabled=need > left or not analysts, key="cm_go"):
        with st.spinner("The committee is working (usually a few minutes)..."):
            try:
                st.session_state["cm_res"] = C.run(tick, analysts)
            except Exception as e:  # noqa: BLE001
                st.error(str(e))
    res = st.session_state.get("cm_res")
    if res:
        rating = res["rating"]
        lean = _v11_lean(res["ticker"], res["asof_session"])
        body = (f'<div class="nx-kv"><span>Rating</span><span>{UI.tag(rating.upper(), COLOUR.get(rating, UI.WARN))}</span></div>'
                f'<div class="nx-kv"><span>Stock · as-of session</span><span>{res["ticker"]} · {res["asof_session"]}</span></div>'
                f'<div class="nx-kv"><span>Model calls · time</span><span>{res["llm_calls"]} · {res["seconds"]:.0f} s</span></div>'
                f'<div class="nx-kv"><span>Logged</span><span>{"late (never scored)" if res["late"] else "on time"}'
                f' · chain {"intact" if res.get("chain_ok") else "CHECK"}</span></div>')
        if lean is not None and rating in C.DIRECTION and rating != "Hold":
            agree = (C.DIRECTION[rating] == 2) == (lean > 0)
            body += (f'<div class="nx-kv"><span>v1.1 5D lean (p_up - p_down)</span><span>{lean:+.3f} · '
                     f'{"agrees" if agree else "differs"}</span></div>')
        UI.frame("Committee decision", f"{res['ticker']}: {rating}", body,
                 "Unvalidated. The frozen models and the risk engine above are unchanged by this rating.")
        for key, label in ORDER:
            text = res["reports"].get(key)
            if text:
                with st.expander(label, expanded=key == "final_trade_decision"):
                    st.markdown(text)
    st.subheader("Logged runs (public record, E27)")
    h = C.history()
    if len(h):
        st.dataframe(h[["run_id", "ticker", "asof_session", "rating", "late", "llm_calls", "analysts"]],
                     hide_index=True, width="stretch")
    else:
        st.caption("No runs logged yet.")
    summ = SH.SHADOW_ROOT / "committee" / "summary.json"
    if summ.exists():
        s = json.loads(summ.read_text(encoding="utf-8"))
        scored = "; ".join(f"{k}: {v.get('scored', 0)} scored" for k, v in s.get("results", {}).items())
        st.caption(f"Forward scoring: {s.get('runs_logged', 0)} runs logged" + (f"; {scored}" if scored else "")
                   + ". No claim before 100 scored runs (E27).")

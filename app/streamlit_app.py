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

st.set_page_config(page_title="NEXUS Alpha", layout="wide", page_icon=":material/radar:")
import ui as UI  # noqa: E402  (app/ is on sys.path when Streamlit runs this script)

UI.inject_css()
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
st.sidebar.markdown('<div style="font-family:Oxanium,sans-serif;font-size:1.5rem;font-weight:700;letter-spacing:.12em" '
                    'class="nx-shim">NEXUS ALPHA</div>', unsafe_allow_html=True)
st.sidebar.caption("Research and paper-trading decision support. Not investment advice. No orders are ever placed.")
try:
    predictor, kb = load_predictor()
    bundle_ok = True
except Exception as e:
    bundle_ok, predictor, kb = False, None, None
    st.sidebar.error(f"No inference bundle: {e}")
mode = st.sidebar.segmented_control("Data mode", ["historical", "live"], default="historical", help="historical: sealed bars to 2022-12-30; live: Yahoo Finance bars")
mode = mode or "historical"
tick = st.sidebar.selectbox("Ticker (s500 universe)", universe(), index=universe().index("AAPL") if "AAPL" in universe() else 0)
asof = None
if mode == "historical":
    asof = st.sidebar.date_input("As-of session", value=pd.Timestamp("2021-06-15"), min_value=pd.Timestamp("2004-01-02"),
                                 max_value=pd.Timestamp("2022-12-29"))
with st.sidebar.expander(":material/shield: Risk settings (paper)"):
    capital = st.number_input("Account capital", 1000.0, 1e9, 100_000.0, step=1000.0)
    risk_pct = st.slider("Risk per trade (%)", 0.1, 2.0, 0.5, 0.1) / 100
    max_pos = st.slider("Max position (% of capital)", 1, 50, 10) / 100
    allow_short = st.checkbox("Short selling allowed", True)
    cost_sc = st.selectbox("Cost scenario", ["base", "stressed", "severe"])
    occl = st.checkbox("Chart occlusion attribution (slower)", False)

st.sidebar.subheader("News articles (1-7, optional)")
import datetime as _dt  # noqa: E402
from nexus_alpha import text_time as _TX  # noqa: E402
n_art = st.sidebar.number_input("Number of articles", 0, 7, 0)
articles = []
for i in range(int(n_art)):
    with st.sidebar.expander(f"Article {i + 1}"):
        h = st.text_input("Headline", key=f"h{i}")
        t = st.text_area("Text", key=f"t{i}", height=80)
        day = st.date_input("Published date", key=f"wd{i}",
                            value=_dt.date(2021, 6, 14) if mode == "historical" else _dt.date.today())
        known = st.checkbox("time known", value=True, key=f"wk{i}")
        clock = st.time_input("time", value=_dt.time(16, 30), key=f"wt{i}", step=60, disabled=not known)
        tz = st.selectbox("time zone", list(_TX.TIMEZONES), key=f"wz{i}")
        src = st.text_input("Source", key=f"s{i}")
        if (h or t) and day is not None:
            articles.append({"headline": h, "text": t, "published_utc": _TX.to_utc(day, clock if known else None, tz)
                             .isoformat(), "time_known": known, "source": src})
shots = st.sidebar.file_uploader("Chart screenshots (weekly, daily, hourly; shown only)", type=["png", "jpg"],
                                 accept_multiple_files=True)
go = st.sidebar.button("Run analysis", type="primary", icon=":material/play_arrow:", width="stretch", disabled=not bundle_ok)

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
all_tabs = st.tabs([":material/radar: Mission Control", ":material/candlestick_chart: Trade Desk", ":material/compare_arrows: Compare 1-5",
                    ":material/dashboard: Desk 1-5", ":material/query_stats: A Forecast", ":material/show_chart: B Charts",
                    ":material/newspaper: C News", ":material/description: D SEC", ":material/speed: E Indicators",
                    ":material/lightbulb: F Explain", ":material/gavel: G Trade planner", ":material/analytics: H Model diagnostics",
                    ":material/history: I Backtesting", ":material/menu_book: J Knowledge base", ":material/tune: K Configuration",
                    ":material/fact_check: L Audit", ":material/groups: M LLM committee", ":material/ssid_chart: N Options",
                    ":material/neurology: O Vision + NLP",
                    ":material/account_balance: P Fundamentals"])
RISK_DEFAULTS = dict(account_capital=capital, risk_pct_per_trade=risk_pct, max_position_pct=max_pos,
                     allow_short=allow_short, cost_scenario=cost_sc)
with all_tabs[16]:
    import committee_tab  # noqa: E402
    committee_tab.render(universe())
with all_tabs[17]:
    import options_tab  # noqa: E402
    options_tab.render(universe())
with all_tabs[18]:
    import vision_nlp_tab  # noqa: E402
    vision_nlp_tab.render(universe())
with all_tabs[19]:
    import fundamentals_tab  # noqa: E402
    fundamentals_tab.render(universe())
with all_tabs[0]:
    import mission_control  # noqa: E402
    mission_control.render()
with all_tabs[1]:
    import trade_desk_tab  # noqa: E402  (app/ is on sys.path when Streamlit runs this script)
    trade_desk_tab.render(predictor if bundle_ok else None, universe(), RISK_DEFAULTS)
with all_tabs[2]:
    import compare_tab  # noqa: E402
    compare_tab.render(universe(), predictor if bundle_ok else None, RISK_DEFAULTS)
with all_tabs[3]:
    import desk_multi_tab  # noqa: E402
    desk_multi_tab.render(predictor if bundle_ok else None, universe(), RISK_DEFAULTS)
tabs = all_tabs[4:]
import plotly.graph_objects as pgo   # noqa: E402  (not "go": that name is the sidebar's Run analysis button)

import skin as SK  # noqa: E402


def _need_run() -> None:
    SK.cards([{"head": "Press Run analysis", "color": UI.ACCENT,
               "body": "Pick a ticker in the sidebar and press the cyan button. Every panel here fills in from that one run."}])


def _num(v) -> float:
    return float("nan") if v is None else float(v)


def _ok(*xs) -> bool:
    return all(x == x for x in xs)


def _skill_pill(h: str, f) -> tuple[str, str]:
    if not f.direction_skill_validated:
        return f"{h} · direction skill not validated", UI.BAD
    if f.direction_skill_replicated is None:
        return f"{h} · validated, not checked out of sample", UI.WARN
    return (f"{h} · validated + replicated", UI.GOOD) if f.direction_skill_replicated else (
        f"{h} · validated, NOT replicated", UI.WARN)


with tabs[0]:
    SK.banner("A · Forecast", "What the frozen model expects",
              "Probabilities from 1 day to 1 month. A range of outcomes, not a promise.")
    if res is None:
        _need_run()
    else:
        i = res.instrument
        SK.tiles([{"label": "Symbol", "value": i.symbol, "sub": f"session {i.asof_session}"},
                  {"label": "Reference close", "value": f"{i.reference_price:,.2f}", "color": UI.ACCENT2,
                   "sub": str(i.reference_price_source)},
                  {"label": "Data quality", "value": str(i.data_quality),
                   "color": UI.GOOD if str(i.data_quality).lower() in ("ok", "good", "full", "clean") else UI.WARN},
                  {"label": "Calibration", "value": str(res.model.calibration_status), "color": UI.WARN,
                   "sub": f"trained through {res.model.trained_through}"}])
        SK.section("Probability by horizon")
        SK.prob_rings({h: (f.down_probability, f.sideways_probability, f.up_probability)
                       for h, f in res.forecasts.items()})
        SK.pills([("DOWN", UI.BAD), ("SIDEWAYS", SK.SIDE), ("UP", UI.GOOD)])
        SK.section("Expected return and its 10-90% range")
        hz = list(res.forecasts)
        med = [_num(res.forecasts[h].expected_return) for h in hz]
        q10 = [_num(res.forecasts[h].return_q10) for h in hz]
        q90 = [_num(res.forecasts[h].return_q90) for h in hz]
        both = [k for k in range(len(hz)) if _ok(q10[k], q90[k])]
        fig = pgo.Figure()
        if len(both) >= 2:
            xs = [hz[k] for k in both]
            fig.add_trace(pgo.Scatter(x=xs + xs[::-1], y=[q90[k] for k in both] + [q10[k] for k in both][::-1],
                                      fill="toself", fillcolor="rgba(63,208,224,.14)", line=dict(width=0),
                                      hoverinfo="skip", name="10-90% range"))
        fig.add_trace(pgo.Scatter(x=hz, y=med, mode="lines+markers", name="median", line=dict(color=UI.ACCENT, width=2.5),
                                  marker=dict(size=9, color=UI.ACCENT), hovertemplate="%{x} median %{y:+.2%}<extra></extra>"))
        fig.add_hline(y=0, line=dict(color=UI.MUTED, dash="dot", width=1))
        fig.update_yaxes(tickformat=".0%")
        UI.plot(fig, height=300)
        SK.section("How sure is the model, and is its direction call trustworthy?")
        SK.gauge_rings([{"caption": h, "frac": min(1.0, f.uncertainty), "center": f"{f.uncertainty:.2f}",
                         "sub": "uncertainty", "color": UI.WARN if f.uncertainty > 0.95 else UI.ACCENT}
                        for h, f in res.forecasts.items()])
        SK.pills([_skill_pill(h, f) for h, f in res.forecasts.items()])
        SK.section("Risk-engine decision by horizon")
        dcol = {"LONG": UI.GOOD, "SHORT": UI.BAD}
        SK.pills([(f"{h} · {p.decision}", dcol.get(p.decision, UI.MUTED)) for h, p in res.trade_plans.items()])
        with SK.fold("All numbers by horizon", "table_rows"):
            df = pd.DataFrame({h: {"UP": f.up_probability, "SIDEWAYS": f.sideways_probability, "DOWN": f.down_probability,
                                   "predicted": f.predicted_direction, "uncertainty (normalised entropy)": f.uncertainty,
                                   "direction skill validated": f.direction_skill_validated,
                                   "replicated 2020-22": f.direction_skill_replicated,
                                   "median return": f.expected_return, "return q10": f.return_q10,
                                   "return q90": f.return_q90} for h, f in res.forecasts.items()}).T
            st.dataframe(df, width="stretch")
            st.caption("Return range: quantile heads of the fusion seeds (10%, 50%, 90% of the horizon's simple return), shown "
                       "only where the 10-90% band covered 80% ± 8 points on 2017-2018 validation. A range, not a direction call.")
        with SK.fold("Stacked probability bars", "stacked_bar_chart"):
            pr = df[["DOWN", "SIDEWAYS", "UP"]].astype(float)
            fig = pgo.Figure([pgo.Bar(y=pr.index, x=pr[c], name=c, orientation="h", marker_color=col,
                                      hovertemplate=f"{c} %{{x:.1%}}<extra></extra>")
                              for c, col in (("DOWN", UI.BAD), ("SIDEWAYS", SK.SIDE), ("UP", UI.GOOD))])
            fig.update_layout(barmode="stack", xaxis=dict(tickformat=".0%", range=[0, 1]), yaxis=dict(autorange="reversed"))
            UI.plot(fig, height=260)
        with SK.fold("Decision table (separate risk engine)", "gavel"):
            st.dataframe(pd.DataFrame({h: {"decision": p.decision, "first reason": (p.reasons or [""])[0]}
                                       for h, p in res.trade_plans.items()}).T, width="stretch")
        with SK.fold("Model, bundle and limitations", "info"):
            st.caption(f"Decision timestamp {i.forecast_timestamp} (09:00 New York, next session). Bundle {res.model.bundle_id}; "
                       f"calibration {res.model.calibration_status}; trained through {res.model.trained_through}.")
            for lim in res.limitations:
                st.warning(lim)

with tabs[1]:
    SK.banner("B · Charts", "What the network actually sees",
              "Three binary OHLC views of the stock: weekly, daily and hourly.")
    trip = st.session_state.get("trip")
    if trip is not None:
        cols = st.columns(3)
        for col, v, lab in zip(cols, ("W", "D", "H"), ("Weekly (104 bars)", "Daily (126 bars)", "Hourly (140 bars)")):
            col.image(chart_png(trip[v]), caption=f"{lab}: model input (renderer r1, binary OHLC bars)")
    elif res is not None:
        st.info("In live mode the model renders the charts from Yahoo bars; they are not cached for display.")
    else:
        _need_run()
    if shots:
        SK.section("Your screenshots (displayed only)")
        st.caption("Screenshots are never parsed for price levels; the model uses structured OHLCV (spec 13.4).")
        for s in shots:
            st.image(s, width="stretch")
    if res is not None:
        nb = res.evidence.get("charts", {}).get("n_valid_bars", {})
        SK.tiles([{"label": f"{n} bars used", "value": str(nb.get(k, "-")), "color": c}
                  for k, n, c in (("W", "Weekly", UI.ACCENT), ("D", "Daily", UI.ACCENT2), ("H", "Hourly", UI.WARN))])
        occ = res.evidence.get("chart_occlusion")
        if occ:
            SK.section("Which stretch of each chart moved the 5D score? (occlusion, a diagnostic)")
            order = [v for v in ("W", "D", "H") if v in occ]
            z = [occ[v]["attribution"] for v in order]
            names = {"W": "Weekly", "D": "Daily", "H": "Hourly"}
            n = len(z[0])
            fig = pgo.Figure(pgo.Heatmap(z=z, x=[f"{k + 1}" + (" oldest" if k == 0 else " newest" if k == n - 1 else "")
                                                for k in range(n)], y=[names[v] for v in order], zmid=0,
                                         colorscale=[[0, UI.BAD], [0.5, "#0F1D2C"], [1, UI.GOOD]],
                                         hovertemplate="%{y} strip %{x}: %{z:+.4f}<extra></extra>",
                                         colorbar=dict(title="push to UP")))
            UI.plot(fig, height=220)
            st.caption("Green: this stretch pushed the 5D score towards UP. Red: towards DOWN. Not proof of a pattern.")
        with SK.fold("Chart evidence record", "data_object"):
            st.json(res.evidence.get("charts", {}))

with tabs[2]:
    SK.banner("C · News", "Company news the model read",
              "Missing news is a mask, never neutral news.")
    if res is None:
        _need_run()
    else:
        news = res.evidence.get("news") or []
        if not news:
            SK.cards([{"head": "No articles supplied", "color": UI.WARN,
                       "body": "Add articles in the sidebar. In training data a missing article is a mask, never a neutral one."}])
        else:
            SK.tiles([{"label": "Articles used", "value": str(len(news)), "color": UI.ACCENT}])
            st.dataframe(pd.DataFrame(news), width="stretch")
            st.caption("User-supplied articles are inference-time inputs; the model was trained on the 2003-2015 corpus.")

with tabs[3]:
    SK.banner("D · SEC filings", "What each filing was expected to do",
              "The SEC price-impact expert says whether a filing moves a stock, not which way.")
    if res is None:
        _need_run()
    else:
        sec = res.evidence.get("sec_filings") or []
        from nexus_alpha import language_view as LV
        tl = LV.filing_timeline(res.instrument.symbol, res.instrument.asof_session)
        SK.tiles([{"label": "Filings in view", "value": str(len(sec)), "color": UI.ACCENT},
                  {"label": "Scored by the expert", "value": str(len(tl)), "color": UI.ACCENT2}])
        if not tl.empty:
            d = tl.tail(40).copy()
            d["when"] = pd.to_datetime(d["available_utc"])
            fig = pgo.Figure(pgo.Scatter(
                x=d["when"], y=d["p_move_h5"], mode="markers+lines", line=dict(color="#23415C", width=1),
                marker=dict(size=12, color=d["lean_h5"], colorscale=[[0, UI.BAD], [0.5, "#6D88A3"], [1, UI.GOOD]], cmid=0,
                            line=dict(color=UI.BG, width=1), colorbar=dict(title="P(up)-P(down)")),
                customdata=d["event_type"], hovertemplate="%{x|%Y-%m-%d} · %{customdata}<br>P(move, 5 sessions) %{y:.2f}"
                                                          "<extra></extra>"))
            fig.update_yaxes(title_text="P(move) over 5 sessions", range=[0, 1])
            UI.plot(fig, height=320)
            st.caption("Height: how likely the filing was to move the stock. Colour: the lean, green up and red down.")
            st.subheader("SEC price-impact expert: what each filing was expected to do (E28)")
            st.dataframe(tl.tail(12).iloc[::-1].rename(columns={"available_utc": "public (UTC)",
                         "p_move_h1": "P(move) 1 session", "p_move_h5": "P(move) 5 sessions",
                         "lean_h1": "P(up)-P(down) 1", "lean_h5": "P(up)-P(down) 5"}).drop(columns="event_type"),
                         width="stretch")
        if sec:
            st.subheader("Filings the model read")
            st.dataframe(pd.DataFrame(sec), width="stretch")
        SK.pills([(m, UI.WARN) for m in res.confluence.get("missing_information", []) if "SEC" in m])

with tabs[4]:
    SK.banner("E · Indicators", "The classic gauges, computed not guessed",
              "Deterministic calculations on daily bars.")
    if res is None:
        _need_run()
    else:
        ind = res.evidence["indicators"]
        close, lo_b, up_b = ind["close"], ind["bb_lo"], ind["bb_up"]
        pctb = (close - lo_b) / ((up_b - lo_b) or 1.0)
        rsi = ind["rsi14"]
        atr_pct = ind["atr14"] / close
        SK.gauge_rings([
            {"caption": "RSI 14", "frac": rsi / 100, "center": f"{rsi:.0f}", "color": UI.WARN if rsi > 70 or rsi < 30 else UI.ACCENT,
             "sub": "overbought" if rsi > 70 else "oversold" if rsi < 30 else "neutral"},
            {"caption": "Bollinger %B", "frac": max(0.0, min(1.0, pctb)), "center": f"{pctb:.0%}",
             "color": UI.WARN if pctb > 1 or pctb < 0 else UI.ACCENT2, "sub": "outside band" if pctb > 1 or pctb < 0 else "inside band"},
            {"caption": "Volume vs 20D", "frac": min(1.0, ind["vol_ratio"] / 2), "center": f"{ind['vol_ratio']:.2f}x",
             "color": UI.GOOD if ind["vol_ratio"] > 1 else UI.MUTED, "sub": "of average"},
            {"caption": "ATR / price", "frac": min(1.0, atr_pct / 0.06), "center": f"{atr_pct:.1%}", "color": UI.ACCENT,
             "sub": "daily range"}])
        SK.tiles([{"label": "MACD histogram", "value": f"{ind['macd_hist']:+.2f}", "color": UI.GOOD if ind["macd_hist"] > 0 else UI.BAD,
                   "sub": "momentum up" if ind["macd_hist"] > 0 else "momentum down"},
                  {"label": "Close vs SMA 20", "value": f"{close / ind['sma20'] - 1:+.1%}",
                   "color": UI.GOOD if close > ind["sma20"] else UI.BAD},
                  {"label": "SMA 20 vs SMA 50", "value": f"{ind['sma20'] / ind['sma50'] - 1:+.1%}",
                   "color": UI.GOOD if ind["sma20"] > ind["sma50"] else UI.BAD}])
        SK.section("Where the close sits")
        fig = pgo.Figure()
        fig.add_trace(pgo.Bar(y=["price"], x=[up_b - lo_b], base=[lo_b], orientation="h", marker_color="rgba(155,140,255,.25)",
                              marker_line=dict(color=UI.ACCENT2, width=1), name="Bollinger band",
                              hovertemplate=f"band {lo_b:.2f} .. {up_b:.2f}<extra></extra>"))
        for nm, x, c, sym in (("close", close, UI.TEXT, "diamond"), ("SMA 20", ind["sma20"], UI.ACCENT, "line-ns"),
                              ("SMA 50", ind["sma50"], UI.WARN, "line-ns")):
            fig.add_trace(pgo.Scatter(x=[x], y=["price"], mode="markers", name=nm,
                                      marker=dict(symbol=sym, size=16, color=c, line=dict(color=c, width=3)),
                                      hovertemplate=f"{nm} {x:.2f}<extra></extra>"))
        fig.update_layout(showlegend=True, barmode="overlay", yaxis=dict(visible=False))
        UI.plot(fig, height=170)
        with SK.fold("Every indicator value", "data_object"):
            st.json(ind)
        sf = res.evidence.get("structured_features", {})
        if sf:
            with SK.fold("Model inputs (scale-free features), strongest first", "bar_chart"):
                s = pd.Series(sf, name="value").astype(float)
                top = s.reindex(s.abs().sort_values(ascending=False).index).head(24).iloc[::-1]
                fig = pgo.Figure(pgo.Bar(x=top.values, y=top.index, orientation="h",
                                         marker_color=[UI.GOOD if v >= 0 else UI.BAD for v in top.values],
                                         hovertemplate="%{y} %{x:+.3f}<extra></extra>"))
                UI.plot(fig, height=120 + 20 * len(top))
                st.dataframe(s, width="stretch")

with tabs[5]:
    SK.banner("F · Explain", "Why the model leans this way",
              "Which inputs agree, which disagree, and what is missing.")
    if res is None:
        _need_run()
    else:
        cf = res.confluence
        SK.tiles([{"label": "Reference horizon", "value": cf.get("reference_horizon", "-")},
                  {"label": "Model lean", "value": str(cf.get("model_lean", "-")),
                   "color": {"UP": UI.GOOD, "DOWN": UI.BAD}.get(str(cf.get("model_lean")), UI.MUTED)}])
        st.markdown(f'<div class="nx-note nx-shim" style="font-size:1rem;margin:2px 0 10px">{cf["summary"]}</div>',
                    unsafe_allow_html=True)
        SK.cards([{"head": "Supporting", "color": UI.GOOD, "bullets": cf["supporting_signals"] or ["none"]},
                  {"head": "Conflicting", "color": UI.BAD, "bullets": cf["conflicting_signals"] or ["none"]},
                  {"head": "Missing", "color": UI.WARN, "bullets": cf["missing_information"] or ["none"]}])
        diag = res.evidence.get("modality_diagnostics", {})
        if diag:
            SK.section("What each input pushed: change in P(UP) - P(DOWN) when it is removed")
            names = {"img_W": "Weekly chart", "img_D": "Daily chart", "img_H": "Hourly chart", "num": "Numbers",
                     "sec": "SEC filings", "news": "News"}
            keys = list(diag["by_modality"])
            z = [diag["by_modality"][k]["delta_up_minus_down"] for k in keys]
            lim = max(0.01, max(abs(v) for row in z for v in row))
            fig = pgo.Figure(pgo.Heatmap(z=z, x=diag["horizons"], y=[names.get(k, k) for k in keys], zmid=0, zmin=-lim, zmax=lim,
                                         colorscale=[[0, UI.BAD], [0.5, "#0F1D2C"], [1, UI.GOOD]],
                                         text=[[f"{v:+.3f}" for v in row] for row in z], texttemplate="%{text}",
                                         hovertemplate="%{y} at %{x}: %{z:+.4f}<extra></extra>"))
            UI.plot(fig, height=120 + 44 * len(keys))
            st.caption("Attributions describe this model's sensitivity, not causes.")
        with SK.fold("Rule events", "rule"):
            st.json(res.evidence.get("rules", {}))
        with SK.fold("RAG citations (explanation only; RAG never changes a probability)", "menu_book"):
            st.dataframe(pd.DataFrame(res.evidence.get("rag_sources", [])), width="stretch")

with tabs[6]:
    SK.banner("G · Trade planner", "A paper plan from the risk engine",
              "Levels and the reasons for them. Nothing is ever sent to a broker.")
    if res is None:
        _need_run()
    else:
        dcol = {"LONG": UI.GOOD, "SHORT": UI.BAD}
        SK.pills([(f"{k} · {p.decision}", dcol.get(p.decision, UI.MUTED)) for k, p in res.trade_plans.items()])
        h = st.segmented_control("Horizon", HZ, default="5D", key="planner_h") or "5D"
        p = res.trade_plans[h]
        SK.tiles([{"label": f"Decision at {h}", "value": p.decision, "color": dcol.get(p.decision, UI.WARN)}])
        import plan_view as PV  # noqa: E402
        import trade_desk_tab  # noqa: E402
        try:
            daily = trade_desk_tab._bars_for_check(tick, mode, asof)["D"]
        except Exception:  # noqa: BLE001  (the chart is optional; the reason card is the record)
            daily = None
        PV.explainer(res, h, RiskConfig(**RISK_DEFAULTS), daily, key="planner")
        with SK.fold("Full plan record (risk engine output)", "receipt_long"):
            st.dataframe(pd.Series({k: v for k, v in p.model_dump().items() if v is not None and k not in ("reasons",)},
                                   dtype=object), width="stretch")

with tabs[7]:
    SK.banner("H · Diagnostics", "How well did the models do on data they never saw",
              "Log loss: lower is better. 2017-18 is validation, 2020-22 is the test.")
    r = latest_run("e03_fusion")
    if r:
        m = r["metrics"]
        st.caption(f"Run {r['id']}: {r['manifest'].get('summary', '')}")
        rows = []
        for name, v in m.items():
            if isinstance(v, dict) and "valid" in v and "test_raw" in v:
                rows.append({"model": name, "modalities": ",".join(v["modalities"]), "valid mean log loss": v["valid"]["mean_log_loss"],
                             "test mean log loss": v["test_raw"]["mean_log_loss"]})
        if rows:
            d = pd.DataFrame(rows).sort_values("valid mean log loss")
            best = d.iloc[0]
            SK.tiles([{"label": "Models compared", "value": str(len(d))},
                      {"label": "Best on validation", "value": str(best["model"]), "color": UI.GOOD,
                       "sub": f"log loss {best['valid mean log loss']:.4f}"},
                      {"label": "Its test log loss", "value": f"{best['test mean log loss']:.4f}", "color": UI.ACCENT2}])
            fig = pgo.Figure([pgo.Bar(x=d["model"], y=d["valid mean log loss"], name="validation 2017-18", marker_color=UI.ACCENT),
                              pgo.Bar(x=d["model"], y=d["test mean log loss"], name="test 2020-22", marker_color=UI.ACCENT2)])
            lo = float(min(d["valid mean log loss"].min(), d["test mean log loss"].min()))
            hi = float(max(d["valid mean log loss"].max(), d["test mean log loss"].max()))
            fig.update_layout(barmode="group", yaxis=dict(range=[lo - (hi - lo) * 0.6, hi + (hi - lo) * 0.2], title="mean log loss"))
            UI.plot(fig, height=340)
            st.caption("The axis is cut to show the differences. They are small: that is the finding.")
            st.dataframe(d, width="stretch")
        if "hypotheses" in m:
            hy = m["hypotheses"]
            SK.section("Registered hypotheses: does the extra input help? (right of zero = improvement)")
            names = list(hy["tests"])
            est = [hy["tests"][k]["estimate"] for k in names]
            lo_ = [hy["tests"][k]["ci95"][0] for k in names]
            hi_ = [hy["tests"][k]["ci95"][1] for k in names]
            cols = [UI.GOOD if hy["holm"][k]["p_holm"] < 0.05 and e > 0 else UI.MUTED for k, e in zip(names, est)]
            fig = pgo.Figure(pgo.Scatter(x=est, y=names, mode="markers", marker=dict(size=12, color=cols, symbol="diamond"),
                                         error_x=dict(type="data", symmetric=False, array=[h - e for h, e in zip(hi_, est)],
                                                      arrayminus=[e - l for l, e in zip(lo_, est)], color=UI.MUTED, thickness=1.6),
                                         customdata=[hy["holm"][k]["p_holm"] for k in names],
                                         hovertemplate="%{y}<br>%{x:+.4f}<br>Holm p %{customdata:.3f}<extra></extra>"))
            fig.add_vline(x=0, line=dict(color=UI.MUTED, dash="dot", width=1))
            fig.update_yaxes(autorange="reversed")
            UI.plot(fig, height=120 + 52 * len(names))
            st.caption("Green: survives Holm correction. Grey: does not. Whiskers: 95% interval.")
            st.dataframe(pd.DataFrame({k: {"estimate": v["estimate"], "ci_low": v["ci95"][0], "ci_high": v["ci95"][1],
                                           "p_holm": hy["holm"][k]["p_holm"]} for k, v in hy["tests"].items()}).T)
        if "missing_modality_stress" in m:
            st.subheader("Missing-modality stress test (test block)")
            st.dataframe(pd.DataFrame(m["missing_modality_stress"]).T)
    SK.section("Data coverage")
    cov = ROOT / "data_manifests" / "s500_frame_summary.json"
    if cov.exists():
        st.json(json.loads(cov.read_text(encoding="utf-8")))
    log = ROOT / "docs" / "EXPERIMENT_LOG.md"
    if log.exists():
        with SK.fold("Experiment log", "history_edu"):
            st.markdown(log.read_text(encoding="utf-8"))

with tabs[8]:
    SK.banner("I · Backtesting", "Would the rules have made money?",
              "Test period 2020-2022, after costs. Equity curve on top, drawdown underneath.")
    versions = [v for v in ("v1.1", "v1") if (ROOT / "reports" / "backtest" / ("e04_backtest.json" if v == "v1" else f"e04_backtest_{v}.json")).exists()]
    ver = st.segmented_control("Backtest version", versions, default=versions[0] if versions else None, key="bt_ver",
                               help="v1.1: audited engine (prior-bar targets, corrected P&L, kill-switch, >= 30-trade threshold rule) with the "
                                    "deployed E08 model. v1: original engine with the E03 model, kept for comparison.") if versions else None
    bt = ROOT / "reports" / "backtest" / ("e04_backtest.json" if ver == "v1" else f"e04_backtest_{ver}.json")
    if ver and bt.exists():
        b = json.loads(bt.read_text(encoding="utf-8"))
        tf = pd.DataFrame(b["test_final"]).T if b["test_final"] else pd.DataFrame()
        if not tf.empty:
            bench = b["benchmark_ew_survivors"]["test"]
            best = tf["sharpe"].astype(float).idxmax()
            SK.tiles([{"label": "Best policy Sharpe", "value": f"{float(tf.loc[best, 'sharpe']):.2f}", "sub": str(best), "color": UI.ACCENT},
                      {"label": "Buy-and-hold Sharpe", "value": f"{float(bench.get('sharpe', float('nan'))):.2f}", "color": UI.GOOD,
                       "sub": "equal-weight survivors"},
                      {"label": "Policies tested", "value": str(len(tf)), "color": UI.ACCENT2},
                      {"label": "Overfit probability (PBO)", "value": f"{float(b['pbo'] if not isinstance(b['pbo'], dict) else b['pbo'].get('pbo', float('nan'))):.2f}",
                       "color": UI.WARN, "sub": "lower is better"}])
        eq = ROOT / "reports" / "backtest" / f"e04_daily_returns_{ver}.csv"
        if eq.exists():
            d = pd.read_csv(eq, index_col=0, parse_dates=True)
            from plotly.subplots import make_subplots
            curve = (1 + d).cumprod()
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28], vertical_spacing=0.04)
            palette = [UI.ACCENT, UI.WARN, UI.ACCENT2, UI.GOOD, UI.BAD, "#5B8DEF", "#E58BD8", "#9FB3C8", "#C9A227"]
            for i, c in enumerate(curve.columns):
                bench_ = "benchmark" in str(c)
                style = (dict(width=2.6, color=UI.TEXT, dash="dash") if bench_
                         else dict(width=1.4, color=palette[i % len(palette)]))
                fig.add_trace(pgo.Scatter(x=curve.index, y=curve[c], name=str(c), mode="lines", line=style,
                                          legendgroup=str(c)), row=1, col=1)
                fig.add_trace(pgo.Scatter(x=curve.index, y=curve[c] / curve[c].cummax() - 1, name=f"{c} drawdown",
                                          mode="lines", line={**style, "width": 1.0 if not bench_ else 1.8},
                                          legendgroup=str(c), showlegend=False), row=2, col=1)
            fig.update_yaxes(title_text="growth of 1", row=1, col=1)
            fig.update_yaxes(title_text="drawdown", tickformat=".0%", row=2, col=1)
            UI.plot(fig, height=460)
        if not tf.empty:
            SK.section("Sharpe by policy")
            sh = tf["sharpe"].astype(float).sort_values()
            fig = pgo.Figure(pgo.Bar(x=sh.values, y=sh.index, orientation="h",
                                     marker_color=[UI.GOOD if v > 0 else UI.BAD for v in sh.values],
                                     hovertemplate="%{y} Sharpe %{x:.2f}<extra></extra>"))
            fig.add_vline(x=float(b["benchmark_ew_survivors"]["test"].get("sharpe", 0)), line=dict(color=UI.TEXT, dash="dash", width=1.5),
                          annotation_text="buy and hold", annotation_font_color=UI.MUTED)
            UI.plot(fig, height=120 + 26 * len(sh))
        with SK.fold("Assumptions and the full results table", "table_rows"):
            st.caption("Entry at the next open; gap-through stops fill at the open; stop-first on ambiguous bars; "
                       "costs 10/15/20 bps round trip plus 0.5%/yr short borrow. Benchmark: equal-weight survivors.")
            if not tf.empty:
                st.dataframe(tf[["trades", "sharpe", "sharpe_ci95_month_blocks", "cagr", "max_drawdown",
                                 "win_rate", "profit_factor", "expectancy_R", "psr_vs_0"]], width="stretch")
            st.json({"frozen_thresholds": b["frozen_thresholds"], "pbo": b["pbo"], "benchmark_test": b["benchmark_ew_survivors"]["test"]})
    else:
        st.info("Backtest not run yet (scripts/experiments/e04_backtest.py).")

with tabs[9]:
    SK.banner("J · Knowledge base", "The books and rules behind the explanations",
              "Search only. The knowledge base explains; it never changes a probability.")
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
        meth = st.segmented_control("Method", ["bm25", "hybrid_rrf", "dense", "rule_lookup", "hybrid_rerank"],
                                    default="hybrid_rrf", key="kb_meth") or "hybrid_rrf"
        if q:
            r = kb.search(meth, q, k=8)
            r["snippet"] = r.text.str.slice(0, 200)
            SK.tiles([{"label": "Passages found", "value": str(len(r)), "color": UI.ACCENT}])
            st.dataframe(r[["doc_id", "page", "rule_id", "section", "score", "snippet"]], width="stretch")
            st.caption("Snippets are limited to 200 characters; the books are private research material.")

with tabs[10]:
    SK.banner("K · Configuration", "Experiment settings", "View, validate and export. Heavy training runs in Colab, never here.")
    cfgs = sorted((ROOT / "configs").glob("*.yaml"))
    sel = st.selectbox("Config file", [c.name for c in cfgs]) if cfgs else None
    if sel:
        import yaml
        with SK.fold("Edit the YAML", "edit_note", expanded=True):
            txt = st.text_area("YAML", (ROOT / "configs" / sel).read_text(encoding="utf-8"), height=300)
            try:
                yaml.safe_load(txt)
                st.success("Valid YAML")
                st.download_button("Export", txt, file_name=sel)
            except yaml.YAMLError as e:
                st.error(f"Invalid YAML: {e}")

with tabs[11]:
    SK.banner("L · Audit", "Curves, diagnostics and evaluation figures",
              "Generated from saved artifacts. Findings and SWOT: docs/CHIEF_AUDIT_REPORT.md.")
    figs = sorted((ROOT / "reports" / "figures").glob("*.png"))
    if not figs:
        st.info("Run scripts/report/audit_report.py to build the figures.")
    else:
        SK.tiles([{"label": "Figures", "value": str(len(figs)), "color": UI.ACCENT}])
        cols = st.columns(2)
        for k, f in enumerate(figs):
            with cols[k % 2]:
                with SK.fold(f.stem.replace("_", " "), "image", expanded=k < 2):
                    st.image(str(f), width="stretch")

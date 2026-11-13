"""Compare tab: 1 to 5 stocks side by side (the TriFrame comparison view, rebuilt on NEXUS's models).

Instant part: everything the nightly forward test logged for one session (the frozen models' probabilities, return
ranges and cross-sectional ranks), the cached bars, tonight's universe features, fundamentals and option-implied
volatility. On demand: the full v1.1 analysis of each stock with its trade plan and reason card.
Display only; data layer in `nexus_alpha.compare`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import skin as SK
from plotly.subplots import make_subplots

import ui as UI
from nexus_alpha import compare as C

COLORS = [UI.ACCENT, UI.WARN, UI.ACCENT2, UI.GOOD, "#E58BD8"]
WINDOWS = {"3M": 63, "6M": 126, "1Y": 252, "2Y": 504}
V2_H = ["1D", "2D", "5D", "10D", "15D", "20D", "3M", "6M", "1Y"]
DEFAULT = ["AAPL", "MSFT", "JPM", "KO", "XOM"]


def _color(i: int) -> str:
    return COLORS[i % len(COLORS)]


@st.cache_data(ttl=900, show_spinner=False)
def _load(session: str, picks: tuple[str, ...]):
    fc = C.forecasts(session, list(picks))
    bars = C.bars(list(picks))
    return fc, bars


def _kpis(picks, stats, fc, ranks, hz) -> None:
    cols = st.columns(len(picks))
    v11 = fc.get("v1.1")
    for i, t in enumerate(picks):
        with cols[i]:
            if t not in stats.index:
                st.metric(t, "no bars")
                continue
            s = stats.loc[t]
            st.markdown(f'<div style="height:3px;background:{_color(i)};margin-bottom:6px"></div>', unsafe_allow_html=True)
            st.metric(t, f"{s['close']:,.2f}", f"{s['1-day change']:+.2%}")
            lines = [("vol 20D", f"{s['volatility 20D (a year)']:.0%}"), ("from 1Y high", f"{s['drawdown now']:+.1%}")]
            if ranks is not None and t in ranks.index:
                lines.append((f"v2 rank {hz}", f"{ranks[t]:.0f} pct"))
            if v11 is not None:
                r = v11[(v11.ticker == t) & (v11.horizon == "5D")]
                if len(r):
                    lines.append(("v1.1 5D lean", f"{float(r.p_up.iloc[0] - r.p_down.iloc[0]):+.3f}"))
            st.markdown("".join(f'<div class="nx-kv"><span>{a}</span><span>{b}</span></div>' for a, b in lines),
                        unsafe_allow_html=True)


def _price_chart(bars, picks, window) -> None:
    p = C.normalized(C.close_panel(bars, WINDOWS[window]))
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.04)
    for i, t in enumerate([t for t in picks if t in p]):
        s = p[t].dropna()
        fig.add_trace(go.Scatter(x=s.index, y=s.values, name=t, mode="lines", line=dict(color=_color(i), width=2),
                                 legendgroup=t, hovertemplate=f"{t} %{{y:.1f}}<extra></extra>"), row=1, col=1)
        dd = s / s.cummax() - 1
        fig.add_trace(go.Scatter(x=dd.index, y=dd.values, name=f"{t} drawdown", mode="lines", legendgroup=t,
                                 showlegend=False, line=dict(color=_color(i), width=1),
                                 hovertemplate=f"{t} %{{y:.1%}}<extra></extra>"), row=2, col=1)
    fig.add_hline(y=100, line=dict(color=UI.MUTED, dash="dot", width=1), row=1, col=1)
    fig.update_yaxes(title_text="price, start = 100", row=1, col=1)
    fig.update_yaxes(title_text="drawdown", tickformat=".0%", row=2, col=1)
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    fig.update_layout(hovermode="x unified")
    UI.plot(fig, height=480)


def _lean_heatmap(fc, picks) -> None:
    v2 = fc.get("v2")
    if v2 is None:
        st.info("No v2 rows logged for this session.")
        return
    m = C.lean_matrix(v2, picks, V2_H)
    if m.empty:
        return
    z = m.to_numpy(float)
    lim = max(0.05, float(np.nanmax(np.abs(z))))
    fig = go.Figure(go.Heatmap(z=z, x=list(m.columns), y=list(m.index), zmid=0, zmin=-lim, zmax=lim,
                               colorscale=[[0, "#B23A3A"], [0.5, "#0F1D2C"], [1, "#1E8A5C"]],
                               text=[[f"{v:+.3f}" for v in row] for row in z], texttemplate="%{text}",
                               textfont=dict(color=UI.TEXT, size=12),
                               hovertemplate="%{y} %{x}: P(up) - P(down) %{z:+.3f}<extra></extra>",
                               colorbar=dict(title="lean", tickformat="+.2f")))
    fig.update_layout(title="Term structure of the v2 lean, P(up) - P(down), by horizon")
    UI.plot(fig, height=120 + 46 * len(m))
    st.caption("Green leans up, red leans down. v2's probabilities beat the class prior at 1D-20D, but stock direction "
               "was at chance on 2020-22, so read the colours as the model's view, not as a validated call.")


def _prob_bars(fc, picks, hz) -> None:
    src = "v1.1" if hz in ("1D", "2D", "5D", "10D", "20D") else "v2"
    d = fc.get(src)
    if d is None:
        return
    d = d[(d.horizon == hz) & d.ticker.isin(picks)].set_index("ticker").reindex(picks).dropna(subset=["p_up"])
    fig = go.Figure([go.Bar(y=d.index, x=d[c], name=n, orientation="h", marker_color=col,
                            text=[f"{v:.0%}" for v in d[c]], textposition="inside",
                            hovertemplate=f"%{{y}} {n} %{{x:.1%}}<extra></extra>")
                     for c, n, col in (("p_down", "DOWN", UI.BAD), ("p_side", "SIDEWAYS", "#3A4F66"),
                                       ("p_up", "UP", UI.GOOD))])
    fig.update_layout(barmode="stack", title=f"{src} probabilities at {hz}", xaxis=dict(tickformat=".0%", range=[0, 1]),
                      yaxis=dict(autorange="reversed"), legend=dict(orientation="h", y=-0.18, x=0))
    UI.plot(fig, height=150 + 44 * len(d))


def _range_chart(fc, picks, hz) -> None:
    rows = []
    for v in ("v1.1", "v2"):
        d = fc.get(v)
        if d is None:
            continue
        d = d[(d.horizon == hz) & d.ticker.isin(picks)].dropna(subset=["ret_q10", "ret_q90"])
        for r in d.itertuples():
            rows.append({"ticker": r.ticker, "model": v, "q10": r.ret_q10, "q50": r.ret_q50, "q90": r.ret_q90})
    if not rows:
        st.caption(f"No validated return range is logged at {hz} for these stocks.")
        return
    r = pd.DataFrame(rows)
    fig = go.Figure()
    for i, t in enumerate(picks):
        s = r[r.ticker == t]
        for j, row in enumerate(s.itertuples()):
            y = f"{t} ({row.model})"
            fig.add_trace(go.Scatter(x=[row.q10, row.q90], y=[y, y], mode="lines+markers", showlegend=False,
                                     line=dict(color=_color(i), width=6), marker=dict(size=9, color=_color(i)),
                                     hovertemplate=f"{y}: 10% %{{x:+.1%}}<extra></extra>"))
            fig.add_trace(go.Scatter(x=[row.q50], y=[y], mode="markers", showlegend=False,
                                     marker=dict(symbol="diamond", size=12, color=UI.TEXT),
                                     hovertemplate=f"{y}: median %{{x:+.2%}}<extra></extra>"))
    fig.add_vline(x=0, line=dict(color=UI.MUTED, dash="dot"))
    fig.update_layout(title=f"10%-90% return range at {hz} (diamond = median)", xaxis=dict(tickformat="+.0%"))
    UI.plot(fig, height=120 + 36 * len(r))
    st.caption("Quantile heads, logged only where the 10-90% band's coverage was validated (80% +/- 8 points on 2017-18).")


def _risk_map(stats, ranks, picks, hz) -> None:
    if stats.empty:
        return
    fig = go.Figure()
    for i, t in enumerate(picks):
        if t not in stats.index:
            continue
        x = stats.loc[t, "volatility 20D (a year)"]
        y = ranks.get(t, np.nan) if ranks is not None else np.nan
        fig.add_trace(go.Scatter(x=[x], y=[y], mode="markers+text", text=[t], textposition="top center", name=t,
                                 marker=dict(size=26, color=_color(i), line=dict(color=UI.BG, width=2), opacity=0.9),
                                 hovertemplate=f"{t}: vol %{{x:.0%}}, rank %{{y:.0f}}<extra></extra>"))
    fig.add_hline(y=50, line=dict(color=UI.MUTED, dash="dot"))
    fig.update_layout(title=f"Risk vs rank ({hz})", showlegend=False,
                      xaxis=dict(title="realised volatility (20 sessions), a year", tickformat=".0%"),
                      yaxis=dict(title=f"v2 rank percentile at {hz}", range=[0, 105]))
    UI.plot(fig, height=440)
    st.caption("Right = more volatile; up = ranked higher by v2 among tonight's 500 stocks. The dotted line is the "
               "universe median rank.")


def _radar(prof, picks) -> None:
    if prof is None or prof.empty:
        st.info("No universe features for this session yet.")
        return
    cats = list(prof.columns)
    fig = go.Figure()
    for i, t in enumerate(picks):
        if t not in prof.index or prof.loc[t].isna().all():
            continue
        v = prof.loc[t].fillna(0).tolist()
        fig.add_trace(go.Scatterpolar(r=v + v[:1], theta=cats + cats[:1], name=t, fill="toself", opacity=0.55,
                                      line=dict(color=_color(i), width=2),
                                      hovertemplate=f"{t} %{{theta}}: %{{r:.0f}} pct<extra></extra>"))
    fig.update_layout(title="Factor profile (universe percentile)", margin=dict(l=70, r=70, t=60, b=70),
                      legend=dict(orientation="h", y=-0.12, x=0),
                      polar=dict(bgcolor="rgba(0,0,0,0)", radialaxis=dict(range=[0, 100], gridcolor="#13233A",
                                                                          tickfont=dict(color=UI.MUTED, size=9)),
                                 angularaxis=dict(gridcolor="#13233A", tickfont=dict(color=UI.TEXT, size=11))))
    UI.plot(fig, height=500)
    st.caption("Percentile among tonight's 500 stocks; higher = more of the named quality. Mom 3M / 1M: 63- and "
               "21-session return. Trend: close vs SMA50. RSI: RSI 14. Near high: close vs the 52-week high. Low vol: "
               "low 20-session realised volatility. Value: earnings yield. Quality: ROIC. Growth: revenue growth. Low "
               "leverage: low debt to equity. Technical factors from tonight's feature rows; fundamentals from the SEC "
               "XBRL panel, point in time.")


def _corr(bars) -> None:
    if len(bars) < 2:
        return
    c = C.correlation(bars)
    fig = go.Figure(go.Heatmap(z=c.to_numpy(), x=list(c.columns), y=list(c.index), zmin=-1, zmax=1, zmid=0,
                               colorscale=[[0, "#B23A3A"], [0.5, "#0F1D2C"], [1, "#1F7F8C"]],
                               text=[[f"{v:.2f}" for v in row] for row in c.to_numpy()], texttemplate="%{text}",
                               textfont=dict(color=UI.TEXT, size=12),
                               hovertemplate="%{y} vs %{x}: %{z:.2f}<extra></extra>"))
    fig.update_layout(title="Correlation of daily log returns (last 126 sessions)")
    UI.plot(fig, height=120 + 50 * len(c))
    st.caption("High correlation means the stocks move together: holding both adds less diversification.")


def _scoreboard(fc, session, picks) -> pd.DataFrame:
    rows = {t: {} for t in picks}
    v11 = fc.get("v1.1")
    if v11 is not None:
        for t in picks:
            for hz in ("5D", "20D"):
                r = v11[(v11.ticker == t) & (v11.horizon == hz)]
                if len(r):
                    rows[t][f"v1.1 {hz} P(up)"] = float(r.p_up.iloc[0])
                    rows[t][f"v1.1 {hz} P(sideways)"] = float(r.p_side.iloc[0])
                    rows[t][f"v1.1 {hz} P(down)"] = float(r.p_down.iloc[0])
    for v, hz in (("v2", "5D"), ("v2", "20D"), ("v2", "3M"), ("v3_fund", "20D"), ("v3_alpha", "20D"), ("v3_alpha", "3M")):
        rk = C.rank_percentile(session, v, hz)
        for t in picks:
            if t in rk.index:
                rows[t][f"{v} {hz} rank pct"] = float(rk[t])
    return pd.DataFrame(rows).T


def _options(session, picks, stats) -> None:
    iv = C.iv_summary(session, picks)
    if iv is None or iv.empty or "iv30" not in iv:
        st.caption("No option-implied snapshot for this session (the nightly CBOE snapshot runs after the forecasts).")
        return
    show = pd.DataFrame({"IV30 (a year)": iv["iv30"]})
    show["implied 20D move (1 sd)"] = iv["iv30"] * np.sqrt(20 / 252)
    if not stats.empty:
        show["realised vol 20D"] = stats["volatility 20D (a year)"].reindex(show.index)
        show["IV / realised"] = show["IV30 (a year)"] / show["realised vol 20D"]
    if "put_call_oi" in iv:
        show["put/call open interest"] = iv["put_call_oi"]
    st.dataframe(show.style.format("{:.2f}"), width="stretch")
    st.caption("CBOE delayed chains via OpenBB (private data, shown locally). IV above realised volatility means the "
               "options market prices more movement than the stock has shown lately.")


def _full_analysis(predictor, picks, risk_defaults, hz) -> None:
    import plan_view as PV
    from nexus_alpha.trading.plan import RiskConfig
    hz_m = hz if hz in PV.H_SESSIONS else "5D"
    st.caption(f"The frozen v1.1 model on live Yahoo bars, one stock at a time (about 10-30 s each). Plans at {hz_m} "
               "(v1.1 covers 1D-1M).")
    if st.button("Run the full model and trade plans for these stocks", key="cmp_full", disabled=predictor is None):
        risk = RiskConfig(**risk_defaults)
        out, bar = {}, st.progress(0.0, text="starting")
        for i, t in enumerate(picks):
            bar.progress(i / len(picks), text=f"{t}: rendering charts, embedding, scoring")
            try:
                out[t] = predictor.predict(t, mode="live", risk=risk, explain=False)
            except Exception as e:  # noqa: BLE001  (one failed stock must not stop the others)
                out[t] = e
        bar.progress(1.0, text="done")
        st.session_state["cmp_full_res"] = (tuple(picks), hz_m, out)
    got = st.session_state.get("cmp_full_res")
    if not got or got[0] != tuple(picks):
        return
    _, hz_m, out = got
    rows = []
    for t, r in out.items():
        if isinstance(r, Exception):
            rows.append({"ticker": t, "decision": "ERROR", "first reason": f"{type(r).__name__}: {r}"})
            continue
        p = r.trade_plans[hz_m]
        rows.append({"ticker": t, "decision": p.decision, "first reason": (p.reasons or [""])[0],
                     "setup": p.setup_family or "-", "entry": p.entry_reference, "stop": p.stop, "T1": p.target_1,
                     "net R:R": p.net_rr, "quantity": p.quantity})
    st.dataframe(pd.DataFrame(rows).set_index("ticker"), width="stretch")
    risk = RiskConfig(**risk_defaults)
    tabs = st.tabs([t for t in out])
    for tab, (t, r) in zip(tabs, out.items()):
        with tab:
            if isinstance(r, Exception):
                st.error(f"{t}: {type(r).__name__}: {r}")
                continue
            daily = C.bars([t]).get(t)
            PV.explainer(r, hz_m, risk, None if daily is None else daily.reset_index(), key=f"cmp_{t}")


def render(universe: list[str], predictor=None, risk_defaults: dict | None = None) -> None:
    SK.banner("Compare", "Up to five stocks, side by side", "Logged forecasts, risk, options and fundamentals on one board.")
    with SK.fold("About this view", "info"):
        st.caption("Side by side: price paths, drawdowns, the frozen models' probabilities and ranks for the latest "
                   "forward-test session, return ranges, a factor profile, correlations and option-implied volatility. "
                   "Then, on demand, the full model with trade plans and their reasons. Research only; no orders.")
    sess = C.sessions("v1.1")
    if not sess:
        st.warning("No forward-test session has been logged yet.")
        return
    c1, c2, c3 = st.columns([3, 1, 1])
    default = [t for t in DEFAULT if t in universe][:5]
    picks = c1.multiselect("Stocks (up to 5)", universe, default=default, max_selections=C.MAX_STOCKS, key="cmp_picks")
    session = c2.selectbox("Logged session", sess[::-1], key="cmp_session")
    hz = c3.selectbox("Horizon", V2_H, index=V2_H.index("20D"), key="cmp_hz")
    if not picks:
        st.info("Pick between one and five stocks.")
        return
    fc, bars = _load(session, tuple(picks))
    stats = C.risk_stats(bars)
    ranks = C.rank_percentile(session, "v2", hz)
    _kpis(picks, stats, fc, ranks, hz)
    UI.frame("verdict", "In one paragraph", f'<div class="nx-note" style="font-size:.9rem">'
             f'{C.verdict(stats, None, ranks.reindex(picks) if len(ranks) else None).replace("**", "")}</div>')
    window = st.radio("Price window", list(WINDOWS), index=2, horizontal=True, key="cmp_window")
    _price_chart(bars, picks, window)
    left, right = st.columns(2)
    with left:
        _prob_bars(fc, picks, hz)
    with right:
        _range_chart(fc, picks, hz)
    _lean_heatmap(fc, picks)
    left, right = st.columns(2)
    with left:
        _risk_map(stats, ranks, picks, hz)
    with right:
        _radar(C.factor_profile(session, picks), picks)
    _corr(bars)
    SK.section("Scoreboard (logged forecasts, session " + session + ")")
    sb = _scoreboard(fc, session, picks)
    if not sb.empty:
        fmt = {c: ("{:.0f}" if "pct" in c else "{:.2f}") for c in sb.columns}
        st.dataframe(sb.style.format(fmt, na_rep="-").background_gradient(
            cmap="viridis", subset=[c for c in sb.columns if "pct" in c], vmin=0, vmax=100), width="stretch")
    SK.section("Risk statistics (cached Yahoo bars)")
    if not stats.empty:
        pct = [c for c in stats.columns if c not in ("as of", "close")]
        st.dataframe(stats.style.format({c: "{:+.1%}" if "return" in c or "change" in c or "drawdown" in c else "{:.1%}"
                                         for c in pct} | {"close": "{:,.2f}"}), width="stretch")
    SK.section("Options market view")
    _options(session, picks, stats)
    SK.section("Full model, trade plans and reasons")
    _full_analysis(predictor, picks, risk_defaults or {}, hz)

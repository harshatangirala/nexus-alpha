"""P Fundamentals: valuation, quality, growth, health and forensic scores for one stock, point in time.

Historical mode reads the development store (SEC XBRL facts filed before the seal, the stored daily panel and its
universe and industry percentiles). Live mode reads today's SEC facts and the nightly live cross-section. Composite
scores and the E30 replication result come from `fundamentals/extended.py` and reports/e30. Display only: nothing here
changes a forecast, a frozen model or the risk engine.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import skin as SK
import ui as UI

ROOT = Path(__file__).resolve().parents[1]
GROUPS = {  # radar axes: (ratio, higher is better)
    "Value": [("earnings_yield", 1), ("book_to_price", 1), ("ebitda_to_ev", 1), ("fcf_yield", 1), ("sales_yield", 1)],
    "Profitability": [("gross_margin", 1), ("operating_margin", 1), ("roe", 1), ("roa", 1), ("roic", 1)],
    "Growth": [("revenue_growth", 1), ("operating_income_growth", 1), ("eps_growth", 1)],
    "Health": [("debt_to_equity", -1), ("net_debt_to_ebitda", -1), ("interest_coverage", 1), ("current_ratio", 1)],
    "Efficiency": [("asset_turnover", 1), ("receivables_to_sales", -1)],
    "Earnings quality": [("accruals", -1), ("piotroski_f", 1), ("altman_z", 1)],
}
COMPOSITE_LABEL = {"value_composite": "Value composite", "quality_composite": "Quality (QMJ-style)",
                   "gross_profitability_rank": "Gross profitability (Novy-Marx)", "magic_formula": "Magic formula (Greenblatt)",
                   "piotroski_rank": "Piotroski F", "beneish_safety_rank": "Low manipulation risk (Beneish)"}


def _f(v) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return x if np.isfinite(x) else float("nan")


def _fmt(v, kind: str = "x") -> str:
    x = _f(v)
    if not np.isfinite(x):
        return "-"
    return {"pct": f"{x:+.1%}", "x": f"{x:,.1f}", "bn": f"{x / 1e9:,.1f}", "int": f"{x:.0f}", "2": f"{x:,.2f}"}[kind]


@st.cache_data(ttl=6 * 3600, show_spinner="Loading SEC XBRL fundamentals...")
def _load(ticker: str, mode: str, asof: str) -> dict:
    from nexus_alpha.fundamentals import extended as X, service as FS
    try:
        d = FS.historical(ticker, asof) if mode == "historical" else FS.live(ticker)
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}
    if not d:
        return {"error": "no XBRL facts for this ticker"}
    s, r = d["snapshot"], d["ratios"]
    row = s.to_dict()
    if r is not None:
        row.update({k: r.get(k) for k in ("market_cap", "enterprise_value", "pe", "eps_growth")})
    ext = X.metrics(pd.DataFrame([row])).iloc[0].to_dict()
    return {"snapshot": s.to_dict(), "ratios": None if r is None else r.to_dict(), "ext": ext, "source": d["source"]}


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def _history(ticker: str, asof: str) -> pd.DataFrame:
    import pyarrow.parquet as pq
    from nexus_alpha import paths
    cols = ["date", "pe", "ev_ebitda", "ps", "pb", "gross_margin", "operating_margin", "net_margin", "roic", "roe",
            "revenue_growth", "piotroski_f", "altman_z", "market_cap"]
    t = pq.read_table(paths.ARTIFACTS / "fundamentals" / "panel.parquet", columns=cols,
                      filters=[("ticker", "=", ticker), ("date", "<=", pd.Timestamp(asof).to_pydatetime())])
    d = t.to_pandas().sort_values("date")
    return d.set_index("date").resample("W-FRI").last().dropna(how="all")


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def _peers(ticker: str, asof: str, mode: str) -> pd.DataFrame:
    import pyarrow.parquet as pq
    from nexus_alpha import paths
    cols = ["ticker", "date", "sic2", "ev_ebitda", "roic", "pe", "market_cap", "gross_margin", "revenue_growth"]
    if mode == "live":
        p = paths.ARTIFACTS / "fundamentals" / "live_panel.parquet"
        d = pd.read_parquet(p) if p.exists() else pd.DataFrame()
        return d[[c for c in cols if c in d]] if len(d) else d
    f = pq.read_table(paths.ARTIFACTS / "fundamentals" / "panel.parquet", columns=["date"],
                      filters=[("ticker", "=", ticker), ("date", "<=", pd.Timestamp(asof).to_pydatetime())]).to_pandas()
    if f.empty:
        return pd.DataFrame()
    day = pd.Timestamp(f.date.max()).to_pydatetime()
    return pq.read_table(paths.ARTIFACTS / "fundamentals" / "panel.parquet", columns=cols,
                         filters=[("date", "=", day)]).to_pandas()


@st.cache_data(ttl=3600, show_spinner=False)
def _composites_live(ticker: str) -> dict:
    from nexus_alpha import paths
    p = paths.ARTIFACTS / "fundamentals" / "live_panel.parquet"
    if not p.exists():
        return {}
    d = pd.read_parquet(p)
    r = d[d.ticker == ticker]
    if r.empty:
        return {}
    return {k: _f(r[f"x_{k}"].iloc[0]) for k in COMPOSITE_LABEL if f"x_{k}" in r}


def _e30() -> dict | None:
    p = ROOT / "reports" / "e30" / "e30_results.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _radar(r: dict) -> None:
    axes, uni, ind = [], [], []
    for grp, items in GROUPS.items():
        u = [(_f(r.get(f"fr_{k}")) if s > 0 else 1 - _f(r.get(f"fr_{k}"))) for k, s in items]
        i = [(_f(r.get(f"fi_{k}")) if s > 0 else 1 - _f(r.get(f"fi_{k}"))) for k, s in items]
        u, i = [x for x in u if np.isfinite(x)], [x for x in i if np.isfinite(x)]
        axes.append(grp)
        uni.append(100 * np.mean(u) if u else np.nan)
        ind.append(100 * np.mean(i) if i else np.nan)
    fig = go.Figure()
    for vals, name, col, fill in ((uni, "vs the universe", UI.ACCENT, "rgba(63,208,224,.18)"),
                                  (ind, "vs its industry", UI.ACCENT2, "rgba(155,140,255,.14)")):
        v = [0 if not np.isfinite(x) else x for x in vals]
        fig.add_trace(go.Scatterpolar(r=v + v[:1], theta=axes + axes[:1], name=name, fill="toself", fillcolor=fill,
                                      line=dict(color=col, width=2), hovertemplate="%{theta}: %{r:.0f}th percentile<extra></extra>"))
    fig.update_layout(polar=dict(bgcolor="rgba(0,0,0,0)", radialaxis=dict(range=[0, 100], tickvals=[25, 50, 75],
                                                                          gridcolor="#1B2B3D", tickfont=dict(color=UI.MUTED)),
                                 angularaxis=dict(gridcolor="#1B2B3D", tickfont=dict(color=UI.TEXT, size=12))),
                      legend=dict(orientation="h", y=-0.08))
    UI.plot(fig, height=420)
    st.caption("Percentiles that day, 100 = best by each ratio's own logic (cheaper, more profitable, safer). "
               "Purple: the same within its two-digit SIC industry.")


def _beneish_chart(e: dict) -> None:
    from nexus_alpha.fundamentals import extended as X
    names = {"dsri": "Receivables vs sales", "gmi": "Gross margin decline", "aqi": "Asset quality", "sgi": "Sales growth",
             "depi": "Depreciation slowdown", "sgai": "SG&A vs sales", "lvgi": "Leverage growth", "tata": "Accruals"}
    contrib = {names[k]: X.BENEISH[k] * (_f(e.get(k)) - (0 if k == "tata" else 1)) for k in X.BENEISH_PARTS}
    contrib = {k: v for k, v in contrib.items() if np.isfinite(v)}
    if not contrib:
        return
    s = pd.Series(contrib).sort_values()
    fig = go.Figure(go.Bar(x=s.values, y=s.index, orientation="h",
                           marker_color=[UI.BAD if v > 0 else UI.GOOD for v in s.values],
                           hovertemplate="%{y}: %{x:+.3f} to M<extra></extra>"))
    fig.add_vline(x=0, line=dict(color=UI.MUTED, dash="dot", width=1))
    fig.update_layout(title=dict(text="What moves the Beneish M-score away from a neutral firm", font=dict(size=13, color=UI.MUTED)))
    UI.plot(fig, height=320)
    st.caption("Red pushes M up (more manipulation-like), green down. A neutral firm has every index at 1 and accruals at 0.")


def _history_chart(h: pd.DataFrame) -> None:
    fig = make_subplots(rows=2, cols=2, shared_xaxes=True, vertical_spacing=0.12, horizontal_spacing=0.08,
                        subplot_titles=["P/E and EV/EBITDA", "Margins", "ROIC and ROE", "Revenue growth"])
    lines = [(1, 1, "pe", "P/E", UI.ACCENT), (1, 1, "ev_ebitda", "EV/EBITDA", UI.ACCENT2),
             (1, 2, "gross_margin", "gross", UI.GOOD), (1, 2, "operating_margin", "operating", UI.ACCENT),
             (1, 2, "net_margin", "net", UI.WARN), (2, 1, "roic", "ROIC", UI.ACCENT), (2, 1, "roe", "ROE", UI.ACCENT2),
             (2, 2, "revenue_growth", "revenue growth", UI.GOOD)]
    for r, c, k, n, col in lines:
        if k in h and h[k].notna().any():
            y = h[k].clip(upper=h[k].quantile(0.98)) if k in ("pe", "ev_ebitda") else h[k]
            fig.add_trace(go.Scatter(x=h.index, y=y, name=n, mode="lines", line=dict(color=col, width=1.8)), row=r, col=c)
    for r, c in ((1, 2), (2, 1), (2, 2)):
        fig.update_yaxes(tickformat=".0%", row=r, col=c)
    for a in fig.layout.annotations:
        a.update(font=dict(family=UI.FONT_MONO, size=11, color=UI.ACCENT), x=a.x)
    fig.update_layout(legend=dict(orientation="h", y=-0.1, x=0, yanchor="top"), margin=dict(t=40))
    UI.plot(fig, height=520)


def _peer_chart(peers: pd.DataFrame, ticker: str) -> None:
    if peers.empty or "sic2" not in peers:
        return
    me = peers[peers.ticker == ticker]
    if me.empty or pd.isna(me.sic2.iloc[0]):
        return
    g = peers[(peers.sic2 == me.sic2.iloc[0])].dropna(subset=["ev_ebitda", "roic"])
    g = g[(g.ev_ebitda > 0) & (g.ev_ebitda < g.ev_ebitda.quantile(0.97) * 1.01)]
    if len(g) < 3:
        return
    other = g[g.ticker != ticker]
    size = 8 + 30 * np.sqrt(g.market_cap.fillna(0) / max(g.market_cap.max(), 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=other.ev_ebitda, y=other.roic, mode="markers+text", text=other.ticker,
                             textposition="top center", textfont=dict(size=10, color=UI.MUTED), name="industry peers",
                             marker=dict(size=size[other.index], color="rgba(155,140,255,.45)", line=dict(color=UI.ACCENT2, width=1)),
                             hovertemplate="%{text}: EV/EBITDA %{x:.1f}, ROIC %{y:.1%}<extra></extra>"))
    sel = g[g.ticker == ticker]
    if len(sel):
        fig.add_trace(go.Scatter(x=sel.ev_ebitda, y=sel.roic, mode="markers+text", text=sel.ticker, textposition="top center",
                                 textfont=dict(size=13, color=UI.ACCENT), name=ticker,
                                 marker=dict(size=22, color=UI.ACCENT, symbol="diamond", line=dict(color=UI.BG, width=2)),
                                 hovertemplate="%{text}: EV/EBITDA %{x:.1f}, ROIC %{y:.1%}<extra></extra>"))
    lo, hi = float(g.roic.quantile(0.04)), float(g.roic.quantile(0.96))
    pad = (hi - lo) * 0.15 or 0.05
    if len(sel):
        lo, hi = min(lo, float(sel.roic.iloc[0])), max(hi, float(sel.roic.iloc[0]))
    fig.update_layout(xaxis_title="EV/EBITDA (cheaper to the left)",
                      yaxis=dict(title="ROIC", tickformat=".0%", range=[lo - pad, hi + pad]),
                      title=dict(text=f"Industry peers (SIC {int(me.sic2.iloc[0])}): price paid vs return on capital",
                                 font=dict(size=13, color=UI.MUTED)))
    UI.plot(fig, height=420)
    st.caption("Bottom-right is expensive and unprofitable, top-left cheap and profitable. Bubble size: market cap.")


def _e30_panel(e30: dict) -> None:
    P = e30["primary"]
    names = list(P)
    est = [P[k]["mean_ic"] for k in names]
    lo = [P[k]["ci95"][0] for k in names]
    hi = [P[k]["ci95"][1] for k in names]
    cols = [UI.GOOD if P[k]["replicates"] else UI.MUTED for k in names]
    fig = go.Figure(go.Scatter(x=est, y=[COMPOSITE_LABEL.get(k, k) for k in names], mode="markers",
                               marker=dict(size=13, color=cols, symbol="diamond"),
                               error_x=dict(type="data", symmetric=False, array=[h - e for h, e in zip(hi, est)],
                                            arrayminus=[e - l for l, e in zip(lo, est)], color=UI.MUTED, thickness=1.6),
                               customdata=[[P[k]["p_holm"], P[k]["mean_q5_minus_q1_20d"]] for k in names],
                               hovertemplate="%{y}<br>mean monthly IC %{x:+.4f}<br>Holm p %{customdata[0]:.3f}"
                                             "<br>top-minus-bottom quintile, 20 sessions %{customdata[1]:+.2%}<extra></extra>"))
    fig.add_vline(x=0, line=dict(color=UI.MUTED, dash="dot", width=1))
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(title=dict(text=f"E30: does each published composite rank this universe? ({e30['signal_dates']} months, "
                                      f"{e30['first'][:4]}-{e30['last'][:4]})", font=dict(size=13, color=UI.MUTED)))
    UI.plot(fig, height=340)
    st.caption("Mean monthly rank correlation with the next 20-session return, 95% block-bootstrap interval. Green: "
               "replicates after Holm correction (pre-registered, D-060). Grey: does not.")


def render(universe: list[str]) -> None:
    SK.banner("P · Fundamentals", "Valuation, quality and forensic scores",
              "SEC XBRL as it was known on the day. Published formulas, no fitted weights.")
    c = st.columns([2, 1.2, 1.4], vertical_alignment="bottom")
    tick = c[0].selectbox("Ticker", universe, index=universe.index("AAPL") if "AAPL" in universe else 0, key="fu_tick")
    mode = c[1].segmented_control("Data", ["historical", "live"], default="historical", key="fu_mode") or "historical"
    asof = c[2].date_input("As-of date (historical)", value=pd.Timestamp("2021-06-15"), min_value=pd.Timestamp("2006-01-03"),
                           max_value=pd.Timestamp("2022-12-30"), key="fu_asof", disabled=mode == "live")
    asof_s = str(pd.Timestamp(asof).date()) if mode == "historical" else str(pd.Timestamp.now().date())
    d = _load(tick, mode, asof_s)
    if d.get("error"):
        st.info(f"Fundamentals unavailable for {tick}: {d['error']}")
        return
    s, r, e = d["snapshot"], d["ratios"] or {}, d["ext"]
    st.caption(d["source"])
    if _f(r.get("foreign_filer")) == 1 or _f(r.get("shares_basis_ok")) == 0:
        st.info("Price-based ratios are not computed for this listing (foreign filer, or a share basis that cannot be "
                "matched to the price; audit 3). Statement-based scores below still apply.")

    SK.section("Valuation")
    SK.tiles([{"label": "P/E", "value": _fmt(r.get("pe")), "color": UI.ACCENT},
              {"label": "EV/EBITDA", "value": _fmt(r.get("ev_ebitda")), "color": UI.ACCENT2},
              {"label": "EV/EBIT", "value": _fmt(e.get("ev_ebit")), "color": UI.ACCENT2},
              {"label": "PEG", "value": _fmt(e.get("peg"), "2"), "color": UI.WARN, "sub": "P/E over EPS growth"},
              {"label": "P/S", "value": _fmt(r.get("ps")), "color": UI.ACCENT},
              {"label": "P/B", "value": _fmt(r.get("pb")), "color": UI.ACCENT},
              {"label": "P/FCF", "value": _fmt(r.get("p_fcf")), "color": UI.ACCENT},
              {"label": "Market cap", "value": _fmt(r.get("market_cap"), "bn"), "sub": "$ billion", "color": UI.GOOD}])

    SK.section("Scores")
    pf, az, bm, mc = _f(r.get("piotroski_f")), _f(r.get("altman_z")), _f(e.get("beneish_m")), _f(e.get("montier_c"))
    gp, cc = _f(e.get("gross_profitability")), _f(e.get("cash_conversion"))
    rings = []
    if np.isfinite(pf):
        rings.append({"caption": "Piotroski F", "frac": pf / 9, "center": f"{pf:.0f}/9",
                      "color": UI.GOOD if pf >= 7 else UI.BAD if pf <= 3 else UI.WARN, "sub": "strong" if pf >= 7 else "weak" if pf <= 3 else "middling"})
    if np.isfinite(az):
        rings.append({"caption": "Altman Z", "frac": min(1.0, max(0.0, az / 6)), "center": f"{az:.1f}",
                      "color": UI.GOOD if az > 2.99 else UI.BAD if az < 1.81 else UI.WARN,
                      "sub": "safe" if az > 2.99 else "distress" if az < 1.81 else "grey zone"})
    if np.isfinite(bm):
        rings.append({"caption": "Beneish M", "frac": min(1.0, max(0.0, (bm + 4) / 4)), "center": f"{bm:.2f}",
                      "color": UI.BAD if bm > -1.78 else UI.GOOD, "sub": "flag" if bm > -1.78 else "no flag"})
    if np.isfinite(mc):
        rings.append({"caption": "Montier C", "frac": mc / 6, "center": f"{mc:.0f}/6",
                      "color": UI.BAD if mc >= 4 else UI.GOOD if mc <= 2 else UI.WARN, "sub": "risky" if mc >= 4 else "clean" if mc <= 2 else "watch"})
    if np.isfinite(gp):
        rings.append({"caption": "Gross profit / assets", "frac": min(1.0, max(0.0, gp / 0.8)), "center": f"{gp:.0%}",
                      "color": UI.ACCENT, "sub": "Novy-Marx"})
    if np.isfinite(cc):
        rings.append({"caption": "Cash conversion", "frac": min(1.0, max(0.0, cc / 2)), "center": f"{cc:.2f}x",
                      "color": UI.GOOD if cc >= 1 else UI.WARN, "sub": "CFO / net income"})
    if rings:
        SK.gauge_rings(rings)
    else:
        st.caption("Not enough tagged statement items for the scores.")

    cols = st.columns([3, 2])
    with cols[0]:
        if r:
            _radar(r)
    with cols[1]:
        comp = _composites_live(tick) if mode == "live" else {}
        if comp:
            SK.section("Published composites tonight (percentile in the universe)")
            vals = pd.Series({COMPOSITE_LABEL[k]: v for k, v in comp.items() if np.isfinite(v)}).sort_values()
            fig = go.Figure(go.Bar(x=100 * vals.values, y=vals.index, orientation="h", marker_color=UI.ACCENT,
                                   hovertemplate="%{y}: %{x:.0f}th percentile<extra></extra>"))
            fig.update_xaxes(range=[0, 100])
            UI.plot(fig, height=300)
        SK.section("Return on equity, taken apart (DuPont)")
        nm = _f(r.get("net_margin"))
        at = _f(r.get("asset_turnover"))
        em = _f(e.get("equity_multiplier"))
        SK.tiles([{"label": "Net margin", "value": _fmt(nm, "pct"), "color": UI.GOOD},
                  {"label": "× Asset turnover", "value": _fmt(at, "2"), "color": UI.ACCENT},
                  {"label": "× Equity multiplier", "value": _fmt(em, "2"), "color": UI.WARN},
                  {"label": "= ROE", "value": _fmt(r.get("roe"), "pct"), "color": UI.ACCENT2}])
        SK.tiles([{"label": "Days sales outstanding", "value": _fmt(e.get("dso"), "int"), "color": UI.ACCENT},
                  {"label": "Days inventory", "value": _fmt(e.get("dio"), "int"), "color": UI.ACCENT},
                  {"label": "Quick ratio", "value": _fmt(e.get("quick_ratio"), "2"), "color": UI.GOOD}])

    _beneish_chart(e)
    if mode == "historical":
        h = _history(tick, asof_s)
        if len(h) > 10:
            SK.section("How the numbers moved (point in time, weekly)")
            _history_chart(h)
    _peer_chart(_peers(tick, asof_s, mode), tick)

    e30 = _e30()
    if e30:
        SK.section("Do these scores predict returns? (E30, pre-registered)")
        _e30_panel(e30)

    with SK.fold("Every extended metric (Beneish parts, Montier, DuPont, efficiency, liquidity)", "table_rows"):
        st.dataframe(pd.Series({k: v for k, v in e.items()}, name="value").to_frame(), width="stretch")
    with SK.fold("Every base ratio with its universe and industry percentile", "table_rows"):
        keys = sorted({k for k in r if not k.startswith(("fr_", "fi_"))})
        st.dataframe(pd.DataFrame({"value": [r.get(k) for k in keys],
                                   "universe pct": [r.get(f"fr_{k}") for k in keys],
                                   "industry pct": [r.get(f"fi_{k}") for k in keys]}, index=keys), width="stretch")
    with SK.fold("Statement snapshot as known that day", "receipt_long"):
        st.dataframe(pd.Series(s, name="value").astype(str).to_frame(), width="stretch")
    with SK.fold("Definitions and limits", "info"):
        st.markdown(
            "* **Beneish M** (1999): eight indices with the published coefficients; flag above -1.78. Missing SG&A or "
            "D&A tags count as neutral.\n* **Montier C** (2008): six warning signs, 0-6.\n* **Gross profitability** "
            "(Novy-Marx 2013): (revenue - COGS) / assets.\n* **Magic formula** (Greenblatt 2005): EBIT/EV and "
            "EBIT/(net working capital + net PP&E).\n* **Quality** after Asness, Frazzini and Pedersen (2019), from what "
            "XBRL supports: no beta, no earnings volatility, one-year growth.\n* Banks, insurers and REITs (SIC 60-67) "
            "get no Beneish, Altman or magic-formula score.\n* Tag coverage: COGS, receivables and operating income are "
            "tagged for only 52-67% of filings in the store, so some scores are missing (A-112).")

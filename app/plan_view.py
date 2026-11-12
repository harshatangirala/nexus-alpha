"""Trade-plan explainer shared by the Trade Desk, the Trade planner and the Compare tab (display only).

Draws the plan on the price chart (entry, stop, targets, invalidation, chase limit, risk and reward zones) and prints
the reason card from `nexus_alpha.trading.explain_plan`. Approved plans are drawn in full colour. When the engine says
NO_TRADE, the levels of the best-formed rule setup are drawn dashed and labelled CONDITIONAL, with the gates that failed
and the backtest evidence on rule-only plans. Nothing here changes a decision.
"""
from __future__ import annotations

import html

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import ui as UI
from nexus_alpha.trading import explain_plan as EP

H_SESSIONS = {"1D": 1, "2D": 2, "5D": 5, "10D": 10, "20D": 20, "1M": 21}


def levels_chart(daily: pd.DataFrame, plan, setup: dict | None = None, conditional: bool = False, f=None,
                 h: str = "5D", title: str | None = None, height: int = 420, n_bars: int = 120) -> go.Figure:
    d = daily.tail(n_bars).copy()
    x = pd.to_datetime(d["date"]) if "date" in d else pd.to_datetime(d.index)
    fig = go.Figure(go.Candlestick(x=x, open=d.open, high=d.high, low=d.low, close=d.close, name="daily bars",
                                   increasing=dict(line=dict(color=UI.GOOD, width=1), fillcolor=UI.GOOD),
                                   decreasing=dict(line=dict(color=UI.BAD, width=1), fillcolor=UI.BAD)))
    for n, col in ((20, UI.ACCENT), (50, UI.ACCENT2)):
        if len(daily) >= n:
            sma = daily.close.rolling(n).mean().tail(n_bars)
            fig.add_trace(go.Scatter(x=x, y=sma.values, mode="lines", line=dict(color=col, width=1.1),
                                     name=f"SMA{n}", hovertemplate=f"SMA{n} %{{y:.2f}}<extra></extra>"))
    last, ref = x.iloc[-1], float(d.close.iloc[-1])
    fut = pd.bdate_range(last + pd.Timedelta(days=1), periods=max(H_SESSIONS.get(h, 5), 3))
    if f is not None and getattr(f, "return_q10", None) is not None and getattr(f, "return_q90", None) is not None:
        fig.add_shape(type="rect", x0=last, x1=fut[-1], y0=ref * (1 + f.return_q10), y1=ref * (1 + f.return_q90),
                      fillcolor="rgba(63,208,224,0.08)", line=dict(color=UI.ACCENT, width=1, dash="dot"))
    if plan is not None and plan.decision in ("LONG", "SHORT"):
        dash = "dash" if conditional else "solid"
        x0, x1 = x.iloc[max(0, len(x) - 15)], fut[-1]
        fig.add_shape(type="rect", x0=x0, x1=x1, y0=min(plan.entry_reference, plan.stop),
                      y1=max(plan.entry_reference, plan.stop), fillcolor="rgba(255,107,107,0.13)", line=dict(width=0))
        fig.add_shape(type="rect", x0=x0, x1=x1, y0=min(plan.entry_reference, plan.target_1),
                      y1=max(plan.entry_reference, plan.target_1), fillcolor="rgba(61,220,151,0.11)", line=dict(width=0))
        rows = [("entry", plan.entry_reference, UI.TEXT, dash), ("stop", plan.stop, UI.BAD, dash),
                ("T1", plan.target_1, UI.GOOD, dash), ("T2", plan.target_2, UI.GOOD, "dot"),
                ("max chase", plan.max_chase_price, UI.WARN, "dot")]
        if setup and setup.get("invalidation") is not None:
            rows.append(("invalidation", setup["invalidation"], "#E58BD8", "dot"))
        for lab, val, col, ds in rows:
            if val:
                fig.add_shape(type="line", x0=x0, x1=x1, y0=val, y1=val, line=dict(color=col, width=1.4, dash=ds))
                left = lab == "invalidation"                  # close to the stop: label it on the other side
                fig.add_annotation(x=x0 if left else x1, y=val, text=f"{lab} {val:,.2f}", showarrow=False,
                                   xanchor="right" if left else "left", yshift=-8 if left else 0,
                                   font=dict(color=col, size=11, family=UI.FONT_MONO))
        if setup and setup.get("zone"):
            z0, z1 = setup["zone"]
            fig.add_hrect(y0=z0, y1=z1, fillcolor="rgba(155,140,255,0.08)", line_width=0,
                          annotation_text=f"{setup['family']} zone", annotation_position="top left",
                          annotation_font=dict(color=UI.ACCENT2, size=10))
        if conditional:
            fig.add_annotation(xref="paper", yref="paper", x=0.01, y=0.98, xanchor="left", yanchor="top",
                               text="CONDITIONAL - NOT APPROVED", showarrow=False,
                               font=dict(color=UI.WARN, size=13, family=UI.FONT_HEAD),
                               bgcolor="rgba(5,9,15,.75)", bordercolor=UI.WARN, borderwidth=1)
    if setup:
        fig.add_annotation(x=last, y=float(d.high.iloc[-1]), text=f"{setup['family']}", showarrow=True, arrowhead=2,
                           ay=-34, font=dict(color=UI.ACCENT2, size=11), arrowcolor=UI.ACCENT2)
    fig.update_layout(xaxis_rangeslider_visible=False, title=title, margin=dict(l=10, r=110, t=40, b=10),
                      legend=dict(orientation="h", y=1.02, x=0))
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    fig.update_layout(template=UI.TEMPLATE, height=height)
    return fig


def render_card(card: dict) -> None:
    order = ["Decision", "Entry", "Stop loss", "Targets", "Exit", "Size", "Gates the real decision failed",
             "Evidence on rule-only plans", "Chart patterns and rules", "Indicators", "Model",
             "Company (micro) evidence", "Macro context", "Book and rulebook references (RAG; explanation only)"]
    keys = [k for k in order if k in card] + [k for k in card if k not in order]
    left, right = st.columns(2)
    for i, k in enumerate(keys):
        body = "".join(f'<div class="nx-kv"><span>{html.escape(str(v))}</span></div>' for v in card[k])
        with (left if i % 2 == 0 else right):
            UI.frame("why", k, body)


def explainer(res, h: str, risk, daily: pd.DataFrame | None, key: str = "x") -> None:
    """The plan on the chart plus its reason card, for one horizon of one InferenceResult."""
    plan = res.trade_plans[h]
    f = res.forecasts[h]
    cond = EP.conditional_plans(res.evidence, h, H_SESSIONS.get(h, 5), risk) if plan.decision not in ("LONG", "SHORT") else []
    card = EP.reason_card(plan, res.evidence, f, cond)
    shown, setup, is_cond = plan, None, False
    if plan.decision in ("LONG", "SHORT"):
        setup = next((s for s in res.evidence.get("rules", {}).get("daily_setups", []) if s["family"] == plan.setup_family), None)
    else:
        ok = [c for c in cond if c["geometry_ok"]]
        if ok:
            shown, setup, is_cond = ok[0]["plan"], ok[0]["setup"], True
    if daily is not None and len(daily):
        fig = levels_chart(daily, shown, setup, is_cond, f, h,
                           title=f"{res.instrument.symbol} {h}: {plan.decision}" + (" (conditional levels shown)" if is_cond else ""))
        st.plotly_chart(fig, width="stretch", theme=None, key=f"plan_{key}_{h}")
    if plan.decision not in ("LONG", "SHORT") and not is_cond:
        st.info("No rule setup with valid geometry today, so there is no level to draw: the engine never invents a level.")
    render_card(card)

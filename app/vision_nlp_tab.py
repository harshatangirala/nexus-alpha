"""Vision + NLP tab: the E28 multimodal network (charts, indicators, news and SEC filings in one model), the SEC
price-impact expert and a point-in-time filing explorer. Display only; data layer in `nexus_alpha.language_view`."""
from __future__ import annotations

import html

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import ui as UI
from nexus_alpha import language_view as LV

GROUP_COLOR = {"vision": UI.ACCENT, "indicators": UI.GOOD, "language": UI.ACCENT2, "other": UI.MUTED}
KIND_COLOR = {"8-K": UI.ACCENT, "10-Q": UI.WARN, "10-K": UI.ACCENT2}


def _ci(d: dict) -> str:
    lo, hi = d["ci95"]
    color = UI.GOOD if lo > 0 else (UI.BAD if hi < 0 else UI.MUTED)
    return (f'<span style="font-family:{UI.FONT_MONO};color:{color}">{d["estimate"]:+.5f}</span> '
            f'<span style="color:{UI.MUTED}">[{lo:+.5f}, {hi:+.5f}]</span>')


def _sankey(gt: pd.DataFrame) -> go.Figure:
    labels = list(gt.label) + ["gated fusion", "6 horizon heads"]
    n = len(gt)
    colors = [GROUP_COLOR[g] for g in gt.group]
    # fixed columns: inputs at the left, fusion in the middle, heads at the right (labels of the last two no longer collide)
    xs = [0.01] * n + [0.5, 0.99]
    g = gt.gate.to_numpy(float)
    gap = 0.03                                          # node centres stacked by gate size, so nodes never overlap
    scale = (1 - gap * (n - 1)) / g.sum()
    ys = [float((g[:i].sum() + g[i] / 2) * scale + gap * i) for i in range(n)] + [0.5, 0.5]
    fig = go.Figure(go.Sankey(
        arrangement="fixed",
        node=dict(label=labels, x=xs, y=ys, pad=14, thickness=14, color=colors + [UI.WARN, UI.TEXT], line=dict(width=0),
                  hovertemplate="%{label}<extra></extra>"),
        link=dict(source=list(range(n)) + [n], target=[n] * n + [n + 1], value=list(gt.gate) + [float(gt.gate.sum())],
                  color=[_alpha(c, .35) for c in colors] + [_alpha(UI.WARN, .25)],
                  hovertemplate="%{source.label} -> %{target.label}: gate %{value:.3f}<extra></extra>")))
    fig.update_layout(title="How much each input feeds the network (mean gate weight, validation rows)")
    return fig


def _alpha(hex_color: str, a: float) -> str:
    h = hex_color.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{a})"


def _decision(m: dict) -> None:
    dec = m["decision"]
    v, t = m["validation"], m["test_after_decision"]
    body = (f'<div style="display:flex;gap:28px;flex-wrap:wrap;margin-top:6px">'
            f'<div><div class="nx-kicker">validation 2017-18: vision + NLP vs current six blocks</div>{_ci(v["MM_vs_BASE"])}</div>'
            f'<div><div class="nx-kicker">validation: vs placebo (language moved to another stock)</div>{_ci(v["MM_vs_PLACEBO"])}</div>'
            f'<div><div class="nx-kicker">test 2020-22, after the decision</div>{_ci(t["MM_vs_BASE"])}</div></div>')
    UI.frame("E28, pre-registered (D-053)", f"Keep test of the language blocks: {dec}",
             UI.chip("supported" if dec == "KEPT" else "not supported") + body,
             note="Gain = mean six-horizon log loss of the other model minus the vision + NLP model; positive = better. "
                  "Kept only if both validation intervals lie above zero. Month-block bootstrap, 95%. Beating the "
                  "placebo but not the six blocks means the language carries real but redundant information: what it "
                  "says about the stock is already in the charts, indicators and classical text features."
                  + (f" Run {m.get('run_note')}." if m.get("run_note") else ""))


def _ablations(m: dict) -> None:
    ct = LV.config_table(m).sort_values("valid gain vs BASE")
    fig = go.Figure()
    fig.add_bar(y=ct.config, x=ct["valid gain vs BASE"], orientation="h", name="validation 2017-18",
                marker_color=[UI.GOOD if x > 0 else UI.BAD for x in ct["valid gain vs BASE"]],
                hovertext=ct.blocks, hovertemplate="%{y}: %{x:+.5f}<br>%{hovertext}<extra></extra>")
    fig.add_scatter(y=ct.config, x=ct["test gain vs BASE"], mode="markers", name="test 2020-22 (reported)",
                    marker=dict(symbol="diamond", size=10, color=UI.WARN))
    fig.add_vline(x=0, line_color=UI.MUTED, line_width=1)
    fig.update_layout(title="Every configuration against the current six blocks (log-loss gain, positive = better)",
                      xaxis_title="mean six-horizon log-loss gain", legend=dict(y=-0.25, yanchor="top"))
    UI.plot(fig, height=360)


def _expert() -> None:
    meta = LV.expert_meta()
    if meta is None:
        st.info("The SEC expert's build is not on this machine (artifacts/features/sec_impact).")
        return
    d = meta["diagnostics"]
    rows = []
    for k, v in d.items():
        h, per = k.split("_", 1)
        per = {"oof_2003_2015": "out of fold 2003-15", "valid_2017_2018": "validation 2017-18"}.get(per, per.replace("_", " "))
        rows += [{"horizon": h.replace("h", "") + " session(s)", "period": per, "what": w, "AUC": v[key]}
                 for w, key in (("does it move?", "auc_move"), ("up vs down", "auc_up_vs_down"))]
    df = pd.DataFrame(rows)
    fig = go.Figure()
    for (per, what), g in df.groupby(["period", "what"]):
        fig.add_bar(x=g.horizon, y=g.AUC, name=f"{what} ({per})",
                    marker_color=(UI.ACCENT if "move" in what else UI.MUTED) if "valid" in per else
                    (_alpha(UI.ACCENT, .45) if "move" in what else _alpha(UI.MUTED, .45)))
    fig.add_hline(y=0.5, line_dash="dot", line_color=UI.MUTED)
    fig.add_hline(y=0.635, line_dash="dash", line_color=UI.WARN,
                  annotation_text="SEC study, locked test: 0.635", annotation_font_color=UI.WARN)
    fig.update_layout(barmode="group", title="SEC price-impact expert: filings say whether a stock moves, not which way",
                      yaxis=dict(range=[0.45, 0.72], title="AUC"), legend=dict(y=-0.2, yanchor="top"))
    UI.plot(fig, height=360)
    st.caption(f"LightGBM on the SEC study's {meta['n_features']} event features (no year, no timestamps, no accounting family), "
               f"labels T_trade at 1 and 5 sessions; fitted before 2016, trees chosen on 2016; "
               f"before 2016 the predictions are out of fold. Present for {meta['present_share']:.0%} of samples.")


def _explorer(universe: list[str]) -> None:
    st.subheader("Filing explorer: what the language models read, stock by stock")
    tickers = [t for t in universe if t] or ["AAPL"]
    c1, c2 = st.columns([1, 3])
    t = c1.selectbox("Stock", tickers, index=tickers.index("AAPL") if "AAPL" in tickers else 0, key="vn_ticker")
    tl = LV.filing_timeline(t)
    if tl.empty:
        c2.info(f"No filings of {t} in the SEC study's 500-firm coverage (2000-2022 development period).")
        return
    fig = go.Figure()
    for kind, g in tl.groupby("kind"):
        fig.add_scatter(x=g.available_utc, y=g.p_move_h1, mode="markers", name=kind,
                        marker=dict(size=6 + 18 * g.lean_h1.abs().clip(0, .5), color=KIND_COLOR.get(kind, UI.MUTED),
                                    opacity=.75, line=dict(width=0)),
                        customdata=np.stack([g.lean_h1, g.p_move_h5], 1),
                        hovertemplate="%{x|%Y-%m-%d}<br>P(move, 1 session) %{y:.2f}<br>"
                                      "P(up) - P(down) %{customdata[0]:+.2f}<br>P(move, 5 sessions) %{customdata[1]:.2f}"
                                      "<extra>" + kind + "</extra>")
    fig.update_layout(title=f"{t}: the SEC expert's P(price moves in the next session) for every filing",
                      yaxis_title="P(move)", legend=dict(y=-0.2, yanchor="top"))
    with c2:
        UI.plot(fig, height=340)
    st.caption("Marker size = |P(up) - P(down)|. Predictions before 2016 are out of fold; from 2016 the model never saw "
               "the event. Research display only.")
    enc = LV.encoded_filings(t)
    if enc.empty:
        return
    opts = enc.head(40)
    pick = st.selectbox("A filing of this stock, to find the most similar earlier filings of other firms (ChronoBERT)",
                        opts.event_id, format_func=lambda e: f"{opts.set_index('event_id').loc[e, 'kind']} "
                                                             f"{opts.set_index('event_id').loc[e, 'available_utc'].date()}",
                        key="vn_filing")
    sim = LV.similar_filings(pick, k=5)
    if sim.empty:
        st.info("No earlier filing to compare with.")
        return
    texts = LV.filing_text([pick] + sim.event_id.tolist(), chars=420)
    st.markdown(f'<div class="nx-frame"><div class="nx-kicker">selected filing</div>'
                f'<div style="font-size:13px">{html.escape(texts.get(pick, "(text not on this machine)"))}</div></div>',
                unsafe_allow_html=True)
    cols = st.columns(len(sim))
    for c, r in zip(cols, sim.itertuples()):
        with c:
            pm = "" if pd.isna(getattr(r, "p_move_h1", np.nan)) else f" | P(move) {r.p_move_h1:.2f}"
            st.markdown(f'<div class="nx-frame"><div class="nx-kicker">{html.escape(r.ticker)} '
                        f'{r.available_utc.date()} | cosine {r.cosine:.3f}{pm}</div>'
                        f'<div style="font-size:12px">{html.escape(texts.get(r.event_id, "")[:300])}</div></div>',
                        unsafe_allow_html=True)
    st.caption("Only filings public before the selected one are searched (point in time). ChronoBERT's 2010 text cutoff "
               "means it never read a filing from the validation or test years. One filing per firm is shown. The SEC study "
               "masked firm names where it recognised them ([FIRM]); some names remain.")


def _analogue_box() -> None:
    st.subheader("Read any filing or news text with ChronoBERT")
    st.caption("Paste the text of a new 8-K, an earnings release or a news story. It is cleaned and encoded exactly as "
               "the SEC filings were, then compared with every encoded filing. Descriptive evidence only: no "
               "probability or plan changes.")
    txt = st.text_area("Text", height=140, key="vn_text",
                       placeholder="Item 2.02 Results of Operations and Financial Condition. On ... the Company reported ...")
    if st.button("Find historical analogues", key="vn_go") and txt.strip():
        with st.spinner("Encoding with ChronoBERT (CPU)..."):
            vec = LV.encode_text(txt)
            an, summ = LV.analogues(vec, k=12)
        if an.empty:
            st.warning("Nothing to compare with: the encoded filings are not on this machine, or the text was empty.")
            return
        c = st.columns(4)
        c[0].metric("mean cosine of the 12 nearest (one per firm)", f"{summ['mean_cosine']:.3f}")
        if summ.get("expert_p_move_mean") is not None:
            c[1].metric("SEC expert P(move), analogues", f"{summ['expert_p_move_mean']:.2f}")
        if summ.get("share_moved") is not None:
            c[2].metric(f"moved next session ({summ['analogues_with_outcome']} with outcome)", f"{summ['share_moved']:.0%}")
        if summ.get("base_rate_moved") is not None:
            c[3].metric("base rate, all labelled filings", f"{summ['base_rate_moved']:.0%}")
        show = an.assign(date=an.available_utc.dt.date, kind=an.event_type.map(
            {"8K": "8-K", "periodic_10Q": "10-Q", "periodic_10K": "10-K"})).drop(columns=["available_utc", "event_type"])
        st.dataframe(show, width="stretch", hide_index=True)
        st.caption("Outcomes are shown only for filings whose label window ended before 2019 (the label-read limit). "
                   "Twelve analogues are a small sample: compare the share with the base rate before reading anything "
                   "into it.")


def render(universe: list[str]) -> None:
    st.header("Vision + NLP: one network that reads charts and language")
    st.caption("Charts (three timeframes through a frozen ConvNeXt), 80 indicators, company news (TF-IDF and FinBERT) "
               "and SEC filings (TF-IDF, ChronoBERT and the SEC price-impact expert) enter one gated fusion network.")
    m = LV.e28()
    if m is None:
        st.info("The documented E28 run is not in this copy of the repository yet.")
    else:
        _decision(m)
        gt = LV.gate_table(m)
        if not gt.empty:
            c1, c2 = st.columns([3, 2])
            with c1:
                UI.plot(_sankey(gt), height=420)
            with c2:
                share = gt.groupby("group").gate.sum()
                fig = go.Figure(go.Pie(labels=share.index, values=share.to_numpy(), hole=.55, sort=False,
                                       marker=dict(colors=[GROUP_COLOR[g] for g in share.index])))
                fig.update_layout(title="Gate share by modality", showlegend=True, legend=dict(y=-0.1, yanchor="top"))
                UI.plot(fig, height=420)
            st.caption("Gates are a model diagnostic: what the network reads, not why prices move.")
        _ablations(m)
        ric = m.get("rank_ic", {})
        if ric:
            st.dataframe(pd.DataFrame({k: {"BASE": v["BASE"]["estimate"], "vision + NLP": v["MM"]["estimate"],
                                           "difference": v["MM_minus_BASE"]["estimate"],
                                           "95% low": v["MM_minus_BASE"]["ci95"][0], "95% high": v["MM_minus_BASE"]["ci95"][1]}
                                       for k, v in ric.items()}).T.rename(index=lambda k: k.replace("valid_", "valid 2017-18, ")
                                                                          .replace("test_", "test 2020-22, "))
                         .style.format("{:+.4f}"), width="stretch")
            st.caption("Daily cross-sectional rank IC of P(up) - P(down): reported, decides nothing.")
    _expert()
    _explorer(universe)
    _analogue_box()

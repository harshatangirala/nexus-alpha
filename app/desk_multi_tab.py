"""Desk 1-5: one to five stocks at once, each with its own chart screenshots (upload or paste) and 1-7 news articles,
plus an optional SEC filing per stock and shared macro texts. Every text carries its date, time and time zone.

What each input does (stated on the page as well):
* **bars**: the frozen v1.1 model renders its own weekly, daily and hourly charts from real bars (the images it was
  trained on). Screenshots are checked against those bars through their legend values; the model never reads pixels.
* **company news**: the model's news branch reads every article public before the decision time (09:00 New York, the
  session after the as-of close) and inside the 5-day window it was trained with; up to 7 after de-duplication.
* **SEC filing and macro texts**: timed and placed in the chronology, scored for tone and (optionally) matched to
  similar historical filings by ChronoBERT. They are context: the frozen models read SEC and macro data from EDGAR and
  ALFRED vintages, point in time, not from pasted text.
* **trade plan**: the deterministic risk engine (entry, stop-loss, targets, size, time exit) with its reasons.
Display only: nothing here changes a model.
"""
from __future__ import annotations

import datetime as dt
import html

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import skin as SK

import paste_image as PI
import ui as UI
from nexus_alpha import text_time as TX
from nexus_alpha import trade_desk as TD
from nexus_alpha.llm import gemini as GM
from nexus_alpha.trading.plan import RiskConfig

MAX_STOCKS, MAX_NEWS, MAX_MACRO = 5, 7, 5
DEFAULT = ["AAPL", "MSFT", "JPM"]
SLOTS = (("W", "weekly chart (1W)"), ("D", "daily chart (1D)"), ("H", "hourly chart (1H)"))
COLORS = [UI.ACCENT, UI.WARN, UI.ACCENT2, UI.GOOD, "#E58BD8"]
KIND_COLOR = {"company news": UI.ACCENT, "SEC filing": UI.ACCENT2, "macro / market": UI.WARN}
DECISION_COLOR = {"LONG": UI.GOOD, "SHORT": UI.BAD, "NO_TRADE": UI.MUTED, "NO_SIGNAL": UI.MUTED}
H6 = ["1D", "2D", "5D", "10D", "20D", "1M"]


def _k(t: str, name: str) -> str:
    return f"dk_{t}_{name}"


# ------------------------------------------------------------------------------------------------ inputs
def _default(key: str, **kw) -> dict:
    """A widget's default only when Session State does not already hold its value (set by the headline fetcher)."""
    return {} if key in st.session_state else kw


def _time_fields(prefix: str, tz_default: str, label: str = "Published") -> tuple:
    c = st.columns([1.1, 0.9, 0.7, 1.3])
    k = {s: f"{prefix}_{s}" for s in ("date", "time", "known", "tz")}
    day = c[0].date_input(f"{label} date", key=k["date"], **_default(k["date"], value=dt.date.today()))
    known = c[2].checkbox("time known", key=k["known"], **_default(k["known"], value=True),
                          help="Untick if you only know the date: the text then counts as public at the end of that "
                               "day (US Pacific time), the rule used in training.")
    clock = c[1].time_input("time", key=k["time"], disabled=not known, step=60, **_default(k["time"], value=dt.time(9, 0)))
    tzs = list(TX.TIMEZONES)
    tz = c[3].selectbox("time zone", tzs, key=k["tz"],
                        **_default(k["tz"], index=tzs.index(tz_default) if tz_default in tzs else 0))
    return day, clock if known else None, tz, known


def _fetch_headlines(t: str) -> None:
    """Yahoo Finance headlines (UTC times) written into this stock's article fields."""
    from nexus_alpha import trade_tools as TT
    try:
        items = TT.yahoo_headlines(t, MAX_NEWS)
    except Exception as e:  # noqa: BLE001  (network)
        st.session_state[_k(t, "fetch_err")] = f"{type(e).__name__}: {e}"
        return
    st.session_state[_k(t, "fetch_err")] = None
    for i, x in enumerate(items[:MAX_NEWS]):
        ts = pd.Timestamp(x["published_utc"]) if x.get("published_utc") else None
        st.session_state[_k(t, f"n{i}_head")] = x.get("headline", "")
        st.session_state[_k(t, f"n{i}_body")] = x.get("text", "")
        st.session_state[_k(t, f"n{i}_src")] = x.get("source", "Yahoo Finance")
        if ts is not None:
            st.session_state[_k(t, f"n{i}_date")] = ts.date()
            st.session_state[_k(t, f"n{i}_time")] = ts.time().replace(second=0, microsecond=0)
            st.session_state[_k(t, f"n{i}_tz")] = "UTC"
            st.session_state[_k(t, f"n{i}_known")] = True
    st.session_state[_k(t, "n")] = max(1, min(MAX_NEWS, len(items)))


def _stock_inputs(t: str, tz_default: str) -> dict:
    st.markdown(f"**{t}: chart screenshots** - upload a file or click a box and paste (Ctrl+V). TriFrame's weekly, "
                "daily and hourly views fit the three slots.")
    shots, legend = {}, {}
    cols = st.columns(3)
    for col, (s, lab) in zip(cols, SLOTS):
        with col:
            up = st.file_uploader(lab, type=["png", "jpg", "jpeg", "webp"], key=_k(t, f"up_{s}"))
            pasted = PI.paste_image(lab, key=_k(t, f"paste_{s}"))
            if up is not None:
                shots[s] = (up.getvalue(), up.type or "image/png", "uploaded")
            elif pasted is not None:
                shots[s] = (pasted[0], pasted[1], "pasted")
            if s in shots:
                st.image(shots[s][0], caption=f"{lab} ({shots[s][2]})", width="stretch")
    with st.expander("Legend values on the screenshots (optional: used to check them against real bars)"):
        for s, lab in SLOTS:
            cc = st.columns(4)
            legend[s] = {f: cc[i].number_input(f"{s} {f}", value=0.0, format="%.4f", key=_k(t, f"lg_{s}_{f}"))
                         for i, f in enumerate(("open", "high", "low", "close"))}
    st.markdown(f"**{t}: company news** (1-7 articles; each with its date, time and time zone)")
    c = st.columns([1, 1.4, 3])
    n = c[0].number_input("articles", 0, MAX_NEWS, key=_k(t, "n"), **_default(_k(t, "n"), value=1))
    c[1].button("Fetch Yahoo headlines", key=_k(t, "fetch"), on_click=_fetch_headlines, args=(t,),
                help="Fills the article fields with this stock's latest Yahoo Finance headlines and their UTC times.")
    if st.session_state.get(_k(t, "fetch_err")):
        c[2].caption(f"Headline fetch failed: {st.session_state[_k(t, 'fetch_err')]}")
    news = []
    for i in range(int(n)):
        with st.expander(f"{t} article {i + 1}", expanded=i == 0):
            head = st.text_input("Headline", key=_k(t, f"n{i}_head"))
            body = st.text_area("Text (optional)", key=_k(t, f"n{i}_body"), height=80)
            day, clock, tz, known = _time_fields(_k(t, f"n{i}"), tz_default)
            src = st.text_input("Source", key=_k(t, f"n{i}_src"))
            if head.strip() or body.strip():
                news.append({"kind": "news", "ticker": t, "headline": head.strip(), "text": body.strip(),
                             "published_utc": TX.to_utc(day, clock, tz), "time_known": known, "tz_label": tz,
                             "source": src})
    sec = None
    if st.checkbox(f"Add an SEC filing for {t} (8-K, 10-Q, 10-K)", key=_k(t, "sec_on")):
        c = st.columns([1, 3])
        form = c[0].selectbox("Form", ["8-K", "10-Q", "10-K", "other"], key=_k(t, "sec_form"))
        title = c[1].text_input("Title / item (e.g. Item 2.02 Results of Operations)", key=_k(t, "sec_title"))
        body = st.text_area("Filing text (the main section is enough)", key=_k(t, "sec_body"), height=110)
        day, clock, tz, known = _time_fields(_k(t, "sec"), tz_default, label="Accepted by EDGAR")
        if title.strip() or body.strip():
            sec = {"kind": "sec", "ticker": t, "headline": f"{form}: {title.strip()}".strip(": "), "text": body.strip(),
                   "published_utc": TX.to_utc(day, clock, tz), "time_known": known, "tz_label": tz, "source": "EDGAR",
                   "form": form}
    return {"shots": shots, "legend": legend, "news": news, "sec": sec}


def _macro_inputs(tz_default: str) -> list[dict]:
    st.markdown("**Macro and market texts shared by every stock** (a CPI release, a Fed statement, an earnings-season "
                "note...). Timed and scored for tone; context only, not model inputs.")
    n = st.number_input("macro texts", 0, MAX_MACRO, 0, key="dk_macro_n")
    out = []
    for i in range(int(n)):
        with st.expander(f"Macro text {i + 1}", expanded=i == 0):
            head = st.text_input("Headline", key=f"dk_macro{i}_head")
            body = st.text_area("Text", key=f"dk_macro{i}_body", height=80)
            day, clock, tz, known = _time_fields(f"dk_macro{i}", tz_default)
            src = st.text_input("Source", key=f"dk_macro{i}_src")
            if head.strip() or body.strip():
                out.append({"kind": "macro", "ticker": "market", "headline": head.strip(), "text": body.strip(),
                            "published_utc": TX.to_utc(day, clock, tz), "time_known": known, "tz_label": tz,
                            "source": src})
    return out


# ------------------------------------------------------------------------------------------------ analysis
def _analogues(text: str) -> dict | None:
    from nexus_alpha import language_view as LV
    vec = LV.encode_text(text)
    if vec is None:
        return None
    tab, summ = LV.analogues(vec, k=12)
    return {"table": tab, "summary": summ}


def _analyse(predictor, picks, inputs, macro, mode, asof, choice, risk, use_llm, use_analogues) -> dict:
    from trade_desk_tab import _bars_for_check
    out, bar = {}, st.progress(0.0, text="starting")
    for i, t in enumerate(picks):
        bar.progress(i / len(picks), text=f"{t}: rendering charts from real bars, embedding, scoring")
        inp = inputs[t]
        arts = [{"headline": a["headline"], "text": a["text"], "time_known": a["time_known"], "source": a["source"],
                 "published_utc": None if a["published_utc"] is None else a["published_utc"].isoformat()}
                for a in inp["news"]]
        rec = {"inputs": inp}
        try:
            res = predictor.predict(t, asof=str(asof) if mode == "historical" else None, mode=mode, articles=arts,
                                    risk=risk, explain=False)
        except Exception as e:  # noqa: BLE001  (one failed stock must not stop the others)
            out[t] = rec | {"error": f"{type(e).__name__}: {e}"}
            continue
        rec["res"] = res
        decision = pd.Timestamp(res.instrument.forecast_timestamp)
        decision = decision.tz_localize("UTC") if decision.tzinfo is None else decision.tz_convert("UTC")
        items = inp["news"] + ([inp["sec"]] if inp["sec"] else []) + macro
        ch = TX.chronology(items, decision, res.instrument.asof_session)
        listing = res.evidence.get("news") or []
        for j, it in enumerate(listing[:len(inp["news"])]):       # the model's own verdict on each article
            ch.loc[j, "status"] = ("model input (news branch)" if it.get("used") else f"excluded: {it.get('reason')}")
        rec["chrono"], rec["decision_utc"] = ch, decision
        try:
            rec["bars"] = _bars_for_check(t, mode, asof)
        except Exception as e:  # noqa: BLE001  (the chart and the screenshot check are optional)
            rec["bars_err"] = f"{type(e).__name__}: {e}"
        if use_llm and GM.available() and (items or inp["shots"]):
            try:
                texts = [{"headline": x["headline"], "text": x["text"], "source": x.get("source", ""),
                          "published_utc": "" if x["published_utc"] is None else str(x["published_utc"])} for x in items]
                images = [(s, v[0], v[1]) for s, v in inp["shots"].items()]
                rec["llm"] = GM.analyze(t, choice.label, res.instrument.forecast_timestamp, texts, images)
            except Exception as e:  # noqa: BLE001
                rec["llm_err"] = f"{type(e).__name__}: {e}"
        if use_analogues and inp["sec"] and inp["sec"]["text"]:
            bar.progress((i + .7) / len(picks), text=f"{t}: ChronoBERT reads the filing (CPU)")
            try:
                rec["analogues"] = _analogues(inp["sec"]["headline"] + "\n" + inp["sec"]["text"])
            except Exception as e:  # noqa: BLE001
                rec["analogues_err"] = f"{type(e).__name__}: {e}"
        out[t] = rec
    bar.progress(1.0, text="done")
    return out


# ------------------------------------------------------------------------------------------------ results
def nz(v):
    """None -> NaN, so tables show '-' for the levels of a NO_TRADE plan."""
    return np.nan if v is None else v


def _lean(f) -> float:
    return float((f.up_probability or 0) - (f.down_probability or 0))


def _skill(f) -> str:
    """Gate 3 in words (D-057): the 2017-18 flag and whether it replicated on 2020-22."""
    if not f.direction_skill_validated:
        return "not validated"
    if f.direction_skill_replicated is None:                 # a research bundle: never checked out of sample
        return "validated 2017-18 (not checked out of sample)"
    return "validated and replicated" if f.direction_skill_replicated else "validated 2017-18, not replicated 2020-22"


def _card(t: str, i: int, rec: dict, h: str | None) -> str:
    color = COLORS[i % len(COLORS)]
    if "error" in rec:
        return (f'<div class="nx-frame"><div class="nx-kicker" style="color:{color}">{html.escape(t)}</div>'
                f'<div class="nx-note">{html.escape(rec["error"])}</div></div>')
    res = rec["res"]
    hh = h or "5D"
    f, p = res.forecasts[hh], res.trade_plans[hh]
    dec_c = DECISION_COLOR.get(p.decision, UI.MUTED)
    ch = rec.get("chrono", pd.DataFrame())
    used = int((ch.status == "model input (news branch)").sum()) if len(ch) else 0
    n_news = len(rec["inputs"]["news"])
    rows = [("as-of session", res.instrument.asof_session), ("reference close", f"{res.instrument.reference_price:,.2f}"),
            (f"P(up) / P(side) / P(down), {hh}",
             f"{f.up_probability:.2f} / {f.sideways_probability:.2f} / {f.down_probability:.2f}"),
            ("lean (up - down)", f"{_lean(f):+.3f}"),
            ("direction skill at this horizon", _skill(f)),
            ("return range q10..q90", "-" if f.return_q10 is None else f"{f.return_q10:+.1%} .. {f.return_q90:+.1%}"),
            ("news read by the model", f"{used} of {n_news}"), ("screenshots", f"{len(rec['inputs']['shots'])} of 3")]
    if p.decision in ("LONG", "SHORT"):
        rows += [("entry (next open)", f"{p.entry_reference:,.2f}"), ("stop-loss", f"{p.stop:,.2f}"),
                 ("target 1 / 2", f"{p.target_1:,.2f} / {p.target_2:,.2f}" if p.target_2 else f"{p.target_1:,.2f}"),
                 ("net reward : risk", f"{p.net_rr:.2f}"), ("time exit", f"{p.holding_sessions} sessions")]
    body = "".join(f'<div class="nx-kv"><span>{html.escape(a)}</span><span>{html.escape(str(b))}</span></div>' for a, b in rows)
    why = html.escape((p.reasons or [""])[0])
    return (f'<div class="nx-frame"><div style="height:3px;background:{color};margin:-14px -16px 10px"></div>'
            f'<div class="nx-kicker">{html.escape(t)} &middot; {hh}</div>'
            f'<div class="nx-title">{UI.tag(p.decision, dec_c)}</div>{body}'
            f'<div class="nx-note" style="margin-top:8px">{why}</div></div>')


def _prob_chart(out: dict, h: str) -> None:
    ok = [t for t, r in out.items() if "res" in r]
    fig = go.Figure()
    for cls, col, attr in (("DOWN", UI.BAD, "down_probability"), ("SIDEWAYS", "#3A4F66", "sideways_probability"),
                           ("UP", UI.GOOD, "up_probability")):
        fig.add_trace(go.Bar(y=ok, x=[getattr(out[t]["res"].forecasts[h], attr) for t in ok], name=cls, orientation="h",
                             marker_color=col, hovertemplate=f"%{{y}} {cls} %{{x:.1%}}<extra></extra>"))
    fig.update_layout(barmode="stack", title=f"Class probabilities, {h}", xaxis=dict(tickformat=".0%", range=[0, 1]),
                      yaxis=dict(autorange="reversed"), legend=dict(orientation="h", y=-0.2))
    UI.plot(fig, height=110 + 48 * len(ok))


def _lean_heatmap(out: dict) -> None:
    ok = [t for t, r in out.items() if "res" in r]
    z = np.array([[_lean(out[t]["res"].forecasts[h]) for h in H6] for t in ok])
    fc = {t: out[t]["res"].forecasts for t in ok}           # the second * needs both (a replicated-only 2D is not starred)
    sk = [[("*" if fc[t][h].direction_skill_validated else "")
           + ("*" if fc[t][h].direction_skill_validated and fc[t][h].direction_skill_replicated else "")
           for h in H6] for t in ok]
    fig = go.Figure(go.Heatmap(z=z, x=H6, y=ok, colorscale=[[0, UI.BAD], [0.5, "#0F1D2C"], [1, UI.GOOD]], zmid=0,
                               zmin=-0.15, zmax=0.15, text=[[f"{v:+.3f}{s}" for v, s in zip(r, q)] for r, q in zip(z, sk)],
                               texttemplate="%{text}", hovertemplate="%{y} %{x}: lean %{z:+.3f}<extra></extra>"))
    fig.update_layout(title="Lean P(up) - P(down) by horizon (* validated 2017-18, ** also replicated 2020-22)",
                      yaxis=dict(autorange="reversed"))
    UI.plot(fig, height=110 + 48 * len(ok))


def _range_chart(out: dict, h: str) -> None:
    ok = [t for t, r in out.items() if "res" in r and r["res"].forecasts[h].return_q10 is not None]
    if not ok:
        st.caption(f"No validated return range at {h}.")
        return
    fig = go.Figure()
    for i, t in enumerate(ok):
        f = out[t]["res"].forecasts[h]
        fig.add_trace(go.Scatter(x=[f.return_q10, f.expected_return, f.return_q90], y=[t] * 3, mode="lines+markers",
                                 line=dict(color=COLORS[i % len(COLORS)], width=6), marker=dict(size=[6, 11, 6]),
                                 name=t, hovertemplate=f"{t}: %{{x:+.2%}}<extra></extra>"))
    fig.add_vline(x=0, line=dict(color=UI.MUTED, width=1, dash="dot"))
    fig.update_layout(title=f"Return range q10 / median / q90, {h}",
                      xaxis=dict(tickformat="+.0%"), showlegend=False, yaxis=dict(autorange="reversed"))
    UI.plot(fig, height=110 + 48 * len(ok))


def _timeline(out: dict) -> pd.DataFrame:
    frames = [r["chrono"] for r in out.values() if "chrono" in r and len(r["chrono"])]
    if not frames:
        st.info("No dated texts were supplied.")
        return pd.DataFrame()
    ch = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["ticker", "kind", "headline", "published (UTC)"])
    ref = next(r for r in out.values() if "decision_utc" in r)
    dec = ref["decision_utc"]
    close = TX.asof_close_utc(ref["res"].instrument.asof_session)
    ch["x"] = -ch["hours before decision"]
    fig = go.Figure()
    fig.add_vrect(x0=-24 * 5, x1=0, fillcolor="rgba(63,208,224,0.06)", line_width=0,
                  annotation_text="5-day news window", annotation_position="top left")
    fig.add_vline(x=0, line=dict(color=UI.GOOD, width=2), annotation_text="decision 09:00 NY",
                  annotation_position="top right")
    fig.add_vline(x=-(dec - close).total_seconds() / 3600, line=dict(color=UI.WARN, width=1, dash="dash"),
                  annotation_text="as-of close", annotation_position="top left")
    sym = {"model input (news branch)": "circle", "context (timed, not a model input)": "diamond"}
    for kind, col in KIND_COLOR.items():
        d = ch[(ch.kind == kind) & ch.x.notna()]
        if not len(d):
            continue
        fig.add_trace(go.Scatter(
            x=d.x, y=d.ticker, mode="markers", name=kind,
            marker=dict(color=col, size=13, symbol=[sym.get(s, "x") for s in d.status], line=dict(color=UI.BG, width=1)),
            customdata=np.stack([d.headline.str[:90], d["published (UTC)"].astype(str), d["available (UTC)"].astype(str),
                                 d.status, d["net tone (LM)"].map(lambda v: "-" if pd.isna(v) else f"{v:+.2f}")], 1),
            hovertemplate="<b>%{customdata[0]}</b><br>published %{customdata[1]}<br>available %{customdata[2]}"
                          "<br>%{customdata[3]}<br>net tone %{customdata[4]}<extra></extra>"))
    lo = float(np.nanmin(ch.x)) if ch.x.notna().any() else -120
    fig.update_layout(title="When each text became public, relative to the decision (circle = read by the model, "
                            "diamond = context, x = excluded)",
                      xaxis=dict(title="hours relative to the decision time (negative = before)",
                                 range=[min(lo - 6, -130), max(12, float(np.nanmax(ch.x)) + 6 if ch.x.notna().any() else 12)]),
                      yaxis=dict(autorange="reversed"), legend=dict(orientation="h", y=-0.25))
    UI.plot(fig, height=170 + 46 * ch.ticker.nunique())
    return ch


def _detail(t: str, rec: dict, h: str | None, risk) -> None:
    import plan_view as PV
    if "error" in rec:
        st.error(f"{t}: {rec['error']}")
        return
    res = rec["res"]
    if h is not None:
        bars = rec.get("bars")
        PV.explainer(res, h, risk, bars["D"] if bars else None, key=f"dk_{t}")
    else:
        st.info("No frozen model covers this horizon; see the probabilities above for every horizon.")
    st.markdown("**Texts in time order**")
    ch = rec.get("chrono", pd.DataFrame())
    if len(ch):
        show = ch.drop(columns=["x"], errors="ignore").sort_values("available (UTC)")
        st.dataframe(show.style.format({"hours before decision": "{:,.1f}", "net tone (LM)": "{:+.2f}"}, na_rep="-"),
                     width="stretch", hide_index=True)
    else:
        st.caption("No texts for this stock.")
    shots = rec["inputs"]["shots"]
    st.markdown("**Screenshots checked against real bars** (the model never reads pixels)")
    if not shots:
        st.caption("No screenshots for this stock.")
    else:
        cols = st.columns(len(shots))
        llm = rec.get("llm") or {}
        for col, (s, (data, _, how)) in zip(cols, shots.items()):
            col.image(data, caption=f"{s} screenshot ({how})", width="stretch")
            legend, src = {x: v for x, v in rec["inputs"]["legend"].get(s, {}).items() if v}, "typed"
            r = next((c for c in llm.get("charts", []) if c.get("slot") == s), None)
            if not legend and r and any(r.get(f"legend_{x}") for x in ("open", "high", "low", "close")):
                legend, src = {x: r.get(f"legend_{x}") for x in ("open", "high", "low", "close") if r.get(f"legend_{x}")}, "read by Gemini"
            if rec.get("bars") is None:
                col.caption(rec.get("bars_err", "bars unavailable"))
            elif legend:
                chk = TD.check_legend(rec["bars"][s], legend, tolerance_pct=0.5 if s == "H" else 0.3)
                col.write(f"Legend ({src}): {legend}")
                col.write(chk)
            else:
                col.caption("No legend values: type them in the inputs (or enable Gemini) to check this screenshot.")
    if rec.get("analogues"):
        a = rec["analogues"]
        s = a["summary"]
        st.markdown("**Historical filings most like the pasted SEC filing** (ChronoBERT, one per firm; descriptive)")
        moved = s.get("share_moved")
        st.write(f"{s['analogues_with_outcome']} of {s['k']} analogues have a known next-session outcome; "
                 + (f"{moved:.0%} of them moved (UP or DOWN) against a base rate of {s['base_rate_moved']:.0%}."
                    if moved is not None else "none has a known outcome."))
        st.dataframe(a["table"], width="stretch", hide_index=True)
    elif rec.get("analogues_err"):
        st.caption(f"ChronoBERT analogues unavailable: {rec['analogues_err']}")
    if rec.get("llm"):
        o = rec["llm"].get("overall", {})
        st.markdown(f"**Gemini read** (live LLM, unvalidated): lean **{o.get('direction_lean')}**, confidence "
                    f"{o.get('confidence')}. {o.get('summary', '')}")
    elif rec.get("llm_err"):
        st.caption(f"Gemini call failed: {rec['llm_err']}")


# ------------------------------------------------------------------------------------------------ page
def render(predictor, universe: list[str], risk_defaults: dict) -> None:
    SK.banner("Desk 1-5", "Your screenshots and texts, one to five stocks", "Each stock gets a card, a chronology and a full read.")
    UI.frame("how this desk reads your inputs", "Bars decide; texts are timed; nothing reads the future",
             '<div class="nx-note" style="font-size:.88rem">The frozen v1.1 model renders its own weekly, daily and '
             "hourly charts from real bars and reads every company-news article that was public before the decision "
             "time (09:00 New York, the session after the as-of close) and inside the 5-day window it was trained "
             "with. Screenshots are checked against real bars; SEC filings and macro texts are placed in the "
             "chronology and scored for tone as context. Each text needs its date, time and time zone: they decide "
             "whether the model may read it.</div>")
    if predictor is None:
        st.error("No inference bundle loaded.")
        return
    c = st.columns([3, 1.1, 1.3, 1.6])
    default = [t for t in DEFAULT if t in universe][:MAX_STOCKS]
    picks = c[0].multiselect("Stocks (1 to 5)", universe, default=default, max_selections=MAX_STOCKS, key="dk_picks")
    mode = c[1].radio("Data", ["live", "historical"], horizontal=True, key="dk_mode")
    asof = c[2].date_input("As-of session (historical)", value=pd.Timestamp("2021-06-15"),
                           min_value=pd.Timestamp("2004-01-02"), max_value=pd.Timestamp("2022-12-29"), key="dk_asof",
                           disabled=mode == "live")
    tzs = list(TX.TIMEZONES)
    tz_default = c[3].selectbox("Your time zone (default for every text)", tzs, index=0, key="dk_tz")
    c = st.columns(3)
    trade = c[0].radio("Trade type", ["Intraday", "Swing", "Long-term"], index=1, horizontal=True, key="dk_trade")
    swing = c[1].slider("Swing: trading days in the trade", 2, 20, 5, key="dk_swing")
    lt = c[2].select_slider("Long-term horizon", options=list(TD.LONG_TERM), value="4 weeks", key="dk_lt")
    if not picks:
        st.info("Pick between one and five stocks.")
        return
    choice = TD.choose_horizon(trade, swing_days=swing, long_term=lt)
    st.caption(f"**{choice.label}**: model horizon {choice.served_by or 'none'} ({choice.note}).")

    tabs = st.tabs([f"{t} inputs" for t in picks] + ["Macro and market (shared)"])
    inputs = {}
    for tab, t in zip(tabs, picks):
        with tab:
            inputs[t] = _stock_inputs(t, tz_default)
    with tabs[-1]:
        macro = _macro_inputs(tz_default)

    c = st.columns([1.4, 2, 2.4])
    run = c[0].button(f"Analyse {len(picks)} stock{'s' if len(picks) > 1 else ''}", type="primary", key="dk_run")
    use_llm = c[1].checkbox("Gemini read of texts and screenshots (1 request per stock)", value=False,
                            disabled=not GM.available(), key="dk_llm",
                            help=None if GM.available() else "Set GEMINI_API_KEY to enable.")
    use_an = c[2].checkbox("ChronoBERT: historical filings like each pasted SEC filing (CPU, about a minute the first "
                           "time)", value=False, key="dk_an")
    risk = RiskConfig(**risk_defaults)
    if run:
        out = _analyse(predictor, picks, inputs, macro, mode, asof, choice, risk, use_llm, use_an)
        st.session_state["dk_result"] = {"picks": tuple(picks), "h": choice.served_by, "out": out}
    got = st.session_state.get("dk_result")
    if not got or got["picks"] != tuple(picks):
        return
    out, h = got["out"], got["h"]
    hh = h or "5D"

    SK.section("1. The board")
    cols = st.columns(len(out))
    for i, (col, (t, rec)) in enumerate(zip(cols, out.items())):
        with col:
            st.markdown(_card(t, i, rec, h), unsafe_allow_html=True)
    st.caption("'Validated' is the frozen v1.1 flag, set on the 2017-18 validation years. On the 2020-22 test years the "
               "up-vs-down AUROC was 0.49-0.51 at every horizon (notebook sections 12 and 15): the direction lean is "
               "weak evidence. What held out of sample is whether a stock moves (sideways AUROC 0.58-0.61) and the "
               "cross-sectional ranking (rank IC 0.02-0.03).")
    if any("res" in r for r in out.values()):
        SK.section("2. Side by side")
        left, right = st.columns(2)
        with left:
            _prob_chart(out, hh)
        with right:
            _range_chart(out, hh)
        _lean_heatmap(out)
        rows = []
        for t, r in out.items():
            if "res" not in r:
                continue
            p = r["res"].trade_plans[hh]
            rows.append({"ticker": t, "decision": p.decision, "setup": p.setup_family or "-", "entry": nz(p.entry_reference),
                         "stop-loss": nz(p.stop), "target 1": nz(p.target_1), "target 2": nz(p.target_2),
                         "net R:R": nz(p.net_rr), "quantity": nz(p.quantity), "time exit (sessions)": nz(p.holding_sessions),
                         "first reason": (p.reasons or [""])[0]})
        st.dataframe(pd.DataFrame(rows).set_index("ticker").style.format(na_rep="-").format(
            {c_: "{:,.2f}" for c_ in ("entry", "stop-loss", "target 1", "target 2", "net R:R")}
            | {"quantity": "{:,.0f}", "time exit (sessions)": "{:.0f}"}, na_rep="-"), width="stretch")
        st.caption("Plans come from the deterministic risk engine: direction skill validated on 2017-18 and replicated "
                   "on 2020-22 (D-057: so far no horizon of v1.1 qualifies), an edge of at least 0.10, "
                   "weekly permission, a daily setup, an hourly trigger, a stop beyond the setup's invalidation and a "
                   "net reward-to-risk of at least 1.5. NO_TRADE lists the first gate that failed.")
        SK.section("3. Chronology: every text against the decision time")
        ch = _timeline(out)
        if len(ch):
            n_in = int((ch.status == "model input (news branch)").sum())
            n_ex = int(ch.status.str.startswith("excluded").sum())
            st.caption(f"{len(ch)} texts: {n_in} read by the model, {n_ex} excluded (after the decision time, older "
                       "than the 5-day window, or undated), the rest context. Times you typed in your zone are converted "
                       "to UTC; a date without a time counts from the end of that day (US Pacific), as in training.")
    SK.section("4. Each stock in detail")
    dtabs = st.tabs(list(out))
    for tab, (t, rec) in zip(dtabs, out.items()):
        with tab:
            _detail(t, rec, h, risk)
    st.caption("Research and paper-trading decision support only; not investment advice; no orders are placed.")

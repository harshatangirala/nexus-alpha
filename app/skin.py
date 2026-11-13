"""Second visual layer of the NEXUS dashboard: page skin, folding, and reusable visual components.

Design references (see THIRD_PARTY_NOTICES.md; no source copied, behaviour re-written in CSS/SVG/JS):
* Arwes (arwes/arwes, MIT): dot-grid background, thin luminous lines, framed panels.
* Magic UI (magicuidesign/magicui, MIT): BorderBeam (a light that travels around a card's edge), OrbitingCircles,
  NumberTicker, shimmering text, animated rings.
* World Monitor (koala73/worldmonitor, AGPL-3.0): layout idea only (a situational-awareness board).

Every component here is display only. Nothing changes a forecast, a probability or the risk engine.
"""
from __future__ import annotations

import html
import math

import streamlit as st
import streamlit.components.v1 as components

import ui as UI

BG, PANEL, BORDER, TEXT, MUTED = UI.BG, UI.PANEL, UI.BORDER, UI.TEXT, UI.MUTED
ACCENT, ACCENT2, GOOD, BAD, WARN = UI.ACCENT, UI.ACCENT2, UI.GOOD, UI.BAD, UI.WARN
FONT_BODY, FONT_HEAD, FONT_MONO = UI.FONT_BODY, UI.FONT_HEAD, UI.FONT_MONO
SIDE = "#6D88A3"            # the "sideways" colour: visible on the dark ground, neutral next to green and red

CSS2 = f"""
<style>
@property --nx-a {{ syntax: '<angle>'; inherits: false; initial-value: 0deg; }}
@keyframes nx-spin {{ to {{ --nx-a: 360deg; }} }}
@keyframes nx-rise {{ from {{ opacity: 0; transform: translateY(8px); }} to {{ opacity: 1; transform: none; }} }}
@keyframes nx-drift {{ from {{ background-position: 0 0, 0 0, 0 0; }} to {{ background-position: 26px 26px, 0 0, 0 0; }} }}
@keyframes nx-ring {{ from {{ stroke-dashoffset: var(--c); }} }}
@keyframes nx-shimmer {{ from {{ background-position: -200% 0; }} to {{ background-position: 200% 0; }} }}

/* page: Arwes-style dot grid with two slow aurora glows */
.stApp {{ background-color: {BG};
  background-image: radial-gradient(rgba(63,208,224,.13) 1px, transparent 1.2px),
    radial-gradient(55% 38% at 12% 0%, rgba(63,208,224,.13), transparent 70%),
    radial-gradient(48% 36% at 92% 4%, rgba(155,140,255,.12), transparent 70%);
  background-size: 26px 26px, auto, auto; background-attachment: fixed; animation: nx-drift 60s linear infinite; }}
[data-testid="stHeader"] {{ background: rgba(5,9,15,.55); backdrop-filter: blur(8px); }}
[data-testid="stSidebar"] {{ background: linear-gradient(180deg, #08111B, #060C14); border-right: 1px solid {BORDER}; }}
.block-container {{ max-width: 1500px; }}
h1, h2, h3 {{ font-family: {FONT_HEAD}; letter-spacing: .01em; }}

/* tabs as pills */
[role=tablist] {{ gap: 4px; border-bottom: 1px solid {BORDER}; }}
button[role=tab] {{ border-radius: 999px 999px 0 0; transition: background .2s, color .2s; }}
button[role=tab]:hover {{ background: rgba(63,208,224,.08); }}
button[role=tab][aria-selected=true] {{ background: linear-gradient(180deg, rgba(63,208,224,.16), transparent); }}

/* expanders: glass panels with an accent edge */
[data-testid="stExpander"] {{ border: 1px solid {BORDER}; border-left: 2px solid {ACCENT}; border-radius: 6px;
  background: linear-gradient(180deg, rgba(15,29,44,.72), rgba(12,22,34,.55)); backdrop-filter: blur(4px);
  margin: 6px 0 10px; transition: border-color .25s, box-shadow .25s; animation: nx-rise .4s ease both; }}
[data-testid="stExpander"]:hover {{ border-color: #2E6475; box-shadow: 0 0 0 1px rgba(63,208,224,.12), 0 6px 22px rgba(0,0,0,.35); }}
[data-testid="stExpander"] summary {{ font-family: {FONT_MONO}; font-size: .78rem; letter-spacing: .06em; color: {MUTED}; }}
[data-testid="stExpander"] summary:hover {{ color: {ACCENT}; }}
[data-testid="stExpander"] details[open] > summary {{ color: {ACCENT}; }}

/* metrics as cards */
[data-testid="stMetric"] {{ background: linear-gradient(160deg, rgba(15,29,44,.85), rgba(12,22,34,.5));
  border: 1px solid {BORDER}; border-radius: 8px; padding: 10px 14px; position: relative; overflow: hidden;
  animation: nx-rise .45s ease both; }}
[data-testid="stMetric"]::after {{ content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 2px;
  background: linear-gradient(180deg, {ACCENT}, {ACCENT2}); }}
[data-testid="stMetricLabel"] p {{ font-family: {FONT_MONO}; font-size: .68rem; letter-spacing: .1em; text-transform: uppercase; color: {MUTED}; }}
[data-testid="stImage"] img {{ border-radius: 8px; border: 1px solid {BORDER}; }}
[data-testid="stPlotlyChart"] {{ background: linear-gradient(180deg, rgba(15,29,44,.45), rgba(12,22,34,.2));
  border: 1px solid {BORDER}; border-radius: 8px; padding: 4px; animation: nx-rise .5s ease both; }}
[data-testid="stDataFrame"] {{ border-radius: 8px; overflow: hidden; }}
.stButton > button, .stDownloadButton > button {{ border-radius: 999px; transition: box-shadow .25s, transform .15s; }}
.stButton > button:hover {{ box-shadow: 0 0 18px rgba(63,208,224,.35); transform: translateY(-1px); }}
::-webkit-scrollbar {{ width: 8px; height: 8px; }} ::-webkit-scrollbar-thumb {{ background: #1F3347; border-radius: 4px; }}

/* section banner */
.nx-banner {{ position: relative; padding: 2px 0 12px; margin: 0 0 14px; animation: nx-rise .4s ease both; }}
.nx-banner::after {{ content: ""; position: absolute; left: 0; right: 0; bottom: 0; height: 1px;
  background: linear-gradient(90deg, {ACCENT}, {ACCENT2} 45%, transparent); }}
.nx-banner .k {{ font-family: {FONT_MONO}; font-size: .7rem; letter-spacing: .18em; color: {ACCENT}; text-transform: uppercase; }}
.nx-banner .t {{ font-family: {FONT_HEAD}; font-size: 1.75rem; font-weight: 600; line-height: 1.15; margin: 2px 0;
  background: linear-gradient(90deg, {TEXT} 35%, {ACCENT}); -webkit-background-clip: text; background-clip: text; color: transparent; }}
.nx-banner .s {{ color: {MUTED}; font-size: .9rem; }}
.nx-sec {{ font-family: {FONT_MONO}; font-size: .72rem; letter-spacing: .16em; text-transform: uppercase; color: {ACCENT};
  margin: 18px 0 6px; display: flex; align-items: center; gap: 10px; }}
.nx-sec::after {{ content: ""; flex: 1; height: 1px; background: linear-gradient(90deg, {BORDER}, transparent); }}

/* tiles with a travelling border beam (Magic UI BorderBeam) */
.nx-tiles {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; margin: 6px 0 12px; }}
.nx-tile {{ position: relative; padding: 12px 14px 10px; border-radius: 10px; border: 1px solid {BORDER};
  background: linear-gradient(160deg, rgba(15,29,44,.9), rgba(10,18,28,.7)); overflow: hidden; animation: nx-rise .5s ease both; }}
.nx-tile::before {{ content: ""; position: absolute; inset: 0; border-radius: inherit; padding: 1px; pointer-events: none;
  background: conic-gradient(from var(--nx-a), transparent 0 80%, var(--tc, {ACCENT}) 92%, transparent 100%);
  -webkit-mask: linear-gradient(#000 0 0) content-box, linear-gradient(#000 0 0); -webkit-mask-composite: xor;
  mask-composite: exclude; animation: nx-spin 7s linear infinite; }}
.nx-tile .l {{ font-family: {FONT_MONO}; font-size: .64rem; letter-spacing: .12em; text-transform: uppercase; color: {MUTED}; }}
.nx-tile .v {{ font-family: {FONT_HEAD}; font-size: 1.75rem; font-weight: 600; line-height: 1.15; color: var(--tc, {TEXT});
  font-variant-numeric: tabular-nums; }}
.nx-tile .s {{ font-size: .76rem; color: {MUTED}; margin-top: 1px; max-width: 60%; }}
.nx-tile svg.sp {{ position: absolute; right: 8px; bottom: 6px; width: 40%; height: 34px; opacity: .9; }}

/* ring gauges */
.nx-rings {{ display: flex; flex-wrap: wrap; gap: 14px; justify-content: center; margin: 4px 0 10px; }}
.nx-ring {{ text-align: center; width: 128px; animation: nx-rise .5s ease both; }}
.nx-ring svg {{ width: 108px; height: 108px; filter: drop-shadow(0 0 8px var(--g, rgba(63,208,224,.25))); }}
.nx-ring circle.arc {{ fill: none; stroke-dasharray: var(--d); stroke-dashoffset: var(--o);
  animation: nx-ring 1.1s cubic-bezier(.2,.7,.2,1) both; transform: rotate(-90deg); transform-origin: 54px 54px; }}
.nx-ring circle.trk {{ fill: none; stroke: #13243A; }}
.nx-ring .cap {{ font-family: {FONT_MONO}; font-size: .7rem; letter-spacing: .1em; color: {MUTED}; text-transform: uppercase; }}
.nx-ring .ctr {{ font-family: {FONT_HEAD}; font-weight: 600; fill: {TEXT}; }}
.nx-ring .sub {{ font-family: {FONT_MONO}; fill: {MUTED}; }}

/* pills, shimmer and idea cards */
.nx-pills {{ display: flex; flex-wrap: wrap; gap: 6px; margin: 4px 0 8px; }}
.nx-pill {{ font-family: {FONT_MONO}; font-size: .72rem; padding: 3px 10px; border-radius: 999px; border: 1px solid currentColor;
  background: color-mix(in srgb, currentColor 9%, transparent); }}
.nx-shim {{ background: linear-gradient(90deg, {MUTED} 30%, {TEXT} 50%, {MUTED} 70%); background-size: 200% 100%;
  -webkit-background-clip: text; background-clip: text; color: transparent; animation: nx-shimmer 5s linear infinite; }}
.nx-cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 10px; margin: 6px 0 12px; }}
.nx-card {{ padding: 12px 14px; border-radius: 10px; border: 1px solid {BORDER}; border-top: 2px solid var(--tc, {ACCENT});
  background: linear-gradient(170deg, rgba(15,29,44,.85), rgba(10,18,28,.6)); animation: nx-rise .5s ease both; }}
.nx-card .h {{ font-family: {FONT_HEAD}; font-weight: 600; font-size: 1rem; color: var(--tc, {TEXT}); margin-bottom: 3px; }}
.nx-card .b {{ font-size: .82rem; color: {MUTED}; line-height: 1.4; }}
.nx-card ul {{ margin: 4px 0 0 16px; padding: 0; font-size: .82rem; color: {TEXT}; }}
@media (prefers-reduced-motion: reduce) {{ .stApp, .nx-tile::before, [data-testid="stExpander"], .nx-ring circle.arc,
  .nx-shim, [data-testid="stMetric"] {{ animation: none !important; }} }}
</style>
"""


def inject() -> None:
    st.markdown(CSS2, unsafe_allow_html=True)


# ---------------------------------------------------------------------------------------------- structure
def fold(title: str, icon: str = "table_rows", expanded: bool = False):
    """A collapsible panel (a context manager). Keeps text and tables out of the way until asked for."""
    return st.expander(f":material/{icon}: {title}", expanded=expanded)


def banner(kicker: str, title: str, sub: str = "") -> None:
    st.session_state["_nx_h"] = title
    st.markdown(f'<div class="nx-banner"><div class="k">{html.escape(kicker)}</div><div class="t">{html.escape(title)}</div>'
                + (f'<div class="s">{html.escape(sub)}</div>' if sub else "") + "</div>", unsafe_allow_html=True)


def section(label: str) -> None:
    st.session_state["_nx_h"] = label
    st.markdown(f'<div class="nx-sec">{html.escape(label)}</div>', unsafe_allow_html=True)


def auto_fold() -> None:
    """Put every st.dataframe / st.json / st.table into a collapsed panel titled after the heading above it.

    Display only: the wrapped call, its arguments and its return value are unchanged. A table already inside an expander,
    or one with row selection, is left alone (Streamlit does not allow nested expanders). Set
    st.session_state['_nx_nofold'] = True around a block to switch it off."""
    if getattr(st, "_nx_folded", False):
        return
    st._nx_folded = True
    orig = {n: getattr(st, n) for n in ("dataframe", "json", "table", "header", "subheader", "expander")}

    def depth() -> int:
        return st.session_state.get("_nx_depth", 0)

    class _Exp:
        """Counts how deep inside expanders the script is."""
        def __init__(self, inner):
            self._i = inner

        def __enter__(self):
            st.session_state["_nx_depth"] = depth() + 1
            return self._i.__enter__()

        def __exit__(self, *a):
            st.session_state["_nx_depth"] = max(0, depth() - 1)
            return self._i.__exit__(*a)

        def __getattr__(self, k):
            return getattr(self._i, k)

    def expander(*a, **k):
        return _Exp(orig["expander"](*a, **k))

    def heading(name):
        def f(body, *a, **k):
            st.session_state["_nx_h"] = str(body)[:70]
            return orig[name](body, *a, **k)
        return f

    def wrap(name):
        def f(*a, **k):
            if depth() > 0 or k.get("on_select") not in (None, "ignore") or st.session_state.get("_nx_nofold"):
                return orig[name](*a, **k)
            title = st.session_state.get("_nx_h", "")
            with orig["expander"](f":material/table_rows: Data · {title}" if title else ":material/table_rows: Data"):
                return orig[name](*a, **k)
        return f

    st.expander = expander
    for n in ("header", "subheader"):
        setattr(st, n, heading(n))
    for n in ("dataframe", "json", "table"):
        setattr(st, n, wrap(n))


# ---------------------------------------------------------------------------------------------- components
def _spark(vals: list[float], color: str) -> str:
    v = [float(x) for x in vals if x is not None and math.isfinite(float(x))]
    if len(v) < 2:
        return ""
    lo, hi = min(v), max(v)
    span = (hi - lo) or 1.0
    pts = " ".join(f"{i / (len(v) - 1) * 100:.1f},{30 - (x - lo) / span * 26 - 2:.1f}" for i, x in enumerate(v))
    return (f'<svg class="sp" viewBox="0 0 100 32" preserveAspectRatio="none" aria-hidden="true">'
            f'<polyline points="{pts} 100,32 0,32" fill="{color}" fill-opacity=".12" stroke="none"/>'
            f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="1.6" vector-effect="non-scaling-stroke"/></svg>')


def tiles(items: list[dict]) -> None:
    """KPI tiles. item: label, value (text), sub (optional), color (optional), spark (optional list of numbers)."""
    out = []
    for it in items:
        c = it.get("color", ACCENT)
        out.append(f'<div class="nx-tile" style="--tc:{c}"><div class="l">{html.escape(str(it["label"]))}</div>'
                   f'<div class="v">{html.escape(str(it["value"]))}</div>'
                   + (f'<div class="s">{html.escape(str(it["sub"]))}</div>' if it.get("sub") else "")
                   + (_spark(it["spark"], c) if it.get("spark") else "") + "</div>")
    st.markdown(f'<div class="nx-tiles">{"".join(out)}</div>', unsafe_allow_html=True)


def cards(items: list[dict]) -> None:
    """Idea cards. item: head, body (text) or bullets (list), color (optional)."""
    out = []
    for it in items:
        c = it.get("color", ACCENT)
        lis = "".join(f"<li>{html.escape(str(b))}</li>" for b in it.get("bullets", []))
        out.append(f'<div class="nx-card" style="--tc:{c}"><div class="h">{html.escape(str(it["head"]))}</div>'
                   + (f'<div class="b">{html.escape(str(it["body"]))}</div>' if it.get("body") else "")
                   + (f"<ul>{lis}</ul>" if lis else "") + "</div>")
    st.markdown(f'<div class="nx-cards">{"".join(out)}</div>', unsafe_allow_html=True)


def pills(items: list[tuple[str, str]]) -> None:
    st.markdown('<div class="nx-pills">' + "".join(
        f'<span class="nx-pill" style="color:{c}">{html.escape(t)}</span>' for t, c in items) + "</div>",
        unsafe_allow_html=True)


_C = 2 * math.pi * 42


def _ring_html(parts: list[tuple[float, str]], center: str, sub: str, caption: str, glow: str) -> str:
    """parts: [(share 0..1, colour)] drawn clockwise from 12 o'clock on a track."""
    arcs, start = [], 0.0
    for share, col in parts:
        ln = max(0.0, min(1.0, share)) * _C
        if ln > 0.4:
            arcs.append(f'<circle class="arc" cx="54" cy="54" r="42" stroke="{col}" stroke-width="10" '
                        f'style="--c:{_C:.1f};--d:{max(ln - 1.2, 0.1):.1f} {_C:.1f};--o:{-start:.1f}"/>')
        start += ln
    return (f'<div class="nx-ring" style="--g:{glow}"><svg viewBox="0 0 108 108" role="img" aria-label="{html.escape(caption)}: {html.escape(center)}">'
            f'<circle class="trk" cx="54" cy="54" r="42" stroke-width="10"/>{"".join(arcs)}'
            f'<text class="ctr" x="54" y="58" text-anchor="middle" font-size="19">{html.escape(center)}</text>'
            f'<text class="sub" x="54" y="73" text-anchor="middle" font-size="8.5">{html.escape(sub)}</text></svg>'
            f'<div class="cap">{html.escape(caption)}</div></div>')


def gauge_rings(items: list[dict]) -> None:
    """Single-value rings. item: caption, frac (0..1), center (text), sub, color."""
    st.markdown('<div class="nx-rings">' + "".join(
        _ring_html([(i["frac"], i.get("color", ACCENT))], i["center"], i.get("sub", ""), i["caption"], i.get("color", ACCENT))
        for i in items) + "</div>", unsafe_allow_html=True)


def prob_rings(rows: dict[str, tuple[float, float, float]], note: str = "") -> None:
    """One three-colour ring per horizon: red = DOWN, grey = SIDEWAYS, green = UP (shares sum to 1).
    rows: {horizon: (p_down, p_side, p_up)}. The centre shows the largest class and its probability."""
    out = []
    for h, (d, s, u) in rows.items():
        names = {"DOWN": d, "SIDE": s, "UP": u}
        top = max(names, key=names.get)
        col = {"DOWN": BAD, "SIDE": SIDE, "UP": GOOD}[top]
        out.append(_ring_html([(d, BAD), (s, SIDE), (u, GOOD)], f"{names[top]:.0%}", {"DOWN": "down", "SIDE": "sideways",
                                                                                      "UP": "up"}[top], h, col))
    st.markdown('<div class="nx-rings">' + "".join(out) + "</div>", unsafe_allow_html=True)
    if note:
        st.caption(note)


def orbit(inner: list[tuple[str, str]], outer: list[tuple[str, str]], core: str = "NEXUS", height: int = 400) -> None:
    """Magic UI OrbitingCircles, written in CSS. inner/outer: [(short label, caption)]. Pure display."""
    def nodes(items, radius, period, color, rev):
        out = []
        for i, (lab, cap) in enumerate(items):
            a = 360 / len(items) * i
            out.append(f'<div class="n" style="--a:{a:.1f}deg;--r:{radius}px"><div class="u"><div class="c" '
                       f'style="animation-duration:{period}s;animation-direction:{"normal" if rev else "reverse"}">'
                       f'<div class="dot" style="border-color:{color};color:{color}">{html.escape(lab)}</div>'
                       f'<div class="cp">{html.escape(cap)}</div></div></div></div>')
        return "".join(out)
    doc = f"""<link rel="stylesheet" href="{UI.FONTS_LINK}">
<style>{UI._BASE_CSS}
html,body {{ background: transparent; overflow: hidden; }}
.stage {{ position: relative; height: {height}px; display: grid; place-items: center; }}
.trk {{ position: absolute; left: 50%; top: 50%; border: 1px dashed #1E3348; border-radius: 50%; transform: translate(-50%,-50%); }}
.orb {{ position: absolute; left: 50%; top: 50%; width: 0; height: 0; animation: spin linear infinite; }}
.n {{ position: absolute; transform: rotate(var(--a)) translateX(var(--r)); }}
.u {{ transform: rotate(calc(-1 * var(--a))); }}
.c {{ animation: spin linear infinite; display: flex; flex-direction: column; align-items: center; width: 90px;
  margin: -24px 0 0 -45px; }}
@keyframes spin {{ to {{ transform: rotate(360deg); }} }}
.dot {{ width: 46px; height: 46px; border-radius: 50%; border: 1.5px solid; display: grid; place-items: center;
  font: 600 11px {UI.FONT_HEAD}; background: radial-gradient(circle at 30% 30%, #12263A, {BG}); box-shadow: 0 0 16px currentColor; }}
.cp {{ font: 10px {UI.FONT_MONO}; color: {MUTED}; margin-top: 4px; text-align: center; white-space: nowrap; }}
.core {{ position: relative; width: 78px; height: 78px; border-radius: 50%; display: grid; place-items: center;
  font: 700 14px {UI.FONT_HEAD}; letter-spacing: .12em; color: {BG};
  background: radial-gradient(circle at 35% 30%, #8CF3FF, {ACCENT} 45%, #2D6FB0); box-shadow: 0 0 40px rgba(63,208,224,.55); }}
.core::after {{ content: ""; position: absolute; inset: -14px; border-radius: 50%; border: 1px solid rgba(63,208,224,.35);
  animation: ping 2.6s ease-out infinite; }}
@keyframes ping {{ from {{ transform: scale(.85); opacity: .9; }} to {{ transform: scale(1.5); opacity: 0; }} }}
@media (prefers-reduced-motion: reduce) {{ .orb, .c, .core::after {{ animation: none; }} }}
</style>
<div class="stage">
<div class="trk" style="width:156px;height:156px"></div><div class="trk" style="width:276px;height:276px"></div>
<div class="orb" style="animation-duration:46s">{nodes(inner, 78, 46, ACCENT2, False)}</div>
<div class="orb" style="animation-duration:70s;animation-direction:reverse">{nodes(outer, 138, 70, ACCENT, True)}</div>
<div class="core">{html.escape(core)}</div></div>"""
    components.html(doc, height=height + 4)

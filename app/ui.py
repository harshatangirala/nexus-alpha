"""Visual layer of the NEXUS dashboard: tokens, CSS, a Plotly template and small self-contained HTML components.

Design references (see THIRD_PARTY_NOTICES.md):
* Arwes (arwes/arwes, MIT): the framed-panel language (corner brackets, thin luminous lines), recreated in CSS.
* Magic UI (magicuidesign/magicui, MIT): the behaviour of NumberTicker, Marquee and AnimatedBeam, ported to plain
  HTML/CSS/JS so they run inside Streamlit components without React.
* World Monitor (koala73/worldmonitor, AGPL-3.0): layout reference only (a situational-awareness board around a
  globe). No code is copied.
* globe.gl (vasturiano/globe.gl, MIT) is loaded from jsDelivr at a pinned version in globe_view.py.

Every component is display only: it never changes a forecast, a probability or the risk engine.
"""
from __future__ import annotations

import html
import json

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
import streamlit.components.v1 as components

# tokens (the same values as .streamlit/config.toml)
BG, PANEL, PANEL2, BORDER = "#05090F", "#0C1622", "#0F1D2C", "#1B2B3D"
TEXT, MUTED = "#D5E3EE", "#8098AE"
ACCENT, ACCENT2 = "#3FD0E0", "#9B8CFF"
GOOD, BAD, WARN = "#3DDC97", "#FF6B6B", "#F2B544"
FONT_BODY = "'IBM Plex Sans', system-ui, -apple-system, 'Segoe UI', sans-serif"
FONT_HEAD = "'Oxanium', 'IBM Plex Sans', system-ui, sans-serif"
FONT_MONO = "'JetBrains Mono', ui-monospace, Consolas, monospace"
FONTS_LINK = ("https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=JetBrains+Mono:wght@400;500"
              "&family=Oxanium:wght@500;600;700&display=swap")
VERDICT = {"supported": (GOOD, "SUPPORTED"), "not supported": (BAD, "NOT SUPPORTED"), "pending": (WARN, "PENDING"),
           "weak": (WARN, "WEAK")}

CSS = f"""
<style>
.nx-frame {{ position: relative; padding: 14px 16px 12px; border: 1px solid {BORDER};
  background: linear-gradient(180deg, rgba(63,208,224,.055), rgba(63,208,224,.012)); margin-bottom: 10px; }}
.nx-frame::before, .nx-frame::after {{ content: ""; position: absolute; width: 14px; height: 14px;
  border-color: {ACCENT}; border-style: solid; opacity: .9; }}
.nx-frame::before {{ top: -1px; left: -1px; border-width: 2px 0 0 2px; }}
.nx-frame::after {{ bottom: -1px; right: -1px; border-width: 0 2px 2px 0; }}
.nx-kicker {{ font-family: {FONT_MONO}; font-size: .72rem; letter-spacing: .14em; text-transform: uppercase;
  color: {ACCENT}; margin-bottom: 4px; }}
.nx-title {{ font-family: {FONT_HEAD}; font-size: 1.15rem; font-weight: 600; color: {TEXT}; margin: 0 0 6px; }}
.nx-note {{ color: {MUTED}; font-size: .84rem; line-height: 1.45; }}
.nx-chip {{ display: inline-block; font-family: {FONT_MONO}; font-size: .68rem; letter-spacing: .08em;
  padding: 1px 7px; border: 1px solid currentColor; border-radius: 2px; white-space: nowrap; }}
.nx-row {{ display: grid; grid-template-columns: minmax(0, 1.4fr) minmax(0, 1fr) minmax(0, 2fr); gap: 10px;
  align-items: center; padding: 7px 0; border-bottom: 1px solid {BORDER}; font-size: .86rem; }}
.nx-row:last-child {{ border-bottom: 0; }}
.nx-row b {{ font-family: {FONT_HEAD}; font-weight: 600; letter-spacing: .02em; }}
.nx-bar {{ position: relative; height: 6px; background: {PANEL2}; border: 1px solid {BORDER}; }}
.nx-bar > i {{ position: absolute; inset: 0 auto 0 0; background: linear-gradient(90deg, {ACCENT}, {ACCENT2}); }}
.nx-bar-l {{ display: flex; justify-content: space-between; color: {MUTED}; font-family: {FONT_MONO};
  font-size: .68rem; margin-top: 3px; font-variant-numeric: tabular-nums; }}
.nx-kv {{ display: flex; justify-content: space-between; gap: 12px; padding: 6px 0; border-bottom: 1px solid {BORDER};
  font-size: .86rem; }}
.nx-kv:last-child {{ border-bottom: 0; }}
.nx-kv span:first-child {{ color: {MUTED}; }}
[data-testid="stMetricValue"] {{ font-family: {FONT_HEAD}; font-variant-numeric: tabular-nums; }}
.stTabs [data-baseweb="tab"] p {{ font-family: {FONT_MONO}; font-size: .74rem; letter-spacing: .06em; text-transform: uppercase; }}
@media (max-width: 640px) {{ .nx-row {{ grid-template-columns: minmax(0, 1fr); gap: 4px; }} }}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    import skin                       # second layer: page skin, folding, ring/tile/orbit components
    skin.inject()
    skin.auto_fold()


def tag(label: str, color: str) -> str:
    return f'<span class="nx-chip" style="color:{color}">{html.escape(label)}</span>'


def chip(verdict: str) -> str:
    color, label = VERDICT.get(verdict, (MUTED, verdict.upper()))
    return tag(label, color)


def frame(kicker: str, title: str, body_html: str, note: str | None = None) -> None:
    """An Arwes-style framed panel with a kicker label, a title and trusted HTML content."""
    st.markdown(f'<div class="nx-frame"><div class="nx-kicker">{html.escape(kicker)}</div>'
                f'<div class="nx-title">{html.escape(title)}</div>{body_html}'
                + (f'<div class="nx-note" style="margin-top:8px">{html.escape(note)}</div>' if note else "")
                + "</div>", unsafe_allow_html=True)


def progress(value: float, total: float, left: str, right: str) -> str:
    share = 0.0 if total <= 0 else max(0.0, min(1.0, value / total))
    return (f'<div class="nx-bar" role="progressbar" aria-valuenow="{value:g}" aria-valuemax="{total:g}">'
            f'<i style="width:{share * 100:.1f}%"></i></div>'
            f'<div class="nx-bar-l"><span>{html.escape(left)}</span><span>{html.escape(right)}</span></div>')


# ---------------------------------------------------------------------------------------------- Plotly
def _register_template() -> str:
    name = "nexus"
    if name not in pio.templates:
        t = go.layout.Template()
        t.layout = go.Layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family=FONT_BODY, color=TEXT, size=12),
            colorway=[ACCENT, WARN, ACCENT2, GOOD, BAD, "#5B8DEF", "#E58BD8"],
            xaxis=dict(gridcolor="#13233A", zerolinecolor="#23384F", linecolor=BORDER, tickfont=dict(color=MUTED),
                       automargin=True),
            yaxis=dict(gridcolor="#13233A", zerolinecolor="#23384F", linecolor=BORDER, tickfont=dict(color=MUTED),
                       automargin=True),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(color=MUTED),
                        bgcolor="rgba(0,0,0,0)"),
            margin=dict(l=10, r=10, t=36, b=10), hoverlabel=dict(bgcolor=PANEL, bordercolor=ACCENT,
                                                                 font=dict(family=FONT_MONO, color=TEXT)))
        pio.templates[name] = t
    return name


TEMPLATE = _register_template()


def plot(fig: go.Figure, height: int | None = None) -> None:
    fig.update_layout(template=TEMPLATE, **({"height": height} if height else {}))
    st.plotly_chart(fig, width="stretch", theme=None)


# ---------------------------------------------------------------------------------------------- components
_BASE_CSS = f"""
:root {{ color-scheme: dark; }}
html, body {{ margin: 0; background: {BG}; color: {TEXT}; font-family: {FONT_BODY}; }}
* {{ box-sizing: border-box; }}
"""


def hero(title: str, subtitle: str, kpis: list[dict], tape: list[dict], height: int = 250) -> None:
    """Title band with animated counters (Magic UI NumberTicker) and an evidence tape (Magic UI Marquee).

    kpis: [{"value": 4, "label": "frozen models", "decimals": 0}]; tape: [{"verdict": "supported", "text": "..."}].
    The HTML holds the final numbers, so the band is complete even if scripts do not run."""
    def num(k):
        d = int(k.get("decimals", 0))
        return f'{k["value"]:,.{d}f}'
    tiles = "".join(f'<div class="k"><div class="v" data-to="{k["value"]}" data-dec="{int(k.get("decimals", 0))}">'
                    f'{num(k)}</div><div class="l">{html.escape(k["label"])}</div></div>' for k in kpis)
    items = "".join(f'<span class="it"><span class="c" style="color:{VERDICT.get(t["verdict"], (MUTED, ""))[0]}">'
                    f'{html.escape(VERDICT.get(t["verdict"], (MUTED, t["verdict"].upper()))[1])}</span>'
                    f'{html.escape(t["text"])}</span>' for t in tape)
    doc = f"""<link rel="stylesheet" href="{FONTS_LINK}">
<style>{_BASE_CSS}
.wrap {{ position: relative; border: 1px solid {BORDER}; padding: 18px 18px 12px;
  background: radial-gradient(120% 140% at 0% 0%, rgba(63,208,224,.10), transparent 55%),
              radial-gradient(90% 120% at 100% 0%, rgba(155,140,255,.08), transparent 60%), {PANEL}; }}
.wrap::before, .wrap::after {{ content: ""; position: absolute; width: 18px; height: 18px; border: solid {ACCENT}; }}
.wrap::before {{ top: -1px; left: -1px; border-width: 2px 0 0 2px; }}
.wrap::after {{ bottom: -1px; right: -1px; border-width: 0 2px 2px 0; }}
.top {{ display: flex; flex-wrap: wrap; gap: 18px; justify-content: space-between; align-items: flex-end; }}
.id {{ min-width: 0; max-width: 560px; }}
.kick {{ font-family: {FONT_MONO}; font-size: 11px; letter-spacing: .16em; color: {ACCENT}; text-transform: uppercase; }}
h1 {{ font-family: {FONT_HEAD}; font-weight: 600; font-size: 26px; line-height: 1.15; margin: 6px 0 6px; text-wrap: balance; }}
p {{ margin: 0; color: {MUTED}; font-size: 13px; line-height: 1.5; }}
.kpis {{ display: flex; flex-wrap: wrap; gap: 8px; }}
.k {{ min-width: 104px; padding: 8px 10px; border: 1px solid {BORDER}; background: rgba(5,9,15,.55); }}
.v {{ font-family: {FONT_HEAD}; font-size: 24px; font-weight: 600; color: {TEXT}; font-variant-numeric: tabular-nums; }}
.l {{ font-family: {FONT_MONO}; font-size: 10px; letter-spacing: .08em; color: {MUTED}; text-transform: uppercase; }}
.tape {{ margin-top: 14px; overflow: hidden; border-top: 1px solid {BORDER}; padding-top: 10px;
  -webkit-mask-image: linear-gradient(90deg, transparent, #000 5%, #000 95%, transparent);
          mask-image: linear-gradient(90deg, transparent, #000 5%, #000 95%, transparent); }}
.track {{ display: flex; gap: 34px; width: max-content; animation: run 75s linear infinite; }}
.tape:hover .track {{ animation-play-state: paused; }}
.it {{ font-size: 12.5px; white-space: nowrap; color: {TEXT}; }}
.c {{ font-family: {FONT_MONO}; font-size: 10.5px; letter-spacing: .08em; border: 1px solid currentColor;
  padding: 1px 6px; margin-right: 8px; }}
@keyframes run {{ from {{ transform: translateX(0); }} to {{ transform: translateX(-50%); }} }}
@media (prefers-reduced-motion: reduce) {{ .track {{ animation: none; flex-wrap: wrap; width: auto; }} }}
</style>
<div class="wrap"><div class="top"><div class="id"><div class="kick">{html.escape(title)}</div>
<h1>{html.escape(subtitle.split("|")[0])}</h1><p>{html.escape(subtitle.split("|")[1] if "|" in subtitle else "")}</p></div>
<div class="kpis">{tiles}</div></div>
<div class="tape" aria-label="Evidence summary"><div class="track">{items}{items}</div></div></div>
<script>
(() => {{
  const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
  const fmt = (x, d) => x.toLocaleString('en-US', {{minimumFractionDigits: d, maximumFractionDigits: d}});
  document.querySelectorAll('[data-to]').forEach(el => {{
    const to = parseFloat(el.dataset.to), d = parseInt(el.dataset.dec || '0');
    if (reduce || !isFinite(to)) return;
    const t0 = performance.now(), dur = 1300;
    const step = t => {{ const k = Math.min(1, (t - t0) / dur), e = 1 - Math.pow(1 - k, 3);
      el.textContent = fmt(to * e, d); if (k < 1) requestAnimationFrame(step); }};
    el.textContent = fmt(0, d); requestAnimationFrame(step);
  }});
}})();
</script>"""
    components.html(doc, height=height)


# nodes: id -> (x, y, w, h, title, subtitle)
_NODES = {
    "src1": (16, 14, 190, 46, "Prices · 1-minute feed", "1H/1D/1W bars · 979 stocks"),
    "src2": (16, 72, 190, 46, "SEC filings + XBRL", "10-K/10-Q/8-K · 31 ratios"),
    "src3": (16, 130, 190, 46, "News corpus", "57.1M rows, ticker-tagged"),
    "src4": (16, 188, 190, 46, "Market + macro", "SPY, ETFs, VIX, FRED, GMD"),
    "feat": (262, 86, 180, 76, "Point-in-time features", "to close t · seal 2023-01-03"),
    "mods": (520, 80, 196, 88, "Frozen models", "v1.1 · v2 · market_e19", "v3_fund · v3_alpha"),
    "risk": (812, 26, 196, 58, "Risk engine", "deterministic gates, sizing"),
    "dash": (812, 132, 196, 58, "Dashboard", "paper only · no broker"),
    "log":  (520, 250, 196, 62, "Forward-test log", "hash chain · before 09:00 NY"),
    "score": (812, 250, 196, 62, "Scorer + SPC", "after each horizon matures"),
}
_EDGES = [("src1", "feat", ""), ("src2", "feat", ""), ("src3", "feat", ""), ("src4", "feat", ""),
          ("feat", "mods", "features"), ("mods", "risk", "calibrated P"), ("risk", "dash", "decision"),
          ("mods", "log", "nightly 07:00 IST"), ("log", "score", "matured labels"), ("score", "dash", "forward evidence")]


def _anchor(n: str, side: str) -> tuple[float, float]:
    x, y, w, h, *_ = _NODES[n]
    return {"r": (x + w, y + h / 2), "l": (x, y + h / 2), "t": (x + w / 2, y), "b": (x + w / 2, y + h)}[side]


def pipeline_beam(height: int = 360) -> None:
    """The system's data flow as an SVG with travelling light beams (Magic UI AnimatedBeam, ported to SVG/CSS)."""
    paths = []
    for i, (a, b, lab) in enumerate(_EDGES):
        if (a, b) == ("mods", "log"):
            (x1, y1), (x2, y2) = _anchor(a, "b"), _anchor(b, "t")
            d = f"M{x1},{y1} L{x2},{y2}"
            lx, ly = x1 + 8, (y1 + y2) / 2
        elif (a, b) in (("risk", "dash"), ("score", "dash")):
            s1, s2 = ("b", "t") if a == "risk" else ("t", "b")
            (x1, y1), (x2, y2) = _anchor(a, s1), _anchor(b, s2)
            d = f"M{x1},{y1} L{x2},{y2}"
            lx, ly = x1 + 8, (y1 + y2) / 2
        else:
            (x1, y1), (x2, y2) = _anchor(a, "r"), _anchor(b, "l")
            mx = (x1 + x2) / 2
            d = f"M{x1},{y1} C{mx},{y1} {mx},{y2} {x2},{y2}"
            lx, ly = mx, (y1 + y2) / 2 - 6
        dashed = ' stroke-dasharray="6 6"' if (a, b) == ("score", "dash") else ""
        paths.append(f'<path d="{d}" class="base"{dashed}/>'
                     f'<path d="{d}" class="beam" style="animation-delay:{(i * 0.37) % 2.6:.2f}s"/>')
        if lab:
            anchor = "start" if (a, b) in (("mods", "log"), ("risk", "dash"), ("score", "dash")) else "middle"
            paths.append(f'<text x="{lx}" y="{ly}" class="el" text-anchor="{anchor}">{html.escape(lab)}</text>')
    nodes = []
    for n, (x, y, w, h, t, *subs) in _NODES.items():
        cls = "node hi" if n in ("mods", "log") else "node"
        top = y + h / 2 - 3 - 8 * (len(subs) - 1)
        lines = "".join(f'<text x="{x + 12}" y="{top + 16 * (k + 1)}" class="ns">{html.escape(s)}</text>'
                        for k, s in enumerate(subs))
        nodes.append(f'<g class="{cls}"><rect x="{x}" y="{y}" width="{w}" height="{h}"/>'
                     f'<path d="M{x},{y + 10} V{y} H{x + 10}" class="cn"/><path d="M{x + w},{y + h - 10} V{y + h} H{x + w - 10}" class="cn"/>'
                     f'<text x="{x + 12}" y="{top}" class="nt">{html.escape(t)}</text>{lines}</g>')
    doc = f"""<link rel="stylesheet" href="{FONTS_LINK}">
<style>{_BASE_CSS}
svg {{ width: 100%; height: auto; display: block; }}
.base {{ fill: none; stroke: #20344A; stroke-width: 1.6; }}
.beam {{ fill: none; stroke: url(#g); stroke-width: 2.4; stroke-linecap: round; stroke-dasharray: 46 420;
  animation: flow 2.6s linear infinite; }}
@keyframes flow {{ from {{ stroke-dashoffset: 466; }} to {{ stroke-dashoffset: 0; }} }}
.node rect {{ fill: {PANEL}; stroke: {BORDER}; }}
.node.hi rect {{ fill: #0E2130; stroke: #2A5468; }}
.cn {{ fill: none; stroke: {ACCENT}; stroke-width: 2; }}
.nt {{ fill: {TEXT}; font-family: {FONT_HEAD}; font-size: 13px; font-weight: 600; }}
.ns {{ fill: {MUTED}; font-family: {FONT_MONO}; font-size: 10.5px; }}
.el {{ fill: {ACCENT}; font-family: {FONT_MONO}; font-size: 10px; letter-spacing: .04em; }}
@media (prefers-reduced-motion: reduce) {{ .beam {{ animation: none; stroke-dasharray: none; opacity: .55; }} }}
</style>
<svg viewBox="0 0 1024 326" role="img" aria-label="NEXUS data flow: sources become point-in-time features, frozen models
forecast, the risk engine decides, and every night's forecasts go to a hash-chained forward-test log that is scored later.">
<defs><linearGradient id="g" gradientUnits="userSpaceOnUse" x1="0" y1="0" x2="1024" y2="0">
<stop offset="0" stop-color="{ACCENT}"/><stop offset="1" stop-color="{ACCENT2}"/></linearGradient></defs>
{''.join(paths)}{''.join(nodes)}
</svg>"""
    components.html(doc, height=height)


def json_script(obj) -> str:
    """JSON for embedding inside a <script> block (closing tags escaped)."""
    return json.dumps(obj, separators=(",", ":")).replace("</", "<\\/")

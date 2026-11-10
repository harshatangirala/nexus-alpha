"""Mission Control: the first tab. Status of the whole system at a glance, and what the evidence does and does not show.

Every number here is read from the repository at run time (forward-test chains, the SPC report, the frozen registry,
experiment metrics) or is a documented result with its source file named next to it (CLAIMS). Nothing on this tab
changes a forecast.
"""
from __future__ import annotations

import html
import json
import re
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import globe_view as GV
import ui as UI

ROOT = Path(__file__).resolve().parents[1]
VERSIONS = ["v1.1", "v2", "market_e19", "v3_fund", "v3_alpha"]
FIRST_INTERVAL_DATES = 20        # FORWARD_SHADOW_TEST.md: a first 1D interval after about 20 matured dates
CRITERION_SESSIONS = 120         # V3_RESEARCH.md: forward criteria after 120 scored sessions

# Documented results (value, 95% interval where the document gives one). Nothing is rounded in the system's favour.
# "evidence" is the exact text in "source"; tests/test_dashboard.py checks that it is there, so no number on this tab
# can drift from its document. "kind" picks the plot panel: auroc (no skill = 0.5) or ic (no skill = 0).
CLAIMS = [
    {"claim": "Will it move: sideways vs move, 5D (LightGBM)", "kind": "auroc", "value": 0.580, "lo": None, "hi": None,
     "period": "2020-22", "verdict": "supported", "source": "reports/RESULTS.md", "ref": "E01",
     "evidence": "+0.0091 [+0.0048, +0.0141] | 0.510 | 0.580",
     "note": "AUROC sideways vs move; log-loss gain over the class prior +0.0091 [+0.0048, +0.0141]"},
    {"claim": "Which way: up vs down, frozen v2, 5D", "kind": "auroc", "value": 0.509, "lo": None, "hi": None,
     "period": "2020-22", "verdict": "not supported", "source": "docs/V2_RESULTS.md", "ref": "v2",
     "evidence": "| Up-vs-down AUROC | 0.510 | 0.509 | 0.488", "note": "1D 0.510, 20D 0.488: chance"},
    {"claim": "Market 20D direction (market_e19)", "kind": "auroc", "value": 0.720, "lo": 0.525, "hi": 0.857,
     "period": "2017-18", "verdict": "supported", "source": "docs/V2_RESULTS.md", "ref": "E19",
     "evidence": "0.720 [0.525, 0.857]", "note": "selection block"},
    {"claim": "Market 20D direction (market_e19)", "kind": "auroc", "value": 0.683, "lo": 0.536, "hi": 0.814,
     "period": "2020-22", "verdict": "supported", "source": "docs/V2_RESULTS.md", "ref": "E19",
     "evidence": "0.683 [0.536, 0.814]", "note": "probabilities no better than the base rate, so shown as a lean"},
    {"claim": "Stock ranking 20D, rank IC (F0 LightGBM, B0)", "kind": "ic", "value": 0.032, "lo": 0.003, "hi": 0.062,
     "period": "2017-18", "verdict": "supported", "source": "docs/V3_RESEARCH.md", "ref": "E21",
     "evidence": "20D 0.032 [0.003, 0.062]", "note": "real but small: below trading costs"},
    {"claim": "Industry-neutral fundamentals, swing rank-IC gain", "kind": "ic", "value": 0.007, "lo": 0.002,
     "hi": 0.012, "period": "2020-22", "verdict": "supported", "source": "docs/V3_RESEARCH.md", "ref": "E23",
     "evidence": "+0.007 [+0.002, +0.012]", "note": "the basis of v3_fund; the forward test decides"},
    {"claim": "Kronos foundation model, swing rank IC", "kind": "ic", "value": 0.005, "lo": -0.028, "hi": 0.039,
     "period": "2017-18", "verdict": "not supported", "source": "docs/V3_RESEARCH.md", "ref": "E24",
     "evidence": "+0.005 [-0.028, +0.039]", "note": "no skill even on data it was trained on"},
    {"claim": "Top-minus-bottom quintile spread after costs, 20D (bp)", "kind": "bp", "value": 36.1, "lo": -36.6,
     "hi": 104.6, "period": "2017-18", "verdict": "not supported", "source": "docs/V3_RESEARCH.md", "ref": "E21",
     "evidence": "+36.1 bp [-36.6, +104.6]", "note": "interval spans zero; 1D: -10.3 bp [-17.4, -3.4], a loss"},
    {"claim": "Best gated policy vs buy-and-hold (Sharpe)", "kind": "text", "value": 0.31, "lo": None, "hi": None,
     "period": "2020-22", "verdict": "not supported", "source": "README.md", "ref": "E04 v1.1",
     "evidence": "0.31 at 5D (41 trades)", "note": "equal-weight buy-and-hold: 0.53; deflated Sharpe about 0"},
    {"claim": "Multimodal fusion beats numbers alone", "kind": "text", "value": None, "lo": None, "hi": None,
     "period": "2020-22", "verdict": "not supported", "source": "README.md", "ref": "E03",
     "evidence": "None of the five pre-registered hypotheses is supported after Holm correction",
     "note": "images, news and filings add 0.0005-0.0012 log loss, about one seed SD"},
    {"claim": "Qlib Alpha158 adds to ranking (LightGBM, long group)", "kind": "text", "value": 0.0134, "lo": 0.0071,
     "hi": 0.0215, "period": "2017-18", "verdict": "weak", "source": "docs/V3_RESEARCH.md", "ref": "E25",
     "evidence": "**+0.0134 [+0.0071, +0.0215]**",
     "note": "kept under the rule; did not replicate on 2020-22: +0.0003 [-0.0097, +0.0188]"},
    {"claim": "Qlib Alpha158 improves the probabilities", "kind": "text", "value": None, "lo": None, "hi": None,
     "period": "2017-18", "verdict": "not supported", "source": "docs/V3_RESEARCH.md", "ref": "E25",
     "evidence": "Log loss: no horizon group improves for either family on 2017-18",
     "note": "log-loss endpoint not kept for either family"},
]


# ---------------------------------------------------------------------------------------------- data
def kpis(root: Path = ROOT) -> dict:
    frozen = len(list((root / "registry" / "frozen_models").glob("*.json")))
    chain = _chain(root, "v1.1")
    sessions = int(chain.asof_session.nunique()) if len(chain) else 0
    last = sorted((root / "shadow" / "v1.1" / "forecasts").glob("????-??-??.csv"))
    stocks = int(pd.read_csv(last[-1], usecols=["ticker"]).ticker.nunique()) if last else 0
    exps = {m.group(1) for p in (root / "experiments").glob("*_e*_*")
            if (m := re.search(r"_e(\d{2})_", p.name)) and (p / "metrics.json").exists()}
    log = (root / "docs" / "DECISION_LOG.md").read_text(encoding="utf-8")
    decisions = len(re.findall(r"^\| D-\d{3} \|", log, flags=re.M))
    tests = sum(len(re.findall(r"^def test_", p.read_text(encoding="utf-8"), flags=re.M))
                for p in (root / "tests").glob("test_*.py"))
    return {"frozen": frozen, "sessions": sessions, "stocks": stocks, "experiments": len(exps),
            "decisions": decisions, "tests": tests}


def _chain(root: Path, version: str) -> pd.DataFrame:
    p = root / "shadow" / version / "chain.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame(columns=["asof_session", "n_rows", "late_rows", "run_utc"])


def forward_status(root: Path = ROOT) -> pd.DataFrame:
    """One row per frozen model: sessions logged, rows, late rows, chain check, matured dates and progress."""
    from nexus_alpha import shadow as SH
    rows = []
    for v in VERSIONS:
        c = _chain(root, v)
        summ = root / "shadow" / v / "summary.json"
        matured = 0
        if summ.exists():
            s = json.loads(summ.read_text(encoding="utf-8"))
            matured = max([int(h.get("n_dates", 0)) for h in s.get("by_horizon", {}).values()] or [0])
        probs = SH.verify_chain(v) if root == ROOT else []
        rows.append({"model": v, "sessions": int(c.asof_session.nunique()) if len(c) else 0,
                     "last_session": str(c.asof_session.max()) if len(c) else "-",
                     "rows": int(pd.to_numeric(c.n_rows, errors="coerce").sum()) if len(c) else 0,
                     "late_rows": int(pd.to_numeric(c.late_rows, errors="coerce").sum()) if len(c) else 0,
                     "chain_intact": not probs, "matured_dates": matured,
                     "last_run_utc": str(c.run_utc.max()) if len(c) else "-"})
    return pd.DataFrame(rows)


def spc_alerts(root: Path = ROOT) -> list[str]:
    p = root / "shadow" / "spc_report.json"
    if not p.exists():
        return ["SPC report missing"]
    r = json.loads(p.read_text(encoding="utf-8"))
    out = []
    for v, d in r.items():
        if not isinstance(d, dict):
            continue
        if not d.get("chain_intact", True):
            out.append(f"{v}: chain broken")
        ooc = (d.get("coverage") or {}).get("out_of_control") or []
        if ooc:
            out.append(f"{v}: coverage out of control on {', '.join(map(str, ooc))}")
    return out


def holdout_status(root: Path = ROOT) -> str:
    from nexus_alpha import paths
    log = paths.GOVERNANCE / "holdout_access_log.csv"
    return "opened (see the access log)" if log.exists() and log.stat().st_size > 0 else "sealed, never opened"


def colab_status() -> str:
    drive = Path("G:/My Drive/antigravity_workspace/colab_results")
    runs = sorted(drive.glob("*/COLAB_RESULTS.json")) if drive.exists() else []
    if runs:
        try:
            r = json.loads(runs[-1].read_text(encoding="utf-8"))
            return f"Colab: {r.get('verdict', '?')} ({runs[-1].parent.name})"
        except Exception:  # noqa: BLE001
            return f"Colab: result unreadable ({runs[-1].parent.name})"
    return "Colab: not run yet (PC rehearsal 2026-11-07: REPRODUCED)"


def e25(root: Path = ROOT) -> dict | None:
    for r in sorted((root / "experiments").glob("*_e25_alpha158_*"), reverse=True):
        mf, mt = r / "manifest.json", r / "metrics.json"
        if mf.exists() and mt.exists() and json.loads(mf.read_text(encoding="utf-8")).get("status") == "COMPLETED":
            return {"run": r.name, **json.loads(mt.read_text(encoding="utf-8"))}
    return None


def tape(k: dict, e: dict | None) -> list[dict]:
    t = [{"verdict": "supported", "text": "Volatility regime: every model beats the class prior, mostly by separating "
                                          "sideways from a move (AUROC about 0.58-0.61)"},
         {"verdict": "not supported", "text": "Stock direction: up-vs-down AUROC 0.49-0.51 on 2020-22"},
         {"verdict": "supported", "text": "Market 20D read (E19): AUROC 0.683 [0.536, 0.814] on 2020-22"},
         {"verdict": "supported", "text": "Cross-sectional ranking: 20D rank IC 0.032 [0.003, 0.062] on 2017-18"},
         {"verdict": "not supported", "text": "No policy beat equal-weight buy-and-hold after costs (Sharpe 0.31 vs 0.53)"},
         {"verdict": "not supported", "text": "Multimodal fusion: 0 of 5 hypotheses survive Holm correction"},
         {"verdict": "supported", "text": "Leakage controls: a global label shuffle gains nothing; a planted leak is caught"},
         {"verdict": "pending", "text": f"Forward test: {k['sessions']} sessions logged; about {CRITERION_SESSIONS} "
                                        "needed before any claim"}]
    if e is not None:
        t.append({"verdict": "weak" if e.get("kept") else "not supported",
                  "text": "Qlib Alpha158 (E25): " + ("kept for ranking under the pre-registered rule; no gain in the "
                                                     "probabilities; partial replication on 2020-22" if e.get("kept")
                                                     else "adds nothing beyond the hand-built features")})
    else:
        t.append({"verdict": "pending", "text": "Qlib Alpha158 factor library (E25): running"})
    return t


# ---------------------------------------------------------------------------------------------- views
def _forward_panel(fs: pd.DataFrame) -> None:
    rows = []
    for r in fs.itertuples():
        ok = UI.tag("CHAIN OK", UI.GOOD) if r.chain_intact else UI.tag("CHAIN BROKEN", UI.BAD)
        late = "" if r.late_rows == 0 else f' · <span style="color:{UI.WARN}">{r.late_rows} late rows (never scored)</span>'
        bars = (UI.progress(r.matured_dates, FIRST_INTERVAL_DATES, f"{r.matured_dates} matured dates",
                            f"first interval at {FIRST_INTERVAL_DATES}")
                + UI.progress(r.sessions, CRITERION_SESSIONS, f"{r.sessions} sessions logged",
                              f"criterion at {CRITERION_SESSIONS}"))
        rows.append(f'<div class="nx-row"><div><b>{html.escape(r.model)}</b><br><span class="nx-note">last '
                    f'{html.escape(r.last_session)} · {r.rows:,} rows{late}</span></div><div>{ok}</div><div>{bars}</div></div>')
    UI.frame("Forward shadow test", "Logged before the next open, scored only after each horizon ends", "".join(rows),
             "Each night's file is hash-chained (run time, file SHA-256, previous row). Editing any logged forecast "
             "breaks the chain, and the scorer refuses a broken chain.")


def _status_panel(fs: pd.DataFrame, root: Path) -> None:
    alerts = spc_alerts(root)
    last = fs.last_run_utc.max()
    kv = [("Last nightly run (UTC)", str(last).replace("T", " ")[:16]),
          ("Next run", "07:00 IST, Tue-Sat (scheduled task)"),
          ("SPC monitor", "no alerts" if not alerts else f"{len(alerts)} alert(s)"),
          ("Sealed 2023+ period", holdout_status(root)),
          ("Reproduction", colab_status()),
          ("Orders sent to a broker", "none, ever")]
    body = "".join(f'<div class="nx-kv"><span>{html.escape(a)}</span><span>{html.escape(b)}</span></div>' for a, b in kv)
    if alerts:
        body += "".join(f'<div class="nx-note" style="color:{UI.WARN}">{html.escape(a)}</div>' for a in alerts)
    UI.frame("System status", "Integrity checks", body)


SHORT = {  # chart labels (the ledger below carries the full wording)
    ("Will it move: sideways vs move, 5D (LightGBM)", "2020-22"): "Will it move, 5D",
    ("Which way: up vs down, frozen v2, 5D", "2020-22"): "Which way, frozen v2, 5D",
    ("Market 20D direction (market_e19)", "2017-18"): "Market 20D read (E19)",
    ("Market 20D direction (market_e19)", "2020-22"): "Market 20D read (E19)",
    ("Stock ranking 20D, rank IC (F0 LightGBM, B0)", "2017-18"): "Stock ranking 20D (B0)",
    ("Industry-neutral fundamentals, swing rank-IC gain", "2020-22"): "Fundamentals gain, swing (E23)",
    ("Kronos foundation model, swing rank IC", "2017-18"): "Kronos, swing (E24)",
}


def _e25_points(e: dict | None) -> list[dict]:
    """E25's rank-IC gain (B + A158 vs B) in each group that carried a keep, on 2017-18 and on 2020-22, read from the
    run's metrics, so the board shows both the selection result and whether it replicated."""
    if e is None:
        return []
    out = []
    for key, k in e.get("keep", {}).items():
        fam, ep = key.split("|")
        if ep != "rank_ic" or not k.get("kept"):
            continue
        for grp in k.get("passing_groups", []):
            for split, period in (("valid", "2017-18"), ("test", "2020-22")):
                g = e.get("rank_ic_gain", {}).get(f"{fam}|B_A158_vs_B|{split}|{grp}")
                if g:
                    lo, hi = g["ci95"]
                    out.append({"label": f"Alpha158 gain, {'LightGBM' if fam == 'lgbm' else 'logistic'} {grp} (E25)",
                                "period": period, "value": g["gain"], "lo": lo, "hi": hi,
                                "verdict": "supported" if lo > 0 else "not supported",
                                "source": f"experiments/{e['run']}"})
    return out


def _evidence_board(e: dict | None) -> None:
    from plotly.subplots import make_subplots
    panels = [("auroc", "AUROC   ·   no skill = 0.5", 0.5), ("ic", "RANK IC   ·   no skill = 0", 0.0)]
    pts = {k: [] for k, _, _ in panels}
    for c in CLAIMS:
        if c["kind"] in pts:
            pts[c["kind"]].append({"label": SHORT.get((c["claim"], c["period"]), c["claim"]), "period": c["period"],
                                   "value": c["value"], "lo": c["lo"], "hi": c["hi"], "verdict": c["verdict"],
                                   "source": f'{c["source"]} ({c["ref"]})'})
    pts["ic"] += _e25_points(e)
    n = [len(pts[k]) for k, _, _ in panels]
    fig = make_subplots(rows=2, cols=1, vertical_spacing=0.16, row_heights=[x / sum(n) for x in n],
                        subplot_titles=[p[1] for p in panels])
    for ri, (kind, _, ref) in enumerate(panels, start=1):
        for p in pts[kind]:
            col = UI.VERDICT[p["verdict"]][0]
            err = None
            if p["lo"] is not None:
                err = dict(type="data", symmetric=False, array=[p["hi"] - p["value"]], arrayminus=[p["value"] - p["lo"]],
                           color=col, thickness=1.8, width=7)
            ci = f" [{p['lo']:.3f}, {p['hi']:.3f}]" if p["lo"] is not None else ""
            fig.add_trace(go.Scatter(
                x=[p["value"]], y=[f'{p["label"]} · {p["period"]}'], mode="markers", error_x=err, showlegend=False,
                marker=dict(size=12, color=col, symbol="diamond", line=dict(color=UI.BG, width=1)),
                hovertemplate=f"{html.escape(p['label'])} · {p['period']}<br>{p['value']:.3f}{ci}<br>"
                              f"{html.escape(p['source'])}<extra></extra>"), row=ri, col=1)
        fig.add_vline(x=ref, line=dict(color=UI.MUTED, dash="dot", width=1), row=ri, col=1)
    fig.update_yaxes(autorange="reversed", automargin=True, tickfont=dict(size=12, color=UI.TEXT))
    fig.update_xaxes(automargin=True)
    for a in fig.layout.annotations:
        a.update(font=dict(family=UI.FONT_MONO, size=11, color=UI.ACCENT), x=0, xanchor="left")
    fig.update_layout(margin=dict(l=10, r=20, t=34, b=10))
    UI.plot(fig, height=120 + 46 * sum(n))
    def fmt(c: dict) -> str:       # the precision the source document uses
        if c["value"] is None:
            return "-"
        f = {"auroc": "{:.3f}", "ic": "{:.3f}", "bp": "{:.1f} bp"}.get(c["kind"], "{:g}")
        out = f.format(c["value"])
        if c["lo"] is not None:
            out += " [" + f.format(c["lo"]).replace(" bp", "") + ", " + f.format(c["hi"]).replace(" bp", "") + "]"
        return out
    led = pd.DataFrame([{"verdict": c["verdict"].upper(), "claim": c["claim"], "period": c["period"], "value": fmt(c),
                         "note": c["note"], "source": f'{c["source"]} ({c["ref"]})'} for c in CLAIMS])
    st.dataframe(led, hide_index=True, width="stretch", height=38 + 35 * len(led),
                 column_config={"verdict": st.column_config.TextColumn(width="medium"),
                                "claim": st.column_config.TextColumn(width="large")})
    st.caption("2020-22 was read by TriFrame and re-scored several times here, so it is a historical diagnostic, not a "
               "clean test. The forward test is the clean one.")


def _e25_panel(e: dict | None) -> None:
    if e is None:
        UI.frame("Qlib benchmark (E25)", "Alpha158 factor library vs the 80 hand-built features",
                 '<div class="nx-note">Running, or not run yet: scripts/experiments/e25_alpha158.py. Pre-registered in '
                 "docs/V3_RESEARCH.md before the features were built.</div>")
        return
    def verdict(v: dict) -> str:
        if not v["kept"]:
            return UI.tag("NOT KEPT", UI.BAD)
        return UI.tag("WEAK KEEP", UI.WARN) if v.get("weak_keep") else UI.tag("KEPT", UI.GOOD)
    chips = "".join(f'<div class="nx-kv"><span>{html.escape(k.replace("|", " · "))}</span><span>{verdict(v)}</span></div>'
                    for k, v in e["keep"].items())
    UI.frame("Qlib benchmark (E25)", "Alpha158 factor library vs the 80 hand-built features (F0)", chips,
             f"Run {e['run']}. Keep rule unchanged from E22/E23: a block is kept if either endpoint passes for a family "
             "whose placebo (Alpha158 shuffled across stocks within each date) does not. The 2020-22 replication is "
             "partial (evidence board above; docs/V3_RESEARCH.md, E25 results), so the forward test decides.")
    period = {"valid": "2017-18", "test": "2020-22"}
    rows = []
    for ep, key in (("rank IC", "rank_ic_gain"), ("log loss", "logloss_gain")):
        for k, v in e[key].items():
            mdl, comp, split, grp = k.split("|")
            rows.append({"family": "LightGBM" if mdl == "lgbm" else "logistic", "comparison": comp.replace("_vs_", " vs "),
                         "period": period[split], "group": grp, "endpoint": ep, "gain": v["gain"],
                         "95% interval": f'[{v["ci95"][0]:+.4f}, {v["ci95"][1]:+.4f}]',
                         "_o": ({"intraday": 0, "swing": 1, "long": 2}[grp], split != "valid")})
    d = pd.DataFrame(rows)
    c = st.columns([3, 1])
    with c[0]:
        sel = st.segmented_control("Comparison", ["B_A158 vs B", "B_A158_placebo vs B", "B_A158 vs B_A158_placebo",
                                                  "A158 vs B"], default="B_A158 vs B", key="mc_e25_comp")
    with c[1]:
        ep = st.segmented_control("Endpoint", ["rank IC", "log loss"], default="rank IC", key="mc_e25_ep")
    show = d[(d.comparison == (sel or "B_A158 vs B")) & (d.endpoint == (ep or "rank IC"))]
    show = show.sort_values(["family", "_o"]).drop(columns=["_o", "comparison", "endpoint"])
    st.dataframe(show.style.format({"gain": "{:+.4f}"}), hide_index=True, width="stretch",
                 height=38 + 35 * len(show))
    q = e.get("qlib_style", {})
    if q:
        qr = []
        for key, v in q.items():
            h, name, mdl, split = key.split("|")
            if mdl == "lgbm" and h in ("1D", "5D", "20D") and name in ("B", "B_A158", "A158"):
                qr.append({"horizon": h, "set": name, "split": split, **{m: v[m] for m in ("IC", "ICIR", "RankIC", "RankICIR")}})
        st.caption("Qlib-style report (LightGBM): IC and rank IC are daily cross-sectional correlations of p_up - p_down "
                   "with the h-session return; ICIR = mean / standard deviation of the daily values. Qlib's own "
                   "benchmarks use Chinese A-shares and a 1-day label, so their levels are not comparable.")
        qd = pd.DataFrame(qr)
        qd["_o"] = [({"valid": 0, "test": 1}[s], {"1D": 0, "5D": 1, "20D": 2}[h], {"B": 0, "B_A158": 1, "A158": 2}[n])
                    for s, h, n in zip(qd.split, qd.horizon, qd["set"], strict=True)]
        qd["split"] = qd.split.map({"valid": "2017-18", "test": "2020-22"})
        qd = qd.sort_values("_o").drop(columns="_o").rename(columns={"split": "period"})
        st.dataframe(qd.style.format({m: "{:+.4f}" for m in ("IC", "ICIR", "RankIC", "RankICIR")}), hide_index=True,
                     width="stretch", height=38 + 35 * len(qd))


@st.cache_data(ttl=3600, show_spinner=False)
def _globe_data(label: str):
    from nexus_alpha import paths
    p = paths.ARTIFACTS / "fundamentals" / "addresses.parquet"
    pts, counts = GV.universe_points(pd.read_parquet(p)) if p.exists() else ([], {"placed": 0, "unplaced": 0})
    try:
        macro, year = GV.macro_by_country(label)
    except Exception:  # noqa: BLE001  (GMD files are local and optional)
        macro, year = {}, "n/a"
    return pts, counts, macro, year


def _globe_panel() -> None:
    c = st.columns([3, 1])
    with c[1]:
        label = st.radio("Country colour", list(GV.MACRO), key="mc_globe_var")
    pts, counts, macro, year = _globe_data(label)
    with c[0]:
        GV.render_globe(pts, macro, label, height=520)
    with c[1]:
        st.markdown(f'<div class="nx-note">Columns: {counts["placed"]} of the forward-test companies by business '
                    f'address (SEC), at the state or country centre; {counts["unplaced"]} have no address on file. The '
                    f'universe is US-listed survivors, so it is concentrated in a few states.<br><br>Countries: Global '
                    f'Macro Database (Mueller, Xu, Lehbib and Chen 2025, NBER WP 33714), release 2026_09, latest actual '
                    f'year (mostly {html.escape(year)}). Context only: macro blocks showed no validated skill in the '
                    f'stock models (D-033).<br><br>The pulse marks New York: every forecast is logged before 09:00 '
                    f'there on the next session.</div>', unsafe_allow_html=True)


@st.cache_data(ttl=300, show_spinner="Checking the forward-test hash chains...")
def _forward_status_cached() -> pd.DataFrame:
    return forward_status()      # hashes every logged file; cached for 5 minutes


def render() -> None:
    k = kpis()
    e = e25()
    words = {4: "Four", 5: "Five", 6: "Six"}
    UI.hero("NEXUS Alpha · Mission Control",
            f"A forecasting system built to be audited|{words.get(k['frozen'], str(k['frozen']))} frozen models forecast "
            "about 500 US stocks every evening, into a hash-chained log, before the next open. Research and paper "
            "trading only: nothing here places an order.",
            [{"value": k["frozen"], "label": "frozen models"}, {"value": k["stocks"], "label": "stocks last night"},
             {"value": k["sessions"], "label": "forward sessions"}, {"value": k["experiments"], "label": "experiments run"},
             {"value": k["decisions"], "label": "logged decisions"}, {"value": k["tests"], "label": "test functions"}],
            tape(k, e))
    fs = _forward_status_cached()
    c = st.columns([3, 2])
    with c[0]:
        _forward_panel(fs)
    with c[1]:
        _status_panel(fs, ROOT)
    st.markdown('<div class="nx-kicker" style="margin-top:8px">Evidence board</div>', unsafe_allow_html=True)
    st.markdown("#### What the evidence supports, and what it does not")
    _evidence_board(e)
    st.markdown('<div class="nx-kicker" style="margin-top:8px">Data flow</div>', unsafe_allow_html=True)
    st.markdown("#### From raw data to a logged forecast")
    UI.pipeline_beam()
    st.markdown('<div class="nx-kicker" style="margin-top:8px">Universe and world</div>', unsafe_allow_html=True)
    st.markdown("#### Where the companies are, and the macro backdrop")
    _globe_panel()
    st.markdown('<div class="nx-kicker" style="margin-top:8px">External benchmark</div>', unsafe_allow_html=True)
    _e25_panel(e)

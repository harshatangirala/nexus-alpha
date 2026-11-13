"""E29 forward scores (D-058): E29a, does prior-shift recalibration improve the frozen v1.1's probabilities? and E29b,
does v1.1 with the archived live headlines beat v1.1 without them on the stocks that have news?

    python scripts/shadow/e29_score.py          # writes artifacts/e29/e29_status.json and prints progress

Reads only the forward test's scored rows (shadow/v1.1/scores.csv: on-time rows whose outcome has matured). For each
row, `calibration.label_shift.corrected` reweights v1.1's probabilities with class shares of rows whose labels were
known before that row's decision time, so the corrected forecast never sees its own or a later outcome.

Endpoint (pre-registered, docs/E29_PREREGISTRATION.md): per decision session, the mean over horizons and stocks of
log loss(raw v1.1) - log loss(E29a), on rows the correction touched; one-sided moving-block bootstrap over sessions
(blocks of 20); decided once there are at least 120 such sessions, Holm across E29's arms. Before that it reports
progress only. Nothing is fitted; nothing here changes a logged forecast.

E29b: on-time rows of shadow/v1.1_news (news_on_forecast.py) joined to v1.1's matured, on-time scored rows on (session,
stock, horizon), horizons 1D-1M; per session the mean of log loss(v1.1) - log loss(v1.1 with news). Start condition
(pre-registered): the first session whose news-on run happened after BOTH (a) the archive's 10th clean nightly run and
(b) the first complete on-time news-on log. A clean archive night (operational definition, fixed here before any E29b
outcome exists): a summary file with at least one headline and fewer than half of the tickers failed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from nexus_alpha import shadow as SH  # noqa: E402
from nexus_alpha.calibration import label_shift as LS  # noqa: E402
from nexus_alpha.evaluation import forward_criteria as FC  # noqa: E402

MIN_SESSIONS = 120
OUT = REPO / "artifacts" / "e29" / "e29_status.json"
ARCHIVE = REPO / "artifacts" / "live_news"
E29B_HORIZONS = ["1D", "2D", "5D", "10D", "20D", "1M"]
CLEAN_NIGHTS = 10


def session_gains(scored: pd.DataFrame) -> pd.DataFrame:
    """One row per decision session: mean log-loss gain of E29a over raw v1.1 on the corrected rows."""
    s = scored[scored.real_y.notna() & ~scored.late.astype(bool)].copy()
    c = LS.corrected(s, s)
    c = c[c.n_prior_rows > 0]
    if c.empty:
        return pd.DataFrame(columns=["asof_session", "rows", "gain"])
    y = c.real_y.astype(int).to_numpy()
    raw = c[["p_down", "p_side", "p_up"]].to_numpy(float)
    cor = c[["p_down_ls", "p_side_ls", "p_up_ls"]].to_numpy(float)
    r_ = np.arange(len(c))
    c["gain"] = -np.log(np.clip(raw[r_, y], 1e-7, 1)) + np.log(np.clip(cor[r_, y], 1e-7, 1))
    return c.groupby("asof_session").agg(rows=("gain", "size"), gain=("gain", "mean")).reset_index()


def _ll(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    return -np.log(np.clip(p[np.arange(len(y)), y], 1e-7, 1))


def e29b_start(archive: Path = ARCHIVE, chain: list[dict] | None = None) -> pd.Timestamp | None:
    """UTC time from which E29b's sessions count (None: the start condition does not hold yet)."""
    nights = []
    for f in sorted(archive.glob("headlines_*.summary.json")) if archive.exists() else []:
        try:
            j = json.loads(f.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        if j.get("headlines", 0) > 0 and j.get("failed", 0) < 0.5 * max(j.get("tickers", 1), 1):
            nights.append(pd.Timestamp(f.name[len("headlines_"):len("headlines_") + 10], tz="UTC") + pd.Timedelta(days=1))
    rows = chain if chain is not None else SH.chain_rows("v1.1_news")
    ontime = [pd.Timestamp(r["run_utc"]) for r in rows if int(float(r.get("late_rows", 1) or 0)) == 0]
    if len(nights) < CLEAN_NIGHTS or not ontime:
        return None
    t = max(sorted(nights)[CLEAN_NIGHTS - 1], min(ontime))
    return t if t.tzinfo else t.tz_localize("UTC")


def news_gains(scored: pd.DataFrame, news: pd.DataFrame, start_utc: pd.Timestamp | None) -> pd.DataFrame:
    """One row per decision session: mean log-loss gain of v1.1 with news over v1.1 without, stocks with news."""
    if start_utc is None or scored.empty or news.empty:
        return pd.DataFrame(columns=["asof_session", "rows", "gain"])
    s = scored[scored.real_y.notna() & ~scored.late.astype(bool) & scored.horizon.isin(E29B_HORIZONS)]
    n = news[~news.late.astype(bool) & (pd.to_datetime(news.run_utc, utc=True) > start_utc)]
    m = s.merge(n[["asof_session", "ticker", "horizon", "p_down", "p_side", "p_up"]],
                on=["asof_session", "ticker", "horizon"], suffixes=("", "_news"))
    if m.empty:
        return pd.DataFrame(columns=["asof_session", "rows", "gain"])
    y = m.real_y.astype(int).to_numpy()
    m["gain"] = (_ll(m[["p_down", "p_side", "p_up"]].to_numpy(float), y)
                 - _ll(m[["p_down_news", "p_side_news", "p_up_news"]].to_numpy(float), y))
    return m.groupby("asof_session").agg(rows=("gain", "size"), gain=("gain", "mean")).reset_index()


def holm(p: dict[str, float]) -> dict[str, float]:
    items = sorted(p.items(), key=lambda kv: kv[1])
    out, run = {}, 0.0
    for i, (k, v) in enumerate(items):
        run = max(run, min(1.0, (len(items) - i) * v))
        out[k] = run
    return out


def main() -> None:
    f = SH.SHADOW_ROOT / "v1.1" / "scores.csv"
    scored = pd.read_csv(f) if f.exists() else pd.DataFrame()
    g = session_gains(scored) if len(scored) else pd.DataFrame(columns=["asof_session", "rows", "gain"])
    n = len(g)
    res = {"arm": "E29a prior-shift recalibration of v1.1", "sessions_with_correction": n, "min_sessions": MIN_SESSIONS,
           "rows": int(g.rows.sum()) if n else 0, "mean_gain_per_row": float(g.gain.mean()) if n else None}
    if n >= MIN_SESSIONS:
        res["p_one_sided"] = FC.block_p_value(g.set_index("asof_session").gain)
    nd = SH.SHADOW_ROOT / "v1.1_news" / "forecasts"
    news = (pd.concat([pd.read_csv(x) for x in sorted(nd.glob("*.csv"))], ignore_index=True)
            if nd.exists() and any(nd.glob("*.csv")) else pd.DataFrame())
    start = e29b_start()
    gb = news_gains(scored, news, start) if len(scored) else pd.DataFrame(columns=["asof_session", "rows", "gain"])
    nb = len(gb)
    resb = {"arm": "E29b v1.1 with archived live headlines vs without", "start_utc": None if start is None else str(start),
            "news_on_sessions_logged": int(news.asof_session.nunique()) if len(news) else 0,
            "sessions_scored": nb, "rows": int(gb.rows.sum()) if nb else 0,
            "mean_gain_per_row": float(gb.gain.mean()) if nb else None}
    if nb >= MIN_SESSIONS:
        resb["p_one_sided"] = FC.block_p_value(gb.set_index("asof_session").gain)
    if n >= MIN_SESSIONS and nb >= MIN_SESSIONS:
        adj = holm({"E29a": res["p_one_sided"], "E29b": resb["p_one_sided"]})
        res["p_holm"], resb["p_holm"] = adj["E29a"], adj["E29b"]
        res["decision"] = "KEPT" if adj["E29a"] < 0.05 and res["mean_gain_per_row"] > 0 else "NOT KEPT"
        resb["decision"] = "KEPT" if adj["E29b"] < 0.05 and resb["mean_gain_per_row"] > 0 else "NOT KEPT"
    else:
        res["decision"] = f"progress only: {n} of {MIN_SESSIONS} sessions"
        resb["decision"] = ("start condition not met yet (10 clean archive nights and one on-time news-on log)"
                            if start is None else f"progress only: {nb} of {MIN_SESSIONS} sessions")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"E29a": res, "E29b": resb}, indent=1) + chr(10), encoding="utf-8")
    print(f"E29a: {res['decision']}; rows corrected {res['rows']}", flush=True)
    print(f"E29b: {resb['decision']}; news-on sessions logged {resb['news_on_sessions_logged']}", flush=True)


if __name__ == "__main__":
    main()

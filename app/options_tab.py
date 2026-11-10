"""Tab N: the options market's view of each stock (OpenBB 5.0.0, CBOE delayed quotes; D-050, E26). Context only."""
from __future__ import annotations

import json
import subprocess

import pandas as pd
import plotly.graph_objects as pgo
import streamlit as st

import ui as UI
from nexus_alpha import envs, options as O, paths, shadow as SH

Z90 = 1.2815515655446004          # the 10%-90% band of a normal distribution is +/- 1.2816 sigma


@st.cache_data(ttl=900, show_spinner=False)
def _chain(ticker: str) -> pd.DataFrame | str:
    py = envs.env_python("openbb")
    out = paths.ARTIFACTS / "openbb_live" / f"{ticker}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run([str(py), "-I", str(paths.REPO_ROOT / "scripts" / "openbb" / "options_chains.py"), "--tickers",
                        ticker, "--out", str(out)], capture_output=True, text=True, timeout=300, env=envs.child_env())
    if r.returncode != 0 or not out.exists():
        ef = out.with_name(out.name + ".errors.json")
        err = (json.loads(ef.read_text(encoding="utf-8")).get("errors", {}).get(ticker, "") if ef.exists()
               else r.stderr[-300:])
        return f"No option chain for {ticker}: {err}"
    return pd.read_parquet(out)


def _model_range(ticker: str) -> dict:
    """v1.1's last logged 10%-90% return range for the stock (only where its coverage was validated)."""
    fs = sorted((SH.SHADOW_ROOT / "v1.1" / "forecasts").glob("????-??-??.csv"))
    if not fs:
        return {}
    v = pd.read_csv(fs[-1])
    v = v[v.ticker == ticker].set_index("horizon")
    return {h: (float(v.ret_q10[h]), float(v.ret_q90[h]), fs[-1].stem) for h in ("1D", "5D", "20D")
            if h in v.index and pd.notna(v.ret_q10[h])}


def _snapshot_status() -> str:
    pubs = sorted((SH.SHADOW_ROOT / "iv_snapshot").glob("????-??-??.json"))
    if not pubs:
        return "No nightly snapshot logged yet; the first is taken by tonight's run."
    last = json.loads(pubs[-1].read_text(encoding="utf-8"))
    b = SH.SHADOW_ROOT / "iv_snapshot" / "benchmark.json"
    dec = ""
    if b.exists():
        res = json.loads(b.read_text(encoding="utf-8")).get("results", {})
        dec = " · ".join(f"{k}: {v.get('decision', '')}" for k, v in res.items())
    return (f"{len(pubs)} nightly snapshots logged; last {last['asof_session']}: {last['tickers_with_iv30']} of "
            f"{last['tickers_requested']} stocks with an implied volatility. " + (dec or "Nothing scored yet."))


def render(tickers: list[str]) -> None:
    st.header("Options market view (OpenBB · CBOE delayed quotes)")
    st.caption("The options market prices the size of future moves, the one thing the models predict well. This view "
               "shows that price next to the model's own range. The forward benchmark E26 tests whether the model adds "
               "anything beyond it. Context only: it never changes a forecast.")
    if envs.env_python("openbb") is None:
        st.info("OpenBB is not installed (artifacts/envs/openbb).")
        return
    tick = st.selectbox("Ticker", tickers, index=tickers.index("AAPL") if "AAPL" in tickers else 0, key="op_tick")
    if st.button("Fetch the option chain", type="primary", key="op_go"):
        st.session_state["op_last"] = tick
    if st.session_state.get("op_last") == tick:
        with st.spinner(f"Fetching {tick}'s option chain from CBOE through OpenBB..."):
            chain = _chain(tick)
        if isinstance(chain, str):
            st.warning(chain)
        else:
            s = O.summarize(chain)
            term = O.atm_term_structure(chain)
            c = st.columns(4)
            c[0].metric("Underlying (delayed)", f"{s['underlying_price']:,.2f}")
            c[1].metric("30-day implied volatility", f"{s['iv30']:.1%}")
            c[2].metric(f"Front expiry ATM IV ({s['front_dte']} days)", f"{s['front_atm_iv']:.1%}")
            c[3].metric("Put/call open interest", f"{s.get('put_call_oi', float('nan')):.2f}")
            mr = _model_range(tick)
            rows = []
            for h, n in (("1D", 1), ("5D", 5), ("20D", 20)):
                sig = O.implied_move(s["iv30"], n)
                rows.append({"horizon": h, "options-implied 1-sigma move": f"±{sig:.2%}",
                             "options-implied 10%-90% range": f"{-Z90 * sig:+.2%} .. {Z90 * sig:+.2%}",
                             "v1.1 10%-90% range (last log)": (f"{mr[h][0]:+.2%} .. {mr[h][1]:+.2%}" if h in mr
                                                               else "not shown (coverage not validated)")})
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            st.caption("Options-implied ranges assume a zero-drift normal model with IV30 scaled by sqrt(h/252). The "
                       "model's range comes from its quantile heads in the last forward-test log"
                       + (f" ({next(iter(mr.values()))[2]})." if mr else "."))
            fig = pgo.Figure(pgo.Scatter(x=term.dte, y=term.atm_iv, mode="lines+markers", line=dict(color=UI.ACCENT),
                                         hovertemplate="%{x} days: %{y:.1%}<extra></extra>"))
            fig.update_layout(xaxis_title="days to expiry", yaxis=dict(title="ATM implied volatility", tickformat=".0%"))
            UI.plot(fig, height=320)
    st.subheader("Forward benchmark (E26)")
    st.caption(_snapshot_status())

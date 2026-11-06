"""Daily point-in-time ratios for one ticker: the latest snapshot filed on or before each session, priced at that
session's as-traded close.

As-traded price = the vendor's split-adjusted close x the product of all later split factors. Shares outstanding from
the cover page are moved to the session's share basis with the splits between the cover date and the session.
The 31 pre-registered ratios (docs/V3_RESEARCH.md, E22) are in RATIOS; the multiples in MULTIPLES are display only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TAX = 0.21
DCF = {"years": 5, "g_lo": -0.05, "g_hi": 0.15, "g_term": 0.025, "r": 0.09}
RATIOS = ["earnings_yield", "sales_yield", "book_to_price", "ebitda_to_ev", "fcf_yield", "dividend_yield",
          "shareholder_yield", "dcf_to_price",
          "gross_margin", "operating_margin", "net_margin", "ebitda_margin", "roe", "roa", "roic",
          "revenue_growth", "operating_income_growth", "eps_growth", "asset_growth", "share_issuance",
          "debt_to_equity", "net_debt_to_ebitda", "interest_coverage", "current_ratio", "cash_to_assets",
          "asset_turnover", "receivables_to_sales",
          "accruals", "piotroski_f", "altman_z", "sue"]
MULTIPLES = ["pe", "ev_ebitda", "ps", "pb", "p_fcf", "market_cap", "enterprise_value"]


def _div(a, b, pos_den: bool = False):
    a, b = np.asarray(a, float), np.asarray(b, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = a / b
    bad = ~np.isfinite(out) | (b == 0) | (pos_den & (b <= 0))
    out[bad] = np.nan
    return out


def split_factor_after(dates: pd.DatetimeIndex, splits: pd.Series) -> np.ndarray:
    """Product of split ratios with split date strictly after each date (splits: ratio indexed by date)."""
    if splits is None or splits.empty:
        return np.ones(len(dates))
    s = splits.sort_index()
    cum_after = np.r_[np.cumprod(s.values[::-1])[::-1], 1.0]          # product of splits from position i onward
    pos = np.searchsorted(s.index.values, dates.values, side="right")   # first split strictly after the date
    return cum_after[pos]


def dcf_equity(fcf, growth, debt, cash):
    """Two-stage FCF DCF with fixed assumptions (E22): equity value."""
    g = np.clip(np.asarray(growth, float), DCF["g_lo"], DCF["g_hi"])
    g = np.where(np.isfinite(g), g, 0.0)
    r, gt, n = DCF["r"], DCF["g_term"], DCF["years"]
    pv = np.zeros_like(np.asarray(fcf, float))
    f = np.asarray(fcf, float)
    for k in range(1, n + 1):
        pv = pv + f * (1 + g) ** k / (1 + r) ** k
    tv = f * (1 + g) ** n * (1 + gt) / (r - gt) / (1 + r) ** n
    return pv + tv - np.nan_to_num(np.asarray(debt, float)) + np.nan_to_num(np.asarray(cash, float))


def daily(snap: pd.DataFrame, bars: pd.DataFrame, splits: pd.Series, sue: pd.DataFrame | None = None,
          start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """bars: date, close (vendor split-adjusted). Returns one row per session with RATIOS + MULTIPLES."""
    if snap is None or snap.empty or bars.empty:
        return pd.DataFrame()
    b = bars[["date", "close"]].copy()
    if start:
        b = b[b.date >= pd.Timestamp(start)]
    if end:
        b = b[b.date <= pd.Timestamp(end)]
    if b.empty:
        return pd.DataFrame()
    s = snap.sort_values("snap_date")
    m = pd.merge_asof(b.sort_values("date"), s, left_on="date", right_on="snap_date", direction="backward")
    m = m[m["snap_date"].notna()].reset_index(drop=True)
    if m.empty:
        return pd.DataFrame()
    dates = pd.DatetimeIndex(m.date)
    price = m.close.to_numpy(float) * split_factor_after(dates, splits)
    sd = pd.DatetimeIndex(m.shares_date)
    sh_adj = np.ones(len(m))
    if splits is not None and not splits.empty:
        f_sd = split_factor_after(sd.fillna(dates[0]), splits)
        f_t = split_factor_after(dates, splits)
        sh_adj = _div(f_sd, f_t)
    shares = m.shares_outstanding.to_numpy(float) * sh_adj
    mcap = shares * price
    debt = np.where(m.lt_debt_total.notna(), m.lt_debt_total,
                    m.lt_debt_noncurrent.fillna(0) + m.lt_debt_current.fillna(0)) + m.st_borrowings.fillna(0)
    debt_prev = np.where(m.lt_debt_total_prev.notna(), m.lt_debt_total_prev, m.lt_debt_noncurrent_prev.fillna(0))
    cash = m.cash.to_numpy(float)
    ev = mcap + debt - np.nan_to_num(cash)
    rev, rev_p = m.ttm_revenue, m.ttm_revenue_prev
    gp = np.where(m.ttm_gross_profit.notna(), m.ttm_gross_profit, rev - m.ttm_cogs)
    gp_p = np.where(m.ttm_gross_profit_prev.notna(), m.ttm_gross_profit_prev, rev_p - m.ttm_cogs_prev)
    ebit, ebit_p = m.ttm_operating_income, m.ttm_operating_income_prev
    ebitda = ebit + m.ttm_da
    ni, ni_p = m.ttm_net_income, m.ttm_net_income_prev
    cfo = m.ttm_cfo
    fcf = cfo - m.ttm_capex.fillna(0)
    liab = np.where(m.liabilities.notna(), m.liabilities, m.assets - m.equity)
    o = pd.DataFrame({"date": dates})
    o["market_cap"], o["enterprise_value"] = mcap, ev
    o["earnings_yield"] = _div(ni, mcap, True)
    o["sales_yield"] = _div(rev, mcap, True)
    o["book_to_price"] = _div(m.equity, mcap, True)
    o["ebitda_to_ev"] = _div(ebitda, ev, True)
    o["fcf_yield"] = _div(fcf, mcap, True)
    o["dividend_yield"] = _div(m.ttm_dividends.fillna(0), mcap, True)
    o["shareholder_yield"] = _div(m.ttm_dividends.fillna(0) + m.ttm_buybacks.fillna(0), mcap, True)
    o["dcf_to_price"] = _div(dcf_equity(fcf, _div(rev, rev_p, True) - 1, debt, cash), mcap, True)
    o["gross_margin"] = _div(gp, rev, True)
    o["operating_margin"] = _div(ebit, rev, True)
    o["net_margin"] = _div(ni, rev, True)
    o["ebitda_margin"] = _div(ebitda, rev, True)
    o["roe"] = _div(ni, m.equity, True)
    o["roa"] = _div(ni, m.assets, True)
    o["roic"] = _div(ebit * (1 - TAX), debt + m.equity - np.nan_to_num(cash), True)
    o["revenue_growth"] = _div(rev, rev_p, True) - 1
    o["operating_income_growth"] = _div(ebit - ebit_p, np.abs(ebit_p), True)
    eps, eps_p = m.ttm_eps_diluted, m.ttm_eps_diluted_prev
    o["eps_growth"] = _div(eps - eps_p, np.abs(eps_p), True)
    o["asset_growth"] = _div(m.assets, m.assets_prev, True) - 1
    o["share_issuance"] = _div(m.q_diluted_shares, m.q_diluted_shares_prev, True) - 1
    o["debt_to_equity"] = _div(debt, m.equity, True)
    o["net_debt_to_ebitda"] = _div(debt - np.nan_to_num(cash), ebitda, True)
    o["interest_coverage"] = _div(ebit, m.ttm_interest_expense, True)
    o["current_ratio"] = _div(m.assets_current, m.liabilities_current, True)
    o["cash_to_assets"] = _div(cash, m.assets, True)
    o["asset_turnover"] = _div(rev, m.assets, True)
    o["receivables_to_sales"] = _div(m.receivables, rev, True)
    o["accruals"] = _div(ni - cfo, m.assets, True)
    # Piotroski F-score: 9 binary signals, current TTM vs the TTM four quarters earlier
    roa, roa_p = _div(ni, m.assets_prev, True), _div(ni_p, m.assets_prev, True)
    lev, lev_p = _div(debt, m.assets, True), _div(debt_prev, m.assets_prev, True)
    cr, cr_p = _div(m.assets_current, m.liabilities_current, True), _div(m.assets_current_prev, m.liabilities_current_prev, True)
    gm, gm_p = _div(gp, rev, True), _div(gp_p, rev_p, True)
    at, at_p = _div(rev, m.assets_prev, True), _div(rev_p, m.assets_prev, True)
    sig = [roa > 0, cfo > 0, roa > roa_p, cfo > ni, lev < lev_p, cr > cr_p,
           m.q_diluted_shares <= m.q_diluted_shares_prev, gm > gm_p, at > at_p]
    o["piotroski_f"] = np.sum([np.nan_to_num(np.asarray(x, float)) for x in sig], axis=0)
    o["piotroski_available"] = np.sum([np.isfinite(np.asarray(v, float)) for v in (roa, cfo, roa_p, ni, lev, lev_p, cr, cr_p, gm)], axis=0)
    o.loc[o.piotroski_available < 6, "piotroski_f"] = np.nan
    ta = m.assets.to_numpy(float)
    o["altman_z"] = (1.2 * _div(m.assets_current - m.liabilities_current, ta, True)
                     + 1.4 * _div(m.retained_earnings, ta, True) + 3.3 * _div(ebit, ta, True)
                     + 0.6 * _div(mcap, liab, True) + 1.0 * _div(rev, ta, True))
    o["sue"] = np.nan
    if sue is not None and len(sue):
        sm = pd.merge_asof(o[["date"]], sue.sort_values("avail_date")[["avail_date", "sue"]], left_on="date",
                           right_on="avail_date", direction="backward")
        age = (sm.date - sm.avail_date).dt.days
        o["sue"] = np.where(age <= 120, sm.sue, np.nan)                 # a surprise older than ~one quarter is stale
    o["days_since_filing"] = (dates - pd.DatetimeIndex(m["snap_date"])).days
    o["pe"] = _div(mcap, ni, True)
    o["ev_ebitda"] = _div(ev, ebitda, True)
    o["ps"] = _div(mcap, rev, True)
    o["pb"] = _div(mcap, m.equity, True)
    o["p_fcf"] = _div(mcap, fcf, True)
    o["price_as_traded"] = price
    return o

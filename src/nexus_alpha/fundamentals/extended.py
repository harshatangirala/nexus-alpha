"""Extended fundamentals: published composite and forensic scores on the point-in-time snapshots.

`metrics(m)` works on rows that carry the snapshot columns of `statements.snapshots` (TTM flows now and four quarters
earlier, balance-sheet instants now and about a year earlier) plus, optionally, `market_cap` and `enterprise_value`
from `ratios.daily`. Everything is as known at the filing date: no new data source, no fitted parameter.
`cross_section(df)` turns one session's rows into universe percentiles and the published composites.

Definitions (fixed here, before any outcome is looked at; docs/E30_PREREGISTRATION.md):
* Beneish (1999) M-score, eight variables, published coefficients:
  M = -4.84 + 0.920 DSRI + 0.528 GMI + 0.404 AQI + 0.892 SGI + 0.115 DEPI - 0.172 SGAI + 4.679 TATA - 0.327 LVGI.
  A firm with no SG&A or no D&A tag gets the neutral 1.0 for SGAI / DEPI (counted in beneish_imputed). Flag: M > -1.78.
* Montier (2008) C-score, 0-6: NI - CFO gap widening; days sales outstanding rising; days inventory rising; other
  current assets / revenue rising; D&A / net PP&E falling (net, not gross, PP&E: the gross figure is not tagged
  consistently); total asset growth above 10%.
* Novy-Marx (2013) gross profitability: (revenue - COGS) / total assets.
* Greenblatt (2005) magic formula: earnings yield EBIT / EV and return on capital EBIT / (net working capital + net PP&E),
  with net working capital = (current assets - cash) - (current liabilities - short-term debt), floored at zero.
* QMJ-style quality (after Asness, Frazzini and Pedersen 2019), from what point-in-time XBRL supports: profitability
  (gross profitability, ROE, ROA, cash ROA, gross margin, low accruals), growth (one-year change in gross profitability,
  ROA and cash ROA; the paper uses five years) and safety (low leverage, Altman Z, low Beneish M). No beta and no
  earnings volatility. Each part is the mean of percentile ranks; quality is the mean of the three parts.
* Value composite: mean percentile of earnings yield, book-to-price, EBITDA/EV, FCF yield and sales yield.
Ohlson's O-score is not computed: its size term needs a 1968-based GNP price index, and its 1970s coefficients are
known to degrade on recent data.

Scores are diagnostics and candidate inputs. None of them changes a frozen model, a forecast or the risk engine.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BENEISH = {"const": -4.84, "dsri": 0.920, "gmi": 0.528, "aqi": 0.404, "sgi": 0.892, "depi": 0.115, "sgai": -0.172,
           "tata": 4.679, "lvgi": -0.327}
BENEISH_FLAG = -1.78
MONTIER_ASSET_GROWTH = 0.10
FINANCIAL_SIC2 = range(60, 68)          # banks, insurers, brokers, REITs: Beneish, Altman and the magic formula exclude them

EXTENDED = ["ev_sales", "ev_ebit", "ebit_to_ev", "peg", "payout_ratio",
            "gross_profitability", "cash_roa", "fcf_margin", "roce", "greenblatt_roc", "equity_multiplier",
            "dso", "dio", "capex_to_da", "rnd_intensity", "sga_intensity", "cash_conversion",
            "quick_ratio", "cash_ratio", "debt_to_assets", "liabilities_to_assets",
            "beneish_m", "montier_c"]
BENEISH_PARTS = ["dsri", "gmi", "aqi", "sgi", "depi", "sgai", "lvgi", "tata"]
COMPOSITES = ["value_composite", "quality_profitability", "quality_growth", "quality_safety", "quality_composite",
              "magic_formula", "gross_profitability_rank", "piotroski_rank", "beneish_safety_rank"]


def _a(x) -> np.ndarray:
    return np.asarray(pd.to_numeric(x, errors="coerce") if isinstance(x, pd.Series) else x, float)


def _div(a, b, pos_den: bool = False) -> np.ndarray:
    a, b = _a(a), _a(b)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = a / b
    bad = ~np.isfinite(out) | (b == 0) | (pos_den & (b <= 0))
    out = np.where(bad, np.nan, out)
    return out


def _col(m: pd.DataFrame, c: str) -> np.ndarray:
    return _a(m[c]) if c in m else np.full(len(m), np.nan)


def _debt(m: pd.DataFrame, prev: bool = False) -> np.ndarray:
    s = "_prev" if prev else ""
    tot = _col(m, f"lt_debt_total{s}")
    parts = np.nan_to_num(_col(m, f"lt_debt_noncurrent{s}")) + np.nan_to_num(_col(m, f"lt_debt_current{s}"))
    out = np.where(np.isfinite(tot), tot, parts) + np.nan_to_num(_col(m, f"st_borrowings{s}"))
    has = np.isfinite(tot) | np.isfinite(_col(m, f"lt_debt_noncurrent{s}")) | np.isfinite(_col(m, f"st_borrowings{s}"))
    return np.where(has, out, np.nan)


def _gp(m: pd.DataFrame, prev: bool = False) -> np.ndarray:
    s = "_prev" if prev else ""
    gp = _col(m, f"ttm_gross_profit{s}")
    return np.where(np.isfinite(gp), gp, _col(m, f"ttm_revenue{s}") - _col(m, f"ttm_cogs{s}"))


def beneish(m: pd.DataFrame) -> pd.DataFrame:
    """The eight Beneish indices and M. SGAI / DEPI fall back to the neutral 1.0 when the tag is missing."""
    rev, rev_p = _col(m, "ttm_revenue"), _col(m, "ttm_revenue_prev")
    rec, rec_p = _col(m, "receivables"), _col(m, "receivables_prev")
    ta, ta_p = _col(m, "assets"), _col(m, "assets_prev")
    ca, ca_p = _col(m, "assets_current"), _col(m, "assets_current_prev")
    ppe, ppe_p = _col(m, "ppe"), _col(m, "ppe_prev")
    da, da_p = _col(m, "ttm_da"), _col(m, "ttm_da_prev")
    sga, sga_p = _col(m, "ttm_sga"), _col(m, "ttm_sga_prev")
    cl, cl_p = _col(m, "liabilities_current"), _col(m, "liabilities_current_prev")
    ltd = np.where(np.isfinite(_col(m, "lt_debt_noncurrent")), _col(m, "lt_debt_noncurrent"), _col(m, "lt_debt_total"))
    ltd_p = np.where(np.isfinite(_col(m, "lt_debt_noncurrent_prev")), _col(m, "lt_debt_noncurrent_prev"),
                     _col(m, "lt_debt_total_prev"))
    gm, gm_p = _div(_gp(m), rev, True), _div(_gp(m, True), rev_p, True)
    o = pd.DataFrame(index=m.index)
    o["dsri"] = _div(_div(rec, rev, True), _div(rec_p, rev_p, True), True)
    o["gmi"] = _div(gm_p, gm, True)
    o["aqi"] = _div(1 - _div(ca + ppe, ta, True), 1 - _div(ca_p + ppe_p, ta_p, True), True)
    o["sgi"] = _div(rev, rev_p, True)
    depi = _div(_div(da_p, da_p + ppe_p, True), _div(da, da + ppe, True), True)
    sgai = _div(_div(sga, rev, True), _div(sga_p, rev_p, True), True)
    o["depi"] = np.where(np.isfinite(depi), depi, 1.0)
    o["sgai"] = np.where(np.isfinite(sgai), sgai, 1.0)
    o["lvgi"] = _div(_div(cl + np.nan_to_num(ltd), ta, True), _div(cl_p + np.nan_to_num(ltd_p), ta_p, True), True)
    o["tata"] = _div(_col(m, "ttm_net_income") - _col(m, "ttm_cfo"), ta, True)
    o["beneish_imputed"] = (~np.isfinite(depi)).astype(int) + (~np.isfinite(sgai)).astype(int)
    core = ["dsri", "gmi", "aqi", "sgi", "lvgi", "tata"]
    ok = np.all([np.isfinite(o[c].to_numpy(float)) for c in core], axis=0)
    mm = BENEISH["const"] + sum(BENEISH[c] * o[c].to_numpy(float) for c in BENEISH_PARTS)
    o["beneish_m"] = np.where(ok, mm, np.nan)
    o["beneish_flag"] = np.where(ok, (mm > BENEISH_FLAG).astype(float), np.nan)
    return o


def montier(m: pd.DataFrame) -> pd.DataFrame:
    """Montier C-score (0-6) and the number of its six signals that could be evaluated."""
    rev, rev_p = _col(m, "ttm_revenue"), _col(m, "ttm_revenue_prev")
    cogs, cogs_p = _col(m, "ttm_cogs"), _col(m, "ttm_cogs_prev")
    ni, ni_p = _col(m, "ttm_net_income"), _col(m, "ttm_net_income_prev")
    cfo, cfo_p = _col(m, "ttm_cfo"), _col(m, "ttm_cfo_prev")
    rec, rec_p = _col(m, "receivables"), _col(m, "receivables_prev")
    inv, inv_p = _col(m, "inventory"), _col(m, "inventory_prev")
    ca, ca_p = _col(m, "assets_current"), _col(m, "assets_current_prev")
    cash, cash_p = _col(m, "cash"), _col(m, "cash_prev")
    oca = ca - np.nan_to_num(cash) - np.nan_to_num(rec) - np.nan_to_num(inv)
    oca_p = ca_p - np.nan_to_num(cash_p) - np.nan_to_num(rec_p) - np.nan_to_num(inv_p)
    da, da_p, ppe, ppe_p = _col(m, "ttm_da"), _col(m, "ttm_da_prev"), _col(m, "ppe"), _col(m, "ppe_prev")
    ag = _div(_col(m, "assets"), _col(m, "assets_prev"), True) - 1
    pairs = [((ni - cfo), (ni_p - cfo_p)),
             (_div(rec, rev, True), _div(rec_p, rev_p, True)),
             (_div(inv, cogs, True), _div(inv_p, cogs_p, True)),
             (_div(oca, rev, True), _div(oca_p, rev_p, True)),
             (-_div(da, ppe, True), -_div(da_p, ppe_p, True))]          # falling D&A / PP&E = rising negative
    sig, avail = [], []
    for now, before in pairs:
        f = np.isfinite(now) & np.isfinite(before)
        sig.append(np.where(f, (now > before).astype(float), 0.0))
        avail.append(f)
    f = np.isfinite(ag)
    sig.append(np.where(f, (ag > MONTIER_ASSET_GROWTH).astype(float), 0.0))
    avail.append(f)
    n = np.sum(avail, axis=0)
    o = pd.DataFrame(index=m.index)
    o["montier_c"] = np.where(n >= 4, np.sum(sig, axis=0), np.nan)
    o["montier_available"] = n
    return o


def metrics(m: pd.DataFrame) -> pd.DataFrame:
    """Extended per-row metrics. `m`: snapshot columns (+ market_cap, enterprise_value, pe, eps_growth when present)."""
    rev, cogs = _col(m, "ttm_revenue"), _col(m, "ttm_cogs")
    ebit, ni, cfo = _col(m, "ttm_operating_income"), _col(m, "ttm_net_income"), _col(m, "ttm_cfo")
    capex, da = np.nan_to_num(_col(m, "ttm_capex")), _col(m, "ttm_da")
    ta, eq = _col(m, "assets"), _col(m, "equity")
    ca, cl = _col(m, "assets_current"), _col(m, "liabilities_current")
    cash, inv, rec, ppe = _col(m, "cash"), _col(m, "inventory"), _col(m, "receivables"), _col(m, "ppe")
    liab = np.where(np.isfinite(_col(m, "liabilities")), _col(m, "liabilities"), ta - eq)
    debt = _debt(m)
    ev, mcap = _col(m, "enterprise_value"), _col(m, "market_cap")
    fcf = cfo - capex
    o = pd.DataFrame(index=m.index)
    o["ev_sales"] = _div(ev, rev, True)
    o["ev_ebit"] = _div(ev, ebit, True)
    o["ebit_to_ev"] = _div(ebit, ev, True)
    pe, eg = _col(m, "pe"), _col(m, "eps_growth")
    o["peg"] = np.where((pe > 0) & (eg > 0), _div(pe, eg * 100, True), np.nan)
    o["payout_ratio"] = _div(np.nan_to_num(_col(m, "ttm_dividends")), ni, True)
    o["gross_profitability"] = _div(_gp(m), ta, True)
    o["cash_roa"] = _div(cfo, ta, True)
    o["fcf_margin"] = _div(fcf, rev, True)
    o["roce"] = _div(ebit, ta - cl, True)
    st_debt = np.nan_to_num(_col(m, "st_borrowings")) + np.nan_to_num(_col(m, "lt_debt_current"))
    nwc = np.maximum((ca - np.nan_to_num(cash)) - (cl - st_debt), 0.0)
    o["greenblatt_roc"] = _div(ebit, nwc + ppe, True)
    o["equity_multiplier"] = _div(ta, eq, True)
    o["dso"] = _div(rec, rev, True) * 365
    o["dio"] = _div(inv, cogs, True) * 365
    o["capex_to_da"] = _div(capex, da, True)
    o["rnd_intensity"] = _div(_col(m, "ttm_rnd"), rev, True)
    o["sga_intensity"] = _div(_col(m, "ttm_sga"), rev, True)
    o["cash_conversion"] = _div(cfo, ni, True)
    o["quick_ratio"] = _div(ca - np.nan_to_num(inv), cl, True)
    o["cash_ratio"] = _div(cash, cl, True)
    o["debt_to_assets"] = _div(debt, ta, True)
    o["liabilities_to_assets"] = _div(liab, ta, True)
    # one-year change of profitability (the growth leg of the QMJ-style score)
    ta_p = _col(m, "assets_prev")
    o["d_gross_profitability"] = o["gross_profitability"] - _div(_gp(m, True), ta_p, True)
    o["d_roa"] = _div(ni, ta, True) - _div(_col(m, "ttm_net_income_prev"), ta_p, True)
    o["d_cash_roa"] = o["cash_roa"] - _div(_col(m, "ttm_cfo_prev"), ta_p, True)
    return pd.concat([o, beneish(m), montier(m)], axis=1)


def _pct(x: pd.Series, higher_is_better: bool = True) -> pd.Series:
    r = pd.to_numeric(x, errors="coerce").rank(pct=True)
    return r if higher_is_better else 1 - r + 1 / max(r.notna().sum(), 1)


def _mean_pct(df: pd.DataFrame, cols: list[tuple[str, bool]], min_n: int) -> pd.Series:
    parts = pd.concat([_pct(df[c], hb) if c in df else pd.Series(np.nan, index=df.index) for c, hb in cols], axis=1)
    return parts.mean(axis=1).where(parts.notna().sum(axis=1) >= min_n)


def cross_section(df: pd.DataFrame, sic2: pd.Series | None = None) -> pd.DataFrame:
    """Published composites for ONE session's cross-section (rows = stocks). Inputs: the RATIOS of `ratios.daily` and
    the EXTENDED metrics. Higher score = more attractive by the composite's own published logic. Financial firms (SIC
    60-67) get no Beneish, Altman or magic-formula score; when `sic2` is not given nobody is excluded."""
    d = df.copy()
    fin = (sic2.reindex(d.index).isin(list(FINANCIAL_SIC2)) if sic2 is not None
           else pd.Series(False, index=d.index))
    for c in ("beneish_m", "altman_z", "ebit_to_ev", "greenblatt_roc"):
        if c in d:
            d.loc[fin, c] = np.nan
    o = pd.DataFrame(index=d.index)
    o["value_composite"] = _mean_pct(d, [("earnings_yield", True), ("book_to_price", True), ("ebitda_to_ev", True),
                                         ("fcf_yield", True), ("sales_yield", True)], 3)
    o["quality_profitability"] = _mean_pct(d, [("gross_profitability", True), ("roe", True), ("roa", True),
                                               ("cash_roa", True), ("gross_margin", True), ("accruals", False)], 3)
    o["quality_growth"] = _mean_pct(d, [("d_gross_profitability", True), ("d_roa", True), ("d_cash_roa", True)], 2)
    o["quality_safety"] = _mean_pct(d, [("debt_to_assets", False), ("altman_z", True), ("beneish_m", False)], 2)
    o["quality_composite"] = o[["quality_profitability", "quality_growth", "quality_safety"]].mean(axis=1).where(
        o[["quality_profitability", "quality_growth", "quality_safety"]].notna().sum(axis=1) >= 2)
    ey, roc = _pct(d.get("ebit_to_ev", pd.Series(np.nan, index=d.index))), _pct(
        d.get("greenblatt_roc", pd.Series(np.nan, index=d.index)))
    o["magic_formula"] = ((ey + roc) / 2).rank(pct=True).where(ey.notna() & roc.notna())
    o["gross_profitability_rank"] = _pct(d.get("gross_profitability", pd.Series(np.nan, index=d.index)))
    o["piotroski_rank"] = _pct(d.get("piotroski_f", pd.Series(np.nan, index=d.index)))
    o["beneish_safety_rank"] = _pct(d.get("beneish_m", pd.Series(np.nan, index=d.index)), higher_is_better=False)
    return o

"""Extended fundamentals: Beneish M, Montier C, gross profitability, the magic formula and the cross-section composites."""
import numpy as np
import pandas as pd
import pytest

from nexus_alpha.fundamentals import extended as X


def _row(**over):
    r = dict(ttm_revenue=1000.0, ttm_revenue_prev=900.0, ttm_cogs=600.0, ttm_cogs_prev=560.0,
             ttm_gross_profit=np.nan, ttm_gross_profit_prev=np.nan, ttm_operating_income=150.0,
             ttm_operating_income_prev=130.0, ttm_net_income=100.0, ttm_net_income_prev=90.0, ttm_cfo=120.0,
             ttm_cfo_prev=110.0, ttm_capex=40.0, ttm_da=50.0, ttm_da_prev=48.0, ttm_sga=200.0, ttm_sga_prev=185.0,
             ttm_rnd=30.0, ttm_dividends=20.0, assets=2000.0, assets_prev=1850.0, assets_current=800.0,
             assets_current_prev=760.0, liabilities=900.0, liabilities_current=400.0, liabilities_current_prev=380.0,
             equity=1100.0, cash=200.0, cash_prev=190.0, receivables=150.0, receivables_prev=130.0, inventory=100.0,
             inventory_prev=95.0, ppe=600.0, ppe_prev=580.0, lt_debt_noncurrent=300.0, lt_debt_noncurrent_prev=310.0,
             lt_debt_current=20.0, st_borrowings=10.0, lt_debt_total=np.nan, lt_debt_total_prev=np.nan,
             market_cap=3000.0, enterprise_value=3130.0, pe=30.0, eps_growth=0.12)
    r.update(over)
    return pd.DataFrame([r])


def test_beneish_matches_hand_calculation():
    m = _row()
    b = X.beneish(m).iloc[0]
    dsri = (150 / 1000) / (130 / 900)
    gmi = ((900 - 560) / 900) / ((1000 - 600) / 1000)
    aqi = (1 - (800 + 600) / 2000) / (1 - (760 + 580) / 1850)
    sgi = 1000 / 900
    depi = (48 / (48 + 580)) / (50 / (50 + 600))
    sgai = (200 / 1000) / (185 / 900)
    lvgi = ((400 + 300) / 2000) / ((380 + 310) / 1850)
    tata = (100 - 120) / 2000
    want = (-4.84 + 0.92 * dsri + 0.528 * gmi + 0.404 * aqi + 0.892 * sgi + 0.115 * depi - 0.172 * sgai
            + 4.679 * tata - 0.327 * lvgi)
    assert b.beneish_m == pytest.approx(want, rel=1e-9)
    assert b.beneish_flag == float(want > -1.78)
    assert b.beneish_imputed == 0


def test_beneish_neutral_when_sga_missing_and_nan_when_core_missing():
    b = X.beneish(_row(ttm_sga=np.nan)).iloc[0]
    assert b.sgai == 1.0 and b.beneish_imputed == 1 and np.isfinite(b.beneish_m)
    assert np.isnan(X.beneish(_row(receivables_prev=np.nan)).iloc[0].beneish_m)


def test_montier_counts_signals():
    c = X.montier(_row()).iloc[0]
    # NI-CFO gap -20 vs -20 (not widening), DSO up, DIO down? 100/600 vs 95/560 -> up, OCA/rev, D&A/PPE, asset growth 8.1%
    oca, oca_p = 800 - 200 - 150 - 100, 760 - 190 - 130 - 95
    expect = [(100 - 120) > (90 - 110), 150 / 1000 > 130 / 900, 100 / 600 > 95 / 560, oca / 1000 > oca_p / 900,
              -(50 / 600) > -(48 / 580), 2000 / 1850 - 1 > 0.10]
    assert c.montier_c == sum(expect) and c.montier_available == 6


def test_metrics_core_values():
    x = X.metrics(_row()).iloc[0]
    assert x.gross_profitability == pytest.approx(400 / 2000)
    assert x.ebit_to_ev == pytest.approx(150 / 3130)
    nwc = (800 - 200) - (400 - 30)
    assert x.greenblatt_roc == pytest.approx(150 / (nwc + 600))
    assert x.peg == pytest.approx(30 / 12)
    assert x.quick_ratio == pytest.approx((800 - 100) / 400)
    assert x.dso == pytest.approx(150 / 1000 * 365)
    assert x.debt_to_assets == pytest.approx((300 + 20 + 10) / 2000)


def test_negative_denominators_are_missing_not_flipped():
    x = X.metrics(_row(ttm_net_income=-50.0, equity=-10.0)).iloc[0]
    assert np.isnan(x.cash_conversion) and np.isnan(x.equity_multiplier) and np.isnan(x.payout_ratio)


def test_cross_section_ranks_and_financial_exclusion():
    rows = pd.concat([_row(), _row(ttm_operating_income=300.0, enterprise_value=2000.0, ttm_cogs=300.0),
                      _row(ttm_operating_income=50.0)], ignore_index=True)
    met = pd.concat([rows, X.metrics(rows)], axis=1)
    met["earnings_yield"] = [0.03, 0.06, 0.02]
    cs = X.cross_section(met)
    assert cs.magic_formula.idxmax() == 1 and cs.gross_profitability_rank.idxmax() == 1
    assert cs.value_composite.isna().all()                 # only one of five value inputs present: below min_n
    sic = pd.Series([20, 61, 35])
    cs2 = X.cross_section(met, sic2=sic)
    assert np.isnan(cs2.magic_formula.iloc[1]) and np.isnan(cs2.beneish_safety_rank.iloc[1])
    assert set(X.COMPOSITES) <= set(cs.columns)

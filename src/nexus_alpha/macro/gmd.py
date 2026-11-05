"""Global Macro Database (Mueller, Xu, Lehbib and Chen 2025, NBER WP 33714), release 2026_09: annual country data.

Licence: GMD Research Use Terms v1.1 (academic, non-commercial; no re-hosting; share-alike for derived work). The CSVs
stay in artifacts/ and are never committed. Cite the GMD in any output that uses these features.

Point-in-time rule (pre-registered, docs/V2_ARCHITECTURE.md): the value for year Y is treated as available from
1 May of year Y+1 (the IMF April World Economic Outlook is the latest major annual source). Known limitations, stated
rather than hidden: (1) only the 2026_09 vintage is held, so revised values stand in for first releases; (2) projection
rows (forecast = 1) are dropped; (3) the banking, currency and sovereign-debt crisis dummies are excluded because they
are classified with hindsight.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import paths

RELEASE = "2026_09"
AVAILABLE_MONTH_DAY = (5, 1)                 # year-Y values usable from 1 May of Y+1
COUNTRIES = ["USA", "CHN", "JPN", "DEU", "GBR", "FRA", "IND", "CAN", "ITA", "KOR", "BRA"]
US_VARS = ["rGDP", "infl", "unemp", "cbrate", "strate", "ltrate", "govdebt_GDP", "govdef_GDP", "CA_GDP", "M2", "rHPI",
           "USDfx", "REER"]


def _dir():
    return paths.ARTIFACTS / "raw" / f"gmd_{RELEASE}"


def load_var(var: str, countries=COUNTRIES) -> pd.DataFrame:
    x = pd.read_csv(_dir() / f"{var}_{RELEASE}.csv", usecols=lambda c: c in ("ISO3", "year", var, "forecast"),
                    low_memory=False)
    if "forecast" in x:
        x = x[x["forecast"].fillna(0) == 0]
    x = x[x.ISO3.isin(countries)].dropna(subset=[var])
    return x.pivot_table(index="year", columns="ISO3", values=var).sort_index()


def annual_panel(first_year: int = 1990) -> pd.DataFrame:
    """One row per year: US levels/changes and a GDP-weighted major-economy growth and inflation aggregate."""
    out = pd.DataFrame(index=pd.RangeIndex(first_year, 2032, name="year"))
    g = load_var("rGDP")
    growth = np.log(g).diff() * 100
    out["us_rgdp_growth"] = growth.get("USA")
    w = load_var("nGDP_USD").reindex(growth.index)
    ww = w[growth.columns].div(w[growth.columns].sum(axis=1), axis=0)
    out["major_rgdp_growth_w"] = (growth * ww).sum(axis=1, min_count=5)
    out["chn_rgdp_growth"] = growth.get("CHN")
    inf = load_var("infl")
    out["us_infl"] = inf.get("USA")
    out["major_infl_w"] = (inf.reindex(ww.index)[ww.columns] * ww).sum(axis=1, min_count=5)
    for v in ["unemp", "cbrate", "strate", "ltrate", "govdebt_GDP", "govdef_GDP", "CA_GDP"]:
        out[f"us_{v.lower()}"] = load_var(v, ["USA"]).get("USA")
    m2 = load_var("M2", ["USA"]).get("USA")
    out["us_m2_growth"] = np.log(m2).diff() * 100 if m2 is not None else np.nan
    hpi = load_var("rHPI", ["USA"]).get("USA")
    out["us_rhpi_growth"] = np.log(hpi).diff() * 100 if hpi is not None else np.nan
    reer = load_var("REER", ["USA"]).get("USA")
    out["us_reer_change"] = np.log(reer).diff() * 100 if reer is not None else np.nan
    out["us_unemp_change"] = out["us_unemp"].diff()
    out["us_term_spread_annual"] = out["us_ltrate"] - out["us_strate"]
    out["us_rgdp_growth_change"] = out["us_rgdp_growth"].diff()
    out["us_infl_change"] = out["us_infl"].diff()
    return out.loc[first_year:]


def asof_features(dates: pd.DatetimeIndex, panel: pd.DataFrame | None = None) -> pd.DataFrame:
    """Latest available year for each date (year Y usable from 1 May of Y+1), prefixed gmd_."""
    panel = annual_panel() if panel is None else panel
    d = pd.DatetimeIndex(dates)
    m, dd = AVAILABLE_MONTH_DAY
    avail_year = np.where((d.month > m) | ((d.month == m) & (d.day >= dd)), d.year - 1, d.year - 2)
    f = panel.reindex(avail_year)
    f.index = d
    f.columns = [f"gmd_{c}" for c in f.columns]
    f["gmd_data_year"] = avail_year
    return f

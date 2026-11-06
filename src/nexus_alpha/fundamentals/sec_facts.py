"""SEC XBRL company facts: download with the owner's declared User-Agent, keep only the concepts NEXUS uses.

SEC requires a User-Agent naming the requester (it rejects generic clients: HTTP 403, measured 2026-11-06). The value
comes only from SEC_USER_AGENT (loaded by `nexus_alpha.secrets` from a git-ignored owner file) and is sent only to
sec.gov / data.sec.gov, at <= 5 requests per second. Raw responses are cached gzip-compressed under
artifacts/fundamentals/raw/ (git-ignored). Facts filed on/after the seal are dropped when facts are loaded, unless the
sealed evaluation is open in this process.
"""
from __future__ import annotations

import gzip
import json
import os
import threading
import time
import urllib.request
from pathlib import Path

import pandas as pd

from .. import paths

CF_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
RAW = "fundamentals/raw"

# concept -> us-gaap tags in priority order (the first tag with a value for a period wins)
FLOW = {
    "revenue": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet",
                "RevenueFromContractWithCustomerIncludingAssessedTax", "SalesRevenueGoodsNet", "RevenuesNetOfInterestExpense"],
    "cogs": ["CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfGoodsSold", "CostOfServices"],
    "gross_profit": ["GrossProfit"],
    "operating_income": ["OperatingIncomeLoss"],
    "pretax_income": ["IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
                      "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments"],
    "net_income": ["NetIncomeLoss", "ProfitLoss", "NetIncomeLossAvailableToCommonStockholdersBasic"],
    "eps_diluted": ["EarningsPerShareDiluted", "EarningsPerShareBasicAndDiluted", "EarningsPerShareBasic"],
    "da": ["DepreciationDepletionAndAmortization", "DepreciationAndAmortization", "DepreciationAmortizationAndAccretionNet",
           "Depreciation"],
    "interest_expense": ["InterestExpense", "InterestExpenseDebt", "InterestExpenseNonoperating"],
    "income_tax": ["IncomeTaxExpenseBenefit"],
    "rnd": ["ResearchAndDevelopmentExpense"],
    "sga": ["SellingGeneralAndAdministrativeExpense"],
    "cfo": ["NetCashProvidedByUsedInOperatingActivities", "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"],
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets"],
    "dividends": ["PaymentsOfDividends", "PaymentsOfDividendsCommonStock"],
    "buybacks": ["PaymentsForRepurchaseOfCommonStock"],
    "diluted_shares": ["WeightedAverageNumberOfDilutedSharesOutstanding"],
}
INSTANT = {
    "assets": ["Assets"],
    "assets_current": ["AssetsCurrent"],
    "liabilities": ["Liabilities"],
    "liabilities_current": ["LiabilitiesCurrent"],
    "equity": ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    "cash": ["CashAndCashEquivalentsAtCarryingValue", "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
             "Cash"],
    "lt_debt_total": ["LongTermDebt"],
    "lt_debt_noncurrent": ["LongTermDebtNoncurrent"],
    "lt_debt_current": ["LongTermDebtCurrent", "DebtCurrent"],
    "st_borrowings": ["ShortTermBorrowings", "CommercialPaper"],
    "inventory": ["InventoryNet"],
    "receivables": ["AccountsReceivableNetCurrent"],
    "retained_earnings": ["RetainedEarningsAccumulatedDeficit"],
    "ppe": ["PropertyPlantAndEquipmentNet"],
}
DEI_INSTANT = {"shares_outstanding": ["EntityCommonStockSharesOutstanding"]}
FORMS = {"10-K", "10-Q", "10-K/A", "10-Q/A", "10-KT", "20-F", "40-F"}
COLS = ["cik", "concept", "tag", "rank", "kind", "unit", "value", "start", "end", "filed", "form", "accn", "fy", "fp"]


class _RateLimiter:
    def __init__(self, rps: float):
        self.dt, self.next, self.lock = 1.0 / rps, 0.0, threading.Lock()

    def wait(self):
        with self.lock:
            now = time.monotonic()
            if now < self.next:
                time.sleep(self.next - now)
            self.next = max(now, self.next) + self.dt


_RL = _RateLimiter(5.0)


def _ua() -> str:
    from .. import secrets
    secrets.load()
    ua = os.environ.get("SEC_USER_AGENT", "").strip()
    if not ua:
        raise RuntimeError("SEC_USER_AGENT is not set (owner's 'Name email' in a git-ignored file via NEXUS_ENV_FILES)")
    return ua


def raw_path(cik: int) -> Path:
    return paths.ARTIFACTS / RAW / f"CIK{int(cik):010d}.json.gz"


def fetch(cik: int, refresh: bool = False, attempts: int = 4) -> Path | None:
    """Download one company's facts to the raw cache (gzip). Returns the path, or None if SEC has no XBRL facts."""
    p = raw_path(cik)
    if p.exists() and not refresh:
        return p
    p.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(CF_URL.format(cik=int(cik)), headers={"User-Agent": _ua(), "Accept-Encoding": "gzip"})
    for i in range(attempts):
        _RL.wait()
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                b = r.read()
                if r.headers.get("Content-Encoding") != "gzip":
                    b = gzip.compress(b)
            tmp = p.with_suffix(".tmp")
            tmp.write_bytes(b)
            os.replace(tmp, p)
            return p
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            time.sleep(2 * (i + 1))
        except Exception:
            time.sleep(2 * (i + 1))
    raise RuntimeError(f"companyfacts download failed for CIK {cik}")


def parse(cik: int) -> pd.DataFrame:
    """Long table of the needed concepts from the cached raw file (all filing dates; see load() for the seal)."""
    p = raw_path(cik)
    if not p.exists():
        return pd.DataFrame(columns=COLS)
    js = json.loads(gzip.decompress(p.read_bytes()))
    facts = js.get("facts") or {}
    rows = []
    for tax, spec_map, kind in (("us-gaap", FLOW, "flow"), ("us-gaap", INSTANT, "instant"), ("dei", DEI_INSTANT, "instant")):
        block = facts.get(tax) or {}
        for concept, tags in spec_map.items():
            for rank, tag in enumerate(tags):
                spec = block.get(tag)
                if not spec:
                    continue
                for unit, vals in (spec.get("units") or {}).items():
                    for v in vals:
                        if v.get("form") not in FORMS:
                            continue
                        rows.append((int(cik), concept, tag, rank, kind, unit, v.get("val"), v.get("start"), v.get("end"),
                                     v.get("filed"), v.get("form"), v.get("accn"), v.get("fy"), v.get("fp")))
    df = pd.DataFrame(rows, columns=COLS)
    if len(df):
        df["value"] = pd.to_numeric(df.value, errors="coerce")
        for c in ("start", "end", "filed"):
            df[c] = pd.to_datetime(df[c], errors="coerce")
        df = df.dropna(subset=["value", "end", "filed"])
    return df


def load(cik: int) -> pd.DataFrame:
    """Parsed facts as allowed by the seal gate: before 2023-01-03 by filing date unless the evaluation is open."""
    from .. import seal
    df = parse(cik)
    if not seal.is_open():
        df = df[df.filed < seal.SEAL_DATE]
    return df.reset_index(drop=True)

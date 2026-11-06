"""Trade Desk process tools (decision D-034): tools, not signals.

Adapted from MIT-licensed skills: tradermonty/claude-trading-skills (`pre-trade-discipline-gate`,
`drawdown-circuit-breaker`, `position-sizer`) and Jakeschincariol/trader-skill (`trader-news`: Yahoo Finance RSS headlines
and the SEC EDGAR 8-K feed). The rule thresholds are the skills' defaults. Nothing here places an order.
"""
from __future__ import annotations

import math
import os
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from email.utils import parsedate_to_datetime

import pandas as pd

ET_TZ = "America/New_York"
YAHOO_RSS = "https://feeds.finance.yahoo.com/rss/2.0/headline?s=%s&region=US&lang=en-US"
SEC_FEED = "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=%s&type=%s&count=%d&output=atom"
SEC_TICKERS = "https://www.sec.gov/files/company_tickers.json"
GENERIC_UA = "Mozilla/5.0 (compatible; nexus-alpha-research/1.0)"


# ------------------------------------------------------------------------------------------------ circuit breaker
@dataclass
class BreakerRules:
    max_daily_loss_pct: float = 2.0
    losing_streak: int = 2
    cooldown_hours: float = 24.0
    weekly_dd_pct: float = 5.0
    monthly_dd_pct: float = 8.0


def circuit_breaker(journal: pd.DataFrame, account: float, now: pd.Timestamp | None = None,
                    rules: BreakerRules = BreakerRules()) -> dict:
    """journal: closed paper trades with exit_utc (timestamp) and pnl (currency). Returns the recommendation
    TRADING_ALLOWED / COOLDOWN / HALTED with the rule that fired and when it releases."""
    now = pd.Timestamp(now or pd.Timestamp.now(tz="UTC"))
    now = now.tz_localize("UTC") if now.tz is None else now
    if journal is None or journal.empty:
        return {"recommendation": "TRADING_ALLOWED", "data_quality": "EMPTY_STATE", "reasons": []}
    j = journal.copy()
    j["exit_utc"] = pd.to_datetime(j.exit_utc, utc=True)
    j = j[j.exit_utc <= now].sort_values("exit_utc")
    et = j.exit_utc.dt.tz_convert(ET_TZ)
    n_et = now.tz_convert(ET_TZ)
    reasons, release = [], []
    day = j[et.dt.date == n_et.date()].pnl.sum()
    if account > 0 and -day / account * 100 >= rules.max_daily_loss_pct:
        reasons.append(f"daily loss {-day / account:.1%} >= {rules.max_daily_loss_pct}%")
        release.append((n_et.normalize() + pd.offsets.BDay(1)).isoformat())
    wk_start = (n_et - pd.Timedelta(days=n_et.weekday())).normalize()
    wk = j[et >= wk_start].pnl.sum()
    if account > 0 and -wk / account * 100 >= rules.weekly_dd_pct:
        reasons.append(f"weekly loss {-wk / account:.1%} >= {rules.weekly_dd_pct}%")
        release.append((wk_start + pd.Timedelta(days=7)).isoformat())
    mo_start = n_et.normalize().replace(day=1)
    mo = j[et >= mo_start].pnl.sum()
    if account > 0 and -mo / account * 100 >= rules.monthly_dd_pct:
        reasons.append(f"monthly loss {-mo / account:.1%} >= {rules.monthly_dd_pct}%")
        release.append((mo_start + pd.offsets.MonthBegin(1)).isoformat())
    if reasons:
        return {"recommendation": "HALTED", "reasons": reasons, "release_et": max(release)}
    streak = 0
    for p in j.pnl.iloc[::-1]:
        if p < 0:
            streak += 1
        else:
            break
    if streak >= rules.losing_streak:
        until = j.exit_utc.iloc[-1] + pd.Timedelta(hours=rules.cooldown_hours)
        if now < until:
            return {"recommendation": "COOLDOWN", "reasons": [f"{streak} losing trades in a row"],
                    "release_et": until.tz_convert(ET_TZ).isoformat()}
    return {"recommendation": "TRADING_ALLOWED", "reasons": []}


# ------------------------------------------------------------------------------------------------ discipline gate
@dataclass
class Checklist:
    entry_in_written_plan: bool | None
    stop_predefined: bool | None
    size_within_plan: bool | None
    planned_risk_dollars: float | None
    actual_risk_dollars: float | None
    notes: str = ""


def discipline_gate(c: Checklist, breaker: dict | None, regime_risk: int | None = None,
                    recent_loss_hours: float | None = None, revenge_window_hours: float = 24.0) -> dict:
    """GO / NO_GO / REVIEW_REQUIRED for one manual paper-trade entry, with every reason."""
    block, review = [], []
    for name in ("entry_in_written_plan", "stop_predefined", "size_within_plan"):
        v = getattr(c, name)
        if v is None:
            review.append(f"{name.replace('_', ' ')}: not answered")
        elif not v:
            block.append(f"{name.replace('_', ' ')}: no")
    pr, ar = c.planned_risk_dollars, c.actual_risk_dollars
    ok = lambda x: x is not None and not isinstance(x, bool) and math.isfinite(float(x)) and float(x) >= 0
    if not (ok(pr) and ok(ar)):
        review.append("planned and actual risk in dollars must both be given (finite, >= 0)")
    elif float(ar) > float(pr):
        block.append(f"actual risk ${float(ar):,.0f} exceeds planned ${float(pr):,.0f}")
    if recent_loss_hours is not None and recent_loss_hours < revenge_window_hours:
        block.append(f"a losing exit {recent_loss_hours:.0f} h ago is inside the {revenge_window_hours:.0f} h revenge window")
    if breaker is None:
        review.append("circuit breaker state unavailable")
    elif breaker.get("recommendation") in ("COOLDOWN", "HALTED", "TRADING_HALTED"):
        block.append(f"circuit breaker: {breaker['recommendation']} ({'; '.join(breaker.get('reasons', []))})")
    if regime_risk is not None and regime_risk >= 3:
        review.append("market regime: SEVERE distribution-day risk (descriptive rule, no validated skill; review size)")
    decision = "NO_GO" if block else ("REVIEW_REQUIRED" if review else "GO")
    return {"decision": decision, "blocked_by": block, "review": review}


def position_size(risk_dollars: float, entry: float, stop: float, max_position_dollars: float | None = None) -> dict:
    """Fixed-dollar-risk sizing: shares = floor(risk / |entry - stop|), optionally capped by a position budget."""
    per = abs(float(entry) - float(stop))
    if per <= 0 or risk_dollars <= 0:
        return {"shares": 0, "reason": "entry equals stop or no risk budget"}
    sh = math.floor(risk_dollars / per)
    capped = False
    if max_position_dollars and sh * entry > max_position_dollars:
        sh, capped = math.floor(max_position_dollars / entry), True
    return {"shares": int(sh), "risk_per_share": per, "position_dollars": sh * entry, "risk_dollars": sh * per,
            "capped_by_budget": capped}


# ------------------------------------------------------------------------------------------------ live text
def _get(url: str, ua: str, timeout: float = 15.0) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": ua})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def yahoo_headlines(ticker: str, limit: int = 10) -> list[dict]:
    """Recent Yahoo Finance headlines for a ticker with their publication time (UTC)."""
    from .inference.pipeline import yahoo_symbol
    root = ET.fromstring(_get(YAHOO_RSS % yahoo_symbol(ticker), GENERIC_UA))
    out = []
    for it in root.iter("item"):
        title = (it.findtext("title") or "").strip()
        pub = it.findtext("pubDate")
        try:
            ts = pd.Timestamp(parsedate_to_datetime(pub)).tz_convert("UTC") if pub else None
        except Exception:
            ts = None
        out.append({"headline": title, "text": (it.findtext("description") or "").strip(),
                    "published_utc": ts.strftime("%Y-%m-%d %H:%M") if ts is not None else "",
                    "source": "Yahoo Finance RSS", "link": (it.findtext("link") or "").strip()})
    return out[:limit]


def sec_filings(ticker: str, form: str = "8-K", count: int = 10) -> list[dict]:
    """Recent EDGAR filings (Atom). SEC requires a User-Agent with the requester's contact: it is read only from the
    SEC_USER_AGENT environment variable, which the owner sets; without it nothing is requested."""
    ua = os.environ.get("SEC_USER_AGENT", "").strip()
    if not ua:
        raise RuntimeError("set SEC_USER_AGENT (your name and email, as SEC requires) to fetch EDGAR filings")
    import json
    tick = json.loads(_get(SEC_TICKERS, ua))
    cik = next((str(v["cik_str"]).zfill(10) for v in tick.values() if v["ticker"].upper() == ticker.upper().replace(".", "-")), None)
    if cik is None:
        return []
    root = ET.fromstring(_get(SEC_FEED % (cik, form, count), ua))
    ns = {"a": "http://www.w3.org/2005/Atom"}
    out = []
    for e in root.findall("a:entry", ns):
        upd = e.findtext("a:updated", default="", namespaces=ns)
        ts = pd.Timestamp(upd).tz_convert("UTC") if upd else None
        out.append({"headline": (e.findtext("a:title", default="", namespaces=ns) or "").strip(),
                    "text": (e.findtext("a:summary", default="", namespaces=ns) or "").strip()[:1500],
                    "published_utc": ts.strftime("%Y-%m-%d %H:%M") if ts is not None else "",
                    "source": f"SEC EDGAR {form}", "link": (e.find("a:link", ns).get("href") if e.find("a:link", ns) is not None else "")})
    return out

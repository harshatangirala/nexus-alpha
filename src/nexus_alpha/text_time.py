"""When was a text public? Date, time and time zone of a user-supplied text (company news, an SEC filing, a macro
release) -> UTC, and its place in the chronology of a forecast.

Every text has two clocks that matter for a trade decided at 09:00 New York on the session after the as-of close:
* **published**: what the user typed, in the user's time zone (IST by default on this desk), converted to UTC;
* **available**: when the model may read it. The project's single availability rule (`data.news.availability`): a
  text with a known minute is available 15 minutes after publication; a text with only a date is available at the end
  of that calendar day in US Pacific time. A text that becomes available after the decision time is excluded, so a
  forecast can never read the future.

Company news also has to fall inside the 5-day window the model was trained with (`news_select.LOOKBACK`). SEC filings
and macro texts typed here are context: they are timed and scored for tone, but they are not inputs of the frozen
models (their SEC and macro inputs come from EDGAR and ALFRED vintages, point in time).
"""
from __future__ import annotations

import datetime as dt

import pandas as pd

from .data import news as N
from .data.news_select import LOOKBACK

TIMEZONES = {"IST (India, UTC+5:30)": "Asia/Kolkata", "ET (New York)": "America/New_York", "UTC": "UTC",
             "London": "Europe/London", "CET (Frankfurt, Paris)": "Europe/Berlin", "SGT (Singapore)": "Asia/Singapore",
             "JST (Tokyo)": "Asia/Tokyo", "PT (California)": "America/Los_Angeles"}
KIND_LABEL = {"news": "company news", "sec": "SEC filing", "macro": "macro / market"}


def zone(tz: str) -> str:
    """A label of TIMEZONES or an IANA name."""
    return TIMEZONES.get(tz, tz)


def to_utc(day: dt.date | str | None, clock: dt.time | str | None = None, tz: str = "UTC") -> pd.Timestamp | None:
    """Local date and (optional) time in `tz` -> a UTC timestamp. Without a time the local date's midnight is returned
    in UTC date terms (the caller then marks the text 'time unknown', and the day rule applies)."""
    if day in (None, ""):
        return None
    d = pd.Timestamp(day).date()
    if clock in (None, ""):
        return pd.Timestamp(d).tz_localize("UTC")
    t = clock if isinstance(clock, dt.time) else pd.Timestamp(str(clock)).time()
    local = pd.Timestamp(dt.datetime.combine(d, t)).tz_localize(zone(tz), ambiguous="NaT", nonexistent="shift_forward")
    if pd.isna(local):                       # the repeated hour when clocks go back: take the earlier one
        local = pd.Timestamp(dt.datetime.combine(d, t)).tz_localize(zone(tz), ambiguous=True)
    return local.tz_convert("UTC")


def asof_close_utc(asof_session: str | pd.Timestamp) -> pd.Timestamp:
    """16:00 New York on the as-of session, in UTC (early closes are ignored: they only make this conservative)."""
    return pd.Timestamp(f"{pd.Timestamp(asof_session).date()} 16:00", tz="America/New_York").tz_convert("UTC")


def available_utc(published_utc: pd.Timestamp | None, time_known: bool) -> pd.Timestamp | None:
    if published_utc is None or pd.isna(published_utc):
        return None
    _, avail, _ = N.availability(pd.Series([published_utc]), pd.Series(["minute" if time_known else "day"]))
    return avail.iloc[0]


def phase(avail: pd.Timestamp | None, close_utc: pd.Timestamp, decision_utc: pd.Timestamp) -> str:
    if avail is None or pd.isna(avail):
        return "no date"
    if avail <= close_utc:
        return "before the as-of close"
    if avail <= decision_utc:
        return "after the close, before the decision"
    return "after the decision time"


def chronology(items: list[dict], decision_utc: pd.Timestamp, asof_session: str | pd.Timestamp) -> pd.DataFrame:
    """items: dicts with kind ('news', 'sec', 'macro'), headline, text, published_utc, time_known, tz_label, ticker.
    Returns one row per text: UTC times, hours before the decision, phase and whether the model reads it."""
    from .nlp.text_features import lm_tone
    close = asof_close_utc(asof_session)
    rows = []
    for it in items:
        pub = it.get("published_utc")
        pub = None if pub in (None, "") else pd.Timestamp(pub)
        if pub is not None and pub.tzinfo is None:
            pub = pub.tz_localize("UTC")
        known = bool(it.get("time_known", True))
        av = available_utc(pub, known)
        ph = phase(av, close, decision_utc)
        kind = it.get("kind", "news")
        if av is None:
            status = "excluded: no publication date"
        elif av > decision_utc:
            status = "excluded: public only after the decision time"
        elif kind == "news" and av <= decision_utc - LOOKBACK:
            status = f"excluded: older than the {LOOKBACK.days}-day news window"
        elif kind == "news":
            status = "model input (news branch)"
        else:
            status = "context (timed, not a model input)"
        tone = lm_tone(f"{it.get('headline', '')} {it.get('text', '')}").get("lm_net_tone") if (
            it.get("headline") or it.get("text")) else None
        rows.append({"ticker": it.get("ticker", ""), "kind": KIND_LABEL.get(kind, kind),
                     "headline": (it.get("headline") or (it.get("text") or "")[:90]).strip(),
                     "published (UTC)": pub, "time known": known, "available (UTC)": av,
                     "hours before decision": None if av is None else (decision_utc - av).total_seconds() / 3600,
                     "phase": ph, "status": status, "net tone (LM)": tone, "source": it.get("source", "")})
    return pd.DataFrame(rows)

"""Gemini as a live analyst for the Trade Desk (docs/V2_ARCHITECTURE.md section 6).

* Reads the user's 1-10 texts (news, macro releases, filings) and up to three chart screenshots and returns a structured
  JSON read: per-text direction lean, catalysts, risks, relevance; an overall lean with confidence; and the OHLC values
  it can read from each chart legend (used only to check the screenshot against real bars).
* Live only. A modern LLM has read about the outcomes of the training years, so its reads are never used as historical
  features and never blended into the model's probabilities. Every read is logged (inputs hashed, not stored) in
  artifacts/llm_reads/ so it can be scored later in the forward test.
* Key: GEMINI_API_KEY (or GOOGLE_API_KEY), set by the owner in the environment. Model: NEXUS_GEMINI_MODEL if set,
  otherwise the newest "flash" model the key can access (listed at run time, never hard-coded).
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import re

from .. import paths

SCHEMA = {
    "type": "object",
    "properties": {
        "texts": {"type": "array", "items": {"type": "object", "properties": {
            "index": {"type": "integer"},
            "kind": {"type": "string", "enum": ["company_news", "macro", "sec_filing", "other"]},
            "relevant_to_ticker": {"type": "boolean"},
            "direction_lean": {"type": "string", "enum": ["bullish", "bearish", "neutral", "unclear"]},
            "confidence": {"type": "number"},
            "horizon_relevance": {"type": "string", "enum": ["intraday", "days", "weeks", "months", "long_term"]},
            "catalysts": {"type": "array", "items": {"type": "string"}},
            "risks": {"type": "array", "items": {"type": "string"}},
            "summary": {"type": "string"}},
            "required": ["index", "kind", "relevant_to_ticker", "direction_lean", "confidence", "summary"]}},
        "charts": {"type": "array", "items": {"type": "object", "properties": {
            "slot": {"type": "string"}, "ticker_shown": {"type": "string"}, "timeframe_shown": {"type": "string"},
            "last_bar_time_shown": {"type": "string"}, "legend_open": {"type": "number"}, "legend_high": {"type": "number"},
            "legend_low": {"type": "number"}, "legend_close": {"type": "number"},
            "legend_is_last_bar": {"type": "boolean"}, "chart_read": {"type": "string"}},
            "required": ["slot", "chart_read"]}},
        "overall": {"type": "object", "properties": {
            "direction_lean": {"type": "string", "enum": ["bullish", "bearish", "neutral", "unclear"]},
            "confidence": {"type": "number"}, "key_drivers": {"type": "array", "items": {"type": "string"}},
            "key_risks": {"type": "array", "items": {"type": "string"}}, "summary": {"type": "string"}},
            "required": ["direction_lean", "confidence", "summary"]},
    },
    "required": ["texts", "charts", "overall"],
}

PROMPT = """You are a careful sell-side analyst assisting a quantitative research desk. Stock: {ticker}. Trade type: {trade}.
Decision time: {decision}. Analyse ONLY the material provided below; do not invent facts, prices or events.
For each text: classify it, say whether it is relevant to {ticker}, give a direction lean for {ticker} over the trade's
horizon with a calibrated confidence in [0,1] (0.5 = no edge), and list concrete catalysts and risks.
For each chart image: read the ticker, timeframe and the legend's OPEN/HIGH/LOW/CLOSE exactly as printed (numbers only;
omit any value you cannot read), say whether the legend shows the last bar, and describe the visible structure briefly.
Then give an overall lean. Say "unclear" when the material does not support a view. Be concise.
{texts}"""


def api_key() -> str | None:
    from .. import secrets
    secrets.load()
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")


def available() -> bool:
    return bool(api_key())


def _client():
    """Client with a 90 s per-request timeout and the SDK's own retries off (generate() handles retries and fallback)."""
    from google import genai
    from google.genai import types
    return genai.Client(api_key=api_key(), http_options=types.HttpOptions(
        timeout=90_000, retry_options=types.HttpRetryOptions(attempts=1)))


def candidate_models(client=None) -> list[str]:
    """Flash models this key can call, best first: stable before preview, newer before older. NEXUS_GEMINI_MODEL, if
    set, goes first. Aliases without a version (e.g. gemini-flash-latest) are skipped because they move silently."""
    client = client or _client()
    ranked = []
    for m in client.models.list():
        name = (getattr(m, "name", "") or "").split("/")[-1]
        acts = getattr(m, "supported_actions", None) or []
        if acts and "generateContent" not in acts:
            continue
        if "flash" not in name or any(x in name for x in ("lite", "tts", "image", "embed", "live", "audio", "exp")):
            continue
        v = re.search(r"gemini-(\d+(?:\.\d+)?)", name)
        if not v:
            continue
        ranked.append(((0 if "preview" in name else 1), float(v.group(1)), name))
    out = [n for _, _, n in sorted(ranked, reverse=True)]
    if os.environ.get("NEXUS_GEMINI_MODEL"):
        out = [os.environ["NEXUS_GEMINI_MODEL"]] + [n for n in out if n != os.environ["NEXUS_GEMINI_MODEL"]]
    if not out:
        raise RuntimeError("no Gemini flash model available to this key; set NEXUS_GEMINI_MODEL")
    return out


def pick_model(client=None) -> str:
    return candidate_models(client)[0]


TRANSIENT = ("429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE", "500", "INTERNAL", "504", "DEADLINE")
SKIP = ("404", "NOT_FOUND")                  # listed but retired for this key: move on at once
DAILY = "PerDay"                             # free tier: 20 requests per model per day (quotaId ...PerDay...)
_STICKY: dict[str, str] = {}                 # the model that last answered, tried first next time (per process)
_EXHAUSTED: dict[str, float] = {}            # model -> epoch seconds when its daily quota resets
_RETIRED: set[str] = set()                   # models that answered 404 (retired for this key)


class QuotaExhausted(RuntimeError):
    """Every candidate model has used its daily free-tier quota."""


def _usage_path():
    return paths.artifact("llm_reads", "usage.json")


def _quota_day() -> str:
    """The free-tier daily quota resets at midnight Pacific time, so usage is counted per Pacific calendar day."""
    from zoneinfo import ZoneInfo
    return dt.datetime.now(ZoneInfo("America/Los_Angeles")).date().isoformat()


def usage_today() -> dict:
    """Successful calls per model in the current quota day (Pacific), from artifacts/llm_reads/usage.json."""
    p = _usage_path()
    d = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    return d.get(_quota_day(), {})


def _count(model: str) -> None:
    p = _usage_path()
    d = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    day = _quota_day()
    d.setdefault(day, {})[model] = d.get(day, {}).get(model, 0) + 1
    p.write_text(json.dumps(d, indent=1), encoding="utf-8")


def generate(parts, cfg, client=None, tries_per_model: int = 2, max_models: int = 5):
    """generate_content with fallback across candidate models. A model that reports its daily quota is skipped until
    the reset; transient errors (rate limit, overload, timeout) get one short retry, then the next model; retired
    models are skipped. Returns (response, model_name). Raises QuotaExhausted when every model is out of quota."""
    import time
    client = client or _client()
    now = time.time()
    cands = [m for m in candidate_models(client)[:max_models] if _EXHAUSTED.get(m, 0) <= now and m not in _RETIRED]
    if _STICKY.get("model") in cands:
        cands = [_STICKY["model"]] + [c for c in cands if c != _STICKY["model"]]
    last = None
    for model in cands:
        for k in range(tries_per_model):
            try:
                r = client.models.generate_content(model=model, contents=parts, config=cfg)
                _STICKY["model"] = model
                _count(model)
                return r, model
            except Exception as e:
                last = e
                msg = str(e)
                if any(t in msg for t in SKIP):
                    _RETIRED.add(model)
                    break
                if DAILY in msg:
                    delay = re.findall(r"'retryDelay': '(\d+)s'", msg)
                    _EXHAUSTED[model] = now + (float(delay[0]) if delay else 3600)
                    break
                if not any(t in msg for t in TRANSIENT):
                    raise
                if k + 1 < tries_per_model:
                    time.sleep(8)
    allc = [m for m in candidate_models(client)[:max_models] if m not in _RETIRED]
    if all(_EXHAUSTED.get(m, 0) > time.time() for m in allc):
        wait = min(_EXHAUSTED[m] for m in allc) - time.time()
        raise QuotaExhausted(f"Gemini free-tier daily quota used up on every model (20 requests per model per day); "
                             f"next reset in about {wait / 3600:.1f} h")
    raise RuntimeError(f"all Gemini candidates failed; last error: {str(last)[:200]}")


def analyze(ticker: str, trade: str, decision_utc: str, texts: list[dict], images: list[tuple[str, bytes, str]],
            log: bool = True) -> dict:
    """texts: [{headline/text, published_utc, source}]; images: [(slot 'W'/'D'/'H', bytes, mime)]."""
    from google.genai import types
    client = _client()
    body = "\n".join(f"[TEXT {i}] published {t.get('published_utc', 'unknown')} source {t.get('source', 'user')}:\n"
                     f"{(t.get('headline') or '')}\n{(t.get('text') or '')[:6000]}" for i, t in enumerate(texts))
    parts = [PROMPT.format(ticker=ticker, trade=trade, decision=decision_utc, texts=body or "(no texts provided)")]
    for slot, data, mime in images:
        parts.append(f"[CHART slot={slot}]")
        parts.append(types.Part.from_bytes(data=data, mime_type=mime))
    cfg = types.GenerateContentConfig(response_mime_type="application/json", response_schema=SCHEMA, temperature=0.2)
    resp, model = generate(parts, cfg, client)
    out = json.loads(resp.text)
    out["_model"] = model
    if log:
        rec = {"logged_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "ticker": ticker,
               "trade": trade, "decision_utc": decision_utc, "model": model,
               "inputs_sha256": hashlib.sha256((body + "".join(hashlib.sha256(d).hexdigest() for _, d, _ in images))
                                               .encode()).hexdigest(),
               "overall": out.get("overall"), "texts": [{k: v for k, v in t.items() if k != "summary"}
                                                         for t in out.get("texts", [])]}
        p = paths.artifact("llm_reads", f"{dt.date.today().isoformat()}.jsonl")
        with open(p, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
    return out


LEGEND_SCHEMA = {"type": "object", "properties": {"charts": {"type": "array", "items": {"type": "object", "properties": {
    "slot": {"type": "string"}, "ticker_shown": {"type": "string"}, "timeframe_shown": {"type": "string"},
    "legend_open": {"type": "number"}, "legend_high": {"type": "number"}, "legend_low": {"type": "number"},
    "legend_close": {"type": "number"}}, "required": ["slot"]}}}, "required": ["charts"]}
LEGEND_PROMPT = ("Each image is a stock chart screenshot, labelled by the [CHART slot=...] line before it. For each chart, "
                 "read the ticker, the timeframe and the OPEN, HIGH, LOW and CLOSE printed in the chart's legend (the "
                 "O/H/L/C values at the top). Copy the numbers exactly as printed. Omit any value you cannot read; never "
                 "estimate from the candles.")


def read_legends(images: list[tuple[str, bytes, str]]) -> dict:
    """Legend-only read (small prompt, few tokens), with model fallback on transient errors (free-tier quotas)."""
    from google.genai import types
    parts = [LEGEND_PROMPT]
    for slot, data, mime in images:
        parts += [f"[CHART slot={slot}]", types.Part.from_bytes(data=data, mime_type=mime)]
    cfg = types.GenerateContentConfig(response_mime_type="application/json", response_schema=LEGEND_SCHEMA, temperature=0)
    r, model = generate(parts, cfg)
    out = json.loads(r.text)
    out["_model"] = model
    return out

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
    from google import genai
    return genai.Client(api_key=api_key())


def pick_model(client=None) -> str:
    if os.environ.get("NEXUS_GEMINI_MODEL"):
        return os.environ["NEXUS_GEMINI_MODEL"]
    client = client or _client()
    best, best_key = None, (-1, -1.0)
    for m in client.models.list():
        name = (getattr(m, "name", "") or "").split("/")[-1]
        acts = getattr(m, "supported_actions", None) or []
        if acts and "generateContent" not in acts:
            continue
        if "flash" not in name or any(x in name for x in ("lite", "tts", "image", "embed", "live", "audio", "exp")):
            continue
        v = re.search(r"gemini-(\d+(?:\.\d+)?)", name)
        if not v:                                   # aliases such as gemini-flash-latest move silently: skip
            continue
        stable = 0 if "preview" in name else 1      # prefer stable releases; previews change and have tighter quotas
        key = (stable, float(v.group(1)))
        if key > best_key:
            best, best_key = name, key
    if not best:
        raise RuntimeError("no Gemini flash model available to this key; set NEXUS_GEMINI_MODEL")
    return best


def analyze(ticker: str, trade: str, decision_utc: str, texts: list[dict], images: list[tuple[str, bytes, str]],
            log: bool = True) -> dict:
    """texts: [{headline/text, published_utc, source}]; images: [(slot 'W'/'D'/'H', bytes, mime)]."""
    from google.genai import types
    client = _client()
    model = pick_model(client)
    body = "\n".join(f"[TEXT {i}] published {t.get('published_utc', 'unknown')} source {t.get('source', 'user')}:\n"
                     f"{(t.get('headline') or '')}\n{(t.get('text') or '')[:6000]}" for i, t in enumerate(texts))
    parts = [PROMPT.format(ticker=ticker, trade=trade, decision=decision_utc, texts=body or "(no texts provided)")]
    for slot, data, mime in images:
        parts.append(f"[CHART slot={slot}]")
        parts.append(types.Part.from_bytes(data=data, mime_type=mime))
    cfg = types.GenerateContentConfig(response_mime_type="application/json", response_schema=SCHEMA, temperature=0.2)
    resp = client.models.generate_content(model=model, contents=parts, config=cfg)
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

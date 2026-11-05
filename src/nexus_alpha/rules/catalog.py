"""Versioned rule catalogue built from the supplied rulebooks (master spec section 3).

Parses 01_ENTRY_RULEBOOK_150.md and 02_EXIT_RULEBOOK_120.md into one record per rule with its exact source text, source
file, family section and line number, plus the family-level implementation profile. Classification fields (side,
timeframe, role, nature, required inputs) are derived by transparent keyword rules and are labelled
`classification_method = "keyword-heuristic-v1 (unreviewed)"`; they are a starting point for review, not a ruling.

A rule in the catalogue is a hypothesis, not a validated signal (both rulebooks say so). `implementation` records whether
NEXUS has an executable, unit-tested detector for it (nexus_alpha.rules.detectors); most rules are catalogued only.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

CATALOG_VERSION = "nexus-rules-catalog-v1"
CLASSIFICATION_METHOD = "keyword-heuristic-v1 (unreviewed)"

_RULE_RE = re.compile(r"^### ([EX])(\d{3}) [—-] (.+?)\s*$")
_FAMILY_RE = re.compile(r"^## (\d{2}) (.+?)\s*$")
_PROFILE_RE = re.compile(r"^## ([EX]\d{3})[–-]([EX]\d{3}) \| (.+?)\s*$")
_FIELD_RE = re.compile(r"^\*\*(Rule definition|Required evidence|Decision use|Model fields|Validation):\*\*\s*(.*)$")
_PROFILE_FIELD_RE = re.compile(r"^\*\*(Required measurable inputs|Operational predicate and state|"
                               r"Reject, invalidate or abstain when|Minimum test suite):\*\*\s*(.*)$")

ENTRY_ROLE = {"01": "data_gate", "02": "weekly_permission", "03": "daily_setup", "04": "location",
              "05": "candle_pattern", "06": "candle_pattern", "07": "trigger", "08": "entry_timing",
              "09": "trend_filter", "10": "momentum_filter", "11": "participation_filter", "12": "volatility_filter",
              "13": "mtf_confluence", "14": "final_gate", "15": "trigger_quality"}
EXIT_ROLE = {"01": "initial_stop", "02": "stop_management", "03": "target", "04": "profit_taking_trailing",
             "05": "deterioration_exit", "06": "candle_exit", "07": "structure_exit", "08": "time_exit",
             "09": "event_execution_exit", "10": "portfolio_control", "11": "lifecycle_governance",
             "12": "model_aware_exit"}
DETERMINISTIC_ROLES = {"data_gate", "initial_stop", "stop_management", "time_exit", "portfolio_control",
                       "final_gate", "event_execution_exit", "lifecycle_governance", "target"}

INPUT_KEYWORDS = {
    "OHLC": r"\bOHLC|candle|bar\b|high|low|close|open\b", "volume": r"volume|participation|OBV|A/D",
    "ATR": r"\bATR\b|true range|volatility", "RSI": r"\bRSI\b", "MACD": r"\bMACD\b", "moving_average": r"moving average|\bSMA|\bEMA|average",
    "bollinger": r"Bollinger|%B|band", "swing_pivots": r"swing|pivot|structure|higher-high|lower-high|higher-low|lower-low|break of structure",
    "support_resistance": r"support|resistance|zone|level|supply|demand", "gap": r"\bgap", "vwap": r"VWAP",
    "spread_liquidity": r"spread|liquidity|depth|slippage|dollar volume", "calendar_events": r"earnings|event|calendar|session|halt",
    "portfolio_state": r"portfolio|exposure|concentration|correlat|capital|drawdown|daily loss",
    "position_state": r"position|unrealized|open trade|partial|trail|entry price|R multiple|\bR\b",
    "model_output": r"model|probability|calibrat|uncertainty|forecast|score",
}


@dataclass
class Rule:
    rule_id: str
    book: str
    title: str
    family_code: str
    family_name: str
    source_file: str
    source_line: int
    definition: str
    required_evidence: str = ""
    decision_use: str = ""
    model_fields: str = ""
    validation: str = ""
    fields_are_boilerplate: bool = False
    side: str = "BOTH"
    timeframes: list = field(default_factory=list)
    role: str = ""
    nature: str = ""
    required_inputs: list = field(default_factory=list)
    classification_method: str = CLASSIFICATION_METHOD
    implementation: str = "catalogued_only"
    detector: str | None = None
    unit_tests: list = field(default_factory=list)
    text_sha256: str = ""


def _side(text: str, role: str) -> str:
    t = text.lower()
    if role in ("data_gate", "portfolio_control", "lifecycle_governance", "final_gate"):
        return "N/A"
    longish = re.search(r"\blong\b|bullish|bottom|hammer|morning star|piercing|demand|support rejection|higher-low|"
                        r"\bbuy|breakout\b|upside|reclaim", t)
    shortish = re.search(r"\bshort\b|bearish|\btop\b|shooting star|evening star|dark cloud|supply|resistance rejection|"
                         r"lower-high|\bsell|breakdown|downside", t)
    if longish and not shortish:
        return "LONG"
    if shortish and not longish:
        return "SHORT"
    return "BOTH"


def _timeframes(text: str, role: str) -> list[str]:
    t = text.lower()
    tf = [k for k, pat in (("1W", r"weekly|week\b"), ("1D", r"daily|\bday\b|session"), ("1H", r"hourly|intraday|hour\b"))
          if re.search(pat, t)]
    if not tf:
        tf = {"weekly_permission": ["1W"], "daily_setup": ["1D"], "trigger": ["1H"], "trigger_quality": ["1H"],
              "entry_timing": ["1D", "1H"]}.get(role, ["1W", "1D", "1H"])
    return tf


def _nature(text: str, role: str) -> str:
    t = text.lower()
    if role == "model_aware_exit" or re.search(r"\bmodel\b|neural|probabilit|calibrat", t):
        return "model_derived"
    if role in DETERMINISTIC_ROLES:
        return "deterministic"
    if role in ("candle_pattern", "location", "daily_setup", "weekly_permission", "mtf_confluence"):
        return "explanatory"
    return "deterministic"


def parse_rulebook(path: str | Path, book: str) -> tuple[list[Rule], list[dict]]:
    p = Path(path)
    lines = p.read_text(encoding="utf-8").splitlines()
    rules: list[Rule] = []
    profiles: list[dict] = []
    fam_code, fam_name = "", ""
    cur: Rule | None = None
    prof: dict | None = None
    last_field = None
    for i, raw in enumerate(lines, start=1):
        line = raw.rstrip()
        m = _PROFILE_RE.match(line)
        if m:
            cur = None
            prof = {"book": book, "first": m.group(1), "last": m.group(2), "family": m.group(3).strip(),
                    "source_file": p.name, "source_line": i}
            profiles.append(prof)
            continue
        m = _FAMILY_RE.match(line)
        if m:
            fam_code, fam_name, cur, prof = m.group(1), m.group(2).strip(), None, None
            continue
        if line.startswith("## ") or line.startswith("# "):
            cur, prof = None, None
            continue
        m = _RULE_RE.match(line)
        if m:
            prof = None
            cur = Rule(rule_id=f"{m.group(1)}{m.group(2)}", book=book, title=m.group(3).strip(), family_code=fam_code,
                       family_name=fam_name, source_file=p.name, source_line=i, definition="")
            rules.append(cur)
            continue
        if cur is not None:
            f = _FIELD_RE.match(line)
            if f:
                key = {"Rule definition": "definition", "Required evidence": "required_evidence",
                       "Decision use": "decision_use", "Model fields": "model_fields", "Validation": "validation"}[f.group(1)]
                setattr(cur, key, f.group(2).strip())
                last_field = key
            elif line.strip() and last_field:
                setattr(cur, last_field, (getattr(cur, last_field) + " " + line.strip()).strip())
        elif prof is not None:
            f = _PROFILE_FIELD_RE.match(line)
            if f:
                prof[f.group(1)] = f.group(2).strip()
    # derived fields
    role_map = ENTRY_ROLE if book == "entry" else EXIT_ROLE
    for r in rules:
        r.role = role_map.get(r.family_code, "unknown")
        text = f"{r.title}. {r.definition}"
        r.side = _side(text, r.role)
        r.timeframes = _timeframes(text, r.role)
        r.nature = _nature(text, r.role)
        r.required_inputs = [k for k, pat in INPUT_KEYWORDS.items() if re.search(pat, text, flags=re.I)]
        r.text_sha256 = hashlib.sha256(text.encode()).hexdigest()[:16]
    # boilerplate detection: a field text shared by more than 5 rules is family boilerplate, not rule-specific
    for fld in ("required_evidence", "decision_use", "model_fields", "validation"):
        counts: dict[str, int] = {}
        for r in rules:
            counts[getattr(r, fld)] = counts.get(getattr(r, fld), 0) + 1
        for r in rules:
            if counts[getattr(r, fld)] > 5:
                r.fields_are_boilerplate = True
    return rules, profiles


def build_catalog(entry_path: str | Path, exit_path: str | Path) -> dict:
    e_rules, e_prof = parse_rulebook(entry_path, "entry")
    x_rules, x_prof = parse_rulebook(exit_path, "exit")
    rules = e_rules + x_rules
    ids = [r.rule_id for r in rules]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate rule ids in rulebooks")
    from . import detectors
    for r in rules:
        if r.rule_id in detectors.IMPLEMENTED:
            d = detectors.IMPLEMENTED[r.rule_id]
            r.implementation, r.detector, r.unit_tests = "executable_detector", d["detector"], d["tests"]
    return {"version": CATALOG_VERSION, "classification_method": CLASSIFICATION_METHOD,
            "sources": {"entry": {"file": Path(entry_path).name,
                                  "sha256": hashlib.sha256(Path(entry_path).read_bytes()).hexdigest()},
                        "exit": {"file": Path(exit_path).name,
                                 "sha256": hashlib.sha256(Path(exit_path).read_bytes()).hexdigest()}},
            "n_rules": len(rules), "rules": [asdict(r) for r in rules], "family_profiles": e_prof + x_prof}

"""Build the versioned rule catalogue from the supplied rulebooks.

Outputs: model_registry/rule_catalog.json (full records), model_registry/rule_catalog.csv (one line per rule)
Run: .venv/Scripts/python -I scripts/build_rule_catalog.py [--print-titles]
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

from nexus_alpha.rag.sources import RULEBOOKS
from nexus_alpha.rules.catalog import build_catalog

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    cat = build_catalog(RULEBOOKS["entry_rulebook_150"], RULEBOOKS["exit_rulebook_120"])
    (REPO / "model_registry" / "rule_catalog.json").write_text(json.dumps(cat, indent=1), encoding="utf-8")
    df = pd.DataFrame(cat["rules"])
    cols = ["rule_id", "book", "family_code", "family_name", "title", "side", "timeframes", "role", "nature",
            "required_inputs", "implementation", "detector", "fields_are_boilerplate", "source_file", "source_line"]
    df[cols].assign(timeframes=df.timeframes.map("|".join), required_inputs=df.required_inputs.map("|".join)) \
        .to_csv(REPO / "model_registry" / "rule_catalog.csv", index=False)
    print(f"{cat['n_rules']} rules; by book {Counter(df.book)}; profiles {len(cat['family_profiles'])}")
    print("nature:", dict(Counter(df.nature)), "| side:", dict(Counter(df.side)))
    print("boilerplate fields:", dict(Counter(zip(df.book, df.fields_are_boilerplate))))
    print("implementation:", dict(Counter(df.implementation)))
    if "--print-titles" in sys.argv:
        for r in cat["rules"]:
            print(f"{r['rule_id']} [{r['family_code']} {r['role']}] {r['title']} :: {r['definition'][:150]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

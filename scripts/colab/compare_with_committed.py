"""Compare a re-run of E22 / E23 / E25 (for example on Colab) with the committed, pre-registered result.

    python scripts/colab/compare_with_committed.py e22_fundamentals e23_industry_fundamentals

For each experiment name: the committed run is the earliest experiments/*_<name>_* folder, the re-run the latest. Prints
whether the keep decisions are identical and the largest differences in mean daily rank IC and in the validation log
losses. A re-run on another machine can differ in the last digits (thread counts change floating-point summation order);
a different keep decision or a difference above 0.002 in log loss is reported as a MISMATCH. The re-run is a
reproduction only: never commit it and never replace the committed record with it.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[2]


def main(names: list[str]) -> None:
    bad = 0
    for n in names:
        runs = sorted(glob.glob(str(REPO / "experiments" / f"*_{n}_*")))
        if len(runs) < 2:
            print(f"{n}: only {len(runs)} run(s) found; nothing to compare (run the experiment first)")
            bad += 1
            continue
        a, b = (json.loads((Path(r) / "metrics.json").read_text(encoding="utf-8")) for r in (runs[0], runs[-1]))
        same_keep = a["keep"] == b["keep"] and a["kept"] == b["kept"]
        key = "mean_daily_ic" if "mean_daily_ic" in a else "mean_daily_rank_ic"     # E25 names it explicitly
        ic = max(abs(a[key][k] - b[key][k]) for k in a[key])
        ll = 0.0
        for h, d in a["by_horizon"].items():
            for s, dd in d.items():
                for m, x in dd.items():
                    y = b["by_horizon"][h][s][m]
                    ll = max(ll, abs(x["valid_ll"] - y["valid_ll"]), abs(x["test_ll"] - y["test_ll"]))
        ok = same_keep and ll < 0.002
        bad += 0 if ok else 1
        print(f"{n}: committed {Path(runs[0]).name} vs re-run {Path(runs[-1]).name}\n"
              f"   keep decisions identical: {same_keep} | max |log-loss diff| {ll:.5f} | max |mean daily IC diff| {ic:.5f} "
              f"-> {'REPRODUCED' if ok else 'MISMATCH'}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main(sys.argv[1:] or ["e22_fundamentals", "e23_industry_fundamentals", "e25_alpha158"])

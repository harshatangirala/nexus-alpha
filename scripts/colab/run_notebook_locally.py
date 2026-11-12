"""Execute notebooks/NEXUS_Alpha_End_to_End.ipynb cell by cell in one Python process (no Jupyter needed).

Used to verify the notebook before it goes to Colab: every code cell runs in order in a shared namespace, figures are
saved as PNG files (so they can be inspected), and the first failing cell stops the run with its traceback.

    python -I scripts/colab/run_notebook_locally.py --quick            # small samples (minutes)
    python -I scripts/colab/run_notebook_locally.py                    # the full run
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
NB = REPO / "notebooks" / "NEXUS_Alpha_End_to_End.ipynb"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--figs", default=str(REPO / "artifacts" / "notebook_run" / "figures"))
    ap.add_argument("--stop-after", type=int, default=None, help="last code cell to run (1-based)")
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figs = Path(a.figs)
    figs.mkdir(parents=True, exist_ok=True)
    for f in figs.glob("*.png"):
        f.unlink()
    state = {"n": 0, "cell": 0}

    def show(*_a, **_k):
        for num in plt.get_fignums():
            state["n"] += 1
            plt.figure(num).savefig(figs / f"c{state['cell']:02d}_{state['n']:02d}.png", bbox_inches="tight")
        plt.close("all")

    plt.show = show

    def display(obj, *_a, **_k):
        import pandas as pd
        if hasattr(obj, "data") and hasattr(obj, "to_html"):         # a pandas Styler
            obj = obj.data
        if isinstance(obj, (pd.DataFrame, pd.Series)):
            with pd.option_context("display.width", 200, "display.max_columns", 14, "display.max_rows", 40):
                print(obj.head(40))
        else:
            print(obj)

    cells = [c for c in json.loads(NB.read_text(encoding="utf-8"))["cells"] if c["cell_type"] == "code"]
    ns = {"__name__": "__main__", "display": display, "Markdown": str}
    total = time.time()
    for i, c in enumerate(cells, 1):
        if a.stop_after and i > a.stop_after:
            break
        src = "".join(c["source"])
        if a.quick:
            src = src.replace("QUICK = False", "QUICK = True", 1)
        src = src.replace("from IPython.display import display, Markdown", "pass")
        state["cell"] = i
        t0 = time.time()
        print(f"\n===== code cell {i}/{len(cells)} =====", flush=True)
        try:
            exec(compile(src, f"<cell {i}>", "exec"), ns)
            show()
        except Exception:
            traceback.print_exc()
            print(f"FAILED in code cell {i} after {time.time() - t0:.0f}s", flush=True)
            return 1
        print(f"----- cell {i} done in {time.time() - t0:.0f}s", flush=True)
    print(f"\nALL CELLS PASSED in {(time.time() - total) / 60:.1f} min; {state['n']} figures in {figs}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

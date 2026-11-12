"""The end-to-end notebook re-implements the forward-test chain check inline (section 18). It must agree with the package's
`shadow.verify_chain` on the committed logs (audit 7, A-77: it looked for forecasts/forecasts/... and reported every
chain as broken)."""
import hashlib
import importlib.util
from pathlib import Path

import pandas as pd

from nexus_alpha import shadow

REPO = Path(__file__).resolve().parents[1]


def _chain_cell() -> str:
    spec = importlib.util.spec_from_file_location("mk_nb", REPO / "scripts" / "colab" / "make_end_to_end_notebook.py")
    mk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mk)
    return next(s for k, s in mk.CELLS if k == "code" and "def verify_chain" in s)


def test_notebook_chain_check_agrees_with_the_package():
    shown = []
    ns = {"REPO_DIR": REPO, "pd": pd, "hashlib": hashlib, "display": shown.append, "print": lambda *a, **k: None,
          "sha256": lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()}
    exec(_chain_cell(), ns)
    table = shown[0].set_index("model")
    for v in table.index:
        assert (table.loc[v, "chain"] == "intact") == (shadow.verify_chain(v) == []), (v, table.loc[v, "chain"])

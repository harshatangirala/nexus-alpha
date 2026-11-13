"""Mission Control tab: every quoted number is in its source document; status readers and the globe helpers work."""
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "app"))

import globe_view as GV  # noqa: E402
import mission_control as MC  # noqa: E402
import ui as UI  # noqa: E402


def test_every_claim_quotes_its_source_verbatim():
    for c in MC.CLAIMS:
        text = (REPO / c["source"]).read_text(encoding="utf-8")
        assert c["evidence"] in text, (c["claim"], c["source"])
        assert c["verdict"] in UI.VERDICT
        if c["lo"] is not None:
            assert c["lo"] <= c["value"] <= c["hi"], c["claim"]


def test_tape_numbers_come_from_the_readme():
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    for s in ("AUROC about 0.58-0.61", "0.49-0.51", "(Sharpe 0.53)", "a planted leak is found"):
        assert s in readme
    t = MC.tape({"sessions": 3}, None)
    assert any(x["verdict"] == "pending" and "E25" in x["text"] for x in t)


def test_kpis_and_forward_status_read_the_repository():
    k = MC.kpis()
    assert k["frozen"] == len(list((REPO / "registry" / "frozen_models").glob("*.json"))) >= 4
    assert k["decisions"] >= 46 and k["tests"] > 100 and k["experiments"] >= 20
    fs = MC.forward_status()
    assert list(fs.model) == MC.VERSIONS
    import json
    import pandas as pd
    frozen = {v: pd.Timestamp(json.loads((REPO / "registry" / "frozen_models" / f"{v}.json").read_text(encoding="utf-8"))
                              ["frozen_utc"]) for v in fs.model}
    new = fs.model.map(lambda v: pd.Timestamp.now(tz="UTC") - frozen[v] < pd.Timedelta(days=3))   # first log still pending
    assert fs.chain_intact.all() and ((fs.sessions >= 1) | new).all()
    assert "missed" in fs and all(isinstance(m, list) for m in fs.missed)


def test_colab_status_reads_the_recorded_reproduction():
    s = MC.colab_status()
    assert s.startswith("Colab: ") and "not run yet" not in s      # the owner's 2026-11-11 run is recorded


def test_missed_sessions_counts_only_passed_decision_times():
    logged = {"2026-11-05", "2026-11-06", "2026-11-09", "2026-11-11"}
    # 12 Nov 2026, 15:00 IST (09:30 UTC): 2026-11-10's decision time (11 Nov 14:00 UTC) has passed; 2026-11-11's has not
    assert MC.missed_sessions(logged, "2026-11-05", pd.Timestamp("2026-11-12 09:30", tz="UTC")) == ["2026-11-10"]
    # before 2026-11-11 is logged and before its decision time, it is pending, not missed
    assert MC.missed_sessions(logged - {"2026-11-11"}, "2026-11-05",
                              pd.Timestamp("2026-11-12 09:30", tz="UTC")) == ["2026-11-10"]
    # after its decision time an unlogged session counts as missed
    assert MC.missed_sessions(logged - {"2026-11-11"}, "2026-11-05",
                              pd.Timestamp("2026-11-12 14:30", tz="UTC")) == ["2026-11-10", "2026-11-11"]


def test_universe_points_group_by_state_or_country():
    a = pd.DataFrame({"ticker": ["A", "B", "C", "D"], "cik": [1, 2, 3, 4], "city": ["x"] * 4,
                      "state_or_country": ["CA", "CA", "D0", None],
                      "state_or_country_description": ["CA", "CA", "Bermuda", None]})
    pts, counts = GV.universe_points(a)
    assert counts == {"placed": 3, "unplaced": 1}
    assert pts[0]["label"] == "CA, United States" and pts[0]["n"] == 2 and pts[0]["tickers"] == "A, B"
    assert {p["label"] for p in pts} == {"CA, United States", "Bermuda"}


def test_macro_by_country_uses_actual_years_only():
    from nexus_alpha.macro import gmd
    if not (gmd._dir() / f"rGDP_{gmd.RELEASE}.csv").exists():
        pytest.skip("Global Macro Database files not present")
    m, year = GV.macro_by_country("Real GDP growth")
    assert "USA" in m and -20 < m["USA"][0] < 20 and m["USA"][1] <= pd.Timestamp.now().year
    assert year != "n/a"


def test_json_for_script_blocks_is_escaped():
    assert "</script>" not in UI.json_script({"x": "</script><script>alert(1)</script>"})

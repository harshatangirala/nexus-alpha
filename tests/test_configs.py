"""Configs must agree with the code constants they document (no silent drift)."""
import json
from pathlib import Path

import yaml

from nexus_alpha import seal, splits, targets
from nexus_alpha.trading import plan
from nexus_alpha.trading.costs import SCENARIOS

ROOT = Path(__file__).resolve().parents[1]


def y(name):
    return yaml.safe_load((ROOT / "configs" / name).read_text(encoding="utf-8"))


def test_configs_parse_and_match_code():
    assert str(seal.SEAL_DATE.date()) == y("data.yaml")["seal_date"]
    t = y("targets.yaml")
    assert t["horizons"] == targets.HORIZONS and t["k_sigma"] == targets.K_SIGMA and t["cost_floor"] == targets.COST_FLOOR
    r = y("risk.yaml")
    rc = plan.RiskConfig()
    assert r["min_edge"] == rc.min_edge and r["min_net_rr"] == rc.min_net_rr and r["risk_pct_per_trade"] == rc.risk_pct_per_trade
    assert r["stop_atr_buffer"] == plan.STOP_ATR_BUFFER and r["chase_atr"] == plan.CHASE_ATR
    assert r["costs_round_trip_bps"] == {k: int(v) for k, v in SCENARIOS.items()}
    tr = y("training.yaml")["splits"]
    assert [str(b.start) for b in splits.PRIMARY] == [str(v[0]) for v in tr.values() if isinstance(v, list)]
    assert tr["purge_sessions"] == targets.MAX_H and tr["embargo_sessions"] == splits.EMBARGO_SESSIONS


def test_cv_registry_matches_recorded_backbone_specs():
    specs = json.loads((ROOT / "model_registry" / "triframe_backbones_s500.json").read_text(encoding="utf-8"))
    for name, v in y("cv_models.yaml")["backbones"].items():
        if "timm" in v:
            assert v["timm"] == specs[name]["timm_name"], name

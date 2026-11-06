"""Build and freeze the `kronos_zs` bundle for the forward test, only if E24's pre-registered necessary condition is met.

    python scripts/kronos/freeze_kronos.py

The bundle is flat (every file at the top level, so the frozen record hashes all of them): kronos_config.json,
kronos.safetensors, tokenizer_config.json, tokenizer.safetensors and bundle_meta.json (configuration, horizons, label
version, the training class prior the scorer compares against, weight revisions, the E24 run). Then
`freeze_model.py --kind kronos --version kronos_zs`.
"""
from __future__ import annotations

import glob
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "experiments"))

from nexus_alpha import paths, shadow as SH  # noqa: E402
from nexus_alpha.kronos import weights as W  # noqa: E402
from nexus_alpha.targets_v2 import LABEL_VERSION_V2  # noqa: E402

import e24_kronos_zero_shot as E24  # noqa: E402


def main() -> None:
    if (SH.FROZEN_DIR / "kronos_zs.json").exists():
        sys.exit("kronos_zs is already frozen: frozen records are never overwritten")
    runs = sorted(glob.glob(str(REPO / "experiments" / "*_e24_kronos_zero_shot_*")))
    if not runs:
        sys.exit("no E24 run: score E24 first")
    m = json.loads((Path(runs[0]) / "metrics.json").read_text(encoding="utf-8"))      # the first (registered) run
    if not m.get("necessary_condition_met"):
        sys.exit("E24's necessary condition is not met: Kronos is not frozen for the forward test (pre-registered)")
    tok = W.TOKENIZER_FOR[E24.MODEL]
    src = {"kronos_config.json": W.verify(E24.MODEL) / "config.json", "kronos.safetensors": W.root() / E24.MODEL / "model.safetensors",
           "tokenizer_config.json": W.verify(tok) / "config.json", "tokenizer.safetensors": W.root() / tok / "model.safetensors"}
    sha = hashlib.sha256(b"".join(hashlib.sha256(p.read_bytes()).digest() for p in src.values())).hexdigest()[:6]
    bid = time.strftime("%Y%m%dT%H%M%S") + f"_kronos_zs_{sha}"
    bdir = paths.ARTIFACTS / "bundles" / bid
    bdir.mkdir(parents=True)
    for name, p in src.items():
        shutil.copyfile(p, bdir / name)
    tr = pd.read_parquet(paths.ARTIFACTS / "samples" / "v2_frame.parquet", columns=["block"] + [f"y_{h}" for h in E24.HORIZONS])
    tr = tr[tr.block == "train"]
    prior = {h: [float(x) for x in np.array([(tr[f"y_{h}"] == c).sum() for c in range(3)]) / tr[f"y_{h}"].notna().sum()]
             for h in E24.HORIZONS}
    man = W.manifest()
    meta = {"bundle_id": bid, "kind": "kronos",
            "recipe": f"{E24.MODEL} zero-shot (vendored Kronos, MIT), NEXUS path sampler, v2 label-rule probabilities",
            "trained_through_block": "pre-trained by its authors on data to 2024-06 (not fitted by NEXUS)",
            "horizons": E24.HORIZONS, "label_version": LABEL_VERSION_V2, "refit_class_prior": prior,
            "config": {"lookback": E24.LOOKBACK, "paths": E24.N_PATHS, "T": E24.T, "top_p": E24.TOP_P,
                       "horizon_sessions": E24.HMAX},
            "max_context": W.MAX_CONTEXT[E24.MODEL],
            "weights": {E24.MODEL: man.get(E24.MODEL), tok: man.get(tok)}, "e24_run": Path(runs[0]).name,
            "decision": "D-043"}
    (bdir / "bundle_meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    subprocess.run([sys.executable, "-B", str(REPO / "scripts" / "shadow" / "freeze_model.py"), "--kind", "kronos",
                    "--version", "kronos_zs", "--bundle", bid,
                    "--note", "E24 necessary condition met (contaminated 2017-18); forward test only; no trading engine"],
                   check=True)
    print("bundle", bid)


if __name__ == "__main__":
    main()

"""Coverage of the deployed bundle's quantile heads (fusion seeds averaged) for the volatility-scaled return z_h.

The app shows an expected-return range for a horizon only if the VALIDATION coverage of the 10-90% band is within 0.08 of
the nominal 0.80 (test coverage is reported, not used to decide). Result written into the bundle's metadata.
Run: .venv/Scripts/python -I scripts/verify/quantile_coverage.py <bundle_run_id>
"""
from __future__ import annotations

import json
import sys

import numpy as np
import torch

from nexus_alpha import paths
from nexus_alpha.inference.pipeline import Bundle
from nexus_alpha.multimodal.dataset import build, load_tables
from nexus_alpha.targets import HORIZONS


def main() -> int:
    bdir = paths.ARTIFACTS / "bundles" / sys.argv[1]
    b = Bundle(bdir)
    if not b.models:
        print("bundle has no fusion component: no quantile heads")
        return 0
    fr = load_tables()
    d, _ = build(fr, (fr.block == "train").to_numpy(), tuple(b.names), b.pre.backbone, pre=b.pre)
    qs = []
    with torch.no_grad():
        for m in b.models:
            out = []
            for s in range(0, len(d), 8192):
                ix = np.arange(s, min(len(d), s + 8192))
                x = {k: torch.from_numpy(v[ix]) for k, v in d.x.items()}
                out.append(m(x, torch.from_numpy(d.mask[ix]))[1].numpy())
            qs.append(np.concatenate(out))
    q = np.sort(np.mean(qs, 0), axis=-1)          # same rearrangement as the inference pipeline
    blk = fr.block.to_numpy()
    cov = {}
    for j, h in enumerate(HORIZONS):
        for name in ("valid", "test"):
            msk = (blk == name) & np.isfinite(d.z[:, j])
            z = d.z[msk, j]
            cov.setdefault(h, {})[name] = {"crossing_rate": float((np.mean(qs, 0)[msk, j, 0] > np.mean(qs, 0)[msk, j, 2]).mean()),
                                           "coverage_10_90": float(((z >= q[msk, j, 0]) & (z <= q[msk, j, 2])).mean()),
                                           "median_abs_error_z": float(np.median(np.abs(z - q[msk, j, 1]))), "n": int(msk.sum())}
        cov[h]["show_in_app"] = bool(abs(cov[h]["valid"]["coverage_10_90"] - 0.80) <= 0.08)
    meta = json.loads((bdir / "bundle_meta.json").read_text(encoding="utf-8"))
    meta["quantile_coverage"] = cov
    (bdir / "bundle_meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    for h, v in cov.items():
        print(h, "valid", round(v["valid"]["coverage_10_90"], 3), "test", round(v["test"]["coverage_10_90"], 3), "show", v["show_in_app"])
    return 0


if __name__ == "__main__":
    sys.exit(main())

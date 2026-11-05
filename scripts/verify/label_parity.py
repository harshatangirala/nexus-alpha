"""Compare NEXUS labels with TriFrame's frozen labels on the s500 rows (through TriFrame's sealed ImageStore).

Expected: sigma_ewma identical (same definition, same bars); 5-session return differs only by dividends (TriFrame adds
dividends with ex-dates t+2..t+5, NEXUS is price-only), so the 3-class labels agree except near the threshold.
Output: reports/verification/label_parity.json
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from nexus_alpha import paths, sources_bridge


def main() -> int:
    fr = pd.read_parquet(paths.ARTIFACTS / "samples" / "s500_frame.parquet",
                         columns=["sample_id", "sigma_ewma", "ret_5D", "y_5D", "end_5D", "end_20D"])
    st = sources_bridge.triframe_image_store(who="nexus_label_parity")
    tf = st.index()[["sample_id", "sigma_ewma", "ret5", "y3_5", "end5", "end20"]]
    m = fr.merge(tf, on="sample_id", how="inner", suffixes=("", "_tf"))
    ok = m.y_5D.notna() & m.y3_5.notna()
    out = {"rows_compared": int(ok.sum()),
           "sigma_max_abs_diff": float(np.nanmax(np.abs(m.sigma_ewma - m.sigma_ewma_tf))),
           "ret5_median_abs_diff": float(np.nanmedian(np.abs(m.ret_5D - m.ret5))),
           "ret5_share_identical_1e-12": float((np.abs(m.ret_5D - m.ret5) < 1e-12)[ok].mean()),
           "y5_agreement": float((m.y_5D[ok] == m.y3_5[ok]).mean()),
           "end5_identical": float((pd.to_datetime(m.end_5D) == pd.to_datetime(m.end5))[ok].mean()),
           "end20_identical": float((pd.to_datetime(m.end_20D) == pd.to_datetime(m.end20)).mean())}
    out["note"] = "TriFrame ret5 includes dividends with ex-dates t+2..t+5; NEXUS labels are price-only"
    p = paths.REPO_ROOT / "reports" / "verification" / "label_parity.json"
    p.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

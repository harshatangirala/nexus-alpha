"""Add per-view valid-bar counts (TriFrame sealed index) to the s500 frame so training applies the same >= 40%-of-window
presence rule for chart views as inference (audit fix A-03). Run: .venv/Scripts/python -I scripts/data/add_nvalid.py"""
import sys
import pandas as pd
from nexus_alpha import paths, sources_bridge

def main():
    p = paths.ARTIFACTS / "samples" / "s500_frame.parquet"
    fr = pd.read_parquet(p).drop(columns=["n_valid_W", "n_valid_D", "n_valid_H"], errors="ignore")
    idx = sources_bridge.triframe_image_store(who="nexus_add_nvalid").index()[["sample_id", "n_valid_W", "n_valid_D", "n_valid_H"]]
    fr = fr.merge(idx, on="sample_id", how="left", validate="one_to_one")
    assert fr[["n_valid_W", "n_valid_D", "n_valid_H"]].notna().all().all()
    fr.to_parquet(p, index=False)
    for v, need in (("W", 42), ("D", 51), ("H", 56)):
        m = fr[f"n_valid_{v}"] < need
        print(v, "rows below threshold:", int(m.sum()), "of which in train:", int((m & (fr.block == "train")).sum()))
    return 0

if __name__ == "__main__":
    sys.exit(main())

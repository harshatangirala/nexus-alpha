"""E28 input: ChronoBERT embeddings of SEC filing text, aggregated per s500 sample (pre-registered, D-053).

Stage 1 (encode, about 6 h on this 12-thread CPU, minutes on a Colab GPU): every filing that is a sample's latest 8-K
(30 days) or latest 10-Q/10-K (120 days) at its decision time, the same filings the classical `sec` block reads
(`nexus_alpha.data.sec.latest_event_ids`). Text cleaning and pooling: `nexus_alpha.nlp.transformer_text`.
Outputs artifacts/features/sec_tx/{events.parquet, event_emb.npy (float16), meta.json}; checkpoints make it resumable.

Stage 2 (aggregate, --aggregate): PCA(32) of the event vectors fitted on events available before 2016-01-01 only, then
per sample: the latest 8-K's components (sec_tx_8k_pc*), the latest periodic filing's (sec_tx_per_pc*), presence flags.
Output artifacts/features/sec_tx_features.parquet (index sample_id) and the PCA (sec_tx_pca.npz) for inference.

Run: .venv/Scripts/python -I scripts/data/build_sec_tx_features.py [--aggregate] [--device cuda] [--threads 10]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from nexus_alpha import paths  # noqa: E402
from nexus_alpha.data import sec as S  # noqa: E402
from nexus_alpha.nlp import transformer_text as TT  # noqa: E402

OUT = paths.ARTIFACTS / "features" / "sec_tx"
FIT_BEFORE = pd.Timestamp("2016-01-01", tz="UTC")
K = 32


def needed_events() -> tuple[pd.DataFrame, pd.DataFrame]:
    fr = pd.read_parquet(paths.ARTIFACTS / "samples" / "s500_frame.parquet", columns=["sample_id", "ticker", "decision_utc"])
    ev = S.load_events(dev_only=True)
    lat = S.latest_event_ids(fr, ev)
    ids = sorted(set(lat.drop(columns="sample_id").stack().dropna()))
    meta = ev.set_index("event_id").loc[ids, ["ticker", "event_type", "available_utc"]].reset_index()
    return meta, lat


def encode_stage(device: str, threads: int, batch: int) -> None:
    import torch
    torch.set_num_threads(threads)
    try:
        import psutil
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)      # the nightly forward test keeps precedence
    except Exception:
        pass
    t0 = time.time()
    meta, _ = needed_events()
    path = next((paths.SEC_FULL500_RUN / "S09").glob("*/event_text.parquet"))
    txt = pq.read_table(path, filters=[("event_id", "in", meta.event_id.tolist())]).to_pandas().set_index("event_id").text
    texts = [TT.clean_filing_text(t) for t in txt.reindex(meta.event_id).tolist()]
    meta["n_chars_clean"] = [len(t) for t in texts]
    print(f"{len(texts):,} filings to encode ({sum(c == 0 for c in meta.n_chars_clean)} empty after cleaning); "
          f"{TT.MODEL}@{TT.REVISION[:10]}, {TT.MAX_TOKENS} tokens, {device}", flush=True)
    tok, mdl = TT.load(device)
    E = TT.encode(texts, tok, mdl, batch=batch, ckpt_dir=OUT / "ckpt")
    E[meta.n_chars_clean.to_numpy() == 0] = np.nan                     # no text: never read as a real vector
    OUT.mkdir(parents=True, exist_ok=True)
    meta.to_parquet(OUT / "events.parquet", index=False)
    np.save(OUT / "event_emb.npy", E)
    (OUT / "meta.json").write_text(json.dumps({
        "model": TT.MODEL, "revision": TT.REVISION, "max_tokens": TT.MAX_TOKENS, "pooling": "masked mean, last layer",
        "cleaning": "safe-harbour paragraphs dropped, whitespace collapsed, first 4,000 characters", "device": device,
        "events": len(meta), "empty_after_cleaning": int((meta.n_chars_clean == 0).sum()),
        "elapsed_s": round(time.time() - t0, 1)}, indent=1), encoding="utf-8")
    print(f"done in {(time.time() - t0) / 3600:.2f} h", flush=True)


def aggregate_stage() -> None:
    meta = pd.read_parquet(OUT / "events.parquet")
    E = np.load(OUT / "event_emb.npy").astype(np.float64)
    ok = np.isfinite(E).all(1)
    fit = ok & (pd.to_datetime(meta.available_utc, utc=True) < FIT_BEFORE).to_numpy()
    mu = E[fit].mean(0)
    _, sv, Vt = np.linalg.svd(E[fit] - mu, full_matrices=False)
    Z = np.full((len(E), K), np.nan)
    Z[ok] = (E[ok] - mu) @ Vt[:K].T
    np.savez(paths.ARTIFACTS / "features" / "sec_tx_pca.npz", mean=mu, components=Vt[:K], model=TT.MODEL,
             revision=TT.REVISION)
    zf = pd.DataFrame(Z, index=meta.event_id)
    _, lat = needed_events()
    lat = lat.set_index("sample_id")
    parts = []
    for kind, short in (("8k", "8k"), ("periodic", "per")):
        col = lat.get(f"event_{kind}", pd.Series(np.nan, index=lat.index, dtype=object))
        V = zf.reindex(col.values).set_axis(col.index)
        V.columns = [f"sec_tx_{short}_pc{i}" for i in range(K)]
        V[f"sec_tx_{short}_present"] = V.iloc[:, 0].notna().astype(float)
        parts.append(V)
    fr = pd.read_parquet(paths.ARTIFACTS / "samples" / "s500_frame.parquet", columns=["sample_id"])
    out = pd.concat(parts, axis=1).reindex(fr.sample_id)
    for short in ("8k", "per"):
        out[f"sec_tx_{short}_present"] = out[f"sec_tx_{short}_present"].fillna(0.0)
    out["sec_tx_present"] = ((out.sec_tx_8k_present + out.sec_tx_per_present) > 0).astype(float)
    out.index.name = "sample_id"
    out.to_parquet(paths.ARTIFACTS / "features" / "sec_tx_features.parquet")
    summ = {"pca_fit_events": int(fit.sum()), "pca_var_explained_32": float((sv[:K] ** 2).sum() / (sv ** 2).sum()),
            "samples": len(out), "present_share": float(out.sec_tx_present.mean()),
            "share_8k": float(out.sec_tx_8k_present.mean()), "share_periodic": float(out.sec_tx_per_present.mean())}
    (OUT / "aggregate_summary.json").write_text(json.dumps(summ, indent=1), encoding="utf-8")
    print(json.dumps(summ, indent=1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=10)
    ap.add_argument("--batch", type=int, default=64)
    a = ap.parse_args()
    if a.aggregate:
        aggregate_stage()
    else:
        encode_stage(a.device, a.threads, a.batch)
        aggregate_stage()
    return 0


if __name__ == "__main__":
    sys.exit(main())

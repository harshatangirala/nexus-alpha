"""Stage the inputs of notebooks/NEXUS_Alpha_End_to_End.ipynb (local artifacts -> a folder Colab reads from Drive).

The notebook trains and evaluates in the open, so it needs data, not scripts. This copies a compact, governed set:

* `samples/s500_frame.parquet`: the 236,520 stock-dates (2001-10 to 2022-12-01): features, labels, purged blocks.
* `img_pca64_convnext_tiny.npy`: the three chart views (W, D, H) of every sample as 64 principal components of the
  frozen ImageNet ConvNeXt-Tiny embeddings (float16, 236,520 x 192). The PCA and its scaling are fitted on the TRAIN
  block only (2003-2015), as `nexus_alpha.multimodal.dataset.build` does. The chart-fine-tuned TriFrame weights are
  not used (they may only be evaluated from 2017; `cv.store.assert_eval_allowed`).
* the news, SEC and rule feature tables (`features/*.parquet`, keyed by `sample_id`);
* saved predictions of the documented runs on the evaluation rows (valid 2017-18, calib 2019, test 2020-22):
  E01 structured baselines, E02 image probes, E03 fusion ablations and variants, E06 seeds and loss landscape,
  E08 = frozen v1.1; and E14 = frozen v2 on its own rows;
* the RAG knowledge base (chunks, dense vectors, manifest). The books are private: keep this folder private.
* the language inputs of sections 8A-8C (E28, D-053), under `nlp/` and `features/`: the cleaned text and ChronoBERT
  vectors of the 68,853 encoded filings, the PCA, the SEC price-impact expert's table (event features and labels whose
  exit is before 2019 only), its documented predictions, and the per-sample news_tx, sec_tx and sec_impact blocks.

Nothing from the sealed 2023+ period is in any of these files (the s500 frame ends 2022-12-01).

    python scripts/colab/stage_notebook_inputs.py               # artifacts/notebook, then a copy on the Drive workspace
    python scripts/colab/stage_notebook_inputs.py --dest D:/x   # another copy
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.dont_write_bytecode = True

from nexus_alpha import paths  # noqa: E402

DRIVE = Path(r"G:\My Drive\antigravity_workspace\nexus_artifacts\notebook")
PRED = paths.ARTIFACTS / "predictions"
H6 = ["1D", "2D", "5D", "10D", "20D", "1M"]
RUNS = {
    "e01": "20261105T194117_e01_structured_baselines_098364",
    "e02": "20261105T194747_e02_cv_branch_93f5ac",
    "e03": "20261106T000935_e03_fusion_8ddffb",
    "e06": "20261106T005012_e06_optimization_fa42d8",
    "e08": "20261106T020411_e08_final_model_5c4327",
    "e14": "20261106T142918_e14_blend_freeze_v2_3298d4",
    "e28": "20261113T035556_e28_vision_nlp_fusion_b1f899",
}
E01 = ["prior", "momentum_lr", "lgbm_structured", "lgbm_struct_x33"]
E02 = ["img_lr__resnet18_random_s0__WDH", "img_lr__resnet18__WDH", "img_lr__convnext_tiny__WDH",
       "img_lr__vit_small_patch14_dinov2__WDH", "img_lr__ft_convnext_tiny__WDH"]
E03 = ["a04_img_WDH", "a05_num", "a06_img_num", "a07_news", "a08_sec", "a12_img_news_sec", "a15_full_no_modality_dropout",
       "a17_full_rules", "variant_late", "variant_concat", "variant_attention", "variant_gated", "variant_hybrid",
       "selected_calibrated"]
E06 = ["best_seed7", "best_seed11", "best_seed23", "best_seed42", "best_seed101", "best_seed_ensemble",
       "num_seed_ensemble", "lr_0.0003", "lr_0.001", "lr_0.003"]
E08 = ["final_deployed_calibrated", "final_for_backtest"]
E28 = ["BASE", "MM", "PLACEBO", "BASE_news_tx", "BASE_sec_tx", "BASE_sec_impact", "text_only", "vision_only"]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def image_block(out: Path, fr: pd.DataFrame) -> dict:
    from nexus_alpha.cv.features import ViewPCA
    train_rows = fr.row.to_numpy()[fr.block.to_numpy() == "train"]
    vp = ViewPCA("convnext_tiny", ("W", "D", "H"), 64).fit(train_rows)
    Z = vp.transform(fr.row.to_numpy()).to_numpy(np.float32)
    np.save(out / "img_pca64_convnext_tiny.npy", Z.astype(np.float16))
    return {"backbone": "convnext_tiny (ImageNet, frozen)", "k_per_view": 64, "views": ["W", "D", "H"],
            "row_order": "samples/s500_frame.parquet", "fit_rows": "block == train (2003-2015)",
            "n_fit_rows": int(len(train_rows)), "explained_variance": vp.explained()}


def nlp_block(out: Path) -> dict:
    """Inputs of the language sections. Filings are public SEC text; the study masked firm names."""
    import pyarrow.parquet as pq
    from nexus_alpha.data import sec as S
    from nexus_alpha.nlp import sec_impact as SI
    from nexus_alpha.nlp import transformer_text as TT
    feat = paths.ARTIFACTS / "features"
    (out / "nlp").mkdir(parents=True, exist_ok=True)
    for name in ("finbert_features", "sec_tx_features", "sec_impact_features"):
        shutil.copy2(feat / f"{name}.parquet", out / "features" / f"{name}.parquet")
    ev = S.load_events(dev_only=True)
    first = ev.set_index("event_id").item_list.map(lambda l: next((i for i in (l or []) if i != "9.01"), None))
    m = pd.read_parquet(feat / "sec_tx" / "events.parquet")
    assert pd.to_datetime(m.available_utc, utc=True).max() < pd.Timestamp("2023-01-03", tz="UTC")
    m["item_main"] = first.reindex(m.event_id).to_numpy()
    m.to_parquet(out / "nlp" / "sec_tx_events.parquet", index=False)
    shutil.copy2(feat / "sec_tx" / "event_emb.npy", out / "nlp" / "sec_tx_event_emb.npy")
    shutil.copy2(feat / "sec_tx_pca.npz", out / "nlp" / "sec_tx_pca.npz")
    path = next((paths.SEC_FULL500_RUN / "S09").glob("*/event_text.parquet"))
    txt = pq.read_table(path, filters=[("event_id", "in", m.event_id.tolist())]).to_pandas().set_index("event_id").text
    texts = pd.DataFrame({"event_id": m.event_id, "text": [TT.clean_filing_text(t) for t in txt.reindex(m.event_id)]})
    texts.to_parquet(out / "nlp" / "filing_texts.parquet", index=False, compression="zstd")
    tab, cols = SI.load_table(ev)
    tab[cols] = tab[cols].astype(np.float32)
    keep = cols + ["ticker", "event_type", "available_utc"] + [f"{k}_h{h}" for h in SI.HORIZONS for k in ("y", "exit")]
    tab[keep].to_parquet(out / "nlp" / "sec_expert_table.parquet", compression="zstd")
    shutil.copy2(feat / "sec_impact" / "meta.json", out / "nlp" / "sec_expert_meta.json")
    shutil.copy2(feat / "sec_impact" / "event_predictions.parquet", out / "nlp" / "sec_impact_event_predictions.parquet")
    return {"filings_encoded": len(m), "encoder": f"{TT.MODEL}@{TT.REVISION}", "expert_events": len(tab),
            "expert_label_exits_before": str(SI.LABEL_READ_BEFORE.date())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default=None, help="extra copy (default: the Drive workspace if it is mounted)")
    ap.add_argument("--keep-images", action="store_true", help="reuse an existing image block")
    a = ap.parse_args()
    t0 = time.time()
    out = paths.ARTIFACTS / "notebook"
    for sub in ("samples", "features", "predictions", "rag"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    src_frame = paths.ARTIFACTS / "samples" / "s500_frame.parquet"
    shutil.copy2(src_frame, out / "samples" / "s500_frame.parquet")
    fr = pd.read_parquet(src_frame, columns=["sample_id", "ticker", "asof_date", "block", "row"])
    assert pd.to_datetime(fr.asof_date).max() < pd.Timestamp("2023-01-01"), "sealed rows must never be staged"
    meta = {"created_utc": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"), "runs": RUNS}
    if not (a.keep_images and (out / "img_pca64_convnext_tiny.npy").exists()):
        meta["images"] = image_block(out, fr)
        print(f"image block done ({time.time() - t0:.0f}s)", flush=True)
    for name in ("news_features", "sec_features", "rule_features"):
        shutil.copy2(paths.ARTIFACTS / "features" / f"{name}.parquet", out / "features" / f"{name}.parquet")
    ev = fr.block.isin(["valid", "calib", "test"]).to_numpy()
    rows = fr.loc[ev, ["sample_id", "ticker", "asof_date", "block"]].reset_index(drop=True)
    rows.to_parquet(out / "predictions" / "rows_eval.parquet", index=False)
    idx = np.flatnonzero(ev)
    for run, names in (("e03", E03), ("e06", E06), ("e08", E08), ("e28", E28)):
        for n in names:
            src = PRED / RUNS[run] / f"{n}__probs.npy"
            if src.exists():
                np.save(out / "predictions" / f"{run}__{n}.npy", np.load(src, mmap_mode="r")[idx].astype(np.float16))
    for run, names in (("e01", E01), ("e02", E02)):
        for n in names:
            src = PRED / RUNS[run] / f"{n}.parquet"
            if not src.exists():
                continue
            d = pd.read_parquet(src).set_index("sample_id").reindex(rows.sample_id)
            arr = np.stack([d[[f"p_down_{h}", f"p_side_{h}", f"p_up_{h}"]].to_numpy(np.float32) for h in H6], 1)
            np.save(out / "predictions" / f"{run}__{n}.npy", arr.astype(np.float16))
    np.save(out / "predictions" / "e06__landscape_2d.npy", np.load(PRED / RUNS["e06"] / "landscape_2d.npy"))
    shutil.copy2(PRED / RUNS["e14"] / "rows.parquet", out / "predictions" / "rows_v2.parquet")
    np.save(out / "predictions" / "e14__v2_probs_calibrated.npy",
            np.load(PRED / RUNS["e14"] / "probs_calibrated.npy").astype(np.float16))
    for f in ("chunks.parquet", "dense_bge_small.npy", "manifest.json"):
        shutil.copy2(paths.ARTIFACTS / "rag" / f, out / "rag" / f)
    meta["nlp"] = nlp_block(out)
    print(f"language inputs done ({time.time() - t0:.0f}s)", flush=True)
    files = sorted(p for p in out.rglob("*") if p.is_file() and p.name != "manifest_notebook.json")
    meta["files"] = {str(p.relative_to(out)).replace("\\", "/"): {"bytes": p.stat().st_size, "sha256": sha256(p)}
                     for p in files}
    old = out / "manifest_notebook.json"
    if "images" not in meta and old.exists():
        meta["images"] = json.loads(old.read_text(encoding="utf-8")).get("images")
    old.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    total = sum(v["bytes"] for v in meta["files"].values())
    print(f"staged {len(files)} files, {total / 1e6:.0f} MB in {out} ({time.time() - t0:.0f}s)", flush=True)
    dest = Path(a.dest) if a.dest else (DRIVE if DRIVE.parent.exists() else None)
    if dest is not None:
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(out, dest)
        print(f"copied to {dest}", flush=True)


if __name__ == "__main__":
    main()

"""E16 input: FinBERT (ProsusAI/finbert) sentiment and embeddings for every selected news headline (local CPU).

Pre-registered comparison (docs/V2_ARCHITECTURE.md section 6): a transformer text encoder vs the v1 classical text
features. Weights are fixed (no fine-tuning). FinBERT was further pre-trained on Reuters TRC2 (2008-2010) and fine-tuned on
Financial PhraseBank; that text lies inside the training block, not in validation or test.

Per headline: P(positive), P(negative), P(neutral) and the attention-masked mean of the last hidden layer (768-d).
Per decision row (sample_id): cluster-weighted mean tone, min/max tone, negative share, count, and a 16-d PCA of the
weighted mean embedding, with the PCA fitted on training-block rows only. Output artifacts/features/finbert_features.parquet.
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
warnings.filterwarnings("ignore")

from nexus_alpha import paths  # noqa: E402

MODEL = "ProsusAI/finbert"


def encode(heads: list[str], batch: int = 256, threads: int = 10) -> tuple[np.ndarray, np.ndarray, dict]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    torch.set_num_threads(threads)
    tok = AutoTokenizer.from_pretrained(MODEL)
    mdl = AutoModelForSequenceClassification.from_pretrained(MODEL).eval()
    id2 = {int(k): v.lower() for k, v in mdl.config.id2label.items()}
    # length-sorted batches: far less padding; results are scattered back to the input order at the end
    order = np.argsort([len(h) for h in heads], kind="stable")
    sh = [heads[j] for j in order]
    probs, embs, t0 = [], [], time.time()
    with torch.inference_mode():
        for i in range(0, len(sh), batch):
            b = tok(sh[i:i + batch], padding=True, truncation=True, max_length=64, return_tensors="pt")
            o = mdl(**b, output_hidden_states=True)
            probs.append(torch.softmax(o.logits, -1).numpy())
            hs, m = o.hidden_states[-1], b["attention_mask"].unsqueeze(-1).float()
            embs.append(((hs * m).sum(1) / m.sum(1)).numpy().astype(np.float16))
            if (i // batch) % 40 == 0:
                print(f"{i + len(sh[i:i + batch])}/{len(sh)} {time.time() - t0:.0f}s", flush=True)
    try:
        from huggingface_hub import model_info
        rev = model_info(MODEL).sha
    except Exception:
        rev = getattr(mdl.config, "_commit_hash", None)
    P, E = np.concatenate(probs), np.concatenate(embs)
    inv = np.empty_like(order)
    inv[order] = np.arange(len(order))
    return P[inv], E[inv], {"id2label": id2, "revision": rev}


def main() -> None:
    reps = pd.read_parquet(paths.ARTIFACTS / "features" / "news_representatives.parquet",
                           columns=["sample_id", "headline", "n_cluster"])
    reps["headline"] = reps.headline.fillna("").str.strip()
    reps = reps[reps.headline != ""]
    uniq = reps.headline.drop_duplicates().tolist()
    P, E, info = encode(uniq)
    lab = info["id2label"]
    ip, ineg = [k for k, v in lab.items() if v == "positive"][0], [k for k, v in lab.items() if v == "negative"][0]
    hid = {h: i for i, h in enumerate(uniq)}
    reps["i"] = reps.headline.map(hid)
    reps["tone"] = P[reps.i, ip] - P[reps.i, ineg]
    reps["neg"] = (P[reps.i, ineg] > 0.5).astype(float)
    reps["w"] = reps.n_cluster.fillna(1).clip(lower=1)
    g = reps.groupby("sample_id")
    agg = pd.DataFrame({"fb_tone_mean": g.apply(lambda x: np.average(x.tone, weights=x.w)),
                        "fb_tone_min": g.tone.min(), "fb_tone_max": g.tone.max(), "fb_neg_share": g.neg.mean(),
                        "fb_n": g.size().astype(float)})
    # weighted mean embedding per sample, then PCA(16) fitted on training-block samples only
    W = reps.w.to_numpy(np.float64)
    order = reps.sample_id.to_numpy()
    sid, inv = np.unique(order, return_inverse=True)
    M = np.zeros((len(sid), E.shape[1]))
    np.add.at(M, inv, E[reps.i.to_numpy()].astype(np.float64) * W[:, None])
    M /= np.bincount(inv, weights=W)[:, None]
    blocks = pd.read_parquet(paths.ARTIFACTS / "samples" / "s500_frame.parquet", columns=["sample_id", "block"]) \
        .set_index("sample_id").block.reindex(sid)
    tr = (blocks == "train").to_numpy()
    mu = M[tr].mean(0)
    U, S, Vt = np.linalg.svd(M[tr] - mu, full_matrices=False)
    Z = (M - mu) @ Vt[:16].T
    emb = pd.DataFrame(Z, index=sid, columns=[f"fb_pc{i}" for i in range(16)])
    out = agg.join(emb).reset_index().rename(columns={"index": "sample_id"})
    out.to_parquet(paths.artifact("features", "finbert_features.parquet"), index=False)
    pd.Series({"model": MODEL, "revision": info["revision"], "headlines": len(uniq), "samples": len(out),
               "pca_fit_rows_train": int(tr.sum()),
               "pca_var_explained_16": float((S[:16] ** 2).sum() / (S ** 2).sum())}).to_json(
        paths.artifact("features", "finbert_features_meta.json"))
    print(out.shape, "done")


if __name__ == "__main__":
    main()

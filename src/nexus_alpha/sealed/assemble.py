"""Evaluation rows -> the full model input frame: bar features + news + SEC + GMD/FRED context (as in development)."""
from __future__ import annotations

import pandas as pd

from .. import paths, shadow as SH
from . import text as TX


def assemble(fr: pd.DataFrame, beyond_seal: bool, fetch_news: bool = False, log=print) -> tuple[pd.DataFrame, dict]:
    from ..inference.pipeline import Bundle
    from ..macro import fred as FR, gmd as G
    b = Bundle(SH.verify_frozen("v1.1")[0])            # frozen TF-IDF/SVD text models
    s = fr[["sample_id", "ticker", "decision_utc"]]
    news, sel = TX.news_for(s, b.news_tf, beyond_seal=beyond_seal, fetch=fetch_news, log=log)
    sec = TX.sec_for(s, b.sec_tf, beyond_seal=beyond_seal, log=log)
    full = fr.merge(news, left_on="sample_id", right_index=True, how="left")
    full = full.merge(sec, left_on="sample_id", right_index=True, how="left")
    dates = pd.DatetimeIndex(sorted(full.asof_date.unique()))
    gm = G.asof_features(dates)
    store = pd.read_parquet(paths.ARTIFACTS / (FR.STORE_FULL if beyond_seal else FR.STORE))
    fe = FR.asof_features(dates, store=store)
    ctx = gm.join(fe, how="outer")
    ctx.index.name = "asof_date"
    full = full.merge(ctx.reset_index(), on="asof_date", how="left")
    info = {"rows": len(full), "news_present_share": float(full.news_present.mean()),
            "sec_covered_share": float(full.sec_covered.mean()), "news_pairs": int(len(sel)),
            "gmd_cols": int(gm.shape[1]), "fred_cols": int(fe.shape[1])}
    log(f"assembled {info}")
    return full, info

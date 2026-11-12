"""SEC price-impact expert (E28 `sec_impact` block, pre-registered in docs/V3_RESEARCH.md, D-053).

The SEC study (`paths.SEC_FULL500_RUN`, read only) labelled every filing event by what it did to the price. This module
learns that mapping on the study's own features and turns it into point-in-time evidence for NEXUS samples.

* Labels: S08 `T_trade` `label3` (0 down, 1 sideways, 2 up; entry at the first tradable price after the decision
  time) at horizons 1 and 5. Only labels whose exit is before LABEL_READ_BEFORE are ever read from disk, so no label
  near or inside the sealed period is loaded.
* Features: S09 event features without `meta_year` (review finding N2) and without the accounting families (finding
  N1: their availability rule is wrong before 2009; D-077 superseded them).
* Model: LightGBM multiclass with fixed parameters; fitted on events available before 2016 whose h5 exit is before 2016;
  trees chosen by early stopping on 2016 events. Events before 2016 get out-of-fold predictions from year-block folds
  (same tree count, a 30-day gap at each fold edge), so no NEXUS training row sees a prediction fitted on its own event.
"""
from __future__ import annotations

import glob

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .. import paths
from ..data import sec as S

HORIZONS = (1, 5)
FIT_BEFORE = pd.Timestamp("2016-01-01", tz="UTC")
ES_BEFORE = pd.Timestamp("2017-01-01", tz="UTC")
LABEL_READ_BEFORE = pd.Timestamp("2019-01-01")
FOLDS = ((2000, 2002), (2003, 2005), (2006, 2008), (2009, 2011), (2012, 2015))
GAP = pd.Timedelta(days=30)
EXCLUDE_PREFIX = ("acc_", "sue_", "ear_")
EXCLUDE = ("meta_year", "_avail_accounting", "_avail_ear", "event_id")
PARAMS = dict(objective="multiclass", num_class=3, learning_rate=0.05, num_leaves=31, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, min_data_in_leaf=200, deterministic=True, force_row_wise=True,
              seed=11, verbose=-1)
MAX_ROUNDS, ES_ROUNDS = 2000, 100
LOOKBACK = pd.Timedelta(days=14)
WINDOW = pd.Timedelta(days=30)
COLUMNS = (["si_pmove_h1", "si_dir_h1", "si_pmove_h5", "si_dir_h5", "si_log_days", "si_is_8k", "si_is_periodic",
            "si_max_pmove_h5_30d", "si_n_events_30d"])


def _stage(stage: str, name: str) -> str:
    hits = glob.glob(str(paths.SEC_FULL500_RUN / stage / "*" / name))
    if len(hits) != 1:
        raise FileNotFoundError(f"{stage}/{name}")
    return hits[0]


def feature_columns(names) -> list[str]:
    return [c for c in names if c not in EXCLUDE and not c.startswith(EXCLUDE_PREFIX)]


def load_table(ev: pd.DataFrame | None = None) -> tuple[pd.DataFrame, list[str]]:
    """One row per development event: features, available_utc, event_type and the h1/h5 labels (NaN when not read)."""
    ev = S.load_events(dev_only=True) if ev is None else ev
    fpath = _stage("S09", "features.parquet")
    cols = feature_columns(pq.read_schema(fpath).names)
    X = pq.read_table(fpath, columns=["event_id"] + cols,
                      filters=[("event_id", "in", ev.event_id.tolist())]).to_pandas().set_index("event_id")
    t = pq.read_table(_stage("S08", "targets.parquet"),
                      columns=["event_id", "horizon", "label3", "censored", "exit_date"],
                      filters=[("target_def", "==", "T_trade"), ("horizon", "in", list(HORIZONS)),
                               ("exit_date", "<", LABEL_READ_BEFORE)]).to_pandas()
    t = t[~t.censored & t.label3.notna()]
    tab = ev.set_index("event_id")[["ticker", "event_type", "available_utc"]].join(X, how="inner")
    for h in HORIZONS:
        th = t[t.horizon == h].drop_duplicates("event_id").set_index("event_id")
        tab[f"y_h{h}"] = th.label3.reindex(tab.index)
        tab[f"exit_h{h}"] = pd.to_datetime(th.exit_date.reindex(tab.index)).dt.tz_localize("UTC")
    return tab, cols


def _fit(X, y, Xes=None, yes=None, rounds=None, threads=2):
    import lightgbm as lgb
    p = {**PARAMS, "num_threads": threads}
    d = lgb.Dataset(X, y.astype(int), free_raw_data=False)
    if rounds is not None:
        return lgb.train(p, d, num_boost_round=rounds)
    v = lgb.Dataset(Xes, yes.astype(int), reference=d)
    return lgb.train(p, d, num_boost_round=MAX_ROUNDS, valid_sets=[v],
                     callbacks=[lgb.early_stopping(ES_ROUNDS, verbose=False)])


def fit_predict(tab: pd.DataFrame, cols: list[str], threads: int = 2, log=print) -> tuple[pd.DataFrame, dict]:
    """Per-event probabilities p{down,side,up}_h{1,5}: out-of-fold before 2016, the final model from 2016 on."""
    out = pd.DataFrame(index=tab.index)
    info = {}
    at = tab.available_utc
    year = at.dt.year
    X = tab[cols].to_numpy(np.float32)
    for h in HORIZONS:
        y = tab[f"y_h{h}"].to_numpy()
        ex = tab[f"exit_h{h}"]
        tr = ((at < FIT_BEFORE) & (ex < FIT_BEFORE)).to_numpy() & np.isfinite(y)
        es = ((at >= FIT_BEFORE) & (at < ES_BEFORE) & (ex < ES_BEFORE)).to_numpy() & np.isfinite(y)
        m = _fit(X[tr], y[tr], X[es], y[es], threads=threads)
        rounds = int(m.best_iteration or m.current_iteration())
        P = np.full((len(tab), 3), np.nan)
        late = (at >= FIT_BEFORE).to_numpy()
        P[late] = m.predict(X[late], num_iteration=rounds)
        for lo, hi in FOLDS:
            f_lo = pd.Timestamp(f"{lo}-01-01", tz="UTC")
            f_hi = pd.Timestamp(f"{hi + 1}-01-01", tz="UTC")
            infold = ((year >= lo) & (year <= hi) & (at < FIT_BEFORE)).to_numpy()
            keep = tr & ~((at >= f_lo - GAP) & (at < f_hi + GAP) | (ex >= f_lo - GAP) & (ex < f_hi + GAP)).to_numpy()
            mf = _fit(X[keep], y[keep], rounds=rounds, threads=threads)
            P[infold] = mf.predict(X[infold])
        for k, c in enumerate(("down", "side", "up")):
            out[f"p{c}_h{h}"] = P[:, k]
        info[f"h{h}"] = {"trees": rounds, "train_events": int(tr.sum()), "early_stop_events": int(es.sum())}
        log(f"SEC expert h{h}: {rounds} trees, {int(tr.sum()):,} training events, {int(es.sum()):,} early-stopping events")
    return out, info


def diagnostics(tab: pd.DataFrame, pred: pd.DataFrame) -> dict:
    """AUC of P(move) and of P(up)-P(down) (up vs down) per period; validation = events available in 2017-18."""
    from sklearn.metrics import roc_auc_score
    res = {}
    yr = tab.available_utc.dt.year
    for h in HORIZONS:
        y = tab[f"y_h{h}"]
        for name, sel in (("oof_2003_2015", (yr >= 2003) & (yr <= 2015)), ("valid_2017_2018", yr.isin([2017, 2018]))):
            m = sel & y.notna() & pred[f"pside_h{h}"].notna()
            mv = (y[m] != 1).astype(int)
            ud = m & (y != 1)
            res[f"h{h}_{name}"] = {
                "n": int(m.sum()),
                "auc_move": float(roc_auc_score(mv, 1 - pred.loc[m, f"pside_h{h}"])),
                "auc_up_vs_down": float(roc_auc_score((y[ud] == 2).astype(int),
                                                      pred.loc[ud, f"pup_h{h}"] - pred.loc[ud, f"pdown_h{h}"]))}
    return res


def sample_features(samples: pd.DataFrame, tab: pd.DataFrame, pred: pd.DataFrame) -> pd.DataFrame:
    """samples: sample_id, ticker, decision_utc. Evidence from events available at the decision time (point in time)."""
    e = tab[["ticker", "event_type", "available_utc"]].join(pred).sort_values(["ticker", "available_utc"])
    rows = []
    for t, g in samples.groupby("ticker"):
        et = e[e.ticker == t]
        if et.empty:
            continue
        at = et.available_utc.to_numpy(dtype="datetime64[ns]")
        typ = et.event_type.to_numpy()
        pm1 = 1 - et.pside_h1.to_numpy()
        d1 = (et.pup_h1 - et.pdown_h1).to_numpy()
        pm5 = 1 - et.pside_h5.to_numpy()
        d5 = (et.pup_h5 - et.pdown_h5).to_numpy()
        for sid, d in zip(g.sample_id.to_numpy(), pd.to_datetime(g.decision_utc, utc=True).to_numpy(dtype="datetime64[ns]")):
            hi = np.searchsorted(at, d, side="right")
            if hi == 0:
                continue
            k = hi - 1
            age = d - at[k]
            if age > LOOKBACK.to_timedelta64():
                continue
            lo = np.searchsorted(at, d - WINDOW.to_timedelta64(), side="left")
            w = pm5[lo:hi]
            rows.append({"sample_id": sid, "si_pmove_h1": pm1[k], "si_dir_h1": d1[k], "si_pmove_h5": pm5[k],
                         "si_dir_h5": d5[k], "si_log_days": float(np.log1p(age / np.timedelta64(1, "D"))),
                         "si_is_8k": float(typ[k] == "8K"), "si_is_periodic": float(typ[k] != "8K"),
                         "si_max_pmove_h5_30d": float(np.nanmax(w)) if np.isfinite(w).any() else np.nan,
                         "si_n_events_30d": float(hi - lo)})
    out = pd.DataFrame(rows, columns=["sample_id"] + COLUMNS).set_index("sample_id")
    out = out.reindex(samples.sample_id)
    out["si_present"] = out.si_pmove_h5.notna().astype(float)
    out.index.name = "sample_id"
    return out

r"""Build notebooks/NEXUS_Alpha_End_to_End.ipynb: the whole research pipeline in one notebook, with the code inline.

The notebook is the presentation copy of the project: every step is code you can read, run and change, with a short
explanation above it. It trains the models itself (logistic regression with a global-optimum certificate, LightGBM
with its learning curves, the multimodal fusion network and a fine-tuned ConvNeXt on chart images, both on the GPU),
evaluates them with every metric of the specification, and compares them with the frozen models' saved predictions.

    python scripts/colab/make_end_to_end_notebook.py        # writes the .ipynb (no outputs)

Inputs it reads are staged by scripts/colab/stage_notebook_inputs.py. Nothing reads the sealed 2023+ period.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "notebooks" / "NEXUS_Alpha_End_to_End.ipynb"
CELLS: list[tuple[str, str]] = []


def md(s: str) -> None:
    CELLS.append(("markdown", s.strip("\n")))


def code(s: str) -> None:
    CELLS.append(("code", s.strip("\n")))


# =====================================================================================================================
md(r"""
# NEXUS Alpha: the end-to-end research pipeline in one notebook

**Neural Exchange for Unified Signal Intelligence.** A multimodal research system for US stocks. It reads weekly, daily
and hourly charts, technical indicators, company news and SEC filings, and forecasts six horizons (1, 2, 5, 10, 20 and
21 sessions) as probabilities of UP, SIDEWAYS or DOWN. A separate, deterministic risk engine turns a forecast into
LONG, SHORT or NO TRADE, with entry, stop, targets and size.

This notebook holds the code of every step, with a short explanation above each one:

| # | Step | What you will see |
|---|---|---|
| 1 | Setup and governance | GPU check, inputs on Google Drive, checksums, the sealed-period guard |
| 2 | Data | 236,520 stock-dates of 500 US stocks, 2001-2022; coverage by year and block |
| 3 | Labels | UP / SIDEWAYS / DOWN from volatility-scaled returns, recomputed and checked |
| 4 | Splits | train, early stopping, validation, calibration and test blocks, purged; the purge is re-verified |
| 5 | Features | 80 indicator features (weekly, daily, hourly); the formulas re-implemented and checked against the stored values |
| 6 | Baselines | class prior; multinomial logistic regression with a **global-minimum certificate** |
| 7 | LightGBM | gradient boosting per horizon, **training and validation loss curves**, feature importance |
| 8 | Fusion neural network (GPU) | charts + indicators + news + SEC, gated fusion, 3 seeds, **loss curves** |
| 8A | Language branch (GPU) | ChronoBERT, a transformer with a 2010 text cutoff, reads 68,853 SEC filings; parity with the documented vectors |
| 8B | SEC price-impact expert | the SEC study's own labels turned into a model: does a filing move the price? |
| 8C | Vision + NLP fusion (E28) | one network over charts, indicators, news and filings; the pre-registered keep test, placebo, gates |
| 9 | Optimisation | convexity, seeds, the loss landscape: what "reaching the global minimum" means here |
| 10 | Fine-tuning (GPU) | ImageNet ConvNeXt-Tiny fine-tuned on rendered chart images, **loss curves** |
| 11 | Calibration | temperature scaling, reliability diagrams, ECE |
| 12 | Evaluation | log loss, Brier, accuracy, balanced accuracy, macro-F1, AUROC, **confusion matrices**, ROC curves |
| 13 | Statistics | month-block bootstrap confidence intervals, placebo, Holm correction, rank IC |
| 14 | Comparison | every model side by side, with the frozen models' saved predictions and the reproduction check |
| 15 | Risk engine | entry, stop, targets and size with the reasons, for real setups |
| 16 | Backtest | equity, drawdown and Sharpe intervals of the documented backtest |
| 17 | Knowledge base (RAG) | BM25, dense and hybrid retrieval over the books and rulebooks, with page citations |
| 18 | Forward test | the public, hash-chained forward test, verified from GitHub |
| 19 | Conclusions | what works, what does not, and what comes next |

**Honest summary before you start.** Volatility regime (sideways or not) is predictable; the cross-sectional ranking of
stocks has a small positive rank IC on validation; the direction of a single stock over 1-20 sessions is close to chance;
the multimodal blocks (charts, news, filings) add little beyond the indicators; and the documented backtests show no
edge over equal-weight buy-and-hold after costs. The notebook shows each of these with numbers you can recompute.

Research and paper trading only. Nothing here is investment advice, and no order is ever sent anywhere.
""")

md(r"""
## How to run it

**On Colab Pro+ (recommended).** Runtime > Change runtime type > **A100 GPU** or **L4 GPU** (T4 also works), High-RAM
on. Then Runtime > Run all. About 45-60 minutes on an A100 and roughly 1.5-2 times that on an L4. The GPU sections are
8, 8A, 8C and 10; without a GPU they run smaller versions (8A checks 400 filings instead of all 68,853), and the
whole notebook takes a few hours on a CPU.

**Inputs (on your Google Drive, under `MyDrive/antigravity_workspace/`):**
* `nexus_artifacts/notebook/`: the compact input set staged by `scripts/colab/stage_notebook_inputs.py` (about 1 GB):
  the sample frame, chart-embedding components, news/SEC/rule features, the language inputs of sections 8A-8C (SEC
  filing text and vectors, the SEC expert's table), saved predictions of the documented runs and the knowledge base. **Keep this folder private:** the knowledge base contains book text.
* `us_stocks_ohlcv/final/1day/`: daily bars, read only up to 2022-12-30 (needed by sections 5 and 10).

The public repository (code, documented metrics and forward-test logs) is cloned in section 1.

**Settings** are in the first code cell: `QUICK = True` runs everything on small samples in a few minutes (a smoke
test); `QUICK = False` is the full run.
""")

# ---------------------------------------------------------------------------------------------------------- setup
md(r"""
## 1. Setup and governance

The project's evidence rests on three rules, and the notebook enforces them in code:
1. **Point in time.** Every input is available at the decision time (09:00 New York on the session after the as-of
   close). Labels use only later prices.
2. **The sealed period stays sealed.** Nothing from 2023-01-03 onward is read; the frozen models will be scored on it
   once, by the owner, after the final freeze. The sample frame ends 2022-12-01 and bars are read with a date filter.
3. **Rules before results.** Model choices are made on the validation years (2017-2018) only; the test years
   (2020-2022) are reported, never used to choose.
""")

code(r"""
# ---- settings ---------------------------------------------------------------------------------------------------
QUICK = False            # True: small samples, a few minutes (smoke test). False: the full run.
SEED = 7
REPO_URL = "https://github.com/harshatangirala/nexus-alpha"

import os, sys, json, math, time, hashlib, warnings, subprocess
from pathlib import Path
warnings.filterwarnings("ignore")
IN_COLAB = "google.colab" in sys.modules
T_START = time.time()

if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    WS = Path("/content/drive/MyDrive/antigravity_workspace")
    REPO_DIR = Path("/content/nexus-alpha")
    if not REPO_DIR.exists():
        subprocess.run(["git", "clone", "-q", REPO_URL, str(REPO_DIR)], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "lightgbm==4.7.0", "timm==1.0.31"], check=True)
else:
    WS = Path(os.environ.get("NEXUS_WS", r"G:\My Drive\antigravity_workspace"))
    REPO_DIR = Path(os.environ.get("NEXUS_REPO", r"C:\Users\harsh\nexus-alpha"))
NB = Path(os.environ.get("NEXUS_NB_INPUTS", str(WS / "nexus_artifacts" / "notebook")))
BARS_DIR = Path(os.environ.get("NEXUS_BARS_1DAY", str(WS / "us_stocks_ohlcv" / "final" / "1day")))
print("inputs:", NB, "| bars:", BARS_DIR, "| repository:", REPO_DIR)
""")

code(r"""
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import torch, torch.nn as nn, torch.nn.functional as F
import lightgbm as lgb
from scipy import optimize, stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (log_loss, roc_auc_score, accuracy_score, balanced_accuracy_score, f1_score,
                             confusion_matrix, roc_curve)
try:
    from IPython.display import display, Markdown
except ImportError:                                   # plain-Python runs
    display = print
    Markdown = str

plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": .3, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 10})
CLR = {"prior": "#9aa5b1", "logistic": "#1f77b4", "lgbm": "#2ca02c", "fusion": "#d62728", "cnn_ft": "#9467bd",
       "v1.1": "#ff7f0e", "UP": "#2ca02c", "SIDEWAYS": "#7f7f7f", "DOWN": "#d62728"}
rng = np.random.default_rng(SEED)
torch.manual_seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    p = torch.cuda.get_device_properties(0)
    print(f"GPU: {p.name}, {p.total_memory / 1e9:.0f} GB")
else:
    print("No GPU: sections 8 and 10 run smaller CPU versions.")
print("torch", torch.__version__, "| lightgbm", lgb.__version__, "| pandas", pd.__version__, "| numpy", np.__version__)

# ---- governance guard ------------------------------------------------------------------------------------------
SEAL_DATE = pd.Timestamp("2023-01-03")                  # first sealed session; nothing on or after it is read
for v in ("NEXUS_HOLDOUT_CONFIG", "NEXUS_HOLDOUT_UNLOCK"):
    assert not os.environ.get(v), f"{v} is set: this notebook never opens the sealed period"

def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

man = json.loads((NB / "manifest_notebook.json").read_text(encoding="utf-8"))
bad = [k for k, v in man["files"].items() if (NB / k).stat().st_size != v["bytes"]]
assert not bad, f"inputs differ from the staging manifest: {bad[:5]}"
check = ["samples/s500_frame.parquet", "img_pca64_convnext_tiny.npy", "predictions/rows_eval.parquet"]
for k in check:
    assert sha256(NB / k) == man["files"][k]["sha256"], f"checksum mismatch: {k}"
print(f"{len(man['files'])} staged files present; checksums of {len(check)} key files verified "
      f"(staged {man['created_utc']}).")
""")

# ---------------------------------------------------------------------------------------------------------- data
md(r"""
## 2. Data

The sample frame is TriFrame's 500-stock set: one decision date every ~10 sessions per stock, staggered, from 2001-10
to 2022-12-01, **236,520 stock-dates**. Each row holds the as-of features, the labels for six horizons and its block.
The 500 stocks are survivors in today's universe, so results carry survivorship bias (stated in
`docs/LIMITATIONS.md`); the benchmark in section 16 has the same bias.
""")

code(r"""
FR = pd.read_parquet(NB / "samples" / "s500_frame.parquet")
FR["asof_date"] = pd.to_datetime(FR.asof_date)
assert FR.asof_date.max() < SEAL_DATE, "a sealed row is in the frame"
H6 = ["1D", "2D", "5D", "10D", "20D", "1M"]
HN = {"1D": 1, "2D": 2, "5D": 5, "10D": 10, "20D": 20, "1M": 21}
CLASSES = ["DOWN", "SIDEWAYS", "UP"]                     # label codes 0, 1, 2
BLOCKS = ["train", "inner", "valid", "calib", "test"]
print(f"{len(FR):,} rows, {FR.ticker.nunique()} stocks, {FR.asof_date.min().date()} to {FR.asof_date.max().date()}")

cov = FR.assign(year=FR.asof_date.dt.year, block=FR.block.fillna("purged")).pivot_table(
    index="year", columns="block", values="sample_id", aggfunc="count").fillna(0)
cov = cov[[c for c in BLOCKS + ["purged"] if c in cov]]
fig, ax = plt.subplots(figsize=(11, 3.4))
bottom = np.zeros(len(cov))
for c, col in zip(cov.columns, ["#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c", "#d9d9d9"]):
    ax.bar(cov.index, cov[c], bottom=bottom, color=col, label=c)
    bottom += cov[c].to_numpy()
ax.set_title("Stock-dates per year and block")
ax.legend(ncol=6, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.1), frameon=False); plt.show()
display(FR.block.fillna("purged").value_counts().rename("rows").to_frame().T)
""")

# ---------------------------------------------------------------------------------------------------------- labels
md(r"""
## 3. Labels

For each horizon h (sessions): the return from the next session's open (the earliest executable price after the
decision time) to the close h sessions later, R_h = close[t+h] / open[t+1] - 1. The threshold scales with the stock's
own volatility: tau_h = max(0.5 * sigma_t * sqrt(h), 10 bps), sigma_t the EWMA (lambda 0.94) daily volatility at t.
**UP** if R_h > tau_h, **DOWN** if R_h < -tau_h, otherwise **SIDEWAYS**. Prices only (no dividends). The rule was fixed
before any model was trained and was not tuned for class balance.

The cell recomputes every label and threshold from the stored returns and volatility and checks they match.
""")

code(r"""
mism = {}
for h in H6:
    tau = np.maximum(0.5 * FR.sigma_ewma * np.sqrt(HN[h]), 0.001)
    ret = FR[f"ret_{h}"]
    y = np.where(ret > tau, 2, np.where(ret < -tau, 0, 1)).astype(float)
    y[ret.isna().to_numpy()] = np.nan
    ok = FR[f"y_{h}"].notna()
    mism[h] = {"tau max abs diff": float(np.nanmax(np.abs(tau - FR[f"tau_{h}"]))),
               "label mismatches": int((y[ok] != FR.loc[ok, f"y_{h}"]).sum()), "labelled rows": int(ok.sum())}
display(pd.DataFrame(mism).T)
assert all(v["label mismatches"] == 0 for v in mism.values()), "label rule not reproduced"

Y = np.stack([FR[f"y_{h}"].fillna(-1).to_numpy(np.int64) for h in H6], 1)       # -1 = no label (not used)
RET = np.stack([FR[f"ret_{h}"].to_numpy(np.float64) for h in H6], 1)
ZV = np.stack([FR[f"z_{h}"].to_numpy(np.float32) for h in H6], 1)              # vol-scaled return (quantile heads)
def shares(b, j):
    y = Y[(FR.block == b).to_numpy(), j]
    return pd.Series(y[y >= 0]).value_counts(normalize=True)
cnt = pd.DataFrame({(b, h): shares(b, j) for b in ["train", "valid", "test"] for j, h in enumerate(H6)}).T
cnt = cnt.rename(columns=dict(enumerate(CLASSES)))[CLASSES]
fig, axs = plt.subplots(1, 3, figsize=(12, 3), sharey=True)
for ax, b in zip(axs, ["train", "valid", "test"]):
    d = cnt.loc[b]
    left = np.zeros(len(d))
    for c in CLASSES:
        ax.barh(d.index, d[c], left=left, color=CLR[c], label=c); left += d[c].to_numpy()
    ax.set_title(f"class shares, {b}"); ax.set_xlim(0, 1)
fig.legend(*axs[0].get_legend_handles_labels(), ncol=3, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, -0.06),
           frameon=False)
plt.tight_layout(); plt.show()
""")

# ---------------------------------------------------------------------------------------------------------- splits
md(r"""
## 4. Splits

| Block | As-of dates | Use |
|---|---|---|
| train | 2003 to 2015 | fitting |
| inner | 2016 | early stopping only |
| valid | 2017 to 2018 | model and configuration choice |
| calib | 2019 | probability calibration and thresholds |
| test | 2020 to 2022-12-01 | reported once per frozen configuration (a historical diagnostic: TriFrame read these years) |
| sealed | from 2023-01-03 | the final evaluation, once, by the owner |

**Purging.** A row stays in a block only if its longest label window (21 sessions) plus a 5-session embargo ends before
the next block starts, so no training outcome overlaps a later block's outcomes. The cell checks it independently.
""")

code(r"""
masks = {b: (FR.block == b).to_numpy() for b in BLOCKS}
rep = []
for a, b in zip(BLOCKS[:-1], BLOCKS[1:]):
    last_end = pd.to_datetime(FR.loc[masks[a], "end_1M"]).max()
    first_next = FR.loc[masks[b], "asof_date"].min()
    gap = int(np.busday_count(last_end.date(), first_next.date()))
    rep.append({"from": a, "to": b, "latest label end": last_end.date(), "next block starts": first_next.date(),
                "business days between": gap, "embargo respected (>= 5)": gap >= 5})
display(pd.DataFrame(rep))
assert all(r["embargo respected (>= 5)"] for r in rep)
EVAL = masks["valid"] | masks["calib"] | masks["test"]                        # rows with saved predictions
EVI = np.flatnonzero(EVAL)
""")

# ---------------------------------------------------------------------------------------------------------- features
md(r"""
## 5. Features

Eighty numeric features per row, all causal (bar t uses bars up to t only):
* **47 indicator features** on the weekly (`w_`), daily (`d_`) and hourly (`h_`) bars: ATR as a share of price, RSI(14),
  MACD(12,26,9) and its histogram in ATR units, distance to SMA20/SMA50, Bollinger %B and width, volume ratio,
  returns, realised volatility, gaps, range, close location, distance to the 52-week high, EWMA volatility.
* **33 extra price/volume features** (`x_`): overnight and intraday returns, abnormal volume, streaks, short-term RSI,
  variance ratio, autocorrelation, and Parkinson and Garman-Klass volatility.

The formulas follow TriFrame's frozen conventions. The cell below re-implements the daily indicators from raw bars and
checks them against the stored values for three stocks.
""")

code(r"""
STRUCTURED = ['d_atr_pct', 'd_rsi14', 'd_macd_atr', 'd_macd_hist_atr', 'd_close_sma20', 'd_close_sma50',
              'd_sma20_sma50', 'd_bb_pctb', 'd_bb_width', 'd_log_vol_ratio', 'd_ret1', 'd_ret5', 'd_ret21', 'd_ret63',
              'd_rv5', 'd_rv20', 'd_gap', 'd_range_atr', 'd_close_loc', 'd_dist_high252', 'd_sigma_ewma',
              'w_atr_pct', 'w_rsi14', 'w_macd_atr', 'w_macd_hist_atr', 'w_close_sma20', 'w_close_sma50',
              'w_sma20_sma50', 'w_bb_pctb', 'w_bb_width', 'w_log_vol_ratio', 'w_ret1', 'w_ret4', 'w_wtd_ret',
              'h_atr_pct', 'h_rsi14', 'h_macd_atr', 'h_macd_hist_atr', 'h_close_sma20', 'h_close_sma50',
              'h_sma20_sma50', 'h_bb_pctb', 'h_bb_width', 'h_log_vol_ratio', 'h_ret7', 'h_ret35',
              'h_last_session_close_loc']
X33 = [c for c in FR.columns if c.startswith("x_")]
NUM = STRUCTURED + X33
print(len(STRUCTURED), "indicator features +", len(X33), "extra =", len(NUM))

def read_daily(ticker):
    # pyarrow pushes the date filter down: no sealed bar is ever loaded into memory
    import pyarrow.parquet as pq
    from datetime import datetime
    t = pq.read_table(BARS_DIR / f"{ticker}.parquet", columns=["date", "open", "high", "low", "close", "volume"],
                      filters=[("date", "<", datetime(2023, 1, 3))])
    d = t.to_pandas().sort_values("date").reset_index(drop=True)
    d["date"] = pd.to_datetime(d["date"]).dt.normalize()
    assert d.date.max() < SEAL_DATE
    return d

def daily_indicators(d):
    c, h, l, v, o = (d[k].astype(float) for k in ("close", "high", "low", "volume", "open"))
    sma20, sma50 = c.rolling(20).mean(), c.rolling(50).mean()
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    hist = macd - macd.ewm(span=9, adjust=False).mean()
    dl = c.diff()
    ag = dl.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    al = (-dl).clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    rsi = (100 - 100 / (1 + ag / al.replace(0, np.nan))).where(al > 0, np.where(ag > 0, 100.0, 50.0))
    sd = c.rolling(20).std(ddof=0)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / 14, adjust=False).mean()
    out = pd.DataFrame({"date": d.date})
    out["d_atr_pct"] = atr / c
    out["d_rsi14"] = rsi
    out["d_macd_hist_atr"] = hist / atr.replace(0, np.nan)
    out["d_close_sma20"] = c / sma20 - 1
    out["d_bb_pctb"] = (c - (sma20 - 2 * sd)) / (4 * sd).replace(0, np.nan)
    out["d_ret21"] = c / c.shift(21) - 1
    out["d_rv20"] = np.log(c).diff().rolling(20).std(ddof=0) * np.sqrt(252)
    out["d_dist_high252"] = c / h.rolling(252, min_periods=126).max() - 1
    out["_close"], out["_sma20"], out["_sma50"] = c, sma20, sma50
    out["_bb_up"], out["_bb_lo"], out["_rsi"] = sma20 + 2 * sd, sma20 - 2 * sd, rsi
    return out

PARITY = {}
if BARS_DIR.exists():
    for t in ["AAPL", "JPM", "KO"]:
        ind = daily_indicators(read_daily(t)).set_index("date")
        rows = FR[FR.ticker == t].set_index("asof_date")
        both = rows.index.intersection(ind.index)
        for c in ["d_atr_pct", "d_rsi14", "d_macd_hist_atr", "d_close_sma20", "d_bb_pctb", "d_ret21", "d_rv20",
                  "d_dist_high252"]:
            a, b = ind.loc[both, c].to_numpy(float), rows.loc[both, c].to_numpy(float)
            okm = np.isfinite(a) & np.isfinite(b)
            PARITY[(t, c)] = float(np.max(np.abs(a[okm] - b[okm]))) if okm.any() else np.nan
    par = pd.Series(PARITY).unstack()
    display(par.style.format("{:.1e}"))
    worst = float(np.nanmax(par.to_numpy()))
    print(f"largest absolute difference over all checks: {worst:.2e}"
          + (" (machine precision: the formulas reproduce the stored features exactly)" if worst < 1e-9 else ""))
    # a chart with the indicators the risk engine reads
    d = daily_indicators(read_daily("AAPL")).set_index("date").loc["2021-01-01":"2021-12-31"]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 5), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    a1.plot(d.index, d._close, color="k", lw=1, label="close")
    a1.plot(d.index, d._sma20, label="SMA20"); a1.plot(d.index, d._sma50, label="SMA50")
    a1.fill_between(d.index, d._bb_lo, d._bb_up, alpha=.12, label="Bollinger(20, 2)")
    a1.set_title("AAPL 2021: the indicators the features and the risk engine use"); a1.legend(ncol=4, fontsize=8)
    a2.plot(d.index, d._rsi, color="#9467bd"); a2.axhline(70, ls="--", c="r", lw=.8); a2.axhline(30, ls="--", c="g", lw=.8)
    a2.set_ylabel("RSI 14"); plt.tight_layout(); plt.show()
else:
    print("Daily bars not found; the parity check is skipped.")
""")

# ---------------------------------------------------------------------------------------------------------- baselines
md(r"""
## 6. Baselines: the class prior and logistic regression, with a global-minimum certificate

* **Class prior.** Each horizon's training class shares, used as the forecast for every row. Every model must beat it.
* **Multinomial logistic regression** on the 80 features (train-median imputation with missing flags,
  standardisation on training rows, clipping at 8 standard deviations), with an L2 penalty of 1e-3 on the mean log
  loss: the logistic component of the frozen v1.1 blend (E09).

**"Did training reach the global minimum?"** For this model the question has an exact answer. The L2-penalised
multinomial log loss is strictly convex, so it has one global minimum. The cell below solves the same objective
from five random starts and checks that (a) all starts reach the same objective value and the same parameters,
(b) the gradient at the solution is numerically zero, and (c) the Hessian there is positive definite (its smallest
eigenvalue is at least the penalty). Together these certify the global minimum. A neural network has no such
certificate; section 9 shows what can be said instead.
""")

code(r"""
class ImputeScale:
    # train-median imputation + missing flags, then standardisation on training rows only
    def fit(self, X):
        self.cols, self.med = list(X.columns), X.median()
        self.miss = [c for c in self.cols if X[c].isna().any()]
        self.sc = StandardScaler().fit(self._z(X)); return self
    def _z(self, X):
        Z = X[self.cols].fillna(self.med)
        for c in self.miss:
            Z[c + "__missing"] = X[c].isna().astype(float)
        return Z.to_numpy(float)
    def transform(self, X):
        return np.clip(self.sc.transform(self._z(X)), -8, 8)

def sample(mask, n):
    ix = np.flatnonzero(mask)
    return ix if (n is None or len(ix) <= n) else np.sort(rng.choice(ix, n, replace=False))

N_TRAIN = 25_000 if QUICK else None
TR = sample(masks["train"], N_TRAIN)
PRE = ImputeScale().fit(FR.iloc[TR][NUM])
XS = PRE.transform(FR[NUM])                                # standardised features for every row
PRED = {}                                                  # model -> probabilities on EVAL rows, shape (n, 6, 3)

prior = np.zeros((len(EVI), 6, 3))
logit = np.zeros((len(EVI), 6, 3))
LOGIT_MODELS = {}
t0 = time.time()
for j, h in enumerate(H6):
    ytr = Y[TR, j]; ok = ytr >= 0
    p = np.bincount(ytr[ok], minlength=3) / ok.sum()
    prior[:, j] = p
    m = LogisticRegression(C=1.0 / (1e-3 * ok.sum()), max_iter=3000)     # sklearn's C: penalty 1e-3 on the mean loss
    m.fit(XS[TR][ok], ytr[ok])
    logit[:, j] = m.predict_proba(XS[EVI])
    LOGIT_MODELS[h] = m
PRED["prior"], PRED["logistic"] = prior, logit
print(f"prior and logistic fitted for 6 horizons in {time.time() - t0:.0f}s")

# ---- the certificate (5D horizon; a 20,000-row training sample keeps the exact Hessian cheap) ----------------------
def certificate(X, y, lam=1e-3, starts=5):
    # objective: mean multinomial log loss + lam/2 * ||W||^2 (all parameters penalised, as in E09)
    n, d = X.shape
    Xb = np.hstack([X, np.ones((n, 1))])
    Yo, k = np.eye(3)[y], (d + 1) * 3
    def probs(w):
        Zl = Xb @ w.reshape(d + 1, 3); Zl -= Zl.max(1, keepdims=True)
        P = np.exp(Zl); return P / P.sum(1, keepdims=True)
    def f_g(w):
        P, W = probs(w), w.reshape(d + 1, 3)
        f = -np.mean(np.log(P[np.arange(n), y] + 1e-300)) + 0.5 * lam * np.sum(W ** 2)
        return f, (Xb.T @ (P - Yo) / n + lam * W).ravel()
    def hessian(w):                                             # exact: data term (positive semi-definite) + lam * I
        P, Hm = probs(w), np.zeros((k, k))
        for a in range(3):
            for b in range(3):
                Hm[a::3, b::3] = (Xb * (P[:, a] * ((a == b) - P[:, b]))[:, None]).T @ Xb / n
        return Hm + lam * np.eye(k)
    sols = []
    for s in range(starts):                                     # five random starts, far apart
        w = optimize.minimize(f_g, np.random.default_rng(100 + s).normal(0, 1, k), jac=True, method="L-BFGS-B",
                              options={"maxiter": 20000, "ftol": 1e-15, "gtol": 1e-10}).x
        for _ in range(3):                                      # Newton steps with the exact Hessian (quadratic convergence)
            w = w - np.linalg.solve(hessian(w), f_g(w)[1])
        sols.append(w)
    fs = np.array([f_g(w)[0] for w in sols]); best = sols[int(fs.argmin())]
    eig = np.linalg.eigvalsh(hessian(best))
    gn = float(np.linalg.norm(f_g(best)[1]))
    dist = float(max(np.abs(w - best).max() for w in sols))
    ok = bool(np.ptp(fs) < 1e-10 and gn < 1e-6 and dist < 1e-5 and eig.min() > 0)
    return {"objective at each start": ", ".join(f"{f:.12f}" for f in fs), "objective range": f"{np.ptp(fs):.1e}",
            "max parameter distance between starts": f"{dist:.1e}", "gradient norm at the solution": f"{gn:.1e}",
            "Hessian smallest eigenvalue": f"{eig.min():.6f} (> 0: strictly convex)",
            "Hessian largest eigenvalue": f"{eig.max():.3f}", "certified global minimum": ok}

j5 = H6.index("5D")
cs = sample(masks["train"] & (Y[:, j5] >= 0), 4_000 if QUICK else 20_000)
CERT = certificate(XS[cs], Y[cs, j5])
display(pd.Series(CERT, name="5D logistic objective, 5 random starts").to_frame())
assert CERT["certified global minimum"], "the certificate failed"
""")

# ---------------------------------------------------------------------------------------------------------- lgbm
md(r"""
## 7. LightGBM: gradient-boosted trees, with training and validation loss curves

One multiclass booster per horizon on the same 80 features (trees need no scaling). Settings are the project's fixed
defaults (learning rate 0.05, 31 leaves, at least 400 rows per leaf, L2 = 5, bagging 0.8). Training stops when the
loss on the **inner** block (2016) has not improved for 100 rounds: the validation years are never used to stop or
tune. The curves show the training loss falling steadily while the held-out loss flattens early; the gap between
them is the variance the early stop controls.
""")

code(r"""
LGB = {"objective": "multiclass", "num_class": 3, "learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 400,
       "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 5.0, "verbose": -1,
       "deterministic": True, "force_row_wise": True, "seed": SEED, "num_threads": os.cpu_count() or 4}
INN = sample(masks["inner"], 6_000 if QUICK else None)
VAL = np.flatnonzero(masks["valid"])
gb = np.zeros((len(EVI), 6, 3))
CURVES, IMP, BEST_IT = {}, {}, {}
Xdf = FR[NUM].astype(np.float32)

def stop_on_inner(patience=100, name="inner (early stop)"):
    # early stopping that watches ONLY the inner block; the validation curve is recorded for display, never used
    best = {"v": float("inf"), "it": 0}
    def _cb(env):
        for ds, metric, val, _ in env.evaluation_result_list:
            if ds != name:
                continue
            if val < best["v"] - 1e-12:
                best.update(v=val, it=env.iteration)
            elif env.iteration - best["it"] >= patience:
                raise lgb.callback.EarlyStopException(best["it"], [(name, metric, best["v"], False)])
    _cb.order = 30
    return _cb

t0 = time.time()
for j, h in enumerate(H6):
    ok_tr, ok_in, ok_va = Y[TR, j] >= 0, Y[INN, j] >= 0, Y[VAL, j] >= 0
    dtr = lgb.Dataset(Xdf.iloc[TR[ok_tr]], Y[TR[ok_tr], j], free_raw_data=False)
    din = lgb.Dataset(Xdf.iloc[INN[ok_in]], Y[INN[ok_in], j], reference=dtr)
    dva = lgb.Dataset(Xdf.iloc[VAL[ok_va]], Y[VAL[ok_va], j], reference=dtr)
    ev = {}
    bst = lgb.train(LGB, dtr, num_boost_round=300 if QUICK else 1500, valid_sets=[dtr, din, dva],
                    valid_names=["train", "inner (early stop)", "valid (shown only)"],
                    callbacks=[stop_on_inner(100), lgb.record_evaluation(ev)])
    CURVES[h], BEST_IT[h] = ev, bst.best_iteration
    gb[:, j] = bst.predict(Xdf.iloc[EVI], num_iteration=bst.best_iteration)
    IMP[h] = pd.Series(bst.feature_importance("gain"), index=NUM)
PRED["lgbm"] = gb
print(f"LightGBM, 6 horizons, {time.time() - t0:.0f}s; best iterations:", BEST_IT)

fig, axs = plt.subplots(2, 3, figsize=(13, 6.2))
for ax, h in zip(axs.ravel(), H6):
    for name, c in (("train", "#1f77b4"), ("inner (early stop)", "#ff7f0e"), ("valid (shown only)", "#2ca02c")):
        ax.plot(CURVES[h][name]["multi_logloss"], color=c, label=name, lw=1.4)
    ax.axvline(BEST_IT[h], color="k", ls=":", lw=1)
    ax.set_title(f"{h}: multiclass log loss by boosting round"); ax.set_xlabel("round")
axs[0, 0].legend(fontsize=8); plt.tight_layout(); plt.show()

imp = pd.DataFrame(IMP)
imp = imp.div(imp.sum(0), axis=1).mean(1).sort_values(ascending=False).head(20)
fig, ax = plt.subplots(figsize=(8, 5)); imp[::-1].plot.barh(ax=ax, color="#2ca02c")
ax.set_title("LightGBM: share of total gain, mean over horizons (top 20)"); plt.tight_layout(); plt.show()
""")

# ---------------------------------------------------------------------------------------------------------- fusion NN
md(r"""
## 8. The multimodal fusion network (GPU)

Six input blocks, each with a presence flag (a missing block is zeroed **and** flagged, so absence is never read as
neutral evidence):

| Block | Content | Dimension |
|---|---|---|
| `img_W`, `img_D`, `img_H` | weekly, daily and hourly chart images through a frozen ImageNet ConvNeXt-Tiny, reduced to 64 principal components per view (fitted on training rows) | 3 x 64 |
| `num` | the 80 indicator features | 80 |
| `news` | aggregated company-news representation (TF-IDF/SVD, Loughran-McDonald tone, counts) available before the decision time | 75 |
| `sec` | SEC filing features (8-K, 10-Q/10-K timing, items, text) inside the SEC study's coverage | 91 |

**Architecture (gated fusion, the variant selected in E03).** Each block is projected to 64 dimensions
(Linear-GELU-Dropout-Linear-LayerNorm) and tagged with a learned type embedding. A gating network scores the present
blocks; a softmax over them weights the projected vectors into one representation. Two heads read it: a classifier
for 6 horizons x 3 classes and quantile heads (10%, 50%, 90%) of the volatility-scaled return.

**Loss.** Cross-entropy over the six horizons (missing labels ignored) plus 0.2 x the pinball loss of the quantiles.
**Training.** AdamW (learning rate 3e-4 with a cosine schedule, weight decay 1e-4, dropout 0.3: the E06 optimum),
modality dropout 0.15, batches of 1,024, early stopping on the inner block (patience 4), three seeds averaged.
""")

code(r"""
img = np.load(NB / "img_pca64_convnext_tiny.npy").astype(np.float32)        # (236520, 192), rows as FR
feats = {k: pd.read_parquet(NB / "features" / f"{k}_features.parquet") for k in ("news", "sec")}
news = FR[["sample_id"]].merge(feats["news"], left_on="sample_id", right_index=True, how="left")
sec = FR[["sample_id"]].merge(feats["sec"], left_on="sample_id", right_index=True, how="left")
NEWS_COLS = [c for c in news.columns if c.startswith("news_") and c not in ("news_present", "news_n_selected_no_text")]
SEC_COLS = [c for c in sec.columns if c.startswith("sec_") and c != "sec_covered"]

def standardise(X, fit_rows):
    mu, sd = np.nanmean(X[fit_rows], 0), np.nanstd(X[fit_rows], 0)
    sd = np.where(np.isfinite(sd) & (sd > 1e-8), sd, 1.0)
    return np.clip(np.nan_to_num((X - np.nan_to_num(mu)) / sd, nan=0.0), -8, 8).astype(np.float32)

blocks, present = {}, {}
for k, v in enumerate("WDH"):
    need = {"W": 42, "D": 51, "H": 56}[v]                      # at least 40% of the window's bars must exist
    present[f"img_{v}"] = (FR[f"n_valid_{v}"].to_numpy() >= need).astype(np.float32)
    blocks[f"img_{v}"] = standardise(img[:, 64 * k: 64 * (k + 1)], TR) * present[f"img_{v}"][:, None]
blocks["num"] = standardise(FR[NUM].to_numpy(np.float64), TR); present["num"] = np.ones(len(FR), np.float32)
for name, df, cols, flag in (("news", news, NEWS_COLS, "news_present"), ("sec", sec, SEC_COLS, "sec_covered")):
    pr = df[flag].fillna(0).to_numpy(np.float32)
    fit = TR[pr[TR] > 0]
    blocks[name] = standardise(df[cols].to_numpy(np.float64), fit) * pr[:, None]
    present[name] = pr
NAMES = list(blocks)
MASK = np.stack([present[n] for n in NAMES], 1)
print({n: blocks[n].shape[1] for n in NAMES})
print("share of rows with the block present:", {n: round(float(present[n].mean()), 3) for n in NAMES})

class Projector(nn.Module):
    def __init__(self, d_in, d, p):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, d), nn.GELU(), nn.Dropout(p), nn.Linear(d, d), nn.LayerNorm(d))
    def forward(self, x):
        return self.net(x)

class GatedFusion(nn.Module):
    def __init__(self, dims, n_h=6, d=64, p=0.3):
        super().__init__()
        self.names = list(dims)
        self.proj = nn.ModuleDict({m: Projector(k, d, p) for m, k in dims.items()})
        self.type_emb = nn.Parameter(torch.randn(len(dims), d) * 0.02)
        self.gate = nn.Sequential(nn.Linear(d, d), nn.Tanh(), nn.Linear(d, 1))
        self.cls, self.q, self.n_h = nn.Linear(d, n_h * 3), nn.Linear(d, n_h * 3), n_h
    def forward(self, xs, mask):
        Hm = torch.stack([self.proj[m](xs[m]) for m in self.names], 1) + self.type_emb[None]
        Hm = Hm * mask[..., None]
        g = self.gate(Hm).squeeze(-1).masked_fill(mask == 0, -1e9).softmax(-1)      # weights over present blocks
        z = (g[..., None] * Hm).sum(1)
        return self.cls(z).view(-1, self.n_h, 3), self.q(z).view(-1, self.n_h, 3), g

QS = torch.tensor([0.1, 0.5, 0.9])
def fusion_loss(logits, quant, y, z, qw=0.2):
    ce = F.cross_entropy(logits.reshape(-1, 3), y.reshape(-1), ignore_index=-1, reduction="sum") / (y >= 0).sum().clamp(min=1)
    zm = torch.isfinite(z); zz = torch.nan_to_num(z)
    d = zz[..., None] - quant
    qs = QS.to(z.device)
    pin = (torch.maximum(qs * d, (qs - 1) * d) * zm[..., None]).sum() / (zm.sum().clamp(min=1) * 3)
    return ce + qw * pin, ce

def to_dev(ix, B=None):
    # B = (blocks, names, mask) of another block set (section 8C); default: this section's six blocks
    bl, nm, mk = B or (blocks, NAMES, MASK)
    return ({n: torch.from_numpy(bl[n][ix]).to(DEVICE) for n in nm}, torch.from_numpy(mk[ix]).to(DEVICE),
            torch.from_numpy(Y[ix]).to(DEVICE), torch.from_numpy(ZV[ix]).to(DEVICE))

@torch.no_grad()
def eval_ce(model, data, bs=8192):
    model.eval(); xs, m, y, z = data; tot, n = 0.0, 0
    for i in range(0, len(m), bs):
        lg, qu, _ = model({k: v[i:i + bs] for k, v in xs.items()}, m[i:i + bs])
        yy = y[i:i + bs]
        tot += F.cross_entropy(lg.reshape(-1, 3), yy.reshape(-1), ignore_index=-1, reduction="sum").item()
        n += int((yy >= 0).sum())
    return tot / max(n, 1)

@torch.no_grad()
def predict_probs(model, ix, bs=8192, B=None):
    model.eval(); xs, m, _, _ = to_dev(ix, B); out = []
    for i in range(0, len(m), bs):
        lg, _, _ = model({k: v[i:i + bs] for k, v in xs.items()}, m[i:i + bs])
        out.append(lg.softmax(-1).float().cpu().numpy())
    return np.concatenate(out)

def train_fusion(seed, epochs=30, bs=1024, lr=3e-4, wd=1e-4, p_drop=0.3, mod_drop=0.15, patience=4, B=None):
    torch.manual_seed(seed); g = torch.Generator(device="cpu").manual_seed(seed)
    bl, nm, _ = B or (blocks, NAMES, MASK)
    model = GatedFusion({n: bl[n].shape[1] for n in nm}, p=p_drop).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    tr, inn, va = to_dev(TR, B), to_dev(INN, B), to_dev(VAL, B)
    hist, best, best_state, bad = [], 1e9, None, 0
    for ep in range(epochs):
        model.train(); perm = torch.randperm(len(TR), generator=g).to(DEVICE); run, nb = 0.0, 0
        for i in range(0, len(TR), bs):
            b = perm[i:i + bs]
            m = tr[1][b]
            drop = (torch.rand(m.shape, device=DEVICE) < mod_drop).float()
            m2 = m * (1 - drop); empty = m2.sum(1) == 0; m2[empty] = m[empty]      # never hide every block
            lg, qu, _ = model({k: v[b] for k, v in tr[0].items()}, m2)
            loss, ce = fusion_loss(lg, qu, tr[2][b], tr[3][b])
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            run += ce.item(); nb += 1
        sched.step()
        e_in, e_va = eval_ce(model, inn), eval_ce(model, va)
        hist.append({"epoch": ep + 1, "train CE": run / nb, "inner CE (early stop)": e_in, "valid CE (shown only)": e_va,
                     "learning rate": sched.get_last_lr()[0]})
        if e_in < best - 1e-5:
            best, bad = e_in, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model, pd.DataFrame(hist)

SEEDS = [7, 11] if QUICK else [7, 11, 23]
FUSION, HIST = {}, {}
t0 = time.time()
for s in SEEDS:
    FUSION[s], HIST[s] = train_fusion(s, epochs=2 if QUICK else 30)
    print(f"seed {s}: {len(HIST[s])} epochs, best inner CE {HIST[s]['inner CE (early stop)'].min():.5f} "
          f"({time.time() - t0:.0f}s)")
PRED["fusion"] = np.mean([predict_probs(FUSION[s], EVI) for s in SEEDS], 0)

fig, axs = plt.subplots(1, len(SEEDS), figsize=(4.6 * len(SEEDS), 3.4), squeeze=False)
for ax, s in zip(axs[0], SEEDS):
    h = HIST[s]
    for c, col in (("train CE", "#1f77b4"), ("inner CE (early stop)", "#ff7f0e"), ("valid CE (shown only)", "#2ca02c")):
        ax.plot(h.epoch, h[c], marker="o", ms=3, label=c, color=col)
    ax.axvline(h.loc[h["inner CE (early stop)"].idxmin(), "epoch"], color="k", ls=":", lw=1)
    ax.set_title(f"fusion seed {s}: mean 6-horizon cross-entropy"); ax.set_xlabel("epoch")
axs[0, 0].legend(fontsize=8); plt.tight_layout(); plt.show()

with torch.no_grad():                                     # what the gates attend to (a diagnostic, not a cause)
    m0 = FUSION[SEEDS[0]]; xs, mk, _, _ = to_dev(VAL[:20000])
    _, _, gates = m0(xs, mk)
gw = pd.Series(gates.float().mean(0).cpu().numpy(), index=NAMES)
fig, ax = plt.subplots(figsize=(6, 2.6)); gw.plot.bar(ax=ax, color="#d62728")
ax.set_title("mean gate weight per block, valid rows (model diagnostic)"); plt.tight_layout(); plt.show()
""")

# ---------------------------------------------------------------------------------------------------------- language
import sys  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nb_sections_nlp  # noqa: E402  (sections 8A-8C: language branch, SEC expert, vision + NLP fusion)

nb_sections_nlp.add(md, code)

# ---------------------------------------------------------------------------------------------------------- optimisation
md(r"""
## 9. Optimisation: what "reaching the global minimum" means for each model

* **Logistic regression (section 6):** certified. Strict convexity, five starts to the same point, zero gradient, positive
  definite Hessian.
* **LightGBM:** greedy tree growth has no global optimum to reach; the right question is when to stop, and the
  held-out curve (section 7) answers it.
* **Neural network:** the loss is non-convex and no certificate exists. What matters in practice is that the solutions
  found are equally good on held-out data, so the choice of minimum does not change the conclusions. The cell shows
  (a) the spread of held-out loss across seeds, (b) the loss along the straight line between two seeds' weights
  (a barrier means separate basins; equal endpoints mean equally good ones), and (c) the 2-D loss landscape around the
  E06 optimum saved by the project (a smooth bowl means the optimiser sits at its bottom).

Further tuning on the same validation years would lower the validation loss and raise the risk of overfitting it:
`docs/TRIAL_LEDGER.md` already counts up to 665 selection trials on 2017-2018. A real improvement has to show up on
data no choice has seen: the forward test (section 18) and the sealed period.
""")

code(r"""
seed_ce = {s: eval_ce(FUSION[s], to_dev(VAL)) for s in SEEDS}
print("valid cross-entropy by seed:", {s: round(v, 5) for s, v in seed_ce.items()},
      "| spread", f"{np.ptp(list(seed_ce.values())):.5f}")
if len(SEEDS) >= 2:
    a, b = FUSION[SEEDS[0]].state_dict(), FUSION[SEEDS[1]].state_dict()
    probe = GatedFusion({n: blocks[n].shape[1] for n in NAMES}).to(DEVICE)
    va_small = to_dev(VAL[:15000])
    path = []
    for t in np.linspace(0, 1, 11):
        probe.load_state_dict({k: (1 - t) * a[k] + t * b[k] if a[k].is_floating_point() else a[k] for k in a})
        path.append(eval_ce(probe, va_small))
    fig, ax = plt.subplots(figsize=(6, 3)); ax.plot(np.linspace(0, 1, 11), path, marker="o")
    ax.set_title(f"valid CE on the line from seed {SEEDS[0]} (0) to seed {SEEDS[1]} (1)"); ax.set_xlabel("t")
    plt.tight_layout(); plt.show()
    print(f"barrier height above the mean of the endpoints: {max(path) - (path[0] + path[-1]) / 2:+.5f}")
L = np.load(NB / "predictions" / "e06__landscape_2d.npy")
e06m = REPO_DIR / "experiments" / "20261106T005012_e06_optimization_fa42d8" / "metrics.json"
grid = (np.asarray(json.loads(e06m.read_text(encoding="utf-8"))["landscape"]["grid"]) if e06m.exists()
        else np.linspace(-1, 1, L.shape[0]))
fig, ax = plt.subplots(figsize=(5.2, 4.2))
cs_ = ax.contourf(grid, grid, L.T, levels=24, cmap="viridis")
ax.plot(0, 0, "r*", ms=12, label="E06 optimum"); plt.colorbar(cs_, ax=ax, label="valid log loss")
ax.set_title("E06: loss landscape around the trained network\n(two random filter-normalised directions)")
ax.legend(fontsize=8); plt.tight_layout(); plt.show()
print(f"landscape: centre {L[L.shape[0] // 2, L.shape[1] // 2]:.5f}, grid minimum {L.min():.5f}")
""")

# ---------------------------------------------------------------------------------------------------------- fine-tuning
md(r"""
## 10. Fine-tuning a pretrained vision network on chart images (GPU)

The fusion network reads **frozen** ImageNet features of the charts. Fine-tuning asks whether adapting the network to
charts helps. The cell renders daily charts with the project's canonical renderer **r1** (TriFrame's binary OHLC
bars: open tick left, high-low line centre, close tick right, with a volume panel; 126 daily bars; log price scaled to
the window, so the chart shows shape, not the volatility level), then fine-tunes an ImageNet ConvNeXt-Tiny on the 5D
label: the stem and first two stages stay frozen, the last two stages and a new 3-class head are trained. Training
rows come from the train block, early stopping uses the inner block, and the result is scored on the validation years.

TriFrame fine-tuned the same network on 500 stocks and found the image gain close to zero; E02 confirms it for NEXUS
(the chart-fine-tuned embeddings are compared in section 14). Expect the same here: the point is the honest test.
""")

code(r"""
FT = {"run": BARS_DIR.exists(), "n_train": 600 if QUICK else (40_000 if DEVICE == "cuda" else 6_000),
      "n_inner": 200 if QUICK else (6_000 if DEVICE == "cuda" else 1_500),
      "n_valid": 400 if QUICK else (12_000 if DEVICE == "cuda" else 3_000),
      "epochs": 1 if QUICK else (4 if DEVICE == "cuda" else 2), "batch": 128 if DEVICE == "cuda" else 32}
RENDER_H, PRICE_H, GAP, VOL_H = 128, 100, 2, 26

def render(o, h, l, c, v, n_bars):
    # TriFrame canonical renderer r1, copied unchanged (nexus_alpha.cv.render.render)
    m = len(o); W = 3 * n_bars
    img = np.zeros((RENDER_H, W), np.uint8)
    if m == 0:
        return img, 0
    o, h, l, c, v = (np.asarray(a, float) for a in (o, h, l, c, v))
    lo, hi = np.log(l.min()), np.log(h.max()); span = hi - lo
    if span > 0:
        f = lambda a: np.rint((np.log(a) - lo) / span * (PRICE_H - 1)).astype(int)
        yo, yh, yl, yc = f(o), f(h), f(l), f(c)
    else:
        yo = yh = yl = yc = np.full(m, PRICE_H // 2)
    x0 = 3 * (n_bars - m) + 3 * np.arange(m)
    rows = np.arange(PRICE_H)[:, None]
    price = np.zeros((PRICE_H, W), np.uint8)
    price[:, x0 + 1] = ((rows >= yl[None, :]) & (rows <= yh[None, :])).astype(np.uint8)
    price[yo, x0] = 1
    price[yc, x0 + 2] = 1
    img[:PRICE_H] = price[::-1]
    vmax = v.max()
    if vmax > 0:
        hv = np.rint(v / vmax * (VOL_H - 1)).astype(int) + (v > 0)
        vol = np.zeros((VOL_H, W), np.uint8)
        vol[:, x0 + 1] = (np.arange(VOL_H)[:, None] < hv[None, :]).astype(np.uint8)
        img[PRICE_H + GAP:] = vol[::-1]
    return img, m

def render_rows(ix, bars_cache):
    out, keep = [], []
    for i in ix:
        t, a = FR.ticker.iat[i], FR.asof_date.iat[i]
        d = bars_cache.get(t)
        if d is None:
            continue
        k = int(np.searchsorted(d.date.to_numpy(), a.to_datetime64(), side="right"))
        w = d.iloc[max(0, k - 126):k]
        if len(w) < 51 or w.date.iat[-1] != a:
            continue
        im, _ = render(w.open.to_numpy(), w.high.to_numpy(), w.low.to_numpy(), w.close.to_numpy(), w.volume.to_numpy(), 126)
        out.append(np.packbits(im, axis=1)); keep.append(i)
    return np.stack(out), np.array(keep)

if FT["run"]:
    import timm
    j5 = H6.index("5D")
    pick = lambda b, n: sample(masks[b] & (Y[:, j5] >= 0) & (FR.n_valid_D.to_numpy() >= 51), n)
    ix_tr, ix_in, ix_va = pick("train", FT["n_train"]), pick("inner", FT["n_inner"]), pick("valid", FT["n_valid"])
    need = sorted(set(FR.ticker.iloc[np.concatenate([ix_tr, ix_in, ix_va])]))
    t0 = time.time()
    bars_cache = {t: read_daily(t) for t in need if (BARS_DIR / f"{t}.parquet").exists()}
    IMG = {k: render_rows(ix, bars_cache) for k, ix in (("train", ix_tr), ("inner", ix_in), ("valid", ix_va))}
    print({k: len(v[1]) for k, v in IMG.items()}, f"charts rendered in {time.time() - t0:.0f}s")
    fig, axs = plt.subplots(1, 3, figsize=(13, 2.2))
    for ax, k in zip(axs, range(3)):
        im = np.unpackbits(IMG["train"][0][k], axis=1)[:, :378]
        ax.imshow(1 - im, cmap="gray", aspect="auto"); ax.set_axis_off()
        ax.set_title(f"{FR.ticker.iat[IMG['train'][1][k]]} {FR.asof_date.iat[IMG['train'][1][k]].date()} "
                     f"-> {CLASSES[Y[IMG['train'][1][k], j5]]} (5D)", fontsize=9)
    plt.tight_layout(); plt.show()

    MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    def batch_tensor(packed):
        x = np.unpackbits(packed, axis=2)[:, :, :384].astype(np.float32)          # 378 columns, padded to 384
        x = torch.from_numpy(1 - x)[:, None].repeat(1, 3, 1, 1)                     # white background, dark bars
        return ((x - MEAN) / STD).to(DEVICE, non_blocking=True)
    net = timm.create_model("convnext_tiny.in12k_ft_in1k", pretrained=True, num_classes=3).to(DEVICE)
    for n_, p_ in net.named_parameters():
        p_.requires_grad = not (n_.startswith("stem") or n_.startswith("stages.0") or n_.startswith("stages.1"))
    opt = torch.optim.AdamW([p_ for p_ in net.parameters() if p_.requires_grad], lr=2e-4, weight_decay=0.05)
    def amp():
        return torch.autocast("cuda", dtype=torch.bfloat16) if DEVICE == "cuda" else torch.autocast("cpu", enabled=False)
    def run_eval(split):
        net.eval(); pk, ix = IMG[split]; out = []
        with torch.no_grad(), amp():
            for i in range(0, len(ix), 256):
                out.append(net(batch_tensor(pk[i:i + 256])).float().softmax(-1).cpu().numpy())
        p = np.concatenate(out); y = Y[ix, j5]
        return log_loss(y, np.clip(p, 1e-7, 1), labels=[0, 1, 2]), p
    FT_HIST, best, best_state = [], 1e9, None
    pk_tr, ix_tr2 = IMG["train"]; ytr_t = torch.from_numpy(Y[ix_tr2, j5])
    for ep in range(FT["epochs"]):
        net.train(); perm = np.random.default_rng(ep).permutation(len(ix_tr2)); run, nb = 0.0, 0
        for i in range(0, len(perm), FT["batch"]):
            b = perm[i:i + FT["batch"]]
            with amp():
                loss = F.cross_entropy(net(batch_tensor(pk_tr[b])).float(), ytr_t[b].to(DEVICE))
            opt.zero_grad(); loss.backward(); opt.step(); run += loss.item(); nb += 1
        ll_in, _ = run_eval("inner"); ll_va, _ = run_eval("valid")
        FT_HIST.append({"epoch": ep + 1, "train CE": run / nb, "inner CE (early stop)": ll_in, "valid CE (shown only)": ll_va})
        print(FT_HIST[-1], f"({time.time() - t0:.0f}s)")
        if ll_in < best:
            best, best_state = ll_in, {k: v.detach().clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)
    ll_va, p_va = run_eval("valid")
    y_va = Y[IMG["valid"][1], j5]
    prior5 = np.bincount(Y[TR, j5][Y[TR, j5] >= 0], minlength=3) / (Y[TR, j5] >= 0).sum()
    ll_prior = log_loss(y_va, np.tile(prior5, (len(y_va), 1)), labels=[0, 1, 2])
    FT_RESULT = {"valid rows": len(y_va), "fine-tuned CNN log loss": ll_va, "prior log loss": ll_prior,
                 "gain vs prior (positive = better)": ll_prior - ll_va,
                 "up-vs-down AUROC": roc_auc_score(y_va[y_va != 1] == 2, (p_va[:, 2] / (p_va[:, 0] + p_va[:, 2]))[y_va != 1])}
    h = pd.DataFrame(FT_HIST)
    fig, ax = plt.subplots(figsize=(6, 3.2))
    for c, col in (("train CE", "#1f77b4"), ("inner CE (early stop)", "#ff7f0e"), ("valid CE (shown only)", "#2ca02c")):
        ax.plot(h.epoch, h[c], marker="o", label=c, color=col)
    ax.axhline(ll_prior, color="k", ls="--", lw=1, label="prior (valid)")
    ax.set_title("ConvNeXt-Tiny fine-tuned on daily charts, 5D label"); ax.legend(fontsize=8); plt.tight_layout(); plt.show()
    display(pd.Series(FT_RESULT).to_frame("value"))
else:
    print("Daily bars not found: the fine-tuning section is skipped.")
""")

# ---------------------------------------------------------------------------------------------------------- calibration
md(r"""
## 11. Calibration

A probability is useful only if 30% means 30%. Each horizon gets one temperature T (logits divided by T) fitted on the
**calibration** block (2019) by minimising its log loss, then applied unchanged to validation and test. The reliability
diagram plots the predicted probability of the top class against how often it was right; ECE is the average gap,
weighted by the number of rows in each bin.
""")

code(r"""
POS = {i: k for k, i in enumerate(EVI)}                      # row index -> position in EVAL arrays
def rows_of(block):
    return np.array([POS[i] for i in np.flatnonzero(masks[block])])
P_CAL, P_VAL, P_TEST = rows_of("calib"), rows_of("valid"), rows_of("test")
YE = Y[EVI]

def fit_temperature(p, y):
    lg = np.log(np.clip(p, 1e-7, 1))
    f = lambda T: log_loss(y, np.exp(lg / T) / np.exp(lg / T).sum(1, keepdims=True), labels=[0, 1, 2])
    return optimize.minimize_scalar(f, bounds=(0.3, 5.0), method="bounded").x

def apply_temperature(p, T):
    lg = np.log(np.clip(p, 1e-7, 1)) / T
    e = np.exp(lg - lg.max(-1, keepdims=True)); return e / e.sum(-1, keepdims=True)

def ece(p, y, bins=15):
    conf, pred = p.max(1), p.argmax(1)
    edges = np.linspace(0, 1, bins + 1); tot = 0.0
    for a, b in zip(edges[:-1], edges[1:]):
        s = (conf > a) & (conf <= b)
        if s.any():
            tot += s.mean() * abs((pred[s] == y[s]).mean() - conf[s].mean())
    return tot

cal = PRED["fusion"].copy(); TEMPS = {}
for j, h in enumerate(H6):
    ok = YE[P_CAL, j] >= 0
    TEMPS[h] = fit_temperature(PRED["fusion"][P_CAL[ok], j], YE[P_CAL[ok], j])
    cal[:, j] = apply_temperature(PRED["fusion"][:, j], TEMPS[h])
PRED["fusion_cal"] = cal
print("temperatures (above 1 = the raw network was over-confident):", {h: round(float(t), 3) for h, t in TEMPS.items()})

fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))
for ax, h in zip(axs, ["1D", "5D", "20D"]):
    j = H6.index(h); ok = YE[P_TEST, j] >= 0; y = YE[P_TEST[ok], j]
    for name, col in (("fusion", "#d62728"), ("fusion_cal", "#1f77b4"), ("prior", "#9aa5b1")):
        p = PRED[name][P_TEST[ok], j]; conf, hit = p.max(1), (p.argmax(1) == y)
        q = pd.qcut(conf, 10, duplicates="drop")
        g = pd.DataFrame({"c": conf, "h": hit}).groupby(q, observed=True).mean()
        ax.plot(g.c, g.h, marker="o", label=f"{name} (ECE {ece(p, y):.3f})", color=col)
    lim = [0.3, 0.7]; ax.plot(lim, lim, "k--", lw=.8); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_title(f"reliability on test, {h}"); ax.set_xlabel("predicted top-class probability"); ax.set_ylabel("observed")
    ax.legend(fontsize=7)
plt.tight_layout(); plt.show()
""")

# ---------------------------------------------------------------------------------------------------------- evaluation
md(r"""
## 12. Evaluation: every metric, every horizon

Models trained in this notebook (prior, logistic, LightGBM, fusion with and without calibration) and the frozen models'
saved predictions on the same rows:
* **E08 = v1.1**, the frozen production model (a blend: 60% fusion over five seeds, 30% logistic, 10% LightGBM). It was
  refitted on 2003-2018 before freezing, so **its validation rows are in-sample**: judge it on test.
* **E03 gated fusion**, the documented run of section 8's architecture (fitted on 2003-2015).

Metrics: **log loss** (the primary endpoint; lower is better), **Brier score**, **accuracy**, **balanced accuracy**,
**macro-F1**, **AUROC** (one-vs-rest, macro), **up-vs-down AUROC** (direction skill among moves), **sideways AUROC**
(volatility-regime skill), **ECE**, and **rank IC** (per-date Spearman correlation of P(up) - P(down) with the
realised return, averaged over dates).

**Reading the confusion matrices.** They count the most likely class. SIDEWAYS is the largest class, and the
probabilities stay close to the class shares, so the most likely class is nearly always SIDEWAYS at short horizons.
A matrix like that says the forecasts are cautious, not that they are useless. The information is in how far the
probabilities move from the prior, which is what log loss, AUROC and rank IC measure.
""")

code(r"""
ROWS = pd.read_parquet(NB / "predictions" / "rows_eval.parquet")
assert (ROWS.sample_id.to_numpy() == FR.sample_id.to_numpy()[EVI]).all(), "saved predictions are not aligned"
def saved(name):
    return np.load(NB / "predictions" / f"{name}.npy").astype(np.float64)
for k, f in (("v1.1 (E08 frozen)", "e08__final_deployed_calibrated"), ("E03 gated (saved)", "e03__variant_gated"),
             ("E01 LightGBM (saved)", "e01__lgbm_struct_x33"), ("E06 seed ensemble (saved)", "e06__best_seed_ensemble")):
    if (NB / "predictions" / f"{f}.npy").exists():
        PRED[k] = saved(f)

def rank_ic(score, ret, dates, min_n=10):
    # per-date Spearman correlation, vectorised: Pearson correlation of within-date ranks
    df = pd.DataFrame({"s": score, "r": ret, "d": dates}).dropna()
    g = df.groupby("d")
    a = g.s.rank() - g.s.rank().groupby(df.d).transform("mean")
    b = g.r.rank() - g.r.rank().groupby(df.d).transform("mean")
    num = (a * b).groupby(df.d).sum()
    den = np.sqrt((a ** 2).groupby(df.d).sum() * (b ** 2).groupby(df.d).sum())
    ics = (num / den)[g.size() >= min_n].replace([np.inf, -np.inf], np.nan).dropna()
    return float(ics.mean()), ics

def metrics(p, y, ret=None, dates=None):
    ok = y >= 0; p, y = np.clip(p[ok], 1e-7, 1), y[ok]; p = p / p.sum(1, keepdims=True)
    yo = np.eye(3)[y]
    out = {"n": int(ok.sum()), "log loss": log_loss(y, p, labels=[0, 1, 2]), "Brier": float(((p - yo) ** 2).sum(1).mean()),
           "accuracy": accuracy_score(y, p.argmax(1)), "balanced accuracy": balanced_accuracy_score(y, p.argmax(1)),
           "macro-F1": f1_score(y, p.argmax(1), average="macro"),
           "AUROC (macro OvR)": roc_auc_score(yo, p, average="macro", multi_class="ovr"), "ECE": ece(p, y)}
    mv = y != 1
    out["up-vs-down AUROC"] = roc_auc_score(y[mv] == 2, (p[mv, 2] / (p[mv, 0] + p[mv, 2])))
    out["sideways AUROC"] = roc_auc_score(y == 1, p[:, 1])
    if ret is not None:
        out["rank IC"] = rank_ic(p[:, 2] - p[:, 0], ret[ok], dates[ok])[0]
    return out

DATES = FR.asof_date.to_numpy()[EVI]; RETE = RET[EVI]
TABLE = []
for name, P in PRED.items():
    for split, ix in (("valid", P_VAL), ("test", P_TEST)):
        for j, h in enumerate(H6):
            TABLE.append({"model": name, "split": split, "horizon": h,
                          **metrics(P[ix, j], YE[ix, j], RETE[ix, j], DATES[ix])})
TABLE = pd.DataFrame(TABLE)
LABEL = {"prior": "class prior", "logistic": "logistic (trained here)", "lgbm": "LightGBM (trained here)",
         "fusion": "fusion network (trained here)", "fusion_cal": "fusion network, calibrated (trained here)",
         "fusion_mm": "vision + NLP fusion network (trained here)"}
mean_ll = TABLE.pivot_table(index="model", columns="split", values="log loss", aggfunc="mean").sort_values("test")
mean_ll.index = [LABEL.get(i, i) for i in mean_ll.index]
display(Markdown("**Mean log loss over the six horizons (lower is better)**")); display(mean_ll.style.format("{:.5f}"))
show = TABLE[TABLE.split == "test"].set_index(["model", "horizon"]).drop(columns=["split"])
display(Markdown("**All metrics on test (2020-2022)**")); display(show.style.format(precision=4))
""")

code(r"""
# confusion matrices of the frozen v1.1 model on test, by horizon (rows: true class; columns: predicted; row-normalised)
name = "v1.1 (E08 frozen)" if "v1.1 (E08 frozen)" in PRED else "fusion_cal"
fig, axs = plt.subplots(1, 6, figsize=(18, 3.3))
for ax, (j, h) in zip(axs, enumerate(H6)):
    ok = YE[P_TEST, j] >= 0
    cm = confusion_matrix(YE[P_TEST[ok], j], PRED[name][P_TEST[ok], j].argmax(1), labels=[0, 1, 2], normalize="true")
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    for a in range(3):
        for b in range(3):
            ax.text(b, a, f"{cm[a, b]:.2f}", ha="center", va="center", color="white" if cm[a, b] > .5 else "black", fontsize=9)
    ax.set_xticks(range(3), ["D", "S", "U"]); ax.set_yticks(range(3), ["D", "S", "U"]); ax.set_title(h); ax.grid(False)
fig.suptitle(f"{name}: confusion matrices on test (row-normalised)", y=1.03); plt.tight_layout(); plt.show()

# ROC curves: direction (up vs down) and volatility regime (sideways vs moves), 5D, test
j = H6.index("5D"); ok = YE[P_TEST, j] >= 0; y = YE[P_TEST[ok], j]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
for nm, col in (("logistic", CLR["logistic"]), ("lgbm", CLR["lgbm"]), ("fusion_cal", CLR["fusion"]), (name, CLR["v1.1"])):
    if nm not in PRED:
        continue
    p = PRED[nm][P_TEST[ok], j]; mv = y != 1
    fpr, tpr, _ = roc_curve(y[mv] == 2, p[mv, 2] / (p[mv, 0] + p[mv, 2]))
    a1.plot(fpr, tpr, color=col, label=f"{nm} {roc_auc_score(y[mv] == 2, p[mv, 2] / (p[mv, 0] + p[mv, 2])):.3f}")
    fpr, tpr, _ = roc_curve(y == 1, p[:, 1])
    a2.plot(fpr, tpr, color=col, label=f"{nm} {roc_auc_score(y == 1, p[:, 1]):.3f}")
for ax, t in ((a1, "5D direction: up vs down (moves only)"), (a2, "5D volatility regime: sideways vs moves")):
    ax.plot([0, 1], [0, 1], "k--", lw=.8); ax.set_title(t); ax.legend(fontsize=8, title="AUROC"); ax.set_xlabel("false positive rate")
plt.tight_layout(); plt.show()
""")

# ---------------------------------------------------------------------------------------------------------- statistics
md(r"""
## 13. Statistics: is any gain real?

* **Month-block bootstrap.** The log-loss gain of a model over the prior, resampling whole calendar months (rows in
  the same month share market moves, so resampling single rows would overstate certainty). 95% intervals and
  one-sided p-values.
* **Holm correction** across the six horizons (one family).
* **Placebo.** The model's forecasts shuffled across stocks within each date: market-level information survives, the
  stock-specific link does not. A real stock-level signal must beat its own placebo.
* **Rank IC** with a month-block interval: the cross-sectional ranking, the part of the signal the project validated.
""")

code(r"""
def month_block_boot(delta, dates, B=1000, seed=SEED):
    months = pd.DatetimeIndex(dates).to_period("M")
    g = pd.DataFrame({"d": delta, "m": months}).groupby("m").d.agg(["sum", "count"])
    s, c = g["sum"].to_numpy(), g["count"].to_numpy()
    r = np.random.default_rng(seed); k = len(g)
    bs = np.array([(s[ix].sum() / c[ix].sum()) for ix in (r.integers(0, k, k) for _ in range(B))])
    return float(delta.mean()), np.percentile(bs, [2.5, 97.5]), float((bs <= 0).mean())

def holm(p):
    p = np.asarray(p); o = np.argsort(p); adj = np.empty_like(p, dtype=float); run = 0.0
    for r, i in enumerate(o):
        run = max(run, (len(p) - r) * p[i]); adj[i] = min(1.0, run)
    return adj

def row_ll(P, y):
    return -np.log(np.clip(P[np.arange(len(y)), y], 1e-7, 1))

def placebo(P, dates, seed=SEED):
    # shuffle forecasts across stocks within each date
    r = np.random.default_rng(seed); out = P.copy()
    for d, ix in pd.Series(np.arange(len(dates))).groupby(dates).groups.items():
        ix = np.asarray(ix); out[ix] = P[r.permutation(ix)]
    return out

name = "v1.1 (E08 frozen)" if "v1.1 (E08 frozen)" in PRED else "fusion_cal"
rows = []
for j, h in enumerate(H6):
    ok = YE[P_TEST, j] >= 0; ix = P_TEST[ok]; y = YE[ix, j]; d = DATES[ix]
    base = row_ll(PRED["prior"][ix, j], y)
    real = base - row_ll(PRED[name][ix, j], y)
    pl = base - row_ll(placebo(PRED[name][ix, j], d), y)
    est, ci, p = month_block_boot(real, d)
    pest, pci, _ = month_block_boot(pl, d)
    ic, ics = rank_ic(PRED[name][ix, j, 2] - PRED[name][ix, j, 0], RETE[ix, j], d)
    icm = ics.groupby(pd.DatetimeIndex(ics.index).to_period("M")).mean()
    bs = [np.random.default_rng(b).choice(icm.to_numpy(), len(icm)).mean() for b in range(1000)]
    rows.append({"horizon": h, "LL gain vs prior": est, "95% CI": f"[{ci[0]:+.4f}, {ci[1]:+.4f}]", "p (one-sided)": p,
                 "placebo gain": pest, "real minus placebo": est - pest,
                 "rank IC": ic, "rank IC 95% CI": f"[{np.percentile(bs, 2.5):+.4f}, {np.percentile(bs, 97.5):+.4f}]"})
ST = pd.DataFrame(rows).set_index("horizon")
ST["p Holm"] = holm(ST["p (one-sided)"].to_numpy())
display(Markdown(f"**{name} on test (2020-2022): gain over the class prior, placebo and rank IC**"))
display(ST.style.format({"LL gain vs prior": "{:+.5f}", "placebo gain": "{:+.5f}", "real minus placebo": "{:+.5f}",
                         "p (one-sided)": "{:.3f}", "p Holm": "{:.3f}", "rank IC": "{:+.4f}"}))
""")

# ---------------------------------------------------------------------------------------------------------- comparison
md(r"""
## 14. Comparison: every model side by side, and the reproduction check

The first chart compares the mean log-loss gain over the prior on test for every model in this notebook and every saved
documented run. The second compares chart encoders from E02 (a linear probe on each frozen backbone's embeddings,
weekly + daily + hourly), including the random-weights control and TriFrame's chart-fine-tuned ConvNeXt (which may be
scored only from 2017, so only validation and test are shown). The last table recomputes the documented E03 numbers
from the saved predictions and compares them with the committed `metrics.json`: they must agree.
""")

code(r"""
for k, f in (("E02 probe: random ResNet18 (control)", "e02__img_lr__resnet18_random_s0__WDH"),
             ("E02 probe: ResNet18", "e02__img_lr__resnet18__WDH"),
             ("E02 probe: ConvNeXt-Tiny", "e02__img_lr__convnext_tiny__WDH"),
             ("E02 probe: DINOv2 ViT-S", "e02__img_lr__vit_small_patch14_dinov2__WDH"),
             ("E02 probe: ConvNeXt-Tiny fine-tuned on charts", "e02__img_lr__ft_convnext_tiny__WDH")):
    if (NB / "predictions" / f"{f}.npy").exists():
        PRED[k] = saved(f)
gain = {}
for name, P in PRED.items():
    g = []
    for j in range(6):
        ok = YE[P_TEST, j] >= 0; ix = P_TEST[ok]
        if np.isnan(P[ix, j]).any():
            g = None; break
        g.append(log_loss(YE[ix, j], PRED["prior"][ix, j], labels=[0, 1, 2]) -
                 log_loss(YE[ix, j], np.clip(P[ix, j], 1e-7, 1), labels=[0, 1, 2]))
    if g is not None:
        gain[name] = np.mean(g)
gain = pd.Series(gain).sort_values()
gain.index = [LABEL.get(i, i) for i in gain.index]
fig, ax = plt.subplots(figsize=(9, 0.42 * len(gain) + 1))
ax.barh(gain.index, gain.to_numpy(), color=["#2ca02c" if v > 0 else "#d62728" for v in gain])
ax.axvline(0, color="k", lw=.8); ax.set_title("mean log-loss gain over the class prior on test (positive = better)")
plt.tight_layout(); plt.show()

repro = []
e03 = REPO_DIR / "experiments" / "20261106T000935_e03_fusion_8ddffb" / "metrics.json"
if e03.exists():
    M = json.loads(e03.read_text(encoding="utf-8"))
    for nm in ("variant_gated", "variant_late", "variant_attention", "a05_num", "a04_img_WDH", "selected_calibrated"):
        f = NB / "predictions" / f"e03__{nm}.npy"
        if not f.exists() or nm not in M:
            continue
        P = saved(f"e03__{nm}")
        for split, ix, key in (("valid", P_VAL, "valid"), ("test", P_TEST, "test_raw" if "test_raw" in M[nm] else "test")):
            if key not in M[nm]:
                continue
            ours = np.mean([log_loss(YE[ix[YE[ix, j] >= 0], j], np.clip(P[ix[YE[ix, j] >= 0], j], 1e-7, 1),
                                     labels=[0, 1, 2]) for j in range(6)])
            repro.append({"run": nm, "split": split, "recomputed here": ours,
                          "committed metrics.json": M[nm][key]["mean_log_loss"]})
    R = pd.DataFrame(repro)
    R["abs diff"] = (R["recomputed here"] - R["committed metrics.json"]).abs()
    display(R.style.format({"recomputed here": "{:.5f}", "committed metrics.json": "{:.5f}", "abs diff": "{:.1e}"}))
    print("all within 2e-4 (float16 storage of the saved probabilities):", bool((R["abs diff"] < 2e-4).all()))
""")

# ---------------------------------------------------------------------------------------------------------- risk
md(r"""
## 15. The risk engine: entry, stop, targets and size, with the reasons

Forecasting and trading are separate. A high P(UP) is never by itself a LONG. The engine applies its gates in order,
and the first failure decides (every reason is reported):

1. data sufficiency (exact levels need structured OHLCV; screenshots alone never give numbers);
2. a calibrated forecast for the horizon;
3. **validated direction skill** at the horizon (the validation up-vs-down AUROC interval must exclude 0.5);
4. an edge |P(UP) - P(DOWN)| of at least 0.10;
5. the weekly regime must permit the side (rulebook E011-E018);
6. a daily setup on the same side supplies the structural invalidation and the first obstacle (E023/E027/E031...);
7. an hourly trigger on the as-of session (E121/E122), else WAIT;
8. stop = invalidation -/+ 0.25 ATR (X001, X011), on the losing side of the entry;
9. target 1 = the first opposing obstacle (X021); net reward-to-risk after costs must be at least 1.5 (E039, E149);
10. size = floor(capital x risk % / (risk per share + costs)), capped by the maximum position (X010);
11. a time exit after the horizon (X071); a maximum chase of 1 ATR beyond the reference (E111).

The engine never widens a stop to fit a size, never invents a level, and never reads prices from chart pixels. The
cell re-implements it and runs four cases:
* an approved long;
* the same setup at 1D, where v1.1's direction skill is not validated (gate 3);
* the same setup at 5D with v1.1's typical edge: on the latest logged session, the median |P(UP) - P(DOWN)| was 0.04
  at 5D, and 2.6% of stocks reached the 0.10 minimum;
* a short when shorting is disabled.

It then recomputes gate 3 from validation predictions and puts the frozen v1.1 flags next to it. On test (section 12)
the direction AUROC is 0.49-0.51 at every horizon: the small validation skill did not replicate.
""")

code(r"""
COST_BPS = {"base": 10, "stressed": 15, "severe": 20}            # round trip, plus 0.5% a year short borrow

def plan_trade(p_up, p_down, skill_ok, weekly_perm, setup, trigger, atr, close, sessions, capital=100_000,
               risk_pct=0.005, max_pos=0.10, allow_short=True, min_edge=0.10, min_rr=1.5, cost="base"):
    reasons = []
    edge = p_up - p_down; side = "LONG" if edge > 0 else "SHORT"
    if not skill_ok:
        reasons.append("no validated directional skill at this horizon")
    if abs(edge) < min_edge:
        reasons.append(f"edge {abs(edge):.3f} below {min_edge:.2f}")
    if side == "SHORT" and not allow_short:
        reasons.append("short selling disabled")
    ok_perm = {"LONG": {"LONG_PREFERRED", "EDGES_ONLY", "REDUCED_CONFIRMATION_REQUIRED"},
               "SHORT": {"SHORT_PREFERRED", "EDGES_ONLY", "REDUCED_CONFIRMATION_REQUIRED"}}[side]
    if weekly_perm not in ok_perm:
        reasons.append(f"weekly permission {weekly_perm} does not allow {side}")
    if setup is None or setup["direction"] != side:
        reasons.append(f"no daily {side.lower()} setup")
    if not trigger:
        reasons.append("no hourly trigger (WAIT)")
    if reasons:
        return {"decision": "NO_TRADE", "reasons": reasons}
    stop = setup["invalidation"] - 0.25 * atr if side == "LONG" else setup["invalidation"] + 0.25 * atr
    t1 = setup["first_obstacle"]
    if not ((stop < close < t1) if side == "LONG" else (t1 < close < stop)):
        return {"decision": "NO_TRADE", "reasons": ["stop/target geometry invalid"]}
    c_ps = close * COST_BPS[cost] / 1e4 + (close * 0.005 * sessions / 252 if side == "SHORT" else 0)
    risk_ps, reward_ps = abs(close - stop), abs(t1 - close)
    net_rr = (reward_ps - c_ps) / (risk_ps + c_ps)
    if net_rr < min_rr:
        return {"decision": "NO_TRADE", "reasons": [f"net reward-to-risk {net_rr:.2f} below {min_rr}"]}
    qty = min(math.floor(capital * risk_pct / (risk_ps + c_ps)), math.floor(capital * max_pos / close))
    if qty < 1:
        return {"decision": "NO_TRADE", "reasons": ["stop too wide for the risk budget (X010)"]}
    return {"decision": side, "entry (next open, if within chase)": close,
            "max chase": close + atr if side == "LONG" else close - atr, "stop": stop, "target 1": t1,
            "target 2": close + 2 * (t1 - close), "net R:R": net_rr, "quantity": qty, "position value": qty * close,
            "loss at stop": qty * (risk_ps + c_ps), "time exit (sessions)": sessions,
            "reasons": [f"setup {setup['family']} {setup['rule_ids']}: invalidation {setup['invalidation']:.2f}",
                        f"stop = invalidation - 0.25 ATR ({atr:.2f}) (X001, X011)", "T1 = first obstacle (X021)",
                        f"size: risk {risk_pct:.1%} of capital over (risk per share + costs) (X010)"]}

setup = {"family": "TREND_PULLBACK_LONG", "direction": "LONG", "rule_ids": ["E023", "E074", "E083"],
         "invalidation": 171.20, "first_obstacle": 189.50}
cases = {
    "approved long (skill validated, edge 0.16)": plan_trade(.42, .26, True, "LONG_PREFERRED", setup, True, 3.10, 176.40, 5),
    "same setup at 1D (direction skill not validated)": plan_trade(.42, .26, False, "LONG_PREFERRED", setup, True, 3.10, 176.40, 1),
    "same setup at 5D with v1.1's typical edge (0.04)": plan_trade(.31, .27, True, "LONG_PREFERRED", setup, True, 3.10, 176.40, 5),
    "short with shorting disabled": plan_trade(.25, .45, True, "SHORT_PREFERRED", None, True, 3.10, 176.40, 5, allow_short=False),
}
for k, v in cases.items():
    display(Markdown(f"**{k}**")); display(pd.Series({a: b for a, b in v.items()}, dtype=object).to_frame("value"))

skill = {}
SK = "E03 gated (saved)" if "E03 gated (saved)" in PRED else "fusion"
for j, h in enumerate(H6):                                 # the gate-3 check, recomputed from validation predictions
    ok = (YE[P_VAL, j] >= 0) & (YE[P_VAL, j] != 1); ix = P_VAL[ok]; y = YE[ix, j] == 2
    p = PRED[SK][ix, j]
    s = p[:, 2] / (p[:, 0] + p[:, 2]); d = DATES[ix]
    m = pd.DatetimeIndex(d).to_period("M"); um = np.unique(m)
    bs = []
    for b in range(300):
        pick_m = np.random.default_rng(b).choice(um, len(um))
        idx = np.concatenate([np.flatnonzero(m == q) for q in pick_m])
        bs.append(roc_auc_score(y[idx], s[idx]))
    lo_, hi_ = np.percentile(bs, [2.5, 97.5])
    skill[h] = {"valid up-vs-down AUROC (recomputed)": roc_auc_score(y, s), "95% CI": f"[{lo_:.3f}, {hi_:.3f}]",
                "validated (recomputed)": lo_ > 0.5}
SKT = pd.DataFrame(skill).T
meta_f = WS / "nexus_artifacts" / "bundles" / "20261106T020411_e08_final_model_5c4327" / "bundle_meta.json"
if meta_f.exists():
    ds = json.loads(meta_f.read_text(encoding="utf-8"))["direction_skill"]
    ds = ds if isinstance(ds, dict) else __import__("ast").literal_eval(ds)
    SKT["frozen v1.1: AUROC [95% CI]"] = [f"{ds[h]['valid_up_vs_down_auroc']:.3f} [{ds[h]['ci95'][0]:.3f}, "
                                          f"{ds[h]['ci95'][1]:.3f}]" for h in SKT.index]
    SKT["frozen v1.1: validated (the flag the engine uses)"] = [ds[h]["validated"] for h in SKT.index]
test_auc = TABLE[(TABLE.model == "v1.1 (E08 frozen)") & (TABLE.split == "test")].set_index("horizon")["up-vs-down AUROC"]
SKT["v1.1 test up-vs-down AUROC"] = test_auc.reindex(SKT.index).round(3)
display(Markdown(f"**Gate 3, direction skill: recomputed on validation ({SK}, fitted 2003-2015, month-block "
                 "bootstrap), the frozen v1.1 flags, and what happened on test**"))
display(SKT)
""")

# ---------------------------------------------------------------------------------------------------------- backtest
md(r"""
## 16. Backtest (the documented run)

The event-driven backtest (`nexus_alpha.trading.backtest`) enters at the next open; stops that gap through fill at the
open; when a bar touches both the stop and the target, the stop is assumed first; costs are 10/15/20 bps round trip
plus 0.5% a year short borrow. Thresholds were chosen on validation and frozen; the test years are shown here. The
benchmark is equal-weight buy-and-hold of the same survivor universe, so it carries the same survivorship bias.
""")

code(r"""
bt = REPO_DIR / "reports" / "backtest" / "e04_backtest_v1.1.json"
eq = REPO_DIR / "reports" / "backtest" / "e04_daily_returns_v1.1.csv"
if bt.exists() and eq.exists():
    b = json.loads(bt.read_text(encoding="utf-8"))
    d = pd.read_csv(eq, index_col=0, parse_dates=True)
    curve = (1 + d).cumprod()
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1.2]})
    for c in curve.columns:
        bench = "benchmark" in c
        a1.plot(curve.index, curve[c], lw=2.4 if bench else 1.2, ls="--" if bench else "-", color="k" if bench else None, label=c)
        a2.plot(curve.index, curve[c] / curve[c].cummax() - 1, lw=1, color="k" if bench else None)
    a1.set_title("growth of 1, test 2020-2022, base costs"); a1.legend(fontsize=7, ncol=2); a2.set_ylabel("drawdown")
    plt.tight_layout(); plt.show()
    t = pd.DataFrame(b["test_final"]).T
    cols = [c for c in ["trades", "sharpe", "sharpe_ci95_month_blocks", "cagr", "max_drawdown", "win_rate",
                        "profit_factor", "expectancy_R", "psr_vs_0"] if c in t]
    display(t[cols])
    print("benchmark (equal-weight survivors), test:", b["benchmark_ew_survivors"]["test"], "| PBO:", b.get("pbo"))
else:
    print("Backtest reports not found in the repository clone.")
""")

# ---------------------------------------------------------------------------------------------------------- RAG
md(r"""
## 17. The knowledge base (RAG)

The books on technical analysis and the two rulebooks (150 entry rules, 120 exit rules) are split into chunks with
their page or rule id. Three retrievers: **BM25** (keyword, k1 = 1.5, b = 0.75, implemented below), **dense** (BGE-small
sentence embeddings, cosine similarity) and **hybrid** (reciprocal-rank fusion of the two). RAG explains; it never
changes a probability (tested in the repository: identical outputs with RAG on or off). Snippets are cut at 160
characters because the books are copyrighted.
""")

code(r"""
import re
from collections import Counter, defaultdict
CH = pd.read_parquet(NB / "rag" / "chunks.parquet")
DV = np.load(NB / "rag" / "dense_bge_small.npy")
TOKEN = re.compile(r"[a-z0-9]+")
STOP = set("a an the of and or to in on for with by is are was were be been it its this that these those as at from into "
           "than then so such can may not no but if when which who what how why where there their they them".split())
tok = lambda t: [w for w in TOKEN.findall((t or "").lower()) if w not in STOP and len(w) > 1]

class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(tok(d)) for d in docs]
        self.len = np.array([sum(t.values()) for t in self.tf], float); self.avg = self.len.mean()
        df = Counter(w for t in self.tf for w in t)
        n = len(docs); self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}
        self.post = defaultdict(list)
        for i, t in enumerate(self.tf):
            for w in t:
                self.post[w].append(i)
    def scores(self, q):
        s = np.zeros(len(self.tf))
        for w in set(tok(q)):
            for i in self.post.get(w, []):
                f = self.tf[i][w]
                s[i] += self.idf[w] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
        return s

bm = BM25(CH.text.tolist())
dense_ok = False
try:
    from transformers import AutoModel, AutoTokenizer
    TK = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5"); EM = AutoModel.from_pretrained("BAAI/bge-small-en-v1.5").eval()
    def qvec(q):
        with torch.no_grad():
            e = TK(["Represent this sentence for searching relevant passages: " + q], return_tensors="pt", truncation=True)
            return F.normalize(EM(**e).last_hidden_state[:, 0], dim=-1).numpy()[0]
    dense_ok = True
except Exception as e:
    print("dense retriever unavailable:", type(e).__name__)

def search(q, k=5, method="hybrid"):
    b_ = bm.scores(q)
    if method == "bm25" or not dense_ok:
        sc = b_
    else:
        d_ = DV @ qvec(q)
        if method == "dense":
            sc = d_
        else:
            sc = np.zeros(len(CH))
            for ranks in (np.argsort(-b_), np.argsort(-d_)):
                sc[ranks] += 1 / (60 + np.arange(1, len(ranks) + 1))
    top = np.argsort(-sc)[:k]
    r = CH.iloc[top][["doc_id", "page", "rule_id", "section"]].copy()
    r["score"] = sc[top]; r["snippet"] = CH.text.iloc[top].str.slice(0, 160).str.replace("\n", " ") + "..."
    return r

for q in ["bullish engulfing at support: where to place the stop", "ATR based trailing stop for a breakout",
          "weekly trend permission before a daily long setup"]:
    display(Markdown(f"**Query:** {q}")); display(search(q, 4))
ev = REPO_DIR / "reports" / "rag" / "rag_eval.json"
if ev.exists():
    E = json.loads(ev.read_text(encoding="utf-8"))
    display(Markdown("**Documented retrieval evaluation** (35 book questions with gold pages; rule lookups)"))
    display(pd.DataFrame(E["book"]).T[["recall@1", "recall@5", "recall@10", "mrr", "ndcg@10"]].style.format("{:.3f}"))
""")

# ---------------------------------------------------------------------------------------------------------- forward
md(r"""
## 18. The forward test: verified from the public repository

Every night the frozen models forecast all ~500 stocks for the session that just closed, before the decision time
(09:00 New York on the next session). Each forecast file is SHA-256 hashed into a chain (each row includes the hash of
the previous row), and the chain is pushed to GitHub, which fixes its time publicly. Nothing can be edited later
without breaking the chain. The cell recomputes every chain from the clone and lists what has been logged.
""")

code(r"""
FIELDS = ["run_utc", "asof_session", "file", "file_sha256", "n_rows", "n_failed", "late_rows"]
def verify_chain(version):
    # row_hash = sha256(previous row_hash + "|" + the row's seven fields joined by "|"); the first row links to GENESIS
    ch = pd.read_csv(REPO_DIR / "shadow" / version / "chain.csv", dtype=str, keep_default_na=False)
    problems, prev = [], "GENESIS"
    for _, r in ch.iterrows():
        f = REPO_DIR / "shadow" / version / r.file           # r.file already starts with "forecasts/"
        if not f.exists():
            problems.append(f"{r.file}: missing")
        elif sha256(f) != r.file_sha256:
            problems.append(f"{r.file}: file hash differs")
        expect = hashlib.sha256((prev + "|" + "|".join(str(r[c]) for c in FIELDS)).encode()).hexdigest()
        if r.prev_hash != prev or r.row_hash != expect:
            problems.append(f"{r.file}: chain link differs")
        prev = r.row_hash
    return ch, problems

if (REPO_DIR / "shadow").exists():
    out = []
    for v in ["v1.1", "v2", "market_e19", "v3_fund", "v3_alpha"]:
        if not (REPO_DIR / "shadow" / v / "chain.csv").exists():
            continue
        ch, probs = verify_chain(v)
        out.append({"model": v, "sessions logged": ch.asof_session.nunique(), "first": ch.asof_session.min(),
                    "latest": ch.asof_session.max(), "late rows": int(ch.late_rows.astype(int).sum()),
                    "chain": "intact" if not probs else "; ".join(probs[:2])})
    display(pd.DataFrame(out))
    print("The promotion rule needs about 120 sessions before any claim (docs/FORWARD_SHADOW_TEST.md).")
""")

# ---------------------------------------------------------------------------------------------------------- end
md(r"""
## 19. Conclusions

**What the evidence supports**
* The pipeline is point in time, purged and reproducible: labels, splits and indicators are re-derived above and match.
* Logistic regression reaches its certified global minimum; LightGBM and the networks stop where held-out loss stops
  improving; the seeds land in equally good minima.
* Volatility regime (sideways vs moves) is predictable: the sideways AUROC is clearly above 0.5 at every horizon
  (section 12).
* The frozen v1.1 probabilities beat the class prior on test at every horizon, significantly at the short horizons
  after Holm (section 13), and calibration holds.
* Cross-sectional ranking has a small positive rank IC, the part kept for the forward test.

**What it does not support**
* Single-stock direction at 1-20 sessions. On validation the up-vs-down AUROC was 0.51-0.54, with intervals just
  above 0.5 from 5D; on the 2020-2022 test years it was 0.49-0.51 (sections 12 and 15). The small validation skill did
  not replicate. Because the probabilities stay close to the prior, the edge gate rejects most plans, and the risk
  engine returns NO TRADE.
* Charts, news and filings add little beyond the indicators (section 14; E03's H1-H3 are not significant after Holm).
* Transformer language does not change that (sections 8A-8C, E28). FinBERT news, ChronoBERT filing vectors and the SEC
  price-impact expert beat a within-date placebo on validation (+0.00023 [+0.00003, +0.00040]) but not the current six
  blocks (-0.00029 [-0.00082, +0.00025]): NOT KEPT. The language is stock-specific but redundant with the charts and
  indicators. The SEC expert reproduces the SEC study inside NEXUS: filings predict whether a stock moves (AUC 0.656)
  but not which way (0.513).
* Fine-tuning the vision network on charts does not beat the prior (section 10), as TriFrame found.
* No backtest edge over equal-weight buy-and-hold after costs (section 16).

**What comes next**
* The forward test runs nightly; about 120 sessions are needed before any promotion (section 18).
* The sealed 2023+ period is opened once, by the owner, after the final freeze.
* `docs/LIMITATIONS.md` lists every known limit (survivorship, vendor differences, reused test years).
""")

code(r"""
print(f"notebook finished in {(time.time() - T_START) / 60:.1f} minutes on {DEVICE}"
      + (" (QUICK mode: small samples)" if QUICK else ""))
""")


def build() -> dict:
    cells = []
    for kind, src in CELLS:
        lines = src.splitlines(keepends=True)
        c = {"cell_type": kind, "metadata": {}, "source": lines}
        if kind == "code":
            c.update({"execution_count": None, "outputs": []})
        cells.append(c)
    return {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "machine_shape": "hm",
                                                                         "gpuType": "A100"},
                                         "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                         "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 5}


if __name__ == "__main__":
    nb = build()
    OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    n_code = sum(1 for k, _ in CELLS if k == "code")
    print(f"wrote {OUT} ({len(CELLS)} cells, {n_code} code)")

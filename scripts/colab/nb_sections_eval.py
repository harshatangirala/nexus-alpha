"""Notebook sections 13A, 16A and 20: which inputs predict (vision vs NLP vs both), what each modality's forecasts are
worth as trades (entry, stop-loss, exit), and a full report of the run written to Google Drive.

Imported by make_end_to_end_notebook.py. 13A comes after section 13 (it needs `metrics`, `month_block_boot`, `holm` and
section 8C's block sets), 16A after section 16 (it reads daily bars), 20 at the very end (it collects every table).

Everything here is descriptive. No configuration is selected from these tables: E28's recorded decision stands, and the
validation years have already served the trial ledger's selection trials.

Audits 9-11 (2026-11-13) hardened both sections after the first A100 run (findings A-90..A-105):
* 13A: five seeds in full mode with the seed-to-seed spread of every number, Holm-adjusted p-values, text coverage by
  year, quantile coverage with and without text, and the gain over the prior year by year.
* 16A: the frozen v1.1 model is shown on test only (its 2017-18 rows are in-sample); a volatility-matched null besides
  the random books; p-values with the (r + 1) / (N + 1) count; gross and net R; the long leg against every stock long
  at the same costs; a daily mark-to-market Sharpe ratio and drawdown; a horizon-scaled bracket as a sensitivity check.
"""
from __future__ import annotations


def add_scorecard(md, code) -> None:
    md(r"""
## 13A. Modality scorecard: vision vs NLP vs both (descriptive)

Which inputs carry the forecastable information: the charts (computer vision), the text (NLP), the indicators, or a
combination? The same gated fusion network as section 8 (same architecture, optimiser, early stopping on the inner
block) is trained on each set of blocks, so the only difference between two rows of the table is the input.

| Configuration | Blocks |
|---|---|
| vision only | weekly, daily and hourly chart embeddings |
| NLP only | news (TF-IDF, tone, timing), SEC (TF-IDF, tone, items, timing), FinBERT news, ChronoBERT filings, the SEC price-impact expert |
| indicators only | the 80 indicator features |
| vision + NLP | charts and every language block, no indicators |
| vision + indicators | charts and indicators, no language |
| indicators + NLP | indicators and language, no charts |
| all | charts, indicators and language (section 8C's MM network) |

**Seeds.** Each configuration is an average of **five seeds** in full mode (7, 11, 23, 42, 101: the seeds of E06 and
E08), so the table also shows how much each number moves from one seed to the next (audit finding A-84: the seed spread
is as large as many of the differences tested). Two configurations whose gap is smaller than their seed spread are not
distinguishable, whatever the interval says.

**Metrics, per configuration, on validation (2017-18) and test (2020-22):** log loss and its gain over the class prior,
Brier score, accuracy, balanced accuracy, macro-F1, AUROC, ECE (calibration error), up-vs-down AUROC (direction),
sideways AUROC (whether the stock moves), rank IC (cross-sectional ranking). The quantile heads are scored too: how often
the realised return lands inside the predicted 10-90% range (target 80%) and below the 10% quantile (target 10%), the
natural distance for a stop-loss.

**A fair test for text.** Language exists only where there was news or a filing before the decision time, and the news
source covers far fewer stocks in 2020-22 than in 2017-18 (the coverage table below). The scorecard is shown on all rows
and on the rows with any text. On rows without text an NLP-only network has no input at all: its output is a constant,
so its quantile coverage there says nothing about language (shown separately).

**Increments** are paired on the same rows: the per-row log-loss difference, averaged over the six horizons, with a
month-block bootstrap 95% interval (positive = the added input helps), a one-sided bootstrap p-value (the share of
resamples at or below zero), its Holm adjustment over the eight comparisons of each row set, and the range of the same
increment computed seed by seed.

**This section is descriptive.** Nothing is selected from it: E28's decision (section 8C) is the recorded one, and the
validation years have already served the trial ledger's selection trials (`docs/TRIAL_LEDGER.md`).
""")
    code(r"""
VISION_B, LANG_B, IND_B = ["img_W", "img_D", "img_H"], ["news", "sec"] + list(NEWB), ["num"]
ALLK = "all (vision + indicators + NLP)"
MOD_CFG = {"vision only": VISION_B, "NLP only": LANG_B, "indicators only": IND_B, "vision + NLP": VISION_B + LANG_B,
           "vision + indicators": VISION_B + IND_B, "indicators + NLP": IND_B + LANG_B}
SC_ORDER = ["prior"] + list(MOD_CFG) + [ALLK]
SEEDS13 = list(SEEDS) if QUICK else [7, 11, 23, 42, 101]

@torch.no_grad()
def predict_quant(model, ix, bs=8192, B=None):
    model.eval(); xs, m, _, _ = to_dev(ix, B); out_ = []
    for i in range(0, len(m), bs):
        _, qu, _ = model({k: v[i:i + bs] for k, v in xs.items()}, m[i:i + bs])
        out_.append(qu.float().cpu().numpy())
    return np.sort(np.concatenate(out_), axis=-1)            # (n, 6, 3): q10, q50, q90 of the vol-scaled return

def row_ll_all(P):
    # per-row, per-horizon log loss on every evaluation row (NaN where the row has no label)
    L = np.full((len(EVI), 6), np.nan, np.float32)
    for j in range(6):
        y = YE[:, j]; ok = y >= 0
        L[ok, j] = -np.log(np.clip(P[np.flatnonzero(ok), j, y[ok]], 1e-7, 1))
    return L

QV, SC_EPOCHS, LLS = {}, {}, {}
t0 = time.time()
for name, names_ in MOD_CFG.items():
    Bk = block_set(names_)
    ms_, hs_ = zip(*[train_fusion(s, epochs=2 if QUICK else 30, B=Bk) for s in SEEDS13])
    ps_ = [predict_probs(m_, EVI, B=Bk) for m_ in ms_]
    PV[name] = np.mean(ps_, 0)
    QV[name] = np.mean([predict_quant(m_, EVI, B=Bk) for m_ in ms_], 0)
    LLS[name] = [row_ll_all(p_) for p_ in ps_]
    SC_EPOCHS[name] = [len(h_) for h_ in hs_]
    print(f"{name}: {len(SEEDS13)} seeds, epochs {SC_EPOCHS[name]} ({time.time() - t0:.0f} s)")
# "all" = section 8C's MM network; its extra seeds are trained here with the same settings
MM_MODELS = {s: FUS2[("MM", s)] for s in SEEDS if ("MM", s) in FUS2}
for s in SEEDS13:
    if s not in MM_MODELS:
        MM_MODELS[s], h_ = train_fusion(s, epochs=2 if QUICK else 30, B=SETS["MM"])
        HIST2[("MM", s)] = h_
ps_ = [predict_probs(MM_MODELS[s], EVI, B=SETS["MM"]) for s in SEEDS13]
PV[ALLK] = np.mean(ps_, 0)
LLS[ALLK] = [row_ll_all(p_) for p_ in ps_]
QV[ALLK] = np.mean([predict_quant(MM_MODELS[s], EVI, B=SETS["MM"]) for s in SEEDS13], 0)
SC_EPOCHS[ALLK] = [len(HIST2[("MM", s)]) for s in SEEDS13]
PV["prior"] = PRED["prior"]
LLS["prior"] = [row_ll_all(PRED["prior"])]
print(f"all: {len(SEEDS13)} seeds ({time.time() - t0:.0f} s)")

ZE = ZV[EVI]                                                 # realised vol-scaled returns (the quantile heads' target)
HAS_TEXT = np.stack([present2[n] for n in LANG_B], 1).max(1)[EVI] > 0
HAS_NEWS = np.maximum(present2["news"], present2["news_tx"])[EVI] > 0
print(f"rows with any text: valid {HAS_TEXT[P_VAL].mean():.1%}, test {HAS_TEXT[P_TEST].mean():.1%}; "
      f"with company news: valid {HAS_NEWS[P_VAL].mean():.1%}, test {HAS_NEWS[P_TEST].mean():.1%}")

# text coverage by year, every row of the sample (the news source thins out in 2020-22: finding A-99)
yr_all = FR.asof_date.dt.year.to_numpy()
TXT_COV = pd.DataFrame({"rows": pd.Series(1, index=yr_all).groupby(level=0).sum(),
                        "any text": pd.Series(np.stack([present2[n] for n in LANG_B], 1).max(1) > 0, index=yr_all).groupby(level=0).mean(),
                        "company news": pd.Series(np.maximum(present2["news"], present2["news_tx"]) > 0, index=yr_all).groupby(level=0).mean(),
                        "SEC filing in 14 days": pd.Series(present2["sec_impact"] > 0, index=yr_all).groupby(level=0).mean()})
TXT_COV.index.name = "year"
TXT_COV["block"] = FR.groupby(yr_all).block.agg(lambda b: b.fillna("purged").mode().iat[0]).reindex(TXT_COV.index)

SC_PER = {}                                                   # (subset, configuration) -> metrics by horizon

def seed_ll(name, ix):
    # mean six-horizon log loss of each single-seed model on rows ix
    return [float(np.mean([np.nanmean(L[ix, j]) for j in range(6)])) for L in LLS[name]]

def scorecard(ix, label):
    rows = {}
    for name in SC_ORDER:
        P = PV[name]
        per = pd.DataFrame([metrics(P[ix, j], YE[ix, j], RETE[ix, j], DATES[ix]) for j in range(6)], index=H6)
        SC_PER[(label, name)] = per
    base = SC_PER[(label, "prior")]["log loss"].mean()
    for name in SC_ORDER:
        per = SC_PER[(label, name)]
        r = per.drop(columns=["n"]).mean().to_dict()
        r = {"mean log loss": r.pop("log loss"), "gain vs prior": float(base - per["log loss"].mean()), **r}
        r["rank IC 5D"], r["rank IC 20D"] = per.loc["5D", "rank IC"], per.loc["20D", "rank IC"]
        r.pop("rank IC", None)
        sl = seed_ll(name, ix)
        r["seed SD of mean log loss"] = float(np.std(sl, ddof=1)) if len(sl) > 1 else np.nan
        if name in QV:
            q, z = QV[name][ix], ZE[ix]; ok = np.isfinite(z)
            r["q10-q90 coverage (target 0.80)"] = float(((z >= q[..., 0]) & (z <= q[..., 2]))[ok].mean())
            r["below q10 (target 0.10)"] = float((z < q[..., 0])[ok].mean())
        rows[name] = r
    return pd.DataFrame(rows).T

t0 = time.time()
SC_TEST, SC_VALID = scorecard(P_TEST, "test"), scorecard(P_VAL, "valid")
SC_TEST_TEXT = scorecard(P_TEST[HAS_TEXT[P_TEST]], "test, text")
SC_VALID_TEXT = scorecard(P_VAL[HAS_TEXT[P_VAL]], "valid, text")
print(f"scorecards: {time.time() - t0:.0f} s")
FMT = {c: "{:.4f}" for c in SC_TEST.columns} | {"gain vs prior": "{:+.5f}", "rank IC 5D": "{:+.4f}", "rank IC 20D": "{:+.4f}",
                                               "seed SD of mean log loss": "{:.5f}"}
display(Markdown("**Test 2020-22, all rows: every metric averaged over the six horizons** (green = better)"))
display(SC_TEST.style.format(FMT, na_rep="-").background_gradient(cmap="RdYlGn", subset=["gain vs prior"])
        .background_gradient(cmap="RdYlGn", subset=["up-vs-down AUROC", "sideways AUROC"]))
COMPACT = ["mean log loss", "gain vs prior", "seed SD of mean log loss", "up-vs-down AUROC", "sideways AUROC",
           "rank IC 5D", "rank IC 20D", "ECE"]
for title, tab in (("Test 2020-22, rows with news or a filing", SC_TEST_TEXT), ("Validation 2017-18, all rows", SC_VALID),
                   ("Validation 2017-18, rows with news or a filing", SC_VALID_TEXT)):
    display(Markdown(f"**{title}**"))
    display(tab[COMPACT].style.format(FMT, na_rep="-").background_gradient(cmap="RdYlGn", subset=["gain vs prior"]))

SC_SEEDS = pd.DataFrame({name: {"seeds": len(LLS[name]), "epochs per seed (cap 30)": SC_EPOCHS.get(name, "-"),
                                "valid: mean log loss per seed": [round(v, 5) for v in seed_ll(name, P_VAL)],
                                "test: mean log loss per seed": [round(v, 5) for v in seed_ll(name, P_TEST)]}
                         for name in SC_ORDER if name != "prior"}).T
display(Markdown("**Seeds and early stopping per configuration** (every network stops on the inner block before its "
                 "30-epoch cap)")); display(SC_SEEDS)
display(Markdown("**Text coverage by year** (share of stock-dates with each kind of text before the decision time)"))
display(TXT_COV.style.format({"rows": "{:,.0f}", "any text": "{:.1%}", "company news": "{:.1%}",
                              "SEC filing in 14 days": "{:.1%}"}))

# quantile coverage with and without text (on rows without text an NLP-only network sees no input at all)
cov_rows = {}
for name in [c for c in SC_ORDER if c in QV]:
    for split, ix0 in (("valid", P_VAL), ("test", P_TEST)):
        for lab, sel in (("with text", HAS_TEXT[ix0]), ("without text", ~HAS_TEXT[ix0])):
            ix = ix0[sel]; q, z = QV[name][ix], ZE[ix]; ok = np.isfinite(z)
            cov_rows[(name, split, lab)] = {"rows": int(len(ix)),
                                            "q10-q90 coverage": float(((z >= q[..., 0]) & (z <= q[..., 2]))[ok].mean())}
SC_COVER = pd.DataFrame(cov_rows).T.unstack([1, 2])
display(Markdown("**Quantile coverage (target 0.80) with and without text**")); display(SC_COVER.round(3))
""")
    code(r"""
# paired increments: what each input adds on top of the others (per-row log loss, six horizons, month-block bootstrap)
def row_ll6(P, ix):
    L = np.full((len(ix), 6), np.nan)
    for j in range(6):
        y = YE[ix, j]; ok = y >= 0
        L[ok, j] = -np.log(np.clip(P[ix[ok], j, y[ok]], 1e-7, 1))
    return L

def increment(with_, without, ix):
    d = np.nanmean(row_ll6(PV[without], ix) - row_ll6(PV[with_], ix), 1)
    ok = np.isfinite(d)
    est, ci, p = month_block_boot(d[ok], DATES[ix][ok])
    # the same increment seed by seed (seed k of one configuration against seed k of the other)
    Lw, Lo = LLS[with_], LLS[without]
    per_seed = [float(np.nanmean(np.nanmean(Lo[min(k, len(Lo) - 1)][ix] - Lw[k][ix], 1))) for k in range(len(Lw))]
    return {"gain": est, "low": float(ci[0]), "high": float(ci[1]), "p (one-sided)": p,
            "seed range low": min(per_seed), "seed range high": max(per_seed)}

INC = [("vision alone beats the prior", "vision only", "prior"),
       ("NLP alone beats the prior", "NLP only", "prior"),
       ("indicators alone beat the prior", "indicators only", "prior"),
       ("NLP added to vision", "vision + NLP", "vision only"),
       ("vision added to NLP", "vision + NLP", "NLP only"),
       ("NLP added to vision + indicators", ALLK, "vision + indicators"),
       ("vision added to indicators + NLP", ALLK, "indicators + NLP"),
       ("indicators added to vision + NLP", ALLK, "vision + NLP")]
SPL = {"valid": P_VAL, "test": P_TEST, "test, rows with text": P_TEST[HAS_TEXT[P_TEST]]}
SC_INC = pd.DataFrame({(lab, sp): increment(a, b, ix) for lab, a, b in INC for sp, ix in SPL.items()}).T
SC_INC.index.names = ["comparison", "rows"]
SC_INC["p Holm (8 per row set)"] = np.nan
for sp in SPL:                                               # A-97: eight comparisons per row set
    m_ = SC_INC.index.get_level_values("rows") == sp
    SC_INC.loc[m_, "p Holm (8 per row set)"] = holm(SC_INC.loc[m_, "p (one-sided)"].to_numpy(float))
SC_INC["clear (interval and seed range above 0, Holm p < 0.05)"] = ((SC_INC["low"] > 0) & (SC_INC["seed range low"] > 0)
                                                                   & (SC_INC["p Holm (8 per row set)"] < 0.05))
display(Markdown("**What each input adds** (gain in mean log loss per row; positive = better; 95% month-block interval; "
                 "Holm-adjusted p; the seed-by-seed range)"))
display(SC_INC.style.format({"gain": "{:+.5f}", "low": "{:+.5f}", "high": "{:+.5f}", "p (one-sided)": "{:.3f}",
                             "p Holm (8 per row set)": "{:.3f}", "seed range low": "{:+.5f}", "seed range high": "{:+.5f}"})
        .apply(lambda r: ["color: #2ca02c" if r["low"] > 0 else ("color: #d62728" if r["high"] < 0 else "")] * len(r), axis=1))

# stability: the gain over the prior year by year (v1.1's 2017-18 rows are in-sample, so they are left blank)
YRS = pd.DatetimeIndex(DATES).year.to_numpy()
yr_rows = {}
for name in [c for c in SC_ORDER if c != "prior"] + (["v1.1 (E08 frozen)"] if "v1.1 (E08 frozen)" in PRED else []):
    P = PV.get(name, PRED.get(name))
    r = {}
    for y_ in sorted(set(YRS[np.concatenate([P_VAL, P_TEST])])):
        ix = np.concatenate([P_VAL, P_TEST]); ix = ix[YRS[ix] == y_]
        if name == "v1.1 (E08 frozen)" and y_ < 2019:
            r[str(y_)] = np.nan
            continue
        lp = np.mean([np.nanmean(row_ll6(PV["prior"], ix)[:, j]) for j in range(6)])
        lm = np.mean([np.nanmean(row_ll6(P, ix)[:, j]) for j in range(6)])
        r[str(y_)] = float(lp - lm)
    yr_rows[name] = r
SC_YEAR = pd.DataFrame(yr_rows).T
SC_YEAR["years with a gain"] = (SC_YEAR > 0).sum(1).astype(str) + " of " + SC_YEAR.notna().sum(1).astype(str)
display(Markdown("**Gain over the class prior, year by year** (mean six-horizon log loss; v1.1 blank where in-sample)"))
display(SC_YEAR.style.format({c: "{:+.5f}" for c in SC_YEAR.columns if c != "years with a gain"}, na_rep="in-sample"))

fig, axs = plt.subplots(1, 3, figsize=(17, 4.2))
cfgs = [c for c in SC_ORDER if c != "prior"]
xx = np.arange(len(cfgs))
axs[0].bar(xx - .2, SC_VALID.loc[cfgs, "gain vs prior"], .4, label="valid 2017-18", color="#1f77b4",
           yerr=SC_VALID.loc[cfgs, "seed SD of mean log loss"].fillna(0), capsize=2)
axs[0].bar(xx + .2, SC_TEST.loc[cfgs, "gain vs prior"], .4, label="test 2020-22", color="#ff7f0e",
           yerr=SC_TEST.loc[cfgs, "seed SD of mean log loss"].fillna(0), capsize=2)
axs[0].axhline(0, color="k", lw=.8); axs[0].set_xticks(xx, cfgs, rotation=35, ha="right"); axs[0].legend(fontsize=8)
axs[0].set_title("log-loss gain over the class prior (bars: +/- 1 seed SD)")
for ax, col, title in ((axs[1], "up-vs-down AUROC", "direction: up-vs-down AUROC, test"),
                       (axs[2], "sideways AUROC", "does it move: sideways AUROC, test")):
    Mx = pd.DataFrame({c: SC_PER[("test", c)][col] for c in cfgs}).T
    im = ax.imshow(Mx.to_numpy(), cmap="RdYlGn", vmin=0.45 if "up" in col else 0.5, vmax=0.55 if "up" in col else 0.65,
                   aspect="auto")
    ax.set_xticks(range(6), H6); ax.set_yticks(range(len(cfgs)), cfgs); ax.set_title(title); ax.grid(False)
    for a_ in range(len(cfgs)):
        for b_ in range(6):
            ax.text(b_, a_, f"{Mx.iat[a_, b_]:.3f}", ha="center", va="center", fontsize=7)
    plt.colorbar(im, ax=ax, fraction=.03)
plt.tight_layout(); plt.show()

single = SC_TEST.loc[["vision only", "NLP only", "indicators only"], "gain vs prior"]
best_single = single.idxmax()
combo = SC_TEST.loc[ALLK, "gain vs prior"]
vn = SC_INC.loc[("NLP added to vision", "test")]
va_ = SC_INC.loc[("vision added to indicators + NLP", "test")]
SC_VERDICT = (f"On test, the strongest single input is **{best_single}** (gain {single.max():+.5f}); vision alone "
              f"{single['vision only']:+.5f}, NLP alone {single['NLP only']:+.5f}, indicators alone "
              f"{single['indicators only']:+.5f}. Vision + NLP: {SC_TEST.loc['vision + NLP', 'gain vs prior']:+.5f} "
              f"(NLP adds {vn['gain']:+.5f} [{vn['low']:+.5f}, {vn['high']:+.5f}] to vision). All three together: "
              f"{combo:+.5f}; the charts add {va_['gain']:+.5f} [{va_['low']:+.5f}, {va_['high']:+.5f}] once indicators "
              "and language are present. Direction (up-vs-down AUROC) stays near 0.5 for every input; the gains come "
              "from predicting whether a stock moves (sideways AUROC) and from cross-sectional ranking (rank IC). "
              f"Seed-to-seed spread of the mean log loss: up to {SC_TEST['seed SD of mean log loss'].max():.5f} (SD), so "
              "smaller gaps between configurations are not distinguishable.")
display(Markdown(SC_VERDICT))
""")


def add_trading(md, code) -> None:
    md(r"""
## 16A. Entry, stop-loss and exit: one trading rule on every modality's forecasts (descriptive)

Section 16 backtests the documented strategies. Here a single, fixed rule turns each configuration's forecasts
(section 13A) and the frozen v1.1 model's into trades, so the results compare the inputs, not the rules:

* **Signal:** s = P(UP) - P(DOWN) at the horizon. On each decision date with at least 10 stocks, **long the top 20%**
  and **short the bottom 20%** (ranking needs no threshold, so nothing is tuned). The risk engine's own gate (|s| at
  least 0.10) is counted separately.
* **Entry:** the next session's open, the first executable price after the 09:00 New York decision time.
* **Primary bracket, "fixed 1.5/3 ATR":** stop-loss 1.5 ATR(14) from the entry, target 3 ATR (reward-to-risk 2:1),
  time exit at the close h sessions later. A bar that gaps through the stop fills at its open; a bar that touches both
  levels counts as a stop; a target fills at the target (conservative both ways). Textbook multiples, fixed before the
  first run.
* **Sensitivity bracket, "sqrt-time":** the same 2:1 bracket scaled with the square root of the holding period
  (stop 1.5 ATR x sqrt(h / 5), target twice that), so it equals the primary bracket at 5D, is wider at 20D (where the
  fixed stop catches routine noise) and tighter at 1D (where the fixed bracket almost never triggers). Reported, never
  chosen from: both brackets were declared before this version ran (audit 9, A-94).
* **Costs:** 10 bps round trip (base) and 20 bps (severe), plus 0.5% a year short borrow.

**Two null tests.** (1) **Random books:** 300 books with the same number of longs and shorts per date, picked at random.
(2) **Volatility-matched books:** 300 random books with the same number of longs and shorts in each (date, ATR-quintile)
cell as the model's book. Costs and R are scaled by each stock's ATR, so a model that merely prefers volatile stocks
would beat plain random books without any skill; the matched books remove that (A-91). p-values count
(r + 1) / (N + 1), so the smallest possible value is 1/301 = 0.003 (A-92). Beating the null books means "ranks better
than chance after the same costs", not "profitable": the "profitable" column requires the expectancy interval to lie
above zero.

**Reading R and bps.** R divides each trade's result by its own stop distance (1.5 ATR), so a quiet stock's trade
weighs more in R than a volatile stock's, and costs weigh more too. Average return in bps weighs every trade by capital.
The two can have opposite signs when quiet stocks lose and volatile stocks win (A-96). The table shows gross R (no
costs) and net R side by side, and the long leg against **every stock long at the same costs** (positive = the model's
longs beat buying everything).

**Sharpe ratio.** A daily mark-to-market book: each day's cohort gets 1/h of the capital (half long, half short),
positions are marked at every close, costs are charged on entry and borrow daily (A-95). Max drawdown on the same book.

The frozen v1.1 model is evaluated on **test only**: it was refitted on 2003-2018, so its 2017-18 rows are in-sample
(A-90). Survivor universe, as everywhere in this notebook.
""")
    code(r"""
from concurrent.futures import ThreadPoolExecutor
TRD_H = ["1D", "5D", "20D"]
COST_BP = {"base": 10, "severe": 20}
BRACKETS = {"fixed 1.5/3 ATR": lambda h: (1.5, 3.0),
            "sqrt-time": lambda h: (1.5 * np.sqrt(HN[h] / 5), 3.0 * np.sqrt(HN[h] / 5))}
PRIMARY = "fixed 1.5/3 ATR"
N_NULL = 60 if QUICK else 300
EVP = np.concatenate([P_VAL, P_TEST]); FRI = EVI[EVP]
BARS16 = dict(globals().get("bars_cache") or {})
need_ = sorted(set(FR.ticker.iloc[FRI]) - set(BARS16))
need_ = [t for t in need_ if (BARS_DIR / f"{t}.parquet").exists()]
t0 = time.time()
with ThreadPoolExecutor(8) as ex_:
    for t_, d_ in zip(need_, ex_.map(read_daily, need_)):
        BARS16[t_] = d_
print(f"daily bars for {len(BARS16)} stocks ({len(need_)} read now, {time.time() - t0:.0f} s)")

HMAX = max(HN[h] for h in TRD_H)
NE = len(EVI)
PATH = {k: np.full((NE, HMAX), np.nan) for k in ("open", "high", "low", "close")}
PATH_DATE = np.full((NE, HMAX), np.datetime64("NaT"), dtype="datetime64[ns]")
sub = pd.DataFrame({"pos": EVP, "ticker": FR.ticker.to_numpy()[FRI], "asof_d": FR.asof_date.to_numpy()[FRI]})
for t_, g_ in sub.groupby("ticker"):
    d_ = BARS16.get(t_)
    if d_ is None:
        continue
    dates_ = d_["date"].to_numpy(dtype="datetime64[ns]")
    a_ = g_["asof_d"].to_numpy(dtype="datetime64[ns]")
    k_ = np.searchsorted(dates_, a_)
    hit = (k_ < len(dates_)) & (dates_[np.minimum(k_, len(dates_) - 1)] == a_)
    idx = k_[:, None] + 1 + np.arange(HMAX)[None, :]
    okk = hit[:, None] & (idx < len(dates_))
    idx = np.minimum(idx, len(dates_) - 1)
    for col in PATH:
        v = d_[col].to_numpy(float)[idx]
        PATH[col][g_.pos.to_numpy()] = np.where(okk, v, np.nan)
    PATH_DATE[g_.pos.to_numpy()] = np.where(okk, dates_[idx], np.datetime64("NaT"))
ENTRY = PATH["open"][:, 0]
ATR_E = (FR.d_atr_pct * FR.close).to_numpy(float)[EVI]
ATRP = FR.d_atr_pct.to_numpy(float)[EVI]
chk = np.isfinite(ENTRY[EVP]) & np.isfinite(FR.entry_open.to_numpy(float)[FRI])
TRD_CHECKS = {"valid + test rows": int(len(EVP)), "price paths found": float(np.isfinite(ENTRY[EVP]).mean()),
              "entry = stored next open (of rows with both)":
                  float(np.isclose(ENTRY[EVP][chk], FR.entry_open.to_numpy(float)[FRI][chk], rtol=1e-6).mean()),
              "first path day after the as-of session": bool((PATH_DATE[EVP, 0][np.isfinite(ENTRY[EVP])]
                                                              > FR.asof_date.to_numpy()[FRI][np.isfinite(ENTRY[EVP])]).all())}
print(TRD_CHECKS)
assert TRD_CHECKS["first path day after the as-of session"], "a trade would enter on or before its as-of session"

def bracket(side, h, k_stop, k_tgt):
    # outcome of a long (side +1) or short (-1) bracket trade on every row; independent of any model
    O, Hh, Lw, C = (PATH[k][:, :h] for k in ("open", "high", "low", "close"))
    risk = k_stop * ATR_E
    stop, tgt = ENTRY - side * risk, ENTRY + side * k_tgt * ATR_E
    hs = (Lw <= stop[:, None]) if side > 0 else (Hh >= stop[:, None])
    ht = (Hh >= tgt[:, None]) if side > 0 else (Lw <= tgt[:, None])
    BIG = h + 1
    fs = np.where(hs.any(1), hs.argmax(1), BIG); ft = np.where(ht.any(1), ht.argmax(1), BIG)
    nfin = np.isfinite(C).sum(1); last = np.maximum(nfin - 1, 0)
    r_ = np.arange(NE)
    stop_first = (fs < BIG) & (fs <= ft); tgt_first = (ft < BIG) & (ft < fs)
    o_at = O[r_, np.minimum(fs, h - 1)]
    gap = (fs > 0) & ((o_at < stop) if side > 0 else (o_at > stop))
    px = np.where(stop_first, np.where(gap, o_at, stop), np.where(tgt_first, tgt, C[r_, last]))
    day = np.where(stop_first, fs, np.where(tgt_first, ft, last)) + 1
    held = np.arange(h)[None, :] < day[:, None]
    fav = (Hh - ENTRY[:, None]) if side > 0 else (ENTRY[:, None] - Lw)
    adv = (ENTRY[:, None] - Lw) if side > 0 else (Hh - ENTRY[:, None])
    with np.errstate(invalid="ignore", divide="ignore"):
        out_ = {"R": side * (px - ENTRY) / risk, "ret": side * (px / ENTRY - 1), "px": px, "days": day, "risk": risk,
                "mfe": np.nanmax(np.where(held, fav, -np.inf), 1) / risk,
                "mae": np.nanmax(np.where(held, adv, -np.inf), 1) / risk,
                "exit": np.where(stop_first, "stop", np.where(tgt_first, "target", "time")),
                "ok": np.isfinite(ENTRY) & np.isfinite(ATR_E) & (ATR_E > 0) & (nfin >= h)}
    return out_

OUTC = {(b, h): {+1: bracket(+1, HN[h], *BRACKETS[b](h)), -1: bracket(-1, HN[h], *BRACKETS[b](h))}
        for b in BRACKETS for h in TRD_H}

def net(o, pos, side, cost):
    c = COST_BP[cost] / 1e4 + (0.005 * o["days"][pos] / 252 if side < 0 else 0)
    return o["ret"][pos] - c, o["R"][pos] - c * ENTRY[pos] / o["risk"][pos]

NETR = {(b, h, c): {s: net(OUTC[(b, h)][s], np.arange(NE), s, c)[1] for s in (+1, -1)}
        for b in BRACKETS for h in TRD_H for c in COST_BP}

def eligible(pos0, h):
    return pos0[OUTC[(PRIMARY, h)][+1]["ok"][pos0]]            # the ok rule does not depend on the bracket

def book_sides(score, E, q=0.2):
    df = pd.DataFrame({"pos": E, "d": DATES[E], "s": score[E]})
    df["n"] = df.groupby("d").s.transform("size"); df = df[df.n >= 10]
    df["r"] = df.groupby("d").s.rank(pct=True, method="first")
    return df.pos[df.r > 1 - q].to_numpy(), df.pos[df.r <= q].to_numpy()

def expectancy(L, S, b, h, cost="base"):
    nr = NETR[(b, h, cost)]
    return float((nr[+1][L].sum() + nr[-1][S].sum()) / max(len(L) + len(S), 1))

def mtm(L, S, b, h, cost="base"):
    # daily mark-to-market book: each cohort (decision date) gets 1/h of capital, half long and half short
    hh = HN[h]; parts_d, parts_r = [], []
    for side, pos in ((+1, L), (-1, S)):
        if not len(pos):
            continue
        o = OUTC[(b, h)][side]
        C = PATH["close"][pos, :hh]; D = PATH_DATE[pos, :hh]; e = o["days"][pos]; px = o["px"][pos]
        prev = np.concatenate([ENTRY[pos][:, None], C[:, :-1]], 1)
        k = np.arange(hh)[None, :]
        cur = np.where(k == (e - 1)[:, None], px[:, None], C)
        r = side * (cur / prev - 1)
        cohort = pd.Series(DATES[pos]); n = cohort.map(cohort.value_counts()).to_numpy()
        w = 1.0 / (hh * 2 * n)
        c = np.zeros_like(r); c[:, 0] = COST_BP[cost] / 1e4
        if side < 0:
            c = c + 0.005 / 252
        act = (k < e[:, None]) & np.isfinite(r)
        parts_d.append(D[act]); parts_r.append((w[:, None] * (r - c))[act])
    if not parts_d:
        return pd.Series(dtype=float)
    return pd.Series(np.concatenate(parts_r)).groupby(np.concatenate(parts_d)).sum().sort_index()

def book_stats(daily):
    if len(daily) < 20 or daily.std() == 0:
        return np.nan, np.nan, np.nan
    eq = (1 + daily).cumprod()
    return (float(daily.mean() / daily.std() * np.sqrt(252)), float(daily.mean() * 252),
            float((eq / eq.cummax() - 1).min()))

def trade_stats(L, S, b, h, cost="base"):
    oL, oS = OUTC[(b, h)][+1], OUTC[(b, h)][-1]
    rl, Rl = net(oL, L, +1, cost); rs, Rs = net(oS, S, -1, cost)
    R = np.concatenate([Rl, Rs]); r = np.concatenate([rl, rs]); d = np.concatenate([DATES[L], DATES[S]])
    if len(R) == 0:
        return {"trades": 0}
    Rg = np.concatenate([oL["R"][L], oS["R"][S]])
    # the round-trip cost per trade at which the book would break even (gross return after short borrow, in bps)
    be = np.concatenate([oL["ret"][L], oS["ret"][S] - 0.005 * oS["days"][S] / 252]).mean() * 1e4
    ex = np.concatenate([oL["exit"][L], oS["exit"][S]])
    est, ci, _ = month_block_boot(R, d)
    pf = r[r > 0].sum() / max(-r[r < 0].sum(), 1e-12)
    sh, ar, mdd = book_stats(mtm(L, S, b, h, cost))
    return {"trades": len(R), "win rate": float((r > 0).mean()), "target hit": float((ex == "target").mean()),
            "stopped out": float((ex == "stop").mean()), "time exit": float((ex == "time").mean()),
            "expectancy R (gross)": float(np.mean(Rg)), "expectancy R": est, "R low": float(ci[0]), "R high": float(ci[1]),
            "profitable (R interval above 0)": bool(ci[0] > 0),
            "avg net return (bps)": float(r.mean() * 1e4), "break-even round trip (bps)": float(be),
            "profit factor": float(pf),
            "MFE (R)": float(np.nanmean(np.concatenate([oL["mfe"][L], oS["mfe"][S]]))),
            "MAE (R)": float(np.nanmean(np.concatenate([oL["mae"][L], oS["mae"][S]]))),
            "avg sessions held": float(np.concatenate([oL["days"][L], oS["days"][S]]).mean()),
            "long leg R": float(Rl.mean()) if len(Rl) else np.nan, "short leg R": float(Rs.mean()) if len(Rs) else np.nan,
            "Sharpe (daily MTM)": sh, "annual return (daily MTM)": ar, "max drawdown (daily MTM)": mdd}

def pval(null, x):
    return float((1 + (np.asarray(null) >= x).sum()) / (1 + len(null)))

def random_null(E, b, h, n, seed=SEED):
    rng_ = np.random.default_rng(seed)
    return np.array([expectancy(*book_sides(rng_.random(NE), E), b, h) for _ in range(n)])

def matched_null(L, S, E, b, h, n, seed=SEED):
    # random books with the model's number of longs and shorts in every (date, ATR-quintile-within-date) cell (A-91)
    df = pd.DataFrame({"d": DATES[E], "a": ATRP[E]})
    qn = np.minimum((df.groupby("d").a.rank(method="first", pct=True).to_numpy() * 5).astype(int), 4)
    cell = pd.factorize(df.d)[0] * 5 + qn
    at = np.full(NE, -1); at[E] = np.arange(len(E))
    ncell = int(cell.max()) + 1
    nL = np.bincount(cell[at[L]], minlength=ncell); nS = np.bincount(cell[at[S]], minlength=ncell)
    cs = np.sort(cell); rank = np.arange(len(E)) - np.searchsorted(cs, cs, side="left")
    isL = rank < nL[cs]; isS = (rank >= nL[cs]) & (rank < nL[cs] + nS[cs])
    rng_ = np.random.default_rng(seed); out_ = np.empty(n)
    for i in range(n):
        order = np.lexsort((rng_.random(len(E)), cell))
        out_[i] = expectancy(E[order[isL]], E[order[isS]], b, h)
    return out_

TRD_MODELS = {c: PV[c] for c in SC_ORDER if c != "prior"}
V11 = "v1.1 (frozen, production)"
if "v1.1 (E08 frozen)" in PRED:
    TRD_MODELS[V11] = PRED["v1.1 (E08 frozen)"]
TRD_ROWS, NULLS = [], {}
t0 = time.time()
for b in BRACKETS:
    for split, pos0 in (("test", P_TEST), ("valid", P_VAL)):
        for h in TRD_H:
            j = H6.index(h); ELIG = eligible(pos0, h)
            NULLS[(b, split, h)] = random_null(ELIG, b, h, N_NULL)
            all_long = float(NETR[(b, h, "base")][+1][ELIG].mean())
            L_, S_ = book_sides(np.random.default_rng(1).random(NE), ELIG)
            TRD_ROWS.append({"bracket": b, "split": split, "horizon": h, "model": "random book (= prior)",
                             **trade_stats(L_, S_, b, h), "expectancy R (severe costs)": expectancy(L_, S_, b, h, "severe"),
                             "every stock long (R, net)": all_long, "edge-gate trades (|s| >= 0.10)": 0})
            for name, P in TRD_MODELS.items():
                if name == V11 and split == "valid":                     # in-sample (refit 2003-2018): not shown
                    continue
                s = P[:, j, 2] - P[:, j, 0]
                L_, S_ = book_sides(s, ELIG)
                st_ = trade_stats(L_, S_, b, h)
                mn = matched_null(L_, S_, ELIG, b, h, N_NULL)
                TRD_ROWS.append({"bracket": b, "split": split, "horizon": h, "model": name, **st_,
                                 "expectancy R (severe costs)": expectancy(L_, S_, b, h, "severe"),
                                 "p vs random books": pval(NULLS[(b, split, h)], st_["expectancy R"]),
                                 "p vs vol-matched books": pval(mn, st_["expectancy R"]),
                                 "every stock long (R, net)": all_long,
                                 "long leg minus every stock long (R)": st_["long leg R"] - all_long,
                                 "edge-gate trades (|s| >= 0.10)": int((np.abs(s[ELIG]) >= 0.10).sum())})
        print(f"{b}, {split}: {time.time() - t0:.0f} s")
TRD = pd.DataFrame(TRD_ROWS).set_index(["bracket", "split", "horizon", "model"])
TFMT = {c: "{:.3f}" for c in TRD.columns if TRD[c].dtype != bool} | {
    "trades": "{:,.0f}", "edge-gate trades (|s| >= 0.10)": "{:,.0f}", "avg net return (bps)": "{:+.1f}",
    "break-even round trip (bps)": "{:+.1f}",
    "expectancy R": "{:+.3f}", "expectancy R (gross)": "{:+.3f}", "R low": "{:+.3f}", "R high": "{:+.3f}",
    "long leg minus every stock long (R)": "{:+.3f}", "Sharpe (daily MTM)": "{:+.2f}", "annual return (daily MTM)": "{:+.2%}",
    "max drawdown (daily MTM)": "{:.1%}"}
TRD_TEST, TRD_VALID = TRD.loc[(PRIMARY, "test")], TRD.loc[(PRIMARY, "valid")]
TRD_SCALED = TRD.loc["sqrt-time"]
COMPACT16 = ["trades", "win rate", "stopped out", "expectancy R (gross)", "expectancy R", "R low", "R high",
             "profitable (R interval above 0)", "p vs random books", "p vs vol-matched books",
             "long leg minus every stock long (R)", "break-even round trip (bps)", "Sharpe (daily MTM)",
             "max drawdown (daily MTM)"]
display(Markdown("**Test 2020-22, primary bracket (fixed 1.5/3 ATR): every model's forecasts** (base costs)"))
display(TRD_TEST.style.format(TFMT, na_rep="-").background_gradient(cmap="RdYlGn", subset=["expectancy R"], vmin=-.1, vmax=.1))
display(Markdown("**Validation 2017-18, primary bracket** (v1.1 omitted: in-sample)"))
display(TRD_VALID[[c for c in COMPACT16 if c in TRD_VALID]].style.format(TFMT, na_rep="-"))
display(Markdown("**Sensitivity: the horizon-scaled bracket (sqrt-time), test and validation**"))
display(TRD_SCALED[[c for c in COMPACT16 if c in TRD_SCALED]].style.format(TFMT, na_rep="-"))
print(f"trade simulation: {time.time() - t0:.0f} s")
""")
    code(r"""
# where the edge would be: expectancy with intervals against the random books, the daily book, the stop diagnostics
fig, axs = plt.subplots(1, len(TRD_H), figsize=(6 * len(TRD_H), 4.4), sharey=True)
for ax, h in zip(axs, TRD_H):
    t_ = TRD_TEST.loc[h].drop(index="random book (= prior)")
    yy = np.arange(len(t_))
    lo, hi = np.percentile(NULLS[(PRIMARY, "test", h)], [2.5, 97.5])
    ax.axvspan(lo, hi, color="#9aa5b1", alpha=.3, label="95% of random books")
    ax.errorbar(t_["expectancy R"], yy, xerr=[t_["expectancy R"] - t_["R low"], t_["R high"] - t_["expectancy R"]],
                fmt="o", color="#d62728", capsize=3)
    ax.axvline(0, color="k", lw=.8); ax.set_yticks(yy, t_.index); ax.set_title(f"{h}: expectancy per trade (R, net), test")
    ax.legend(fontsize=7, loc="lower right")
plt.tight_layout(); plt.show()

h = "5D"; E5 = eligible(P_TEST, h)
fig, axs = plt.subplots(1, 2, figsize=(14, 4))
for name, col in ((ALLK, "#d62728"), ("vision only", "#1f77b4"), ("NLP only", "#9467bd"), ("indicators only", "#2ca02c"),
                  (V11, "#ff7f0e")):
    if name not in TRD_MODELS:
        continue
    s = TRD_MODELS[name][:, H6.index(h), 2] - TRD_MODELS[name][:, H6.index(h), 0]
    daily = mtm(*book_sides(s, E5), PRIMARY, h)
    axs[0].plot(daily.index, (1 + daily).cumprod(), label=name, color=col, lw=1.3)
axs[0].axhline(1, color="k", lw=.6)
axs[0].set_title(f"daily mark-to-market long-short book, {h}, test (growth of 1, base costs)")
axs[0].legend(fontsize=8)
o = OUTC[(PRIMARY, h)][+1]; okp = E5
win = o["R"][okp] > 0
axs[1].hist(o["mae"][okp][win], bins=np.linspace(0, 3, 31), alpha=.6, label="winning longs", color="#2ca02c", density=True)
axs[1].hist(o["mae"][okp][~win], bins=np.linspace(0, 3, 31), alpha=.6, label="losing longs", color="#d62728", density=True)
axs[1].axvline(1.0, color="k", ls=":", lw=1)
axs[1].set_title(f"maximum adverse excursion before the exit (in R = 1.5 ATR), every stock, {h}, test")
axs[1].set_xlabel("MAE (R); 1.0 = the stop"); axs[1].legend(fontsize=8)
plt.tight_layout(); plt.show()

stop_rows = {}
for b in BRACKETS:
    for h in TRD_H:
        o = OUTC[(b, h)][+1]; E_ = eligible(P_TEST, h)
        stop_rows[(b, h)] = {"long: stopped out": float((o["exit"][E_] == "stop").mean()),
                             "long: target hit": float((o["exit"][E_] == "target").mean()),
                             "long: winners that first went 0.75 R against": float((o["mae"][E_] > .75)[o["R"][E_] > 0].mean()),
                             "every stock long: mean R gross (market drift)": float(np.nanmean(o["R"][E_])),
                             "every stock long: mean R net (10 bps)": float(np.nanmean(NETR[(b, h, "base")][+1][E_])),
                             "cost of a round trip in R (mean)": float(np.nanmean(o["R"][E_] - NETR[(b, h, "base")][+1][E_]))}
TRD_STOP = pd.DataFrame(stop_rows)
display(Markdown("**Stop-loss diagnostics, every test stock-date, long bracket** (no model: the market alone; gross and "
                 "net of the same costs as the model rows)"))
display(TRD_STOP.style.format("{:.3f}"))
""")


def add_report(md, code) -> None:
    md(r"""
## 20. Full report: every key number of this run in one file

The last cell collects the tables of every section (data, splits, labels, E28, the language branch, the SEC expert,
optimisation, fine-tuning, calibration, every metric, the statistics, the modality scorecard with seeds, coverage and
year-by-year stability, the trading evaluation with both null tests and both brackets, the backtest, the knowledge base
and the forward test) and writes them to Google Drive as one Markdown file and one JSON file:
`antigravity_workspace/nexus_artifacts/notebook_reports/NOTEBOOK_REPORT_<UTC time>.md` (and `.json`).

Open the Markdown file on Drive, or share it, to review the whole run without scrolling through the notebook. A section
that did not run is listed as missing rather than failing the cell. Notes on how to read the tables come first.
""")
    code(r"""
import datetime as _dt, platform
REPORT_DIR = Path(os.environ.get("NEXUS_REPORT_DIR", str(WS / "nexus_artifacts" / "notebook_reports")))
REPORT_DIR.mkdir(parents=True, exist_ok=True)
STAMP = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
G_ = globals()

def _git(*a):
    try:
        return subprocess.run(["git", "-C", str(REPO_DIR), *a], capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        return "unknown"

def _cell(v):
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    if isinstance(v, (float, np.floating)):
        return "-" if not np.isfinite(v) else (f"{v:.5f}" if abs(v) < 100 else f"{v:,.1f}")
    if isinstance(v, (list, tuple, np.ndarray)):
        return "[" + ", ".join(_cell(x) for x in list(v)[:8]) + "]"
    return str(v).replace("|", "/").replace("\n", " ")

def md_table(obj, max_rows=400):
    df = obj if isinstance(obj, pd.DataFrame) else (obj.to_frame() if isinstance(obj, pd.Series) else pd.Series(obj).to_frame("value"))
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [" / ".join(map(str, c)) for c in df.columns]
    df = df.reset_index()
    df.columns = [" / ".join(map(str, c)) if isinstance(c, tuple) else str(c) for c in df.columns]
    # A-109: a "|" inside a column name (e.g. "|s| >= 0.10") would split the header and shift every column
    lines = ["| " + " | ".join(c.replace("|", "\\|") for c in df.columns) + " |", "|" + "---|" * len(df.columns)]
    for _, r in df.head(max_rows).iterrows():
        lines.append("| " + " | ".join(_cell(v) for v in r.to_numpy()) + " |")
    if len(df) > max_rows:
        lines.append(f"\n({len(df) - max_rows} more rows in the JSON file)")
    return "\n".join(lines)

def jsonable(obj):
    if isinstance(obj, pd.DataFrame):
        d = obj.copy()
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = [" / ".join(map(str, c)) for c in d.columns]
        d = d.reset_index()
        d.columns = [" / ".join(map(str, c)) if isinstance(c, tuple) else str(c) for c in d.columns]
        return json.loads(d.to_json(orient="records", default_handler=str))
    if isinstance(obj, pd.Series):
        return json.loads(obj.to_json(default_handler=str))
    return json.loads(json.dumps(obj, default=lambda o: o.item() if hasattr(o, "item") else str(o)))

def g(name, f=None):
    if name not in G_:
        return None
    try:
        return f(G_[name]) if f else G_[name]
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}

NOTES = ("* **In-sample rows.** v1.1 was refitted on 2003-2018: its 2017-18 (validation) numbers in section 12 are "
         "in-sample and marked so; section 16A shows v1.1 on test only.\n"
         "* **Test years.** 2020-22 is a reused development test (TriFrame read it; every model since v1 was scored on "
         "it). It is a diagnostic, not proof; only the forward test and the sealed 2023+ evaluation are clean.\n"
         "* **p-values.** Section 13A: one-sided month-block bootstrap (the share of resamples at or below zero), Holm-"
         "adjusted over the eight comparisons of each row set. Section 16A: (r + 1) / (N + 1) against 300 null books "
         "(smallest possible 0.003); beating the null books means ranking better than chance after the same costs, "
         "not profitability (see 'profitable').\n"
         "* **Seeds.** Every fusion number moves by about 0.001 between seeds; differences smaller than the seed SD in "
         "13A are not distinguishable.\n"
         "* **R versus bps.** R divides by each trade's stop distance, so quiet stocks weigh more; bps weigh every trade "
         "by capital. Opposite signs mean quiet stocks lost while volatile stocks won.\n"
         "* **Direction skill (gate 3).** No horizon of v1.1 or v2 is both validated (2017-18) and replicated "
         "(2020-22); the dashboard's engine therefore trades no direction (D-057). The frozen models are unchanged.\n"
         "* **Forward test.** Late rows are flagged and never scored; v3_fund's 9 late rows are the 1-stock file of "
         "session 2026-11-05 (incident D-045).")

SECTIONS = [
    ("Run", {"started": _dt.datetime.fromtimestamp(T_START, _dt.timezone.utc).isoformat(timespec="seconds"),
             "minutes": round((time.time() - T_START) / 60, 1), "QUICK mode": QUICK, "device": DEVICE,
             "gpu": torch.cuda.get_device_name(0) if DEVICE == "cuda" else "none", "python": platform.python_version(),
             "torch": torch.__version__, "lightgbm": lgb.__version__, "pandas": pd.__version__, "numpy": np.__version__,
             "repository commit": _git("rev-parse", "--short", "HEAD"), "commit date": _git("log", "-1", "--format=%cI"),
             "notebook build": "audits 9-11 (2026-11-13)", "seeds (sections 8, 8C)": SEEDS,
             "seeds (section 13A)": g("SEEDS13")}),
    ("How to read this report", NOTES),
    ("2. Data", {"rows": len(FR), "stocks": int(FR.ticker.nunique()), "first": str(FR.asof_date.min().date()),
                 "last": str(FR.asof_date.max().date()), **FR.block.fillna("purged").value_counts().to_dict()}),
    ("3. Labels: recomputed vs stored", g("mism", pd.DataFrame)),
    ("4. Splits: purging and embargo between blocks", g("rep", pd.DataFrame)),
    ("6. Logistic regression: global-minimum certificate", g("CERT")),
    ("7. LightGBM: best boosting round per horizon (early stop on the inner block)", g("BEST_IT")),
    ("8. Fusion network: epochs and best inner loss per seed",
     g("HIST", lambda H: {s: {"epochs": len(h), "best inner CE": float(h["inner CE (early stop)"].min())} for s, h in H.items()})),
    ("8A. ChronoBERT re-encoding parity (centred cosine)",
     g("COS", lambda c: {"filings": int(len(c)), "median": float(np.median(c)), "1st percentile": float(np.percentile(c, 1)),
                         "minimum": float(np.min(c))})),
    ("8B. SEC price-impact expert", g("SI_AUC", pd.DataFrame)),
    ("8C. E28 vision + NLP keep test (this run)", g("G28")),
    ("8C. E28 documented run: configurations", g("T28")),
    ("8C. E28 documented run: intervals", g("E28_D")),
    ("8C. E28 documented run: rank IC", g("RIC")),
    ("10. Fine-tuned chart CNN", g("FT_RESULT")),
    ("11. Calibration temperatures", g("TEMPS", lambda T: {h: float(t) for h, t in T.items()})),
    ("12. Mean log loss over six horizons", g("mean_ll")),
    ("12. Every metric, every model, every horizon (valid and test)", g("TABLE")),
    ("13. v1.1 on test: gain over prior, placebo, rank IC, Holm", g("ST")),
    ("13A. Text coverage by year", g("TXT_COV")),
    ("13A. Seeds and epochs per configuration", g("SC_SEEDS")),
    ("13A. Modality scorecard, test, all rows", g("SC_TEST")),
    ("13A. Modality scorecard, test, rows with text", g("SC_TEST_TEXT")),
    ("13A. Modality scorecard, validation, all rows", g("SC_VALID")),
    ("13A. Modality scorecard, validation, rows with text", g("SC_VALID_TEXT")),
    ("13A. Quantile coverage with and without text", g("SC_COVER")),
    ("13A. What each input adds", g("SC_INC")),
    ("13A. Gain over the prior, year by year", g("SC_YEAR")),
    ("13A. Verdict", g("SC_VERDICT")),
    ("14. Mean log-loss gain over the prior on test (every model and saved run)", g("gain")),
    ("14. Reproduction of the documented E03 numbers", g("REPRO")),
    ("15. Gate 3: direction skill on validation vs test (replication, D-057)", g("SKT")),
    ("16. Documented backtest, test", g("BT_TEST")),
    ("16. Benchmark and PBO", g("BT_JSON", lambda b_: {"benchmark (equal-weight survivors), test":
                                                       b_.get("benchmark_ew_survivors", {}).get("test"), "PBO": b_.get("pbo")})),
    ("16A. Data checks", g("TRD_CHECKS")),
    ("16A. One bracket rule on every model: test (primary bracket)", g("TRD_TEST")),
    ("16A. One bracket rule on every model: validation (primary bracket; v1.1 omitted, in-sample)", g("TRD_VALID")),
    ("16A. Sensitivity: horizon-scaled bracket (sqrt-time)", g("TRD_SCALED")),
    ("16A. Stop-loss diagnostics", g("TRD_STOP")),
    ("17. Knowledge base: documented retrieval evaluation",
     g("E", lambda e: pd.DataFrame(e["book"]).T[["recall@1", "recall@5", "recall@10", "mrr", "ndcg@10"]])),
    ("18. Forward test chains", g("FWD")),
]
missing = [t for t, v in SECTIONS if v is None]
lines = [f"# NEXUS Alpha: end-to-end notebook report ({STAMP})", "",
         "Generated by the last cell of `notebooks/NEXUS_Alpha_End_to_End.ipynb`. Research only; nothing here is advice.", ""]
for title, v in SECTIONS:
    if v is None:
        continue
    lines += [f"## {title}", ""]
    lines.append(v if isinstance(v, str) else md_table(v))
    lines.append("")
if missing:
    lines += ["## Sections that did not run", "", *[f"* {m}" for m in missing], ""]
REPORT_MD = REPORT_DIR / f"NOTEBOOK_REPORT_{STAMP}.md"
REPORT_JSON = REPORT_DIR / f"NOTEBOOK_REPORT_{STAMP}.json"
REPORT_MD.write_text("\n".join(lines), encoding="utf-8")
REPORT_JSON.write_text(json.dumps({t: jsonable(v) for t, v in SECTIONS if v is not None} | {"missing": missing},
                                  indent=1, default=str), encoding="utf-8")
print(f"report written: {REPORT_MD}\n                {REPORT_JSON}")
print(f"{len(SECTIONS) - len(missing)} sections, {len(missing)} missing: {missing}")
""")

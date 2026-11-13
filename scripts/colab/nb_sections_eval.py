"""Notebook sections 13A, 16A and 20: which inputs predict (vision vs NLP vs both), what each modality's forecasts are
worth as trades (entry, stop-loss, exit), and a full report of the run written to Google Drive.

Imported by make_end_to_end_notebook.py. 13A comes after section 13 (it needs `metrics`, `month_block_boot` and section
8C's block sets), 16A after section 16 (it reads daily bars), 20 at the very end (it collects every table of the run).

Everything here is descriptive. No configuration is selected from these tables: E28's recorded decision stands, and the
validation years have already served the trial ledger's selection trials.
"""
from __future__ import annotations


def add_scorecard(md, code) -> None:
    md(r"""
## 13A. Modality scorecard: vision vs NLP vs both (descriptive)

Which inputs carry the forecastable information: the charts (computer vision), the text (NLP), the indicators, or a
combination? The same gated fusion network as section 8 (same architecture, optimiser, early stopping on the inner
block, same seeds) is trained on each set of blocks, so the only difference between two rows of the table is the input.

| Configuration | Blocks |
|---|---|
| vision only | weekly, daily and hourly chart embeddings |
| NLP only | news (TF-IDF, tone, timing), SEC (TF-IDF, tone, items, timing), FinBERT news, ChronoBERT filings, the SEC price-impact expert |
| indicators only | the 80 indicator features |
| vision + NLP | charts and every language block, no indicators |
| vision + indicators | charts and indicators, no language |
| indicators + NLP | indicators and language, no charts |
| all | charts, indicators and language (section 8C's MM network) |

**Metrics, per configuration, on validation (2017-18) and test (2020-22):** log loss and its gain over the class prior,
Brier score, accuracy, balanced accuracy, macro-F1, AUROC, ECE (calibration error), up-vs-down AUROC (direction),
sideways AUROC (whether the stock moves), rank IC (cross-sectional ranking). The quantile heads are scored too: how often
the realised return lands inside the predicted 10-90% range (target 80%) and below the 10% quantile (target 10%), the
natural distance for a stop-loss.

**A fair test for text.** Language exists only where there was news or a filing before the decision time. The scorecard
is shown on all rows and on the rows with any text; without text an NLP-only model can only learn the class prior.

**Increments** are paired on the same rows: the per-row log-loss difference, averaged over the six horizons, with a
month-block bootstrap 95% interval (positive = the added input helps).

**This section is descriptive.** Nothing is selected from it: E28's decision (section 8C) is the recorded one, and the
validation years have already served the trial ledger's selection trials (`docs/TRIAL_LEDGER.md`).
""")
    code(r"""
VISION_B, LANG_B, IND_B = ["img_W", "img_D", "img_H"], ["news", "sec"] + list(NEWB), ["num"]
ALLK = "all (vision + indicators + NLP)"
MOD_CFG = {"vision only": VISION_B, "NLP only": LANG_B, "indicators only": IND_B, "vision + NLP": VISION_B + LANG_B,
           "vision + indicators": VISION_B + IND_B, "indicators + NLP": IND_B + LANG_B}
SC_ORDER = ["prior"] + list(MOD_CFG) + [ALLK]

@torch.no_grad()
def predict_quant(model, ix, bs=8192, B=None):
    model.eval(); xs, m, _, _ = to_dev(ix, B); out_ = []
    for i in range(0, len(m), bs):
        _, qu, _ = model({k: v[i:i + bs] for k, v in xs.items()}, m[i:i + bs])
        out_.append(qu.float().cpu().numpy())
    return np.sort(np.concatenate(out_), axis=-1)            # (n, 6, 3): q10, q50, q90 of the vol-scaled return

QV, SC_EPOCHS = {}, {}
t0 = time.time()
for name, names_ in MOD_CFG.items():
    Bk = block_set(names_)
    ms_, hs_ = zip(*[train_fusion(s, epochs=2 if QUICK else 30, B=Bk) for s in SEEDS])
    PV[name] = np.mean([predict_probs(m_, EVI, B=Bk) for m_ in ms_], 0)
    QV[name] = np.mean([predict_quant(m_, EVI, B=Bk) for m_ in ms_], 0)
    SC_EPOCHS[name] = [len(h_) for h_ in hs_]
    print(f"{name}: {len(SEEDS)} seeds, epochs {SC_EPOCHS[name]} ({time.time() - t0:.0f} s)")
PV[ALLK] = PV["MM"]
QV[ALLK] = np.mean([predict_quant(FUS2[("MM", s)], EVI, B=SETS["MM"]) for s in SEEDS], 0)
PV["prior"] = PRED["prior"]

ZE = ZV[EVI]                                                 # realised vol-scaled returns (the quantile heads' target)
HAS_TEXT = np.stack([present2[n] for n in LANG_B], 1).max(1)[EVI] > 0
print(f"rows with any text: valid {HAS_TEXT[P_VAL].mean():.1%}, test {HAS_TEXT[P_TEST].mean():.1%}")

SC_PER = {}                                                   # (subset, configuration) -> metrics by horizon

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
FMT = {c: "{:.4f}" for c in SC_TEST.columns} | {"gain vs prior": "{:+.5f}", "rank IC 5D": "{:+.4f}", "rank IC 20D": "{:+.4f}"}
display(Markdown("**Test 2020-22, all rows: every metric averaged over the six horizons** (green = better)"))
display(SC_TEST.style.format(FMT, na_rep="-").background_gradient(cmap="RdYlGn", subset=["gain vs prior"])
        .background_gradient(cmap="RdYlGn", subset=["up-vs-down AUROC", "sideways AUROC"]))
COMPACT = ["mean log loss", "gain vs prior", "up-vs-down AUROC", "sideways AUROC", "rank IC 5D", "rank IC 20D", "ECE"]
for title, tab in (("Test 2020-22, rows with news or a filing", SC_TEST_TEXT), ("Validation 2017-18, all rows", SC_VALID),
                   ("Validation 2017-18, rows with news or a filing", SC_VALID_TEXT)):
    display(Markdown(f"**{title}**"))
    display(tab[COMPACT].style.format(FMT, na_rep="-").background_gradient(cmap="RdYlGn", subset=["gain vs prior"]))
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
    return {"gain": est, "low": float(ci[0]), "high": float(ci[1]), "p (one-sided)": p}

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
display(Markdown("**What each input adds** (gain in mean log loss per row; positive = better; 95% month-block interval)"))
display(SC_INC.style.format({"gain": "{:+.5f}", "low": "{:+.5f}", "high": "{:+.5f}", "p (one-sided)": "{:.3f}"})
        .apply(lambda r: ["color: #2ca02c" if r["low"] > 0 else ("color: #d62728" if r["high"] < 0 else "")] * len(r), axis=1))

fig, axs = plt.subplots(1, 3, figsize=(17, 4.2))
cfgs = [c for c in SC_ORDER if c != "prior"]
xx = np.arange(len(cfgs))
axs[0].bar(xx - .2, SC_VALID.loc[cfgs, "gain vs prior"], .4, label="valid 2017-18", color="#1f77b4")
axs[0].bar(xx + .2, SC_TEST.loc[cfgs, "gain vs prior"], .4, label="test 2020-22", color="#ff7f0e")
axs[0].axhline(0, color="k", lw=.8); axs[0].set_xticks(xx, cfgs, rotation=35, ha="right"); axs[0].legend(fontsize=8)
axs[0].set_title("log-loss gain over the class prior (higher = better)")
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
SC_VERDICT = (f"On test, the strongest single input is **{best_single}** (gain {single.max():+.5f}); vision alone "
              f"{single['vision only']:+.5f}, NLP alone {single['NLP only']:+.5f}, indicators alone "
              f"{single['indicators only']:+.5f}. Vision + NLP: {SC_TEST.loc['vision + NLP', 'gain vs prior']:+.5f} "
              f"(NLP adds {vn['gain']:+.5f} [{vn['low']:+.5f}, {vn['high']:+.5f}] to vision). All three together: "
              f"{combo:+.5f}. Direction (up-vs-down AUROC) stays near 0.5 for every input; the gains come from "
              "predicting whether a stock moves (sideways AUROC) and from cross-sectional ranking (rank IC).")
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
* **Stop-loss:** 1.5 ATR(14) from the entry. **Target:** 3 ATR (reward-to-risk 2:1). **Time exit:** the close h
  sessions later. A bar that gaps through the stop fills at its open; a bar that touches both levels counts as a stop;
  a target fills at the target (conservative both ways). The ATR multiples are textbook values (Wilder; Kaufman),
  fixed before this section was written.
* **Costs:** 10 bps round trip (base) and 20 bps (severe), plus 0.5% a year short borrow.
* **Null:** 300 random books (the same number of longs and shorts per date, picked at random). The class prior gives
  every stock the same score, so it *is* a random book; a within-date placebo of any model is one too. A useful signal
  beats the random books: the table reports the share of random books with a higher expectancy (permutation p).

Per configuration and horizon: trades, win rate, target / stop / time-exit shares, **expectancy in R** (net, with a
month-block 95% interval), average net return per trade, profit factor, average favourable and adverse excursion (MFE,
MAE in R), the long and short legs, and an approximate Sharpe ratio of the per-date book (positions overlap, so it is
indicative). Test shown in full; validation alongside. Survivor universe, as everywhere in this notebook.
""")
    code(r"""
from concurrent.futures import ThreadPoolExecutor
TRD_H = ["1D", "5D", "20D"]
K_STOP, K_TGT, COST_BP = 1.5, 3.0, {"base": 10, "severe": 20}
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
ENTRY = PATH["open"][:, 0]
ATR_E = (FR.d_atr_pct * FR.close).to_numpy(float)[EVI]
chk = np.isfinite(ENTRY[EVP]) & np.isfinite(FR.entry_open.to_numpy(float)[FRI])
print(f"paths found for {np.isfinite(ENTRY[EVP]).mean():.1%} of valid + test rows; entry = stored next open for "
      f"{np.isclose(ENTRY[EVP][chk], FR.entry_open.to_numpy(float)[FRI][chk], rtol=1e-6).mean():.2%} of them")

def bracket(side, h):
    # outcome of a long (side +1) or short (-1) bracket trade on every row; independent of any model
    O, Hh, Lw, C = (PATH[k][:, :h] for k in ("open", "high", "low", "close"))
    risk = K_STOP * ATR_E
    stop, tgt = ENTRY - side * risk, ENTRY + side * K_TGT * ATR_E
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
        out_ = {"R": side * (px - ENTRY) / risk, "ret": side * (px / ENTRY - 1), "days": day,
                "mfe": np.nanmax(np.where(held, fav, -np.inf), 1) / risk,
                "mae": np.nanmax(np.where(held, adv, -np.inf), 1) / risk,
                "exit": np.where(stop_first, "stop", np.where(tgt_first, "target", "time")),
                "ok": np.isfinite(ENTRY) & np.isfinite(ATR_E) & (ATR_E > 0) & (nfin >= h)}
    return out_

OUTC = {h: {+1: bracket(+1, HN[h]), -1: bracket(-1, HN[h])} for h in TRD_H}

def net(o, pos, side, h, cost):
    c = COST_BP[cost] / 1e4 + (0.005 * o["days"][pos] / 252 if side < 0 else 0)
    return o["ret"][pos] - c, o["R"][pos] - c * ENTRY[pos] / (K_STOP * ATR_E[pos])

def book_sides(score, pos, h, q=0.2):
    o = OUTC[h][+1]
    pos = pos[o["ok"][pos]]
    df = pd.DataFrame({"pos": pos, "d": DATES[pos], "s": score[pos]})
    df["n"] = df.groupby("d").s.transform("size"); df = df[df.n >= 10]
    df["r"] = df.groupby("d").s.rank(pct=True, method="first")
    return df.pos[df.r > 1 - q].to_numpy(), df.pos[df.r <= q].to_numpy()

def trade_stats(longs, shorts, h, cost="base"):
    oL, oS = OUTC[h][+1], OUTC[h][-1]
    rl, Rl = net(oL, longs, +1, h, cost); rs, Rs = net(oS, shorts, -1, h, cost)
    R = np.concatenate([Rl, Rs]); r = np.concatenate([rl, rs]); d = np.concatenate([DATES[longs], DATES[shorts]])
    ex = np.concatenate([oL["exit"][longs], oS["exit"][shorts]])
    mfe = np.concatenate([oL["mfe"][longs], oS["mfe"][shorts]]); mae = np.concatenate([oL["mae"][longs], oS["mae"][shorts]])
    days = np.concatenate([oL["days"][longs], oS["days"][shorts]])
    if len(R) == 0:
        return {"trades": 0}
    est, ci, _ = month_block_boot(R, d)
    pf = r[r > 0].sum() / max(-r[r < 0].sum(), 1e-12)
    bk = pd.DataFrame({"d": d, "r": r, "side": np.r_[np.ones(len(rl)), -np.ones(len(rs))]})
    per = bk.groupby(["d", "side"]).r.mean().unstack()
    book = per.mean(1)
    sharpe = float(book.mean() / book.std() * np.sqrt(252 / HN[h])) if len(book) > 2 and book.std() > 0 else np.nan
    return {"trades": len(R), "win rate": float((r > 0).mean()), "target hit": float((ex == "target").mean()),
            "stopped out": float((ex == "stop").mean()), "time exit": float((ex == "time").mean()),
            "expectancy R": est, "R low": float(ci[0]), "R high": float(ci[1]),
            "avg net return (bps)": float(r.mean() * 1e4), "profit factor": float(pf),
            "MFE (R)": float(np.nanmean(mfe)), "MAE (R)": float(np.nanmean(mae)), "avg sessions held": float(days.mean()),
            "long leg R": float(Rl.mean()) if len(Rl) else np.nan, "short leg R": float(Rs.mean()) if len(Rs) else np.nan,
            "Sharpe (approx.)": sharpe}

def expectancy(longs, shorts, h, cost="base"):
    _, Rl = net(OUTC[h][+1], longs, +1, h, cost); _, Rs = net(OUTC[h][-1], shorts, -1, h, cost)
    return float(np.concatenate([Rl, Rs]).mean())

def random_null(pos, h, n_books=300, seed=SEED):
    # expectancy of books that pick the same number of longs and shorts per date at random
    rng_ = np.random.default_rng(seed)
    return np.array([expectancy(*book_sides(rng_.random(NE), pos, h), h) for _ in range(n_books)])

TRD_MODELS = {c: PV[c] for c in SC_ORDER if c != "prior"}
if "v1.1 (E08 frozen)" in PRED:
    TRD_MODELS["v1.1 (frozen, production)"] = PRED["v1.1 (E08 frozen)"]
TRD_ROWS, NULLS = [], {}
t0 = time.time()
for split, pos0 in (("test", P_TEST), ("valid", P_VAL)):
    for h in TRD_H:
        j = H6.index(h)
        NULLS[(split, h)] = random_null(pos0, h, n_books=60 if QUICK else 300)
        L_, S_ = book_sides(np.random.default_rng(1).random(NE), pos0, h)
        rnd = trade_stats(L_, S_, h)
        TRD_ROWS.append({"split": split, "horizon": h, "model": "random book (= prior)", **rnd,
                         "expectancy R (severe costs)": expectancy(L_, S_, h, "severe"),
                         "permutation p": np.nan, "edge-gate trades (|s| >= 0.10)": 0})
        for name, P in TRD_MODELS.items():
            s = P[:, j, 2] - P[:, j, 0]
            L_, S_ = book_sides(s, pos0, h)
            st_ = trade_stats(L_, S_, h)
            okp = pos0[OUTC[h][+1]["ok"][pos0]]
            gate = int((np.abs(s[okp]) >= 0.10).sum())
            TRD_ROWS.append({"split": split, "horizon": h, "model": name, **st_,
                             "expectancy R (severe costs)": expectancy(L_, S_, h, "severe"),
                             "permutation p": float((NULLS[(split, h)] >= st_["expectancy R"]).mean()),
                             "edge-gate trades (|s| >= 0.10)": gate})
TRD = pd.DataFrame(TRD_ROWS).set_index(["split", "horizon", "model"])
print(f"trade simulation: {time.time() - t0:.0f} s")
TFMT = {c: "{:.3f}" for c in TRD.columns} | {"trades": "{:,.0f}", "edge-gate trades (|s| >= 0.10)": "{:,.0f}",
                                            "avg net return (bps)": "{:+.1f}", "expectancy R": "{:+.3f}",
                                            "R low": "{:+.3f}", "R high": "{:+.3f}", "permutation p": "{:.3f}"}
TRD_TEST, TRD_VALID = TRD.loc["test"], TRD.loc["valid"]
display(Markdown("**Test 2020-22: the same long-short bracket rule on every model's forecasts** (base costs)"))
display(TRD_TEST.style.format(TFMT, na_rep="-").background_gradient(cmap="RdYlGn", subset=["expectancy R"], vmin=-.1, vmax=.1))
display(Markdown("**Validation 2017-18** (compact)"))
display(TRD_VALID[["trades", "win rate", "expectancy R", "R low", "R high", "profit factor", "permutation p"]]
        .style.format(TFMT, na_rep="-"))
""")
    code(r"""
# where the edge would be: expectancy with intervals against the random books, and the stop-loss diagnostics
fig, axs = plt.subplots(1, len(TRD_H), figsize=(6 * len(TRD_H), 4.4), sharey=True)
for ax, h in zip(axs, TRD_H):
    t_ = TRD_TEST.loc[h].drop(index="random book (= prior)")
    yy = np.arange(len(t_))
    lo, hi = np.percentile(NULLS[("test", h)], [2.5, 97.5])
    ax.axvspan(lo, hi, color="#9aa5b1", alpha=.3, label="95% of random books")
    ax.errorbar(t_["expectancy R"], yy, xerr=[t_["expectancy R"] - t_["R low"], t_["R high"] - t_["expectancy R"]],
                fmt="o", color="#d62728", capsize=3)
    ax.axvline(0, color="k", lw=.8); ax.set_yticks(yy, t_.index); ax.set_title(f"{h}: expectancy per trade (R), test")
    ax.legend(fontsize=7, loc="lower right")
plt.tight_layout(); plt.show()

h = "5D"; pos0 = P_TEST
fig, axs = plt.subplots(1, 2, figsize=(14, 4))
for name, col in ((ALLK, "#d62728"), ("vision only", "#1f77b4"), ("NLP only", "#9467bd"), ("indicators only", "#2ca02c")):
    s = PV[name][:, H6.index(h), 2] - PV[name][:, H6.index(h), 0]
    L_, S_ = book_sides(s, pos0, h)
    rl, _ = net(OUTC[h][+1], L_, +1, h, "base"); rs, _ = net(OUTC[h][-1], S_, -1, h, "base")
    bk = pd.concat([pd.DataFrame({"d": DATES[L_], "r": rl, "s": 1}), pd.DataFrame({"d": DATES[S_], "r": rs, "s": -1})])
    book = bk.groupby(["d", "s"]).r.mean().unstack().mean(1).sort_index()
    axs[0].plot(book.index, (book / HN[h]).cumsum(), label=name, color=col, lw=1.3)
axs[0].set_title(f"cumulative per-date long-short book return, {h}, test (sum of daily-scaled returns)")
axs[0].legend(fontsize=8)
o = OUTC[h][+1]; okp = pos0[o["ok"][pos0]]
win = o["R"][okp] > 0
axs[1].hist(o["mae"][okp][win], bins=np.linspace(0, 3, 31), alpha=.6, label="winning longs", color="#2ca02c", density=True)
axs[1].hist(o["mae"][okp][~win], bins=np.linspace(0, 3, 31), alpha=.6, label="losing longs", color="#d62728", density=True)
axs[1].axvline(1.0, color="k", ls=":", lw=1)
axs[1].set_title(f"maximum adverse excursion before the exit (in R = {K_STOP} ATR), every stock, {h}, test")
axs[1].set_xlabel("MAE (R); 1.0 = the stop"); axs[1].legend(fontsize=8)
plt.tight_layout(); plt.show()

TRD_STOP = pd.DataFrame({h: {"long: stopped out": float((OUTC[h][+1]["exit"][P_TEST] == "stop")[OUTC[h][+1]["ok"][P_TEST]].mean()),
                             "long: target hit": float((OUTC[h][+1]["exit"][P_TEST] == "target")[OUTC[h][+1]["ok"][P_TEST]].mean()),
                             "long: winners that first went 0.75 R against": float(
                                 (OUTC[h][+1]["mae"][P_TEST] > .75)[(OUTC[h][+1]["R"][P_TEST] > 0) & OUTC[h][+1]["ok"][P_TEST]].mean()),
                             "long: mean R, every stock (market drift)": float(np.nanmean(OUTC[h][+1]["R"][P_TEST][OUTC[h][+1]["ok"][P_TEST]]))}
                         for h in TRD_H})
display(Markdown("**Stop-loss diagnostics, every test stock-date, long bracket** (no model: the market alone)"))
display(TRD_STOP.style.format("{:.3f}"))
""")


def add_report(md, code) -> None:
    md(r"""
## 20. Full report: every key number of this run in one file

The last cell collects the tables of every section (data, labels, E28, the language branch, the SEC expert,
optimisation, fine-tuning, calibration, every metric, the statistics, the modality scorecard, the trading evaluation,
the backtest and the forward test) and writes them to Google Drive as one Markdown file and one JSON file:
`antigravity_workspace/nexus_artifacts/notebook_reports/NOTEBOOK_REPORT_<UTC time>.md` (and `.json`).

Open the Markdown file on Drive, or share it, to review the whole run without scrolling through the notebook. A section
that did not run is listed as missing rather than failing the cell.
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
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
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

SECTIONS = [
    ("Run", {"started": _dt.datetime.fromtimestamp(T_START, _dt.timezone.utc).isoformat(timespec="seconds"),
             "minutes": round((time.time() - T_START) / 60, 1), "QUICK mode": QUICK, "device": DEVICE,
             "gpu": torch.cuda.get_device_name(0) if DEVICE == "cuda" else "none", "python": platform.python_version(),
             "torch": torch.__version__, "lightgbm": lgb.__version__, "pandas": pd.__version__, "numpy": np.__version__,
             "repository commit": _git("rev-parse", "--short", "HEAD"), "commit date": _git("log", "-1", "--format=%cI"),
             "seeds": SEEDS}),
    ("2. Data", {"rows": len(FR), "stocks": int(FR.ticker.nunique()), "first": str(FR.asof_date.min().date()),
                 "last": str(FR.asof_date.max().date()), **FR.block.fillna("purged").value_counts().to_dict()}),
    ("3. Labels: recomputed vs stored", g("mism", pd.DataFrame)),
    ("6. Logistic regression: global-minimum certificate", g("CERT")),
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
    ("13A. Modality scorecard, test, all rows", g("SC_TEST")),
    ("13A. Modality scorecard, test, rows with text", g("SC_TEST_TEXT")),
    ("13A. Modality scorecard, validation, all rows", g("SC_VALID")),
    ("13A. Modality scorecard, validation, rows with text", g("SC_VALID_TEXT")),
    ("13A. What each input adds", g("SC_INC")),
    ("13A. Verdict", g("SC_VERDICT")),
    ("14. Mean log-loss gain over the prior on test (every model and saved run)", g("gain")),
    ("14. Reproduction of the documented E03 numbers", g("REPRO")),
    ("15. Gate 3: direction skill on validation vs test", g("SKT")),
    ("16. Documented backtest, test", g("BT_TEST")),
    ("16. Benchmark and PBO", g("BT_JSON", lambda b_: {"benchmark (equal-weight survivors), test":
                                                       b_.get("benchmark_ew_survivors", {}).get("test"), "PBO": b_.get("pbo")})),
    ("16A. One bracket rule on every model: test", g("TRD_TEST")),
    ("16A. One bracket rule on every model: validation", g("TRD_VALID")),
    ("16A. Stop-loss diagnostics", g("TRD_STOP")),
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

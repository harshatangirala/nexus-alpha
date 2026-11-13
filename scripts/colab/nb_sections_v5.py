"""Notebook v5 additions (audit 12, 2026-11-14): section 13B, extended fundamentals and the E30 replication.

Section 13B reads one staged table (`fundamentals/e30_stock_months.parquet`, built by `stage_e30_inputs.py`), recomputes
the published composites from the as-known SEC XBRL snapshots with the repository's `fundamentals/extended.py`, repeats
the pre-registered E30 test (docs/E30_PREREGISTRATION.md) and checks the result against the documented one.
"""
from __future__ import annotations


def add_fundamentals(md, code) -> None:
    md(r"""
## 13B. Fundamentals: published scores, and whether they rank returns (E30)

The project already computes 31 point-in-time ratios and 7 multiples from SEC XBRL filings (P/E, EV/EBITDA, P/S, P/B,
P/FCF, ROIC, Piotroski F, Altman Z, accruals and more). Audit 12 added the **published composite and forensic scores**,
each exactly as its authors defined it, with nothing fitted:

| Score | Source | What it measures |
|---|---|---|
| Beneish M-score | Beneish (1999), eight indices | how much a firm's accounts look like those of known manipulators; flag above -1.78 |
| Montier C-score | Montier (2008), six signs | earnings-quality warning signs, 0 to 6 |
| Gross profitability | Novy-Marx (2013) | (revenue - cost of goods sold) / total assets |
| Magic formula | Greenblatt (2005) | earnings yield EBIT / EV together with return on capital |
| Quality (QMJ-style) | after Asness, Frazzini and Pedersen (2019) | profitability, one-year growth of profitability, safety |
| Value composite | classic value | mean percentile of earnings yield, book-to-price, EBITDA/EV, FCF yield, sales yield |
| Piotroski F | Piotroski (2000) | nine accounting signals |

**E30** asked the first question a factor researcher asks: *does each composite rank next month's returns in this
universe?* It was pre-registered before it was run: one test per composite, mean monthly Spearman rank correlation (IC)
with the next 20-session return, a one-sided month-block bootstrap, Holm correction over the six composites. This cell
repeats it from the staged table. SEC XBRL data begin in 2009, there are 164 month-ends (April 2009 to November 2022), and each composite's test uses the 137 to 151 months
with at least 100 stocks that have its score and a return.
""")

    code(r"""
import importlib.util
E30_FILE = NB / "fundamentals" / "e30_stock_months.parquet"
if not E30_FILE.exists():
    raise FileNotFoundError(f"{E30_FILE} is missing: run scripts/colab/stage_e30_inputs.py on the PC "
                            "(it also copies the file to the Drive input folder)")
_spec = importlib.util.spec_from_file_location("nx_extended", REPO_DIR / "src" / "nexus_alpha" / "fundamentals" / "extended.py")
XF = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(XF)                       # numpy and pandas only: the published formulas, no fitted parameter

FM = pd.read_parquet(E30_FILE)
FM["date"] = pd.to_datetime(FM["date"])
if QUICK:
    FM = FM[FM.date.dt.year >= 2018]
print(f"{len(FM):,} stock-months, {FM.ticker.nunique()} stocks, {FM.date.min().date()} to {FM.date.max().date()}")
assert FM.date.max() < pd.Timestamp("2023-01-03"), "the sealed period must never be read"

FMX = pd.concat([FM.reset_index(drop=True), XF.metrics(FM.reset_index(drop=True))], axis=1)
cover = FMX[["beneish_m", "montier_c", "gross_profitability", "greenblatt_roc", "ebit_to_ev", "peg", "dso"]].notna().mean()
print("coverage of the new metrics (share of stock-months with a value):")
display(cover.round(2).to_frame("coverage").T)
""")

    md(r"""
**What the numbers look like on one stock.** Apple on the last session of 2021: each score from the as-known snapshot.
A score that needs a tag the company does not report (MSFT's depreciation, for instance) is missing, not guessed.
""")

    code(r"""
ex = FMX[(FMX.ticker == "AAPL") & (FMX.date == FMX.date[FMX.ticker == "AAPL"].max())].iloc[0] if QUICK else \
     FMX[(FMX.ticker == "AAPL") & (FMX.date <= "2021-12-31")].sort_values("date").iloc[-1]
show = ["pe", "ev_ebitda", "ev_ebit", "peg", "gross_profitability", "greenblatt_roc", "roce", "cash_conversion", "dso",
        "dio", "quick_ratio", "beneish_m", "beneish_flag", "montier_c", "piotroski_f", "altman_z"]
display(pd.Series({k: (float(ex[k]) if k in ex and pd.notna(ex[k]) else np.nan) for k in show}, name=f"AAPL {ex.date.date()}").to_frame().T.round(3))
""")

    code(r"""
PRIMARY = ["value_composite", "quality_composite", "gross_profitability_rank", "magic_formula", "piotroski_rank",
           "beneish_safety_rank"]
H_E30, BLOCK_E30, SEED_E30, MIN_STOCKS = 20, 6, 30, 100
N_BOOT_E30 = 500 if QUICK else 10_000

def spearman_(a, b):
    ok = a.notna() & b.notna()
    return float(a[ok].rank().corr(b[ok].rank())) if ok.sum() >= MIN_STOCKS else np.nan

monthly = []
for d, g_ in FMX.groupby("date"):
    g_ = g_.set_index("ticker")
    cs = XF.cross_section(g_, sic2=g_.sic2)                     # percentile composites within this month's universe
    row = {"date": d}
    for c in PRIMARY:
        row[c] = spearman_(cs[c], g_.R)                          # the month's rank IC with the next 20-session return
    monthly.append(row)
IC = pd.DataFrame(monthly).set_index("date").sort_index()

def block_p(v, block=BLOCK_E30, n_boot=N_BOOT_E30, seed=SEED_E30):
    # one-sided: the share of moving-block bootstrap means at or below zero (as evaluation.forward_criteria.block_p_value)
    v = np.asarray(pd.Series(v).dropna(), float); n = len(v); b = min(block, n); k = int(np.ceil(n / b))
    r = np.random.default_rng(seed)
    means = np.array([np.concatenate([v[s:s + b] for s in r.integers(0, n - b + 1, k)])[:n].mean() for _ in range(n_boot)])
    return float(np.mean(means <= 0))

def holm_(p):
    items = sorted(p.items(), key=lambda kv: kv[1]); out, run = {}, 0.0
    for i, (k, v) in enumerate(items):
        run = max(run, min(1.0, (len(items) - i) * v)); out[k] = run
    return out

def ci_(v, block=BLOCK_E30, n=2000, seed=SEED_E30):
    v = np.asarray(v.dropna(), float); b = block; k = int(np.ceil(len(v) / b)); r = np.random.default_rng(seed)
    m = [np.concatenate([v[s:s + b] for s in r.integers(0, len(v) - b + 1, k)])[:len(v)].mean() for _ in range(n)]
    return np.percentile(m, [2.5, 97.5])

P1 = {c: block_p(IC[c]) for c in PRIMARY}
HO = holm_(P1)
E30_TABLE = pd.DataFrame({c: {"months": int(IC[c].notna().sum()), "mean IC": IC[c].mean(),
                              "95% low": ci_(IC[c])[0], "95% high": ci_(IC[c])[1], "share of months > 0": (IC[c].dropna() > 0).mean(),
                              "p (one-sided)": P1[c], "p (Holm)": HO[c],
                              "replicates": bool(HO[c] < 0.05 and IC[c].mean() > 0)} for c in PRIMARY}).T
display(E30_TABLE.style.format({"mean IC": "{:+.4f}", "95% low": "{:+.4f}", "95% high": "{:+.4f}", "share of months > 0": "{:.2f}",
                                "p (one-sided)": "{:.3f}", "p (Holm)": "{:.3f}"}))
""")

    code(r"""
fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
names = {"value_composite": "Value composite", "quality_composite": "Quality (QMJ-style)",
         "gross_profitability_rank": "Gross profitability", "magic_formula": "Magic formula",
         "piotroski_rank": "Piotroski F", "beneish_safety_rank": "Low Beneish M"}
y = np.arange(len(PRIMARY))[::-1]
est = E30_TABLE["mean IC"].astype(float).to_numpy(); lo = E30_TABLE["95% low"].astype(float).to_numpy(); hi = E30_TABLE["95% high"].astype(float).to_numpy()
ax[0].errorbar(est, y, xerr=[est - lo, hi - est], fmt="D", color="#1f77b4", ecolor="#9aa5b1", capsize=4)
ax[0].axvline(0, color="grey", ls=":")
ax[0].set_yticks(y); ax[0].set_yticklabels([names[c] for c in PRIMARY])
ax[0].set_xlabel("mean monthly rank IC with the next 20-session return"); ax[0].set_title("E30: no composite clears zero after Holm")
for c in ["gross_profitability_rank", "piotroski_rank", "value_composite"]:
    ax[1].plot(IC.index, IC[c].fillna(0).cumsum(), label=names[c])
ax[1].axhline(0, color="grey", ls=":"); ax[1].legend(); ax[1].set_title("cumulative monthly IC (a steady climb would be skill)")
plt.tight_layout(); plt.show()
""")

    code(r"""
DOC30 = REPO_DIR / "reports" / "e30" / "e30_results.json"
if DOC30.exists() and not QUICK:
    doc = json.loads(DOC30.read_text(encoding="utf-8"))["primary"]
    diff = {c: abs(float(E30_TABLE.loc[c, "mean IC"]) - doc[c]["mean_ic"]) for c in PRIMARY}
    E30_PARITY = pd.DataFrame({"this run": E30_TABLE["mean IC"].astype(float), "documented": {c: doc[c]["mean_ic"] for c in PRIMARY},
                               "difference": diff})
    display(E30_PARITY.style.format("{:+.5f}"))
    print("max difference:", f"{max(diff.values()):.2e}", "(float32 staging explains anything below 1e-4)")
    assert max(diff.values()) < 5e-4, "the notebook no longer reproduces the documented E30 result"
else:
    print("documented result not compared (QUICK mode, or the repository clone has no reports/e30)")
""")

    md(r"""
**Reading it.** No composite has a mean IC whose 95% interval stays above zero after correcting for six tests: gross
profitability and Piotroski are the closest, value is about zero, and the growth and Montier parts lean the wrong way
(documented in `docs/E30_RESULTS.md`). Published anomalies weaken after publication and are weaker among large, liquid
stocks (McLean and Pontiff 2016 find returns 58% lower after publication), and this universe is today's large
survivors with price-only returns, which flatters junk and penalises high-yield stocks. The scores are therefore shown
on the dashboard (tab P) as a description of a company and a risk flag, not as a signal. A different horizon (one to
twelve months) would be a new hypothesis and needs its own pre-registration.
""")


FORWARD_MD = r"""
**Two additions since v4.** (1) `v1.2` is the v1.1 recipe, unchanged, refit on 2003 to 2021-11-30 and calibrated on
2022 (E31, D-062): a single forward test of whether refitting on recent data helps. (2) `v1.1_news` logs the frozen v1.1
*with* the archived live headlines, for the stocks that have one (E29b, D-058): a first live test of the text branch.
Both are judged only after 120 sessions, and both have their own hash chains, verified below as soon as they exist.
"""

"""Notebook sections 8A-8C: the language branch, the SEC price-impact expert and the vision + NLP fusion (E28).

Imported by make_end_to_end_notebook.py, which places these cells after section 8 (the fusion network they extend).
"""
from __future__ import annotations

E28_RUN = "PENDING"          # set to the E28 run folder under experiments/ once the documented run has finished


def add(md, code, e28_run: str = E28_RUN) -> None:
    md(r"""
## 8A. The language branch: a chronologically consistent transformer reads the SEC filings (GPU)

Section 8 reads text through classical features only (TF-IDF components and Loughran-McDonald tone). Here a transformer
reads the filings themselves.

**Why ChronoBERT, not SEC-BERT or FinBERT.** A language model that has read text from the years it is evaluated on
carries look-ahead: it may "know" how a company's story ended. ChronoBERT (He, Lv, Manela and Wu, 2025) is a
ModernBERT-base model pretrained only on text dated before a cutoff. NEXUS uses the 2010-12-31 cutoff, so the encoder has
never read anything from the early-stopping (2016), validation (2017-18), calibration (2019) or test (2020-22) years.
SEC-BERT was pretrained on 10-K filings up to 2019, which would include the validation years' own filings.

**What is encoded.** Every filing that is some sample's latest 8-K (30 days) or latest 10-Q/10-K (120 days) at its
decision time: 68,853 filings of the SEC study's 500 firms. The text is the SEC study's event text (the filing's main
sections, with firm names masked as `[FIRM]`). Safe-harbour paragraphs are dropped, and the first 256 tokens are
encoded. The vector is the masked mean of the last layer, 768 numbers.

**Reproducibility.** The documented run (E28) encoded these filings on the project PC's CPU. This cell re-encodes them
on the GPU and checks that the vectors match. Every filing is re-encoded on a GPU (a few minutes on an A100 or L4);
a random 400 on a CPU, and 1,000 (GPU) or 300 (CPU) in QUICK mode. bfloat16 on a GPU gives cosine above 0.99, not bit-identical vectors.
""")
    code(r"""
NLP = NB / "nlp"
EVM = pd.read_parquet(NLP / "sec_tx_events.parquet")                     # one row per encoded filing
E_REF = np.load(NLP / "sec_tx_event_emb.npy")                             # the vectors E28 used (CPU, float16)
TXT = pd.read_parquet(NLP / "filing_texts.parquet").set_index("event_id").text.reindex(EVM.event_id).fillna("")
EVM["available_utc"] = pd.to_datetime(EVM.available_utc, utc=True)
assert EVM.available_utc.max() < pd.Timestamp(SEAL_DATE, tz="UTC"), "a sealed filing is in the inputs"
print(f"{len(EVM):,} filings, {EVM.event_type.value_counts().to_dict()}")
display(pd.DataFrame({"filing": EVM.event_type.iloc[[0, 1, 2]].values,
                      "available (UTC)": EVM.available_utc.iloc[[0, 1, 2]].astype(str).values,
                      "cleaned text, first 160 characters": [t[:160] for t in TXT.iloc[[0, 1, 2]]]}))

CHRONO, CHRONO_REV = "manelalab/chrono-bert-v1-20101231", "1520cf239e7922584ce7762af9e5040e8edd9374"
try:
    from transformers import AutoModel, AutoTokenizer
    import transformers
    assert tuple(int(x) for x in transformers.__version__.split(".")[:2]) >= (4, 48)      # ModernBERT support
except (ImportError, AssertionError):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers>=4.48"], check=True)
    from transformers import AutoModel, AutoTokenizer
TOK = AutoTokenizer.from_pretrained(CHRONO, revision=CHRONO_REV)
ENC = AutoModel.from_pretrained(CHRONO, revision=CHRONO_REV).eval().to(DEVICE)
if DEVICE == "cuda":
    ENC = ENC.to(torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)

@torch.inference_mode()
def chrono_encode(texts, bs=None, max_tokens=256):
    bs = bs or (256 if DEVICE == "cuda" else 16)
    order = np.argsort([len(t) for t in texts], kind="stable")
    out = np.zeros((len(texts), ENC.config.hidden_size), np.float32)
    for i in range(0, len(texts), bs):
        ix = order[i:i + bs]
        b = TOK([texts[k] for k in ix], padding=True, truncation=True, max_length=max_tokens, return_tensors="pt").to(DEVICE)
        hs = ENC(**b).last_hidden_state.float()
        m = b["attention_mask"].unsqueeze(-1).float()
        out[ix] = ((hs * m).sum(1) / m.sum(1).clamp(min=1)).cpu().numpy()
    return out

has_text = np.flatnonzero(EVM.n_chars_clean.to_numpy() > 0)
n_enc = (1_000 if DEVICE == "cuda" else 300) if QUICK else (len(has_text) if DEVICE == "cuda" else 400)
SEL = np.sort(rng.choice(has_text, n_enc, replace=False)) if n_enc < len(has_text) else has_text
t0 = time.time()
E_NEW = chrono_encode(TXT.iloc[SEL].tolist())
dt = time.time() - t0
ref = E_REF[SEL].astype(np.float32)
COS = (E_NEW * ref).sum(1) / (np.linalg.norm(E_NEW, axis=1) * np.linalg.norm(ref, axis=1))
print(f"encoded {len(SEL):,} filings on {DEVICE} in {dt:.0f} s ({len(SEL) / dt:.0f} per second); cosine with the "
      f"stored vectors: median {np.median(COS):.5f}, 1st percentile {np.quantile(COS, .01):.5f}")
assert np.median(COS) > 0.99, "the re-encoding does not reproduce the stored vectors"
""")
    code(r"""
# what the vectors capture: two principal components (fitted on pre-2016 filings) and point-in-time nearest neighbours
PCA_TX = np.load(NLP / "sec_tx_pca.npz", allow_pickle=True)
OKV = np.isfinite(E_REF).all(1)
Z2 = np.full((len(E_REF), 2), np.nan)
Z2[OKV] = (E_REF[OKV].astype(np.float64) - PCA_TX["mean"]) @ PCA_TX["components"][:2].T
ITEM = {"2.02": "8-K results", "5.02": "8-K officers", "1.01": "8-K agreement", "8.01": "8-K other events",
        "7.01": "8-K Reg FD", "2.01": "8-K acquisition"}
KIND = np.where(EVM.event_type == "8K", EVM.item_main.map(ITEM).fillna("8-K other items"),
                EVM.event_type.str.replace("periodic_", "").str.replace("10", "10-"))
fig, ax = plt.subplots(figsize=(8, 5.2))
sub = rng.choice(np.flatnonzero(OKV), min(20_000, int(OKV.sum())), replace=False)
for k, c in zip(pd.Series(KIND[sub]).value_counts().index[:8], plt.cm.tab10.colors):
    m = sub[KIND[sub] == k]
    ax.scatter(Z2[m, 0], Z2[m, 1], s=3, alpha=.35, color=c, label=k)
ax.set_title("ChronoBERT filing vectors: the first two principal components")
ax.set_xlabel("component 1"); ax.set_ylabel("component 2")
ax.legend(markerscale=5, fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.14), frameon=False)
plt.tight_layout(); plt.show()

q = int(np.flatnonzero(OKV & (EVM.item_main == "2.02").to_numpy() & (EVM.available_utc.dt.year == 2018).to_numpy())[0])
past = np.flatnonzero(OKV & (EVM.available_utc < EVM.available_utc.iloc[q]).to_numpy())      # only earlier filings
V = E_REF[past].astype(np.float32); qv = E_REF[q].astype(np.float32)
sim = V @ qv / (np.linalg.norm(V, axis=1) * np.linalg.norm(qv))
nn_ix = past[np.argsort(-sim)[:3]]
display(Markdown(f"**Nearest earlier filings to an 8-K results filing of {EVM.available_utc.iloc[q].date()}** "
                 "(cosine of the ChronoBERT vectors; firm names are masked in the SEC study's text)"))
display(pd.DataFrame({"filing": [KIND[q]] + list(KIND[nn_ix]),
                      "date": [str(EVM.available_utc.iloc[i].date()) for i in [q] + list(nn_ix)],
                      "cosine": [1.0] + [float(s) for s in np.sort(sim)[::-1][:3]],
                      "text, first 140 characters": [TXT.iloc[i][:140] for i in [q] + list(nn_ix)]}))
""")

    md(r"""
## 8B. The SEC price-impact expert: what the SEC study learned, as evidence for NEXUS

The SEC filings study labelled each filing by what happened to the stock after it became public. `T_trade` enters at
the first tradable price after the decision time; `label3` is UP, SIDEWAYS or DOWN of the abnormal return at 1 and 5
sessions. Its headline finding was that filings predict **whether** a stock moves (AUC 0.635 on its locked test), not
**which way** (0.494).

Here that knowledge becomes a model NEXUS can use:
* **Model.** LightGBM on the study's own event features. Two families are left out: `meta_year` (review finding N2),
  and the accounting features (finding N1: their availability rule is wrong before 2009).
* **Fitting.** Fitted on events before 2016; trees are chosen on 2016 events.
* **Predictions.** Every event gets one. Events before 2016 are predicted out of fold, so no NEXUS training row
  sees a prediction fitted on its own event.
* **Sealed period.** No label with an exit after 2018 is read.

The cell refits the expert and checks its AUCs and predictions against the documented build.
""")
    code(r"""
TE_ = pd.read_parquet(NLP / "sec_expert_table.parquet")
SI_META = json.loads((NLP / "sec_expert_meta.json").read_text(encoding="utf-8"))
SI_FEAT = SI_META["features"]
SI_PARAMS = {**SI_META["params"], "num_threads": os.cpu_count() or 2}       # the documented fixed parameters
at = pd.to_datetime(TE_.available_utc, utc=True)
F16, F17 = pd.Timestamp("2016-01-01", tz="UTC"), pd.Timestamp("2017-01-01", tz="UTC")
XE = TE_[SI_FEAT].to_numpy(np.float32)
SI_MODELS, SI_AUC, SI_NEW = {}, [], {}
for h in (1, 5):
    y = TE_[f"y_h{h}"].to_numpy(); ex = pd.to_datetime(TE_[f"exit_h{h}"], utc=True)
    tr_ = ((at < F16) & (ex < F16)).to_numpy() & np.isfinite(y)
    es_ = ((at >= F16) & (at < F17) & (ex < F17)).to_numpy() & np.isfinite(y)
    d_tr = lgb.Dataset(XE[tr_], y[tr_].astype(int)); d_es = lgb.Dataset(XE[es_], y[es_].astype(int), reference=d_tr)
    m = lgb.train(SI_PARAMS, d_tr, num_boost_round=2000, valid_sets=[d_es],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
    SI_MODELS[h] = m
    va_ = at.dt.year.isin([2017, 2018]).to_numpy() & np.isfinite(y)
    p = m.predict(XE[va_], num_iteration=m.best_iteration)
    SI_NEW[h] = (np.flatnonzero(va_), p)
    mv = (y[va_] != 1).astype(int); ud = y[va_] != 1
    SI_AUC.append({"horizon": f"{h} session(s)", "trees": m.best_iteration, "training events": int(tr_.sum()),
                   "valid events (2017-18)": int(va_.sum()), "AUC: does it move?": roc_auc_score(mv, 1 - p[:, 1]),
                   "AUC: up vs down": roc_auc_score((y[va_][ud] == 2).astype(int), p[ud, 2] - p[ud, 0]),
                   "documented AUC (move)": SI_META["diagnostics"][f"h{h}_valid_2017_2018"]["auc_move"]})
display(pd.DataFrame(SI_AUC).style.format(precision=4))

DOC = pd.read_parquet(NLP / "sec_impact_event_predictions.parquet")
ix1, p1 = SI_NEW[1]
diff = np.abs(p1[:, 1] - DOC.pside_h1.reindex(TE_.index[ix1]).to_numpy())
print(f"h1 P(sideways) on 2017-18 events, refit here vs the documented build: median |diff| {np.median(diff):.1e}, "
      f"max {diff.max():.1e} (LightGBM is deterministic within one version and machine type)")

imp = pd.Series(SI_MODELS[1].feature_importance("gain"), index=SI_FEAT).sort_values().tail(15)
fig, axs = plt.subplots(1, 2, figsize=(13, 4.2))
axs[0].barh(imp.index, imp / imp.sum(), color="#8c564b"); axs[0].set_title("expert, 1 session: top features (gain share)")
yv = TE_["y_h1"].to_numpy()[ix1]
dec = pd.qcut(1 - p1[:, 1], 10, labels=False, duplicates="drop")
rate = pd.Series(yv != 1).groupby(dec).mean()
axs[1].plot(rate.index + 1, rate.to_numpy(), marker="o", color="#8c564b")
axs[1].set_xlabel("decile of predicted P(move), 2017-18 events"); axs[1].set_ylabel("share that moved")
axs[1].set_title("does the expert rank filings by how much they move the price?")
plt.tight_layout(); plt.show()
""")

    md(rf"""
## 8C. Vision + NLP: one fusion network over charts, indicators and language (E28)

E28 was pre-registered before any of its features existed (`docs/V3_RESEARCH.md`, decision D-053). It adds three
language blocks to section 8's six:

| Block | What it is | Present for |
|---|---|---|
| `news_tx` | FinBERT tone and a 16-component embedding of the company's headlines before the decision time | samples with news |
| `sec_tx` | ChronoBERT vectors (section 8A) of the latest 8-K and 10-Q/10-K, 32 components each | samples with a recent filing |
| `sec_impact` | the SEC expert's P(move) and P(up) - P(down) for the latest filing (section 8B) | a filing within 14 days |

**The test.** MM (all nine blocks) is compared with BASE (section 8's network: the same settings and seeds). It is also
compared with PLACEBO: MM with the three new blocks moved to another stock of the same date. The placebo keeps their
distribution but breaks their link to the stock. The language blocks are **kept** only if both validation intervals
lie above zero; the test years are reported once, after that decision.

This cell retrains MM and PLACEBO here; BASE is section 8's model. The documented run (`experiments/{e28_run}`) used the
same design with the package code (`scripts/experiments/e28_vision_nlp_fusion.py`). Its table follows, with the
descriptive variants.
""")
    code(r"""
fb = pd.read_parquet(NB / "features" / "finbert_features.parquet").set_index("sample_id")
stx = pd.read_parquet(NB / "features" / "sec_tx_features.parquet")
sim_ = pd.read_parquet(NB / "features" / "sec_impact_features.parquet")
J = FR[["sample_id"]].join(fb, on="sample_id").join(stx, on="sample_id").join(sim_, on="sample_id")
NEWB = {"news_tx": ([c for c in fb.columns if c != "fb_n"], J.fb_n.fillna(0).to_numpy() > 0),
        "sec_tx": ([c for c in stx.columns if c != "sec_tx_present"], J.sec_tx_present.fillna(0).to_numpy() > 0),
        "sec_impact": ([c for c in sim_.columns if c != "si_present"], J.si_present.fillna(0).to_numpy() > 0)}
blocks2, present2 = dict(blocks), dict(present)
for n, (cols, pr) in NEWB.items():
    pr = pr.astype(np.float32)
    blocks2[n] = standardise(J[cols].to_numpy(np.float64), TR[pr[TR] > 0]) * pr[:, None]
    present2[n] = pr
print({n: (blocks2[n].shape[1], round(float(present2[n][masks["valid"]].mean()), 3)) for n in NEWB},
      "(dimension, share present on valid)")

PERM = np.arange(len(FR)); r_pl = np.random.default_rng(11)                 # placebo: within-date shuffle across stocks
for idx in FR.groupby("asof_date").indices.values():
    PERM[idx] = idx[r_pl.permutation(len(idx))]

def block_set(names, placebo=False):
    bl = {n: (blocks2[n][PERM] if placebo and n in NEWB else blocks2[n]) for n in names}
    mk = np.stack([(present2[n][PERM] if placebo and n in NEWB else present2[n]) for n in names], 1)
    return bl, list(names), mk

MM_NAMES = list(NAMES) + list(NEWB)
SETS = {"MM": block_set(MM_NAMES), "PLACEBO": block_set(MM_NAMES, placebo=True)}
FUS2, HIST2, PV = {}, {}, {}
t0 = time.time()
for k, B in SETS.items():
    for s in SEEDS:
        FUS2[(k, s)], HIST2[(k, s)] = train_fusion(s, epochs=2 if QUICK else 30, B=B)
    PV[k] = np.mean([predict_probs(FUS2[(k, s)], EVI, B=B) for s in SEEDS], 0)
    print(f"{k}: {len(SEEDS)} seeds trained ({time.time() - t0:.0f} s)")
PV["BASE"] = PRED["fusion"]
PRED["fusion_mm"] = PV["MM"]

POSV = {i: k for k, i in enumerate(EVI)}
def rows_of(mask):
    return np.array([POSV[i] for i in np.flatnonzero(mask) if i in POSV])
YE_ = Y[EVI]
def mean_ll6(P, ix):
    return float(np.mean([log_loss(YE_[ix[YE_[ix, j] >= 0], j], np.clip(P[ix[YE_[ix, j] >= 0], j], 1e-7, 1),
                                   labels=[0, 1, 2]) for j in range(6)]))

def paired_gain(a, b, ix, n_boot=300 if QUICK else 1000, seed=7):
    # mean LL(b) - mean LL(a) over the six horizons; month-block bootstrap (positive = a better)
    months = pd.to_datetime(FR.asof_date.to_numpy()[EVI[ix]]).to_period("M").astype(str)
    groups = pd.Series(np.arange(len(ix))).groupby(months).indices
    keys = list(groups); r = np.random.default_rng(seed)
    est = mean_ll6(PV[b], ix) - mean_ll6(PV[a], ix)
    bs = []
    for _ in range(n_boot):
        pick = np.concatenate([groups[keys[k]] for k in r.integers(0, len(keys), len(keys))])
        bs.append(mean_ll6(PV[b], ix[pick]) - mean_ll6(PV[a], ix[pick]))
    return {"gain": est, "low": float(np.quantile(bs, .025)), "high": float(np.quantile(bs, .975))}

IXV, IXT = rows_of(masks["valid"]), rows_of(masks["test"])
G28 = pd.DataFrame({("valid 2017-18", "MM vs BASE"): paired_gain("MM", "BASE", IXV),
                    ("valid 2017-18", "MM vs PLACEBO"): paired_gain("MM", "PLACEBO", IXV),
                    ("test 2020-22 (after the decision)", "MM vs BASE"): paired_gain("MM", "BASE", IXT),
                    ("test 2020-22 (after the decision)", "MM vs PLACEBO"): paired_gain("MM", "PLACEBO", IXT)}).T
KEPT_HERE = bool((G28.loc["valid 2017-18", "low"] > 0).all())
display(Markdown(f"**Mean six-horizon log-loss gain of the vision + NLP network (positive = better; 95% month-block "
                 f"intervals).** Keep rule on validation: **{'KEPT' if KEPT_HERE else 'NOT KEPT'}** in this run."))
display(G28.style.format("{:+.5f}"))
""")
    code(r"""
# where the network looks, the loss curves and the confusion matrices of the vision + NLP model
B_MM = SETS["MM"]
with torch.no_grad():
    xs, mk, _, _ = to_dev(VAL[:20000], B_MM)
    _, _, g = FUS2[("MM", SEEDS[0])](xs, mk)
GW = pd.Series(g.float().mean(0).cpu().numpy(), index=MM_NAMES)
GROUP = {"img_W": "vision", "img_D": "vision", "img_H": "vision", "num": "indicators", "news": "language",
         "sec": "language", "news_tx": "language", "sec_tx": "language", "sec_impact": "language"}
fig, axs = plt.subplots(1, 3, figsize=(15, 3.8))
colors = {"vision": "#1f77b4", "indicators": "#2ca02c", "language": "#9467bd"}
axs[0].bar(GW.index, GW.to_numpy(), color=[colors[GROUP[n]] for n in GW.index])
axs[0].set_title("mean gate weight per block (valid rows)"); axs[0].tick_params(axis="x", rotation=45)
share = GW.groupby(GROUP).sum()
axs[1].pie(share.to_numpy(), labels=[f"{k}\n{v:.0%}" for k, v in share.items()], colors=[colors[k] for k in share.index],
           startangle=90, wedgeprops={"width": .45})
axs[1].set_title("gate share by modality")
h = HIST2[("MM", SEEDS[0])]
for c, col in (("train CE", "#1f77b4"), ("inner CE (early stop)", "#ff7f0e"), ("valid CE (shown only)", "#2ca02c")):
    axs[2].plot(h.epoch, h[c], marker="o", ms=3, label=c, color=col)
axs[2].set_title(f"vision + NLP network, seed {SEEDS[0]}: loss curves"); axs[2].set_xlabel("epoch")
axs[2].legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, frameon=False)
plt.tight_layout(); plt.show()
print("Gate weights are a model diagnostic, not a causal explanation: they show what the network reads, not why prices move.")

fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))
for ax, hz in zip(axs, ("1D", "5D", "20D")):
    j = H6.index(hz); ok = YE_[IXT, j] >= 0
    cm = confusion_matrix(YE_[IXT[ok], j], PV["MM"][IXT[ok], j].argmax(1), labels=[0, 1, 2], normalize="true")
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    for a in range(3):
        for b in range(3):
            ax.text(b, a, f"{cm[a, b]:.2f}", ha="center", va="center", color="white" if cm[a, b] > .55 else "black")
    ax.set_xticks(range(3), CLASSES); ax.set_yticks(range(3), CLASSES)
    ax.set_title(f"vision + NLP, {hz}, test (row-normalised)"); ax.set_xlabel("predicted"); ax.set_ylabel("true")
plt.tight_layout(); plt.show()
""")
    code(rf"""
# the documented E28 run (package code, the PC): every configuration, the decision and the test report
E28_DIR = REPO_DIR / "experiments" / "{e28_run}"
if (E28_DIR / "metrics.json").exists():
    M28 = json.loads((E28_DIR / "metrics.json").read_text(encoding="utf-8"))
    res = M28["results"]
    T28 = pd.DataFrame({{k: {{"blocks": ", ".join(v["modalities"]), "valid mean log loss": v["valid_mean_log_loss"],
                            "test mean log loss": v["test_mean_log_loss"]}} for k, v in res.items()}}).T
    display(Markdown(f"**Documented run: decision {{M28['decision']}}** (rule: {{M28['decision_rule']}})"))
    display(T28.style.format({{"valid mean log loss": "{{:.5f}}", "test mean log loss": "{{:.5f}}"}}))
    D = pd.DataFrame({{**{{f"valid: {{k}}": v for k, v in M28["validation"].items()}},
                      **{{f"valid (reported): {{k}}": v for k, v in M28["reported_validation"].items()}},
                      **{{f"test: {{k}}": v for k, v in M28["test_after_decision"].items()}}}}).T
    D = D.assign(low=D.ci95.map(lambda c: c[0]), high=D.ci95.map(lambda c: c[1]))[["estimate", "low", "high"]]
    display(D.style.format("{{:+.5f}}"))
    RIC = pd.DataFrame({{k: {{"BASE": v["BASE"]["estimate"], "MM": v["MM"]["estimate"],
                            "MM - BASE": v["MM_minus_BASE"]["estimate"], "low": v["MM_minus_BASE"]["ci95"][0],
                            "high": v["MM_minus_BASE"]["ci95"][1]}} for k, v in M28["rank_ic"].items()}}).T
    display(Markdown("**Daily cross-sectional rank IC of P(up) - P(down)** (reported; it decides nothing)"))
    display(RIC.style.format("{{:+.4f}}"))
else:
    print("the documented E28 run is not in this copy of the repository")
""")

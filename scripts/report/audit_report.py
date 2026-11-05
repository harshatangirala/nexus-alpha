"""Build the visual audit report: reports/figures/*.png and reports/NEXUS_Audit_Report.html (figures embedded).

Every number and curve is read from saved artifacts (experiments/*/metrics.json, model histories, predictions,
reports/*). A figure that cannot be drawn is listed with its error in the report's status table, never silently dropped.
Run: .venv/Scripts/python -I scripts/report/audit_report.py
"""
from __future__ import annotations

import base64
import glob
import io
import json
import sys
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from nexus_alpha import paths  # noqa: E402

FIG = ROOT / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
H = ["1D", "2D", "5D", "10D", "20D", "1M"]
CLS = ["DOWN", "SIDEWAYS", "UP"]
C = {"train": "#4C72B0", "inner": "#DD8452", "valid": "#55A868", "test": "#C44E52", "prior": "#8C8C8C", "calib": "#8172B3"}
plt.rcParams.update({"figure.dpi": 110, "font.size": 9, "axes.grid": True, "grid.alpha": 0.3})


def latest(prefix):
    for r in sorted((ROOT / "experiments").glob(f"*_{prefix}_*"), reverse=True):
        if not (r / "manifest.json").exists():
            continue
        m = json.loads((r / "manifest.json").read_text(encoding="utf-8"))
        if m.get("status") == "COMPLETED" and (r / "metrics.json").exists():
            return r.name, json.loads((r / "metrics.json").read_text(encoding="utf-8"))
    return None, None


SECTIONS, STATUS = [], []


def figure(title, note=""):
    def deco(fn):
        def wrapped(*a, **k):
            name = fn.__name__
            try:
                fig = fn(*a, **k)
                if fig is None:
                    STATUS.append((name, title, "skipped: inputs not available"))
                    return
                p = FIG / f"{name}.png"
                fig.savefig(p, bbox_inches="tight")
                buf = io.BytesIO()
                fig.savefig(buf, format="png", bbox_inches="tight")
                plt.close(fig)
                SECTIONS.append(("fig", title, note, base64.b64encode(buf.getvalue()).decode()))
                STATUS.append((name, title, "ok"))
            except Exception as e:
                STATUS.append((name, title, f"FAILED: {type(e).__name__}: {e}"))
                traceback.print_exc()
        return wrapped
    return deco


def table(title, df: pd.DataFrame, note=""):
    SECTIONS.append(("table", title, note, df.to_html(index=False, float_format=lambda x: f"{x:.4f}", border=0, classes="t")))


# ------------------------------------------------------------------------------------------------ data
def frame():
    return pd.read_parquet(paths.ARTIFACTS / "samples" / "s500_frame.parquet",
                           columns=["sample_id", "asof_date", "block"] + [f"y_{h}" for h in H])


@figure("Class distribution by horizon (train vs test)", "Registered vol-scaled rule; SIDEWAYS share is stable across regimes.")
def f01_class_distribution(fr):
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.4), sharey=True)
    for a, b in zip(ax, ("train", "test")):
        d = fr[fr.block == b]
        shares = np.array([[(d[f"y_{h}"] == k).sum() / d[f"y_{h}"].notna().sum() for k in range(3)] for h in H])
        bottom = np.zeros(len(H))
        for k, col in zip(range(3), ("#C44E52", "#8C8C8C", "#55A868")):
            a.bar(H, shares[:, k], bottom=bottom, color=col, label=CLS[k])
            bottom += shares[:, k]
        a.set_title(b)
    ax[0].legend(ncol=3, fontsize=8)
    ax[0].set_ylabel("share")
    return fig


@figure("E01 baselines: test log loss by horizon (lower is better)")
def f02_e01_logloss(m):
    names = ["prior", "momentum_lr", "lgbm_structured", "lgbm_x33", "lgbm_struct_x33", "lgbm_struct_x33_mkt"]
    fig, ax = plt.subplots(figsize=(11, 3.6))
    w = 0.13
    for i, n in enumerate(names):
        ax.bar(np.arange(len(H)) + i * w, [m[h]["test"][n]["log_loss"] for h in H], w, label=n)
    ax.set_xticks(np.arange(len(H)) + 2.5 * w, H)
    ax.set_ylim(1.0, 1.1)
    ax.legend(fontsize=7, ncol=3)
    ax.set_ylabel("log loss")
    return fig


@figure("E01 what is predictable: AUROC SIDEWAYS-vs-move and UP-vs-DOWN (test)", "0.5 = chance")
def f03_e01_auc(m):
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.4), sharey=True)
    for n in ("momentum_lr", "lgbm_struct_x33", "lgbm_struct_x33_mkt"):
        ax[0].plot(H, [m[h]["test"][n].get("auroc_SIDEWAYS") for h in H], "o-", label=n)
        ax[1].plot(H, [m[h]["test"][n].get("auroc_up_vs_down") for h in H], "o-", label=n)
    for a, t in zip(ax, ("SIDEWAYS vs move", "UP vs DOWN")):
        a.axhline(0.5, color="k", lw=0.8)
        a.set_title(t)
    ax[0].legend(fontsize=7)
    return fig


@figure("E02 chart branch: mean validation log loss by backbone and view (images only)", "Random-weight control in the last row")
def f04_e02_heatmap(m):
    bbs = ["convnext_tiny", "vit_small_patch14_dinov2", "resnet18", "ft_convnext_tiny", "resnet18_random_s0"]
    vs = ["W", "D", "H", "WDH"]
    M = np.array([[np.mean([m[h]["valid"][f"img_lr__{b}__{v}"]["log_loss"] for h in H]) for v in vs] for b in bbs])
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    im = ax.imshow(M, cmap="viridis_r")
    ax.set_xticks(range(4), vs)
    ax.set_yticks(range(len(bbs)), bbs)
    for i in range(len(bbs)):
        for j in range(4):
            ax.text(j, i, f"{M[i, j]:.4f}", ha="center", va="center", color="w", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.8)
    return fig


@figure("E03 ablations: mean six-horizon log loss (validation and test)")
def f05_e03_ablations(m):
    rows = [(n, v["valid"]["mean_log_loss"], v["test_raw"]["mean_log_loss"]) for n, v in m.items()
            if isinstance(v, dict) and "valid" in v and "test_raw" in v and n != "a13_full"]
    rows.sort(key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(10, 6))
    y = np.arange(len(rows))
    ax.barh(y - 0.2, [r[1] for r in rows], 0.4, color=C["valid"], label="valid")
    ax.barh(y + 0.2, [r[2] for r in rows], 0.4, color=C["test"], label="test")
    ax.set_yticks(y, [r[0] for r in rows], fontsize=8)
    ax.set_xlim(1.05, 1.08)
    ax.invert_yaxis()
    ax.legend()
    return fig


@figure("E03 registered hypotheses: paired month-block bootstrap (positive = improvement)", "None supported after Holm correction")
def f06_e03_forest(m):
    t = m["hypotheses"]["tests"]
    fig, ax = plt.subplots(figsize=(7, 3))
    for i, (k, v) in enumerate(t.items()):
        ax.plot(v["ci95"], [i, i], "k-")
        ax.plot(v["estimate"], i, "o", color=C["test"])
    ax.axvline(0, color="grey", lw=1)
    ax.set_yticks(range(len(t)), list(t))
    ax.set_xlabel("difference in mean log loss")
    return fig


@figure("E03 missing-modality stress test (test block): increase in log loss when a modality is masked")
def f07_e03_stress(m):
    s = m["missing_modality_stress"]
    fig, ax = plt.subplots(figsize=(6, 2.8))
    ax.bar([k.replace("without_", "") for k in s], [v["delta_vs_full"] for v in s.values()], color=C["inner"])
    ax.set_ylabel("Δ log loss")
    return fig


@figure("E03 training curves: training loss and early-stopping (2016) cross-entropy per epoch, every model")
def f08_e03_curves(run_id):
    files = sorted(glob.glob(str(paths.ARTIFACTS / "models" / run_id / "*" / "history.json")))
    if not files:
        return None
    n = len(files)
    cols = 5
    fig, axs = plt.subplots(int(np.ceil(n / cols)), cols, figsize=(15, 2.4 * np.ceil(n / cols)))
    for ax, f in zip(axs.ravel(), files):
        h = json.load(open(f))["hist"]
        ep = [r["epoch"] for r in h]
        ax.plot(ep, [r["train_loss"] for r in h], color=C["train"], label="train (incl. quantile loss)")
        ax2 = ax.twinx()
        ax2.plot(ep, [r["inner_ce"] for r in h], color=C["inner"], label="inner CE")
        ax.set_title(Path(f).parent.name, fontsize=8)
    for ax in axs.ravel()[n:]:
        ax.axis("off")
    fig.tight_layout()
    return fig


@figure("E06 loss and validation curves: training subset, early-stopping (2016) and validation (2017-18) cross-entropy",
        "Cosine schedule; the dashed line marks the restored (best early-stopping) epoch")
def f09_e06_curves(m):
    names = [n for n in m if isinstance(m[n], dict) and "history" in m[n]]
    cols = 4
    fig, axs = plt.subplots(int(np.ceil(len(names) / cols)), cols, figsize=(15, 2.7 * np.ceil(len(names) / cols)))
    for ax, n in zip(axs.ravel(), names):
        hst = m[n]["history"]
        ep = [r["epoch"] for r in hst]
        for k, c in (("train_ce_eval", C["train"]), ("inner_ce", C["inner"]), ("valid_ce", C["valid"])):
            ax.plot(ep, [r[k] for r in hst], color=c, label=k.replace("_ce", "").replace("_eval", ""))
        ax.axvline(m[n]["best_epoch"], ls="--", color="k", lw=0.8)
        ax.set_title(n, fontsize=8)
    axs.ravel()[0].legend(fontsize=7)
    for ax in axs.ravel()[len(names):]:
        ax.axis("off")
    fig.tight_layout()
    return fig


@figure("E06 search: validation log loss of the learning-rate and capacity/regularisation configurations")
def f10_e06_search(m):
    names = [n for n in m if n.startswith(("lr_", "cap_", "reg_"))]
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.bar(names, [m[n]["valid"] for n in names], color=C["valid"])
    ax.bar(names, [m[n]["test"] - m[n]["valid"] for n in names], bottom=[m[n]["valid"] for n in names], alpha=0)
    ax.set_ylim(min(m[n]["valid"] for n in names) - 0.002, max(m[n]["valid"] for n in names) + 0.002)
    ax.set_ylabel("valid mean log loss")
    plt.setp(ax.get_xticklabels(), rotation=20)
    return fig


@figure("E06 seed stability: five seeds of the best configuration and of the numbers-only head, plus seed ensembles")
def f11_e06_seeds(m):
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.2))
    for a, split in zip(ax, ("valid", "test")):
        for grp, c in (("best", C["valid"]), ("num", C["inner"])):
            vals = [m[f"{grp}_seed{s}"][split] for s in (7, 11, 23, 42, 101)]
            a.plot(range(5), vals, "o", color=c, label=f"{grp} seeds")
            a.axhline(m[f"{grp}_seed_ensemble"][split], color=c, ls="--", label=f"{grp} ensemble")
        a.set_title(split)
        a.set_xticks(range(5), ["7", "11", "23", "42", "101"])
    ax[0].legend(fontsize=7)
    return fig


@figure("E06 loss landscape around the best solution (validation cross-entropy)",
        "Left: straight path between two seeds' solutions (alpha 0 and 1). Right: 2-D slice along two filter-normalised random directions.")
def f12_e06_landscape(m, run_id):
    L = m.get("landscape")
    if not L:
        return None
    surf = np.load(paths.ARTIFACTS / "predictions" / run_id / "landscape_2d.npy")
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ax[0].plot([p["alpha"] for p in L["path_1d"]], [p["valid_ce"] for p in L["path_1d"]], "o-", color=C["valid"])
    ax[0].set_xlabel("alpha")
    ax[0].set_ylabel("valid CE")
    g = np.array(L["grid"])
    cs = ax[1].contourf(g, g, surf.T, levels=25, cmap="viridis")
    ax[1].plot(0, 0, "r*", ms=12)
    fig.colorbar(cs, ax=ax[1])
    return fig


@figure("E07 LightGBM validation curves: multi-class log loss per boosting iteration (train, 2016 early-stopping, 2017-18 validation)",
        "Vertical line = iteration chosen on the 2016 block; the train curve keeps falling while validation flattens or rises (overfitting beyond it)")
def f13_e07_curves(m):
    fig, axs = plt.subplots(2, 3, figsize=(14, 6.5))
    for ax, h in zip(axs.ravel(), H):
        c = m["curves"][h]
        for k in ("train", "inner", "valid"):
            ax.plot(c[k], color=C[k], label=k)
        ax.axvline(c["best_iteration_inner"], color="k", ls="--", lw=0.8)
        ax.set_title(f"{h}: best iteration {c['best_iteration_inner']}")
        ax.set_ylim(min(c["train"]) - 0.01, max(c["valid"][:5]) + 0.005)
    axs[0, 0].legend(fontsize=7)
    fig.tight_layout()
    return fig


@figure("E07 Optuna search history (mean validation log loss over 1D/5D/20D)", "Trial 0 = the E01 default parameters")
def f14_e07_optuna():
    p = ROOT / "reports" / "e07_optuna_trials.csv"
    if not p.exists():
        return None
    t = pd.read_csv(p)
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(t.number, t.value, "o", color=C["valid"], label="trial")
    ax.plot(t.number, t.value.cummin(), "-", color="k", label="best so far")
    ax.legend()
    ax.set_xlabel("trial")
    return fig


@figure("E07 learning curves: validation log loss vs number of training stocks", "More data still helps or has flattened?")
def f15_e07_learning(m):
    fig, ax = plt.subplots(figsize=(7, 3))
    for h, d in m["learning_curves"].items():
        fr_ = sorted(d, key=float)
        ax.plot([d[f]["train_rows"] for f in fr_], [d[f]["valid_log_loss"] for f in fr_], "o-", label=h)
    ax.set_xscale("log")
    ax.set_xlabel("training rows")
    ax.legend()
    return fig


@figure("E07 feature importance (share of total gain, mean over horizons): top 25")
def f16_e07_importance(m):
    imp = pd.DataFrame(m["importance"]).fillna(0).mean(axis=1).sort_values(ascending=False).head(25)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(imp.index[::-1], imp.values[::-1], color=C["train"])
    return fig


@figure("E08 candidate models: mean six-horizon log loss on validation (selection) and test (diagnostic)")
def f17_e08_candidates(m):
    c = m["candidates"]
    names = sorted(c, key=lambda n: c[n]["valid"])
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(names))
    ax.plot(x, [c[n]["valid"] for n in names], "o-", color=C["valid"], label="valid")
    ax.plot(x, [c[n]["test"] for n in names], "s-", color=C["test"], label="test")
    ax.set_xticks(x, names, rotation=40, ha="right", fontsize=8)
    ax.legend()
    return fig


def final_probs():
    rid, m = latest("e08_final_model")
    if not rid:
        return None, None, None
    P = np.load(paths.ARTIFACTS / "predictions" / rid / "final_deployed_calibrated__probs.npy")
    return rid, m, P


@figure("Final model: reliability diagrams on the test years (UP and DOWN probabilities)",
        "Equal-count bins (10 per panel, about 3,600 test rows each); error bars = 95% binomial interval; diagonal = perfect calibration")
def f18_reliability(fr, P):
    te = (fr.block == "test").to_numpy()
    fig, axs = plt.subplots(2, 6, figsize=(17, 5.8))
    for j, h in enumerate(H):
        m = te & fr[f"y_{h}"].notna().to_numpy()
        y = fr.loc[m, f"y_{h}"].to_numpy(int)
        for r, k in enumerate((2, 0)):
            pk = P[m, j, k]
            q = np.quantile(pk, np.linspace(0, 1, 11))
            b = np.clip(np.searchsorted(q[1:-1], pk, side="right"), 0, 9)
            mp = np.array([pk[b == i].mean() for i in range(10)])
            of = np.array([(y[b == i] == k).mean() for i in range(10)])
            n = np.array([(b == i).sum() for i in range(10)])
            se = 1.96 * np.sqrt(of * (1 - of) / np.maximum(n, 1))
            ax = axs[r, j]
            lo, hi = min(mp.min(), of.min()) - 0.03, max(mp.max(), of.max()) + 0.03
            ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
            ax.errorbar(mp, of, yerr=se, fmt="o-", ms=3, color=C["test"], ecolor="#999", capsize=2)
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.set_title(f"{h} {CLS[k]}", fontsize=8)
            if j == 0:
                ax.set_ylabel("observed frequency", fontsize=8)
            if r == 1:
                ax.set_xlabel("mean predicted", fontsize=8)
    fig.tight_layout()
    return fig


@figure("Final model: ROC curves on the test years (one-vs-rest per class, and UP vs DOWN among non-sideways outcomes)")
def f19_roc(fr, P):
    te = (fr.block == "test").to_numpy()
    fig, axs = plt.subplots(1, 6, figsize=(17, 3.2))
    for j, h in enumerate(H):
        m = te & fr[f"y_{h}"].notna().to_numpy()
        y = fr.loc[m, f"y_{h}"].to_numpy(int)
        ax = axs[j]
        for k, c in zip(range(3), ("#C44E52", "#8C8C8C", "#55A868")):
            fpr, tpr, _ = roc_curve((y == k).astype(int), P[m, j, k])
            ax.plot(fpr, tpr, color=c, label=CLS[k])
        nd = y != 1
        fpr, tpr, _ = roc_curve((y[nd] == 2).astype(int), P[m, j, 2][nd] - P[m, j, 0][nd])
        ax.plot(fpr, tpr, "k:", label="UP vs DOWN")
        ax.plot([0, 1], [0, 1], color="grey", lw=0.6)
        ax.set_title(h)
    axs[0].legend(fontsize=6)
    fig.tight_layout()
    return fig


@figure("Final model: precision-recall curves on the test years (dashed = base rate)")
def f20_pr(fr, P):
    te = (fr.block == "test").to_numpy()
    fig, axs = plt.subplots(1, 6, figsize=(17, 3.2))
    for j, h in enumerate(H):
        m = te & fr[f"y_{h}"].notna().to_numpy()
        y = fr.loc[m, f"y_{h}"].to_numpy(int)
        for k, c in zip(range(3), ("#C44E52", "#8C8C8C", "#55A868")):
            pr, rc, _ = precision_recall_curve((y == k).astype(int), P[m, j, k])
            axs[j].plot(rc, pr, color=c, label=CLS[k])
            axs[j].axhline((y == k).mean(), color=c, ls="--", lw=0.7)
        axs[j].set_title(h)
    axs[0].legend(fontsize=6)
    fig.tight_layout()
    return fig


@figure("Final model: row-normalised confusion matrices on the test years (argmax class)")
def f21_confusion(m_final):
    fig, axs = plt.subplots(1, 6, figsize=(17, 3))
    for ax, h in zip(axs, H):
        cm = np.array(m_final["final"]["by_horizon"][h]["confusion"], float)
        cmn = cm / cm.sum(1, keepdims=True)
        ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        for i in range(3):
            for k in range(3):
                ax.text(k, i, f"{cmn[i, k]:.2f}", ha="center", va="center", fontsize=7)
        ax.set_xticks(range(3), ["D", "S", "U"])
        ax.set_yticks(range(3), ["D", "S", "U"])
        ax.set_title(h)
    return fig


@figure("Final model by test year: log loss and UP-vs-DOWN AUROC")
def f22_by_year(m_final):
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.2))
    for h in H:
        by = m_final["final"]["by_horizon"][h]["by_year"]
        ys = sorted(by)
        ax[0].plot(ys, [by[y]["log_loss"] for y in ys], "o-", label=h)
        ax[1].plot(ys, [by[y]["auroc_up_vs_down"] for y in ys], "o-", label=h)
    ax[1].axhline(0.5, color="k", lw=0.8)
    ax[0].set_title("log loss")
    ax[1].set_title("UP vs DOWN AUROC")
    ax[0].legend(fontsize=7, ncol=2)
    return fig


@figure("E05 walk-forward 2008-2022: UP-vs-DOWN AUROC and log-loss gain over the prior per test year")
def f23_walk_forward():
    p = ROOT / "reports" / "walk_forward" / "e05_fold_metrics.csv"
    if not p.exists():
        return None
    d = pd.read_csv(p)
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.4))
    for h, g in d.groupby("horizon"):
        ax[0].plot(g.test_year, g.auc_up_vs_down, "o-", label=h)
        ax[1].plot(g.test_year, g.d_logloss, "o-", label=h)
    ax[0].axhline(0.5, color="k", lw=0.8)
    ax[1].axhline(0, color="k", lw=0.8)
    ax[0].set_title("UP vs DOWN AUROC")
    ax[1].set_title("log-loss gain vs prior")
    ax[0].legend()
    return fig


@figure("E09 convexity certificate: objective reached from five random starts (identical = unique global minimum)")
def f24_convexity():
    p = ROOT / "reports" / "e09_convexity_certificate.json"
    if not p.exists():
        return None
    c = json.loads(p.read_text())
    fig, ax = plt.subplots(figsize=(7, 3))
    for h, v in c.items():
        o = np.array(v["objective_per_start"])
        ax.plot(range(1, 6), o - o.min(), "o-", label=f"{h} (range {v['objective_range']:.1e})")
    ax.set_xlabel("start")
    ax.set_ylabel("objective - min")
    ax.legend(fontsize=7)
    return fig


@figure("Backtest (test 2020-2022, base costs): cumulative return and drawdown vs equal-weight survivors")
def f25_equity(tag):
    p = ROOT / "reports" / "backtest" / f"e04_daily_returns_{tag}.csv"
    if not p.exists():
        return None
    d = pd.read_csv(p, index_col=0, parse_dates=True)
    fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    for c in d.columns:
        eq = (1 + d[c]).cumprod()
        ax[0].plot(eq.index, eq.values, lw=2.2 if "benchmark" in c else 1, color="k" if "benchmark" in c else None, label=c)
        ax[1].plot(eq.index, (eq / eq.cummax() - 1).values, lw=1, color="k" if "benchmark" in c else None)
    ax[0].legend(fontsize=7, ncol=3)
    ax[0].set_ylabel("growth of 1")
    ax[1].set_ylabel("drawdown")
    return fig


@figure("RAG retrieval evaluation (within the target document)")
def f26_rag():
    p = ROOT / "reports" / "rag" / "rag_eval.json"
    if not p.exists():
        return None
    e = json.loads(p.read_text())["book"]
    rows = {k.split("|")[1]: v for k, v in e.items() if k.startswith("within")}
    fig, ax = plt.subplots(figsize=(9, 3))
    ms = ["recall@1", "recall@5", "mrr", "ndcg@10"]
    for i, (k, v) in enumerate(rows.items()):
        ax.bar(np.arange(len(ms)) + i * 0.16, [v[x] for x in ms], 0.16, label=k)
    ax.set_xticks(np.arange(len(ms)) + 0.32, ms)
    ax.legend(fontsize=7)
    return fig


@figure("Gated fusion diagnostic: mean gate weight per modality (test vs validation)", "Attribution diagnostic, not causal")
def f27_gates(run_id):
    p = ROOT / "experiments" / run_id / "gates.json"
    g = json.loads(p.read_text())
    if "variant_gated" not in g:
        return None
    v = g["variant_gated"]
    fig, ax = plt.subplots(figsize=(7, 3))
    x = np.arange(len(v["names"]))
    ax.bar(x - 0.2, v["mean_valid"], 0.4, label="valid", color=C["valid"])
    ax.bar(x + 0.2, v["mean_test"], 0.4, label="test", color=C["test"])
    ax.set_xticks(x, v["names"])
    ax.legend()
    return fig


def main() -> int:
    fr = frame()
    f01_class_distribution(fr)
    r1, m1 = latest("e01_structured_baselines")
    if m1:
        f02_e01_logloss(m1)
        f03_e01_auc(m1)
    r2, m2 = latest("e02_cv_branch")
    if m2:
        f04_e02_heatmap(m2)
    r3, m3 = latest("e03_fusion")
    if m3:
        f05_e03_ablations(m3)
        f06_e03_forest(m3)
        f07_e03_stress(m3)
        f08_e03_curves(r3)
        f27_gates(r3)
    r6, m6 = latest("e06_optimization")
    if m6:
        f09_e06_curves(m6)
        f10_e06_search(m6)
        f11_e06_seeds(m6)
        f12_e06_landscape(m6, r6)
        sel = m6["selection"]
        table("E06 selection and seed spread", pd.DataFrame([{**{"best_config": sel["best_config"], "best_lr": sel["best_lr"]},
                                                             **m6["seed_spread"], **{k: m6["landscape"][k] for k in ("center_valid_ce", "min_on_grid")}}]))
    r7, m7 = latest("e07_lgbm_tuning")
    if m7:
        f13_e07_curves(m7)
        f14_e07_optuna()
        f15_e07_learning(m7)
        f16_e07_importance(m7)
        table("E07 tuned LightGBM: per-horizon log loss", pd.DataFrame([{"horizon": h, **{f"{n}_{b}": m7["scores"][n][h][b] for n in ("tuned", "refit")
                                                                         for b in ("valid", "calib", "test")}} for h in H]),
              "refit rows include 2017-18, so its validation score is in-sample")
    r8, m8, P = final_probs()
    if m8:
        f17_e08_candidates(m8)
        f18_reliability(fr, P)
        f19_roc(fr, P)
        f20_pr(fr, P)
        f21_confusion(m8)
        f22_by_year(m8)
        rows = []
        for h in H:
            r = m8["final"]["by_horizon"][h]
            rows.append({"horizon": h, "n": r["n"], "log_loss": r["log_loss"], "gain_vs_prior": r["gain_vs_prior"]["estimate"],
                         "gain_ci_low": r["gain_vs_prior"]["ci95"][0], "gain_ci_high": r["gain_vs_prior"]["ci95"][1],
                         "gain_vs_E03": r["gain_vs_e03"]["estimate"], "brier": r["brier"], "ece": r["ece_top"],
                         "accuracy": r["accuracy"], "balanced_acc": r["balanced_accuracy"], "macro_f1": r["macro_f1"],
                         "auroc_macro": r["auroc_macro"], "prauc_macro": r["prauc_macro"], "auroc_up_down": r.get("auroc_up_vs_down"),
                         "auroc_up_down_ci_low": r["auroc_up_down_ci"]["ci95"][0], "auroc_up_down_ci_high": r["auroc_up_down_ci"]["ci95"][1]})
        table("Final deployed model (refit 2003-2018, calibrated on 2019): test-year metrics with month-block 95% intervals", pd.DataFrame(rows))
        cls = []
        for h in H:
            r = m8["final"]["by_horizon"][h]
            for k in CLS:
                cls.append({"horizon": h, "class": k, "support": r[f"support_{k}"], "predicted": r[f"pred_count_{k}"],
                            "precision": r[f"precision_{k}"], "recall": r[f"recall_{k}"], "f1": r[f"f1_{k}"],
                            "auroc": r.get(f"auroc_{k}"), "prauc": r.get(f"prauc_{k}"), "base_rate": r.get(f"base_rate_{k}")})
        table("Final model: per-class precision, recall, F1, AUROC and PR-AUC (test years)", pd.DataFrame(cls))
        table("Calibration choice on the test years (mean log loss)", pd.DataFrame([m8["calibration_compare_test"]]))
        qc = [{"horizon": h, **{f"{s}_{k}": v for s, d in m8["quantile_coverage_e03"][h].items() for k, v in d.items()}} for h in H]
        table("Quantile heads (E03): coverage of the 10-90% band of the volatility-scaled return (nominal 0.80)", pd.DataFrame(qc))
        bm = ROOT / "artifacts" / "bundles" / r8 / "bundle_meta.json"
        qd = json.loads(bm.read_text(encoding="utf-8")).get("quantile_coverage") if bm.exists() else None
        if qd:
            table("Deployed bundle (5 fusion seeds, refit): 10-90% coverage, median absolute error of z, quantile crossing rate",
                  pd.DataFrame([{"horizon": h, **{f"{b}_{k}": v for b in ("valid", "test") for k, v in qd[h][b].items()},
                                 "shown_in_app": qd[h]["show_in_app"]} for h in H]),
                  "a range is shown only if validation coverage is within 0.80 +/- 0.08 (scripts/verify/quantile_coverage.py)")
    f23_walk_forward()
    f24_convexity()
    c = ROOT / "reports" / "e09_convexity_certificate.json"
    if c.exists():
        cc = json.loads(c.read_text())
        table("E09 convexity certificate (multinomial logistic, numeric features)", pd.DataFrame(
            [{"horizon": h, **{k: v for k, v in d.items() if k != "objective_per_start" and k != "converged_flags"}} for h, d in cc.items()]))
    for tag in ("v1.1", "v1"):
        if (ROOT / "reports" / "backtest" / f"e04_daily_returns_{tag}.csv").exists():
            f25_equity(tag)
            break
    for tag in ("v1.1", "v1"):
        p = ROOT / "reports" / "backtest" / ("e04_backtest.json" if tag == "v1" else f"e04_backtest_{tag}.json")
        if p.exists():
            b = json.loads(p.read_text())
            table(f"Backtest {tag}, test 2020-2022 (thresholds frozen on 2017-2018)", pd.DataFrame(
                [{"policy": k, "trades": v.get("trades", 0), "sharpe": v["sharpe"], "sharpe_ci_low": v["sharpe_ci95_month_blocks"][0],
                  "sharpe_ci_high": v["sharpe_ci95_month_blocks"][1], "sortino": v.get("sortino"), "cagr": v["cagr"],
                  "max_dd": v["max_drawdown"], "calmar": v.get("calmar"), "win_rate": v.get("win_rate"), "profit_factor": v.get("profit_factor"),
                  "expectancy_R": v.get("expectancy_R"), "psr": v.get("psr_vs_0"), "dsr": (v.get("deflated_sharpe") or {}).get("dsr")}
                 for k, v in b["test_final"].items()]), f"benchmark Sharpe {b['benchmark_ew_survivors']['test']['sharpe']:.2f}; PBO {b['pbo'].get('pbo')}")
    rid10, m10 = latest("e10_controls")
    if m10:
        rows10 = [{"horizon": h, "control": k, "valid_LL_minus_prior": v["valid"]["log_loss"] - v["valid"]["prior_log_loss"],
                   "test_LL_minus_prior": v["test"]["log_loss"] - v["test"]["prior_log_loss"], "trees": v["test"]["best_iteration"]}
                  for h, d in m10.items() for k, v in d.items()]
        table("E10 controls: placebos (labels shuffled within date; across all rows) and canary (planted future-return leak); negative = better than the prior",
              pd.DataFrame(rows10), "A working pipeline must show the global placebo at 0, the within-date placebo between 0 and the real model (it keeps date-level timing) and the canary far below 0")
    f26_rag()
    html = ["<!doctype html><html><head><meta charset='utf-8'><title>NEXUS Alpha audit report</title><style>",
            "body{font-family:Segoe UI,Arial,sans-serif;max-width:1500px;margin:24px auto;padding:0 16px;color:#222}",
            "h1{font-size:24px}h2{font-size:17px;margin-top:34px;border-bottom:1px solid #ddd}p.n{color:#555;font-size:13px}",
            "img{max-width:100%;border:1px solid #eee}table.t{border-collapse:collapse;font-size:12px}",
            "table.t td,table.t th{border:1px solid #ddd;padding:3px 6px;text-align:right}</style></head><body>",
            "<h1>NEXUS Alpha: visual audit report</h1><p class='n'>Generated from saved artifacts by scripts/report/audit_report.py. "
            "Test years 2020-2022 are a historical diagnostic. See docs/CHIEF_AUDIT_REPORT.md for findings and the SWOT.</p>"]
    for kind, title, note, body in SECTIONS:
        html.append(f"<h2>{title}</h2>" + (f"<p class='n'>{note}</p>" if note else ""))
        html.append(f"<img src='data:image/png;base64,{body}'>" if kind == "fig" else body)
    html.append("<h2>Figure status</h2>" + pd.DataFrame(STATUS, columns=["figure", "title", "status"]).to_html(index=False, border=0, classes="t"))
    html.append("</body></html>")
    (ROOT / "reports" / "NEXUS_Audit_Report.html").write_text("\n".join(html), encoding="utf-8")
    bad = [s for s in STATUS if s[2] != "ok"]
    print(f"{len(STATUS)} figures, {len(bad)} not ok: {bad}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

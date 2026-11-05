"""E09: certificate that the convex heads reach their unique GLOBAL minimum (the only model class where one can be proved).

Model: multinomial logistic regression on the 80 standardised numeric features, reference-class parameterisation
(class DOWN logits fixed at 0, so the softmax translation invariance is removed), L2 penalty lam/2 * ||W||^2 on weights
and 1e-6 on intercepts => strictly convex objective with a unique minimiser. Evidence, per horizon (1D, 5D, 20D):
(1) L-BFGS from 5 random starts reaches the same objective and parameters; (2) gradient norm at the solution ~ 0;
(3) the analytic Hessian at the solution is positive definite (min eigenvalue > 0), so the stationary point is the
unique global minimum. Also reports validation log loss of the certified model.
Output: reports/e09_convexity_certificate.json
"""
from __future__ import annotations

import json
import sys

import numpy as np
from scipy.optimize import minimize

from nexus_alpha import paths
from nexus_alpha.evaluation.metrics import log_loss
from nexus_alpha.experiments import Run
from nexus_alpha.multimodal.dataset import build, load_tables

LAM = 1e-3


def objective(theta, X, Y1h, lam):
    n, d = X.shape
    W = theta.reshape(d, 2)
    Z = np.concatenate([np.zeros((n, 1)), X @ W], 1)
    Z -= Z.max(1, keepdims=True)
    P = np.exp(Z)
    P /= P.sum(1, keepdims=True)
    pen = np.full((d, 2), lam)
    pen[-1] = 1e-6                                       # intercept row
    f = -np.mean(np.log(np.clip((P * Y1h).sum(1), 1e-300, None))) + 0.5 * (pen * W ** 2).sum()
    g = X.T @ (P[:, 1:] - Y1h[:, 1:]) / n + pen * W
    return f, g.ravel(), P


def hessian(X, P, lam):
    n, d = X.shape
    p = P[:, 1:]
    H = np.zeros((2 * d, 2 * d))
    for a in range(2):
        for b in range(2):
            w = p[:, a] * ((a == b) - p[:, b])
            H[a::2, b::2] = (X * w[:, None]).T @ X / n
    pen = np.full((d, 2), lam)
    pen[-1] = 1e-6
    return H + np.diag(pen.ravel())


def main() -> int:
    fr = load_tables()
    blk = fr.block.to_numpy()
    d, _ = build(fr, blk == "train", ("num",), "convnext_tiny")
    run = Run("e09_convexity_certificate", {"lambda": LAM, "starts": 5, "rows": "60,000 training rows (seeded)"})
    rng = np.random.default_rng(7)
    out = {}
    for j, h in ((0, "1D"), (2, "5D"), (4, "20D")):
        tr = np.flatnonzero((blk == "train") & (d.y[:, j] >= 0))
        tr = rng.choice(tr, min(60000, len(tr)), replace=False)
        X = np.concatenate([d.x["num"][tr].astype(np.float64), np.ones((len(tr), 1))], 1)
        Y1h = np.eye(3)[d.y[tr, j]]
        sols = []
        for s in range(5):
            th0 = np.random.default_rng(100 + s).normal(0, 0.5, X.shape[1] * 2)
            r = minimize(lambda t: objective(t, X, Y1h, LAM)[:2], th0, jac=True, method="L-BFGS-B",
                         options={"maxiter": 5000, "gtol": 1e-10, "ftol": 1e-15})
            sols.append(r)
        fvals = np.array([r.fun for r in sols])
        thetas = np.stack([r.x for r in sols])
        best = sols[int(np.argmin(fvals))]
        f, g, P = objective(best.x, X, Y1h, LAM)
        eig = np.linalg.eigvalsh(hessian(X, P, LAM))
        va = np.flatnonzero((blk == "valid") & (d.y[:, j] >= 0))
        Xv = np.concatenate([d.x["num"][va].astype(np.float64), np.ones((len(va), 1))], 1)
        Pv = objective(best.x, Xv, np.eye(3)[d.y[va, j]], LAM)[2]
        out[h] = {"objective_per_start": fvals.tolist(), "objective_range": float(fvals.max() - fvals.min()),
                  "max_param_distance_between_starts": float(np.abs(thetas - thetas[0]).max()),
                  "gradient_norm_at_solution": float(np.linalg.norm(g)), "hessian_min_eigenvalue": float(eig.min()),
                  "hessian_max_eigenvalue": float(eig.max()), "certified_global_minimum": bool(eig.min() > 0 and np.linalg.norm(g) < 1e-5),
                  "valid_log_loss": log_loss(d.y[va, j], Pv), "converged_flags": [bool(r.success) for r in sols]}
        print(h, {k: v for k, v in out[h].items() if k != "objective_per_start"}, flush=True)
    (paths.REPO_ROOT / "reports" / "e09_convexity_certificate.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    run.finish("COMPLETED", "; ".join(f"{h}: certified {v['certified_global_minimum']}, |grad| {v['gradient_norm_at_solution']:.1e}, "
                                       f"min eig {v['hessian_min_eigenvalue']:.2e}, spread {v['objective_range']:.1e}" for h, v in out.items()), out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

import numpy as np
import pytest
import torch

from nexus_alpha.calibration.scalers import TemperatureScaler, VectorScaler, fit_best
from nexus_alpha.evaluation.metrics import check_probs, classification_report, log_loss
from nexus_alpha.multimodal.fusion import FusionConfig, FusionModel, apply_modality_dropout, fusion_loss

DIMS = {"img_W": 8, "img_D": 8, "img_H": 8, "num": 12, "news": 6, "sec": 5}
VARIANTS = ("late", "concat", "attention", "gated", "hybrid")


def _batch(B=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = {k: torch.randn(B, d, generator=g) for k, d in DIMS.items()}
    mask = torch.ones(B, len(DIMS))
    mask[: B // 2, 4] = 0          # half the rows have no news
    mask[::3, 5] = 0               # some rows outside SEC coverage
    return x, mask


@pytest.mark.parametrize("variant", VARIANTS)
def test_shapes_and_probabilities(variant):
    torch.manual_seed(0)
    m = FusionModel(FusionConfig(modality_dims=DIMS, variant=variant, d_model=16, hidden=32)).eval()
    x, mask = _batch()
    lg, q, diag = m(x, mask)
    assert lg.shape == (16, 6, 3) and q.shape == (16, 6, 3)
    p = torch.softmax(lg, -1).detach().numpy()
    check_probs(p.reshape(-1, 3))
    if variant in ("gated", "hybrid", "late"):
        g = diag.get("gates", diag.get("modality_weights")).detach().numpy()
        assert np.allclose(g.sum(1), 1, atol=1e-5)
        assert (g[mask.numpy() == 0] < 1e-6).all()           # absent modalities get zero weight


@pytest.mark.parametrize("variant", VARIANTS)
def test_masked_modality_has_no_influence(variant):
    torch.manual_seed(1)
    m = FusionModel(FusionConfig(modality_dims=DIMS, variant=variant, d_model=16, hidden=32)).eval()
    x, mask = _batch()
    mask[:, 4] = 0                                           # news absent everywhere
    lg0, _, _ = m(x, mask)
    x2 = dict(x)
    x2["news"] = x["news"] * 100 + 5                         # garbage in the masked block
    lg1, _, _ = m(x2, mask)
    assert torch.allclose(lg0, lg1, atol=1e-5)


def test_modality_dropout_never_hides_everything():
    mask = torch.ones(500, 6)
    mask[:, 1:] = 0
    out = apply_modality_dropout(mask, 0.99)
    assert (out.sum(1) > 0).all()


def test_loss_ignores_missing_labels():
    cfg = FusionConfig(modality_dims=DIMS)
    lg = torch.randn(4, 6, 3)
    q = torch.randn(4, 6, 3)
    y = torch.full((4, 6), -1, dtype=torch.long)
    y[0, 0] = 2
    z = torch.full((4, 6), float("nan"))
    l1 = fusion_loss(lg, q, y, z, cfg)
    lg2 = lg.clone()
    lg2[1:] += 100                                           # rows with no labels cannot move the loss
    assert torch.allclose(l1, fusion_loss(lg2, q, y, z, cfg))


def _overconfident(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 3, n)
    logits = rng.normal(0, 1, (n, 3))
    logits[np.arange(n), y] += 0.6                           # some signal
    p = np.exp(3 * logits)                                   # far too sharp
    return p / p.sum(1, keepdims=True), y


def test_calibrators_reduce_log_loss_of_an_overconfident_model():
    p, y = _overconfident()
    base = log_loss(y[2000:], p[2000:])
    t = TemperatureScaler().fit(p[:2000], y[:2000])
    v = VectorScaler().fit(p[:2000], y[:2000])
    assert t.T > 1.5
    assert log_loss(y[2000:], t.transform(p[2000:])) < base
    assert log_loss(y[2000:], v.transform(p[2000:])) < base
    c, info = fit_best(p[:2000], y[:2000])
    assert info["chosen"] in ("temperature", "vector")
    check_probs(c.transform(p[2000:]))


def test_classification_report_on_known_case():
    y = np.array([0, 1, 2, 2])
    p = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8], [0.6, 0.2, 0.2]])
    r = classification_report(y, p)
    assert r["accuracy"] == 0.75 and r["support_UP"] == 2 and r["confusion"][2][0] == 1
    assert r["log_loss"] == pytest.approx(-np.mean(np.log([0.8, 0.8, 0.8, 0.2])))
    with pytest.raises(ValueError):
        check_probs(np.array([[0.5, 0.5, 0.5]]))

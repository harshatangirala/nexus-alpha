"""End-to-end inference checks against a real bundle (skipped when no bundle or source data exists)."""
import numpy as np
import pytest

from nexus_alpha import paths
from nexus_alpha.inference.pipeline import NexusPredictor, latest_bundle

from .conftest import source_data


def _bundle():
    b = latest_bundle()
    if b is None:
        mini = paths.ARTIFACTS / "bundles_test" / "mini"
        b = mini if (mini / "bundle_meta.json").exists() else None
    return b


needs_bundle = pytest.mark.skipif(_bundle() is None, reason="no inference bundle built")


@pytest.fixture(scope="module")
def predictor():
    return NexusPredictor(_bundle())


@source_data
@needs_bundle
def test_prediction_contract_and_missing_news(predictor):
    r = predictor.predict("MSFT", asof="2021-03-15", mode="historical", articles=[], explain=False)
    assert set(r.forecasts) == {"1D", "2D", "5D", "10D", "20D", "1M"}
    for f in r.forecasts.values():
        p = np.array([f.down_probability, f.sideways_probability, f.up_probability])
        assert np.isfinite(p).all() and abs(p.sum() - 1) < 1e-6
    assert any("no news" in m for m in r.confluence["missing_information"])
    assert r.instrument.forecast_timestamp.startswith("2021-03-16T13:00")      # 09:00 New York, next session
    for plan in r.trade_plans.values():
        if plan.decision in ("LONG", "SHORT"):
            assert plan.stop is not None and plan.quantity >= 1


@source_data
@needs_bundle
def test_sealed_dates_are_refused(predictor):
    with pytest.raises(ValueError):
        predictor.predict("MSFT", asof="2023-02-01", mode="historical")


@source_data
@needs_bundle
def test_future_article_is_excluded_and_rag_never_changes_probabilities(predictor):
    arts = [{"headline": "Company reports results", "text": "Results were in line.", "published_utc": "2021-03-16 15:00"}]
    a = predictor.predict("MSFT", asof="2021-03-15", mode="historical", articles=arts, explain=False)
    assert a.evidence["news"][0]["used"] is False                                 # published after the decision time
    from nexus_alpha.rag.explain import load_kb
    kb = load_kb()
    if kb is None:
        pytest.skip("knowledge base not built")
    b = NexusPredictor(predictor.b.path, kb=kb).predict("MSFT", asof="2021-03-15", mode="historical", articles=arts,
                                                        explain=False)
    for h in a.forecasts:
        assert a.forecasts[h].up_probability == b.forecasts[h].up_probability
        assert a.forecasts[h].down_probability == b.forecasts[h].down_probability
    assert "rag_sources" in b.evidence and "rag_sources" not in a.evidence

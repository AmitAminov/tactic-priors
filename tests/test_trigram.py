"""Trigram model: valid smoothed distributions and pseudo-tactic behaviour."""

from __future__ import annotations

import numpy as np
import pytest

from tactic_priors.ngram_models import TrigramTacticsModel


def assert_valid_distributions(model: TrigramTacticsModel) -> None:
    assert model.model, "model has no contexts"
    for context, dist in model.model.items():
        probs = np.array(list(dist.values()))
        assert (probs >= 0).all(), f"negative probability in context {context}"
        assert (probs <= 1 + 1e-12).all(), f"probability > 1 in context {context}"
        assert np.isclose(probs.sum(), 1.0), f"context {context} sums to {probs.sum()}"


def test_mle_distributions_are_valid(traced_proofs):
    model = TrigramTacticsModel.from_traced_proofs(traced_proofs, smoothing="mle")
    assert_valid_distributions(model)


def test_witten_bell_distributions_are_valid(traced_proofs):
    model = TrigramTacticsModel.from_traced_proofs(traced_proofs, smoothing="witten_bell")
    assert_valid_distributions(model)


def test_witten_bell_interpolates_with_bigram(traced_proofs):
    """WB(w3 | w1, w2) = lambda * MLE + (1 - lambda) * bigram, lambda = n / (n + t).

    For the context ("intro h", "simp") the corpus gives n = 4 observations
    over t = 2 continuation types (ring 3/4, linarith 1/4), so lambda = 2/3.
    The bigram distribution after "simp" is ring 4/5, linarith 1/5.
    """
    mle = TrigramTacticsModel.from_traced_proofs(traced_proofs, smoothing="mle")
    wb = TrigramTacticsModel.from_traced_proofs(traced_proofs, smoothing="witten_bell")
    context = ("intro h", "simp")
    mle_dist = mle.get_next_tactic_probabilities(*context)
    wb_dist = wb.get_next_tactic_probabilities(*context)
    assert set(mle_dist) == set(wb_dist)
    assert mle_dist["ring"] == pytest.approx(3 / 4)
    assert wb_dist["ring"] == pytest.approx((2 / 3) * (3 / 4) + (1 / 3) * (4 / 5))
    assert wb_dist["linarith"] == pytest.approx((2 / 3) * (1 / 4) + (1 / 3) * (1 / 5))
    assert sum(wb_dist.values()) == pytest.approx(1.0)


def test_unknown_smoothing_raises(traced_proofs):
    with pytest.raises(ValueError):
        TrigramTacticsModel.from_traced_proofs(traced_proofs, smoothing="laplace")


def test_context_conditioning(traced_proofs):
    model = TrigramTacticsModel.from_traced_proofs(traced_proofs, smoothing="mle")
    sos = TrigramTacticsModel.SOS
    start_dist = model.get_next_tactic_probabilities(sos, sos)
    # "intro h" opens 6 of 8 proofs, "simp" 1, "nlinarith" 1.
    assert start_dist["intro h"] == pytest.approx(6 / 8)
    assert model.predict_next_tactic(sos, sos) == "intro h"


def test_pseudo_tactic_threshold_behaviour(traced_proofs):
    """Tactics below NGRAM_PROB_TH collapse into the pseudo-tactic pool."""
    original_th = TrigramTacticsModel.NGRAM_PROB_TH
    try:
        # With a high threshold, singleton tactics (1/20 tokens = 0.05)
        # fall below it and become pseudo-tactics.
        TrigramTacticsModel.NGRAM_PROB_TH = 0.09
        model = TrigramTacticsModel.from_traced_proofs(traced_proofs, smoothing="mle")
        assert "norm_num" in model.pseudo_tactics
        assert "omega" in model.pseudo_tactics
        assert "norm_num" not in model.trigram_tactics
        assert TrigramTacticsModel.PSEUDO_TACTIC in model.trigram_tactics

        # With the production threshold (1e-5), nothing in the tiny corpus is rare.
        TrigramTacticsModel.NGRAM_PROB_TH = original_th
        model = TrigramTacticsModel.from_traced_proofs(traced_proofs, smoothing="mle")
        assert model.pseudo_tactics == []
    finally:
        TrigramTacticsModel.NGRAM_PROB_TH = original_th


def test_sampled_pseudo_tactic_materialises_as_rare_tactic(traced_proofs):
    original_th = TrigramTacticsModel.NGRAM_PROB_TH
    try:
        TrigramTacticsModel.NGRAM_PROB_TH = 0.09
        model = TrigramTacticsModel.from_traced_proofs(traced_proofs, smoothing="mle")
        placeholders = set(TrigramTacticsModel.PLACE_HOLDERS)
        np.random.seed(1)
        for _ in range(50):
            tactics, probs = model.get_k_sampled_tactics_and_probabilities(
                "intro h", "simp", 3
            )
            assert len(tactics) <= 3
            assert not placeholders.intersection(tactics)
            assert all(p > 0 for p in probs)
    finally:
        TrigramTacticsModel.NGRAM_PROB_TH = original_th


def test_unseen_context_backs_off_to_sos(traced_proofs):
    model = TrigramTacticsModel.from_traced_proofs(traced_proofs, smoothing="mle")
    np.random.seed(2)
    # ("never seen", "intro h") is unseen; backoff uses (SOS, "never seen"),
    # then (SOS, SOS), which always exists.
    tactics, probs = model.get_k_sampled_tactics_and_probabilities(
        "never seen", "also never seen", 2
    )
    assert len(tactics) == 2
    assert len(probs) == 2

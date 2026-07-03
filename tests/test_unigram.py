"""Unigram model: probabilities sum to 1, respect frequencies, sampling works."""

from __future__ import annotations

import numpy as np
import pytest

from tactic_priors.ngram_models import UnigramTacticsModel, calculate_distribution


def test_probabilities_sum_to_one(frequency_csv):
    model = UnigramTacticsModel(frequency_csv)
    assert np.isclose(sum(model.d_prob.values()), 1.0)


def test_probabilities_respect_frequencies(frequency_csv):
    model = UnigramTacticsModel(frequency_csv)
    assert model.get_tactic_probability("simp") == pytest.approx(0.5)
    assert model.get_tactic_probability("ring") == pytest.approx(0.25)
    assert model.get_tactic_probability("linarith") == pytest.approx(0.05)
    # Frequency order is preserved in probability order.
    assert (
        model.get_tactic_probability("simp")
        > model.get_tactic_probability("ring")
        > model.get_tactic_probability("intro h")
        > model.get_tactic_probability("linarith")
    )


def test_unknown_tactic_has_zero_probability(frequency_csv):
    model = UnigramTacticsModel(frequency_csv)
    assert model.get_tactic_probability("exact?") == 0.0


def test_sampling_returns_distinct_known_tactics(frequency_csv):
    model = UnigramTacticsModel(frequency_csv)
    rng_samples = model.sample_tactics(n_samples=3)
    assert len(rng_samples) == 3
    assert len(set(rng_samples)) == 3  # without replacement
    assert all(t in model.d_prob for t in rng_samples)


def test_sampling_frequency_tracks_probability(frequency_csv):
    np.random.seed(0)
    model = UnigramTacticsModel(frequency_csv)
    draws = [model.sample_tactics(1)[0] for _ in range(2000)]
    empirical = calculate_distribution(draws)
    assert empirical["simp"] == pytest.approx(0.5, abs=0.05)
    assert empirical["ring"] == pytest.approx(0.25, abs=0.05)


def test_calculate_distribution_sums_to_one():
    dist = calculate_distribution(["a", "a", "b", "c"])
    assert sum(dist.values()) == pytest.approx(1.0)
    assert dist["a"] == pytest.approx(0.5)

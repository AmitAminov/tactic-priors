"""CI smoke test for the committed artifacts.

Loads both released pickles through the package's legacy-module shim and
cross-checks the unigram model against the plain-text CSV equivalent, so a
broken pickle / CSV / loader can never ship silently.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from tactic_priors.ngram_models import (
    TrigramTacticsModel,
    UnigramTacticsModel,
    load_legacy_pickle,
)

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"

N_TACTICS = 16_850  # distinct tactics in the released distribution


def test_committed_pickles_load_and_match_csv() -> None:
    """Both artifact pickles deserialise and the unigram matches the CSV."""
    unigram = load_legacy_pickle(
        ARTIFACTS / "empirical_tactics_probability_model_mathlib4.pkl"
    )
    trigram = load_legacy_pickle(ARTIFACTS / "trigram_mathlib4.pkl")
    assert isinstance(unigram, UnigramTacticsModel)
    assert isinstance(trigram, TrigramTacticsModel)

    # The trigram artifact is structurally sane (non-empty conditional table).
    assert len(trigram.model) > 0
    assert len(trigram.pseudo_tactics) > 0

    # Cross-check every unigram probability against the inspectable CSV.
    df = pd.read_csv(ARTIFACTS / "empirical_tactics_probability_mathlib4.csv")
    assert len(df) == N_TACTICS
    assert len(unigram.d_prob) == N_TACTICS
    mismatches = sum(
        1
        for tactic, prob in zip(df["tactic"], df["prob"])
        if not math.isclose(
            unigram.get_tactic_probability(tactic), prob, rel_tol=0.0, abs_tol=1e-12
        )
    )
    assert mismatches == 0

    # Spot value quoted in the analysis, and global normalisation.
    assert math.isclose(
        unigram.get_tactic_probability("simp"), 0.0710, rel_tol=0.0, abs_tol=5e-4
    )
    assert math.isclose(sum(unigram.d_prob.values()), 1.0, rel_tol=0.0, abs_tol=1e-9)

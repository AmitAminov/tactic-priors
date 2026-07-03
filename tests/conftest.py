"""Shared fixtures: a tiny synthetic tactic corpus and a frequency CSV."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

# A small synthetic corpus in the LeanDojo Benchmark 4 schema. Frequencies
# are deliberately skewed so sampling behaviour is testable.
PROOF_SEQUENCES = [
    ["intro h", "simp", "ring"],
    ["intro h", "simp", "linarith"],
    ["intro h", "norm_num"],
    ["simp", "ring"],
    ["intro h", "simp", "ring"],
    ["nlinarith", "ring"],
    ["intro h", "simp", "ring"],
    ["intro h", "omega"],
]


@pytest.fixture()
def traced_proofs() -> list[dict]:
    """Synthetic traced proofs matching the LeanDojo JSON schema."""
    return [
        {"traced_tactics": [{"tactic": t} for t in seq]} for seq in PROOF_SEQUENCES
    ]


@pytest.fixture()
def frequency_csv(tmp_path: Path) -> Path:
    """A tactic-frequency CSV with known probabilities."""
    df = pd.DataFrame(
        {
            "tactic": ["simp", "ring", "intro h", "linarith"],
            "counter": [50, 25, 20, 5],
        }
    )
    df["prob"] = df["counter"] / df["counter"].sum()
    path = tmp_path / "freqs.csv"
    df.to_csv(path, index=False)
    return path

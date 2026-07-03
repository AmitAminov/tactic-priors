"""tactic_priors: empirical tactic-distribution baselines for Lean 4 theorem proving.

Subpackage overview:
    - :mod:`tactic_priors.ngram_models`: unigram and trigram tactic models
      (runnable offline, no Lean toolchain required).
    - :mod:`tactic_priors.build_distribution`: builds the empirical tactic
      distribution and the trigram model from LeanDojo Benchmark 4.
    - :mod:`tactic_priors.search`: best-first proof-search evaluation driver
      (requires ``lean_dojo``; cluster/GPU reference code).
"""

from tactic_priors.ngram_models import (
    TrigramTacticsModel,
    UnigramTacticsModel,
    load_legacy_pickle,
)

__all__ = [
    "TrigramTacticsModel",
    "UnigramTacticsModel",
    "load_legacy_pickle",
]

__version__ = "0.1.0"

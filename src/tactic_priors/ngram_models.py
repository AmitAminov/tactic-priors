"""N-gram models over Lean 4 tactic sequences.

Two statistical baselines for tactic prediction, both estimated from the
traced proofs of LeanDojo Benchmark 4 (Mathlib4):

    - :class:`UnigramTacticsModel`: samples tactics from the empirical
      (frequency-based) tactic distribution, ignoring context.
    - :class:`TrigramTacticsModel`: predicts the next tactic from the two
      preceding tactics. Rare tactics (empirical probability below
      :attr:`TrigramTacticsModel.NGRAM_PROB_TH`) are collapsed into a single
      ``@PSEUDO_TACTIC@`` pseudo-word during training; at sampling time a
      drawn pseudo-tactic is materialised as a uniformly random rare tactic.
      Witten-Bell interpolation with the bigram distribution is available at
      training time (see :meth:`TrigramTacticsModel.from_traced_proofs`).

Pickle compatibility: the released artifacts (``trigram_mathlib4.pkl``,
``empirical_tactics_probability_model_mathlib4.pkl``) were serialised from a
module named ``ngram_tactics_models`` on the research cluster. Use
:func:`load_legacy_pickle` to deserialise them against this module.
"""

from __future__ import annotations

import pickle
import random
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

TracedProof = Mapping[str, object]

__all__ = [
    "UnigramTacticsModel",
    "TrigramTacticsModel",
    "calculate_distribution",
    "load_legacy_pickle",
]


def calculate_distribution(text: Sequence[str]) -> dict[str, float]:
    """Compute the empirical probability distribution of items in a sequence.

    Args:
        text: A sequence of items (e.g. tactic strings).

    Returns:
        A mapping from each distinct item to its relative frequency. The
        values sum to 1 for a non-empty input.
    """
    histogram = Counter(text)
    n = len(text)
    return {key: count / n for key, count in histogram.items()}


def return_zero() -> int:
    """Return 0. Module-level so that ``defaultdict`` instances pickle."""
    return 0


def create_default_dict() -> defaultdict:
    """Create a ``defaultdict(int-like)``. Module-level for pickling."""
    return defaultdict(return_zero)


def load_legacy_pickle(path: str | Path) -> object:
    """Load a model pickled on the research cluster under the legacy module name.

    The original training pipeline lived in a top-level module called
    ``ngram_tactics_models``; pickles created there reference classes and
    helper functions by that module path. This function aliases the legacy
    name to this module before unpickling.

    Args:
        path: Path to the pickle file (e.g. ``artifacts/trigram_mathlib4.pkl``).

    Returns:
        The deserialised model object.
    """
    sys.modules.setdefault("ngram_tactics_models", sys.modules[__name__])
    with open(path, "rb") as file:
        return pickle.load(file)


class UnigramTacticsModel:
    """Empirical (unigram) distribution over Mathlib4 tactics.

    The model is backed by a CSV with columns ``tactic``, ``counter`` and
    ``prob`` (the empirical probability of each tactic among all traced
    tactic invocations).

    Attributes:
        model_name: Human-readable model identifier.
        df_tactics_data: The underlying tactic-frequency table.
        tactics: Column of tactic strings.
        probabilities: Column of empirical probabilities (sums to 1).
        d_prob: Mapping from tactic string to empirical probability.
    """

    def __init__(self, csv_path: str | Path) -> None:
        """Initialise the model from a tactic-frequency CSV.

        Args:
            csv_path: Path to ``empirical_tactics_probability_mathlib4.csv``
                (or any CSV with ``tactic`` and ``prob`` columns whose
                probabilities sum to 1).
        """
        self.model_name: str = "empirical_model"
        self.df_tactics_data: pd.DataFrame = pd.read_csv(csv_path)
        self.tactics: pd.Series = self.df_tactics_data["tactic"]
        self.probabilities: pd.Series = self.df_tactics_data["prob"]
        self.d_prob: dict[str, float] = dict(zip(self.tactics, self.probabilities))

    def sample_tactics(self, n_samples: int = 1) -> list[str]:
        """Sample distinct tactics proportionally to their empirical frequency.

        Args:
            n_samples: Number of tactics to draw (without replacement).

        Returns:
            A list of ``n_samples`` distinct tactic strings.

        Raises:
            ValueError: If ``n_samples`` exceeds the number of known tactics.
        """
        probs = self.probabilities.to_numpy(dtype=float)
        probs = probs / probs.sum()  # guard against floating-point drift
        sampled = np.random.choice(
            self.tactics.to_numpy(), size=n_samples, p=probs, replace=False
        )
        return list(sampled)

    def get_tactic_probability(self, tactic: str) -> float:
        """Return the empirical probability of a tactic (0 if unseen).

        Args:
            tactic: The tactic string to look up.

        Returns:
            The empirical probability, or 0.0 for out-of-vocabulary tactics.
        """
        return self.d_prob.get(tactic, 0.0)


class TrigramTacticsModel:
    """Trigram model over tactic sequences with a pseudo-tactic mechanism.

    Proof tactic sequences are padded with ``@PROOF_START@`` / ``@PROOF_END@``
    markers. Tactics whose empirical probability is below
    :attr:`NGRAM_PROB_TH` are replaced by the ``@PSEUDO_TACTIC@`` pseudo-word
    before counting trigrams, which keeps the context space tractable while
    still reserving probability mass for the long tail. When a pseudo-tactic
    is drawn at sampling time it is materialised as a uniformly random tactic
    from the rare-tactic pool.

    Attributes:
        model_name: Human-readable model identifier.
        model: Mapping ``(w1, w2) -> {w3: P(w3 | w1, w2)}``.
        trigram_tactics: Vocabulary of frequent tactics plus placeholders.
        pseudo_tactics: Pool of rare tactics represented by the pseudo-word.
    """

    EOS: str = "@PROOF_END@"
    SOS: str = "@PROOF_START@"
    PSEUDO_TACTIC: str = "@PSEUDO_TACTIC@"
    PLACE_HOLDERS: list[str] = [EOS, SOS, PSEUDO_TACTIC]
    N_PLACE_HOLDERS: int = len(PLACE_HOLDERS)

    #: Tactics with empirical probability below this threshold are collapsed
    #: into the pseudo-tactic during training.
    NGRAM_PROB_TH: float = 1e-5

    #: Number of SOS/EOS padding tokens per proof (trigram context width - 1).
    N_PAD_TRIGRAM: int = 2

    def __init__(
        self,
        model: Mapping[tuple[str, str], Mapping[str, float]],
        trigram_tactics: list[str],
        pseudo_tactics: list[str],
    ) -> None:
        """Initialise from precomputed conditional distributions.

        Args:
            model: Mapping from a context ``(w1, w2)`` to a probability
                distribution over next tactics ``w3``.
            trigram_tactics: Frequent-tactic vocabulary (with placeholders).
            pseudo_tactics: Rare tactics represented by ``@PSEUDO_TACTIC@``.
        """
        self.model_name: str = "trigram_model"
        self.model = model
        self.trigram_tactics = trigram_tactics
        self.pseudo_tactics = pseudo_tactics

    def get_next_tactic_probabilities(self, w1: str, w2: str) -> dict[str, float]:
        """Return the conditional next-tactic distribution for a context.

        Args:
            w1: Tactic two steps back.
            w2: Immediately preceding tactic.

        Returns:
            Mapping ``{w3: P(w3 | w1, w2)}``; empty if the context is unseen.
        """
        # .get avoids materialising empty entries in legacy defaultdict models.
        return dict(self.model.get((w1, w2), {}))

    def predict_next_tactic(
        self, w1: str, w2: str, include_place_holders: bool = True
    ) -> str:
        """Return the most probable next tactic for a context.

        Args:
            w1: Tactic two steps back.
            w2: Immediately preceding tactic.
            include_place_holders: If False, skip SOS/EOS/pseudo-tactic
                placeholders and return the best real tactic.

        Returns:
            The argmax tactic, or an explanatory message for unseen contexts.
        """
        next_tactic_probs = self.get_next_tactic_probabilities(w1, w2)
        if not next_tactic_probs:
            return f"No prediction available for {w1}, {w2}"
        if include_place_holders:
            return max(next_tactic_probs, key=next_tactic_probs.get)
        tactics, _ = self.get_top_k_tactics_and_probabilities(w1, w2, 1, False)
        return tactics[0]

    def get_k_sampled_tactics_and_probabilities(
        self, w1: str, w2: str, k: int
    ) -> tuple[list[str], list[float]]:
        """Sample ``k`` next tactics from the conditional distribution.

        Placeholders are filtered from the sample when the support is large
        enough; otherwise sampled placeholders are materialised as random
        rare tactics with probability floor :attr:`NGRAM_PROB_TH`. Unseen
        contexts back off to the ``(SOS, w1)`` context.

        Args:
            w1: Tactic two steps back.
            w2: Immediately preceding tactic.
            k: Number of tactics to sample.

        Returns:
            Tuple ``(tactics, probabilities)`` of length at most ``k``.
        """
        next_tactic_probs = self.get_next_tactic_probabilities(w1, w2)
        if not next_tactic_probs:
            return self.get_k_sampled_tactics_and_probabilities(self.SOS, w1, k)
        sorted_tactics, sorted_probs = self._sorted_support(next_tactic_probs)
        probs = np.array(sorted_probs, dtype=float)
        probs = probs / probs.sum()
        n_sample = k + self.N_PLACE_HOLDERS
        if n_sample <= len(sorted_tactics):
            sampled = np.random.choice(sorted_tactics, size=n_sample, p=probs, replace=False)
            sampled_tactics = [t for t in sampled if t not in self.PLACE_HOLDERS][:k]
            sampled_probs = [next_tactic_probs[t] for t in sampled_tactics]
        else:
            sampled = np.random.choice(sorted_tactics, size=k, p=probs, replace=True)
            sampled_probs = [
                next_tactic_probs[t] if t not in self.PLACE_HOLDERS else self.NGRAM_PROB_TH
                for t in sampled
            ]
            sampled_tactics = [
                t if t not in self.PLACE_HOLDERS else random.choice(self.pseudo_tactics)
                for t in sampled
            ]
        return sampled_tactics, sampled_probs

    def get_top_k_tactics_and_probabilities(
        self, w1: str, w2: str, k: int, include_place_holders: bool = True
    ) -> tuple[list[str], list[float]]:
        """Return the ``k`` most probable next tactics for a context.

        Args:
            w1: Tactic two steps back.
            w2: Immediately preceding tactic.
            k: Number of tactics to return.
            include_place_holders: If False, skip SOS/EOS/pseudo-tactic
                placeholders.

        Returns:
            Tuple ``(tactics, probabilities)`` sorted by descending
            probability (at most ``k`` entries).

        Raises:
            KeyError: If the context ``(w1, w2)`` was never observed.
        """
        next_tactic_probs = self.get_next_tactic_probabilities(w1, w2)
        if not next_tactic_probs:
            raise KeyError(f"No prediction available for context ({w1!r}, {w2!r})")
        sorted_tactics, sorted_probs = self._sorted_support(next_tactic_probs)
        if include_place_holders:
            return sorted_tactics[:k], sorted_probs[:k]
        top_tactics: list[str] = []
        top_probs: list[float] = []
        for tactic, prob in zip(sorted_tactics, sorted_probs):
            if tactic in self.PLACE_HOLDERS:
                continue
            top_tactics.append(tactic)
            top_probs.append(prob)
            if len(top_tactics) == k:
                break
        return top_tactics, top_probs

    @staticmethod
    def _sorted_support(
        distribution: Mapping[str, float],
    ) -> tuple[list[str], list[float]]:
        """Sort a distribution's support by descending probability.

        Args:
            distribution: Mapping from tactic to probability.

        Returns:
            Tuple of (tactics, probabilities), highest probability first.
        """
        pairs = sorted(zip(distribution.values(), distribution.keys()), reverse=True)
        return [t for _, t in pairs], [p for p, _ in pairs]

    @classmethod
    def from_traced_proofs(
        cls,
        proofs: Iterable[TracedProof],
        smoothing: str = "witten_bell",
        show_progress: bool = False,
    ) -> TrigramTacticsModel:
        """Train a trigram tactic model from LeanDojo traced proofs.

        Each proof is a dict with a ``traced_tactics`` list whose elements
        carry a ``tactic`` string (the LeanDojo Benchmark 4 JSON schema).

        Smoothing options:
            - ``"mle"``: per-context maximum-likelihood conditional
              distributions (the estimator used to train the released
              ``trigram_mathlib4.pkl`` artifact).
            - ``"witten_bell"``: Witten-Bell interpolation of the trigram MLE
              with the bigram MLE, using the standard mixing weight
              ``lambda = n / (n + t)`` where ``n`` is the context count and
              ``t`` the number of distinct continuation types. The backoff
              distribution is restricted to the observed trigram support and
              renormalised; probability mass for genuinely unseen
              continuations is instead handled by the pseudo-tactic
              mechanism at sampling time.

        Args:
            proofs: Iterable of traced-proof dicts.
            smoothing: ``"witten_bell"`` or ``"mle"``.
            show_progress: If True, wrap iteration in a tqdm progress bar.

        Returns:
            A trained :class:`TrigramTacticsModel`.

        Raises:
            ValueError: If ``smoothing`` is not a recognised option.
        """
        if smoothing not in ("witten_bell", "mle"):
            raise ValueError(f"Unknown smoothing {smoothing!r}; use 'witten_bell' or 'mle'")

        proofs = list(proofs)
        iterator: Iterable[TracedProof] = proofs
        if show_progress:
            from tqdm import tqdm

            iterator = tqdm(proofs, desc="counting tactics")

        d_tactics_counter: dict[str, int] = {}
        for proof in iterator:
            for traced_tactic in proof["traced_tactics"]:
                tactic = traced_tactic["tactic"]
                d_tactics_counter[tactic] = d_tactics_counter.get(tactic, 0) + 1

        df = pd.DataFrame(
            {"tactic": list(d_tactics_counter.keys()), "counter": list(d_tactics_counter.values())}
        ).sort_values(by="counter", ascending=False)
        n_sum = df["counter"].sum()
        df["prob"] = df["counter"] / n_sum

        frequent_tactics = df[df["prob"] >= cls.NGRAM_PROB_TH]["tactic"].to_list()
        pseudo_word_tactics = df[df["prob"] < cls.NGRAM_PROB_TH]["tactic"].to_list()
        frequent_set = set(frequent_tactics)
        trigram_tactics = frequent_tactics + [cls.EOS, cls.SOS, cls.PSEUDO_TACTIC]

        token_stream: list[str] = []
        iterator = tqdm(proofs, desc="building token stream") if show_progress else proofs
        for proof in iterator:
            traced_tactics = proof["traced_tactics"]
            if len(traced_tactics) == 0:
                continue
            token_stream.extend([cls.SOS] * cls.N_PAD_TRIGRAM)
            for traced_tactic in traced_tactics:
                tactic = traced_tactic["tactic"]
                token_stream.append(tactic if tactic in frequent_set else cls.PSEUDO_TACTIC)
            token_stream.extend([cls.EOS] * cls.N_PAD_TRIGRAM)

        trigram_counts: dict[tuple[str, str], Counter] = defaultdict(Counter)
        bigram_counts: dict[str, Counter] = defaultdict(Counter)
        for w1, w2, w3 in zip(token_stream, token_stream[1:], token_stream[2:]):
            trigram_counts[(w1, w2)][w3] += 1
            bigram_counts[w2][w3] += 1

        model: dict[tuple[str, str], dict[str, float]] = {}
        for context, counts in trigram_counts.items():
            n = sum(counts.values())
            if smoothing == "mle":
                model[context] = {w3: c / n for w3, c in counts.items()}
                continue
            # Witten-Bell interpolation with the bigram distribution,
            # restricted to the observed trigram support.
            t = len(counts)
            lam = n / (n + t)
            _, w2 = context
            bg = bigram_counts[w2]
            bg_mass = sum(bg[w3] for w3 in counts)
            distribution = {}
            for w3, c in counts.items():
                p_mle = c / n
                p_backoff = bg[w3] / bg_mass if bg_mass > 0 else 1.0 / t
                distribution[w3] = lam * p_mle + (1.0 - lam) * p_backoff
            model[context] = distribution

        return cls(model, trigram_tactics, pseudo_word_tactics)

    # Backwards-compatible alias matching the original cluster code name.
    from_mathlib4_data = from_traced_proofs

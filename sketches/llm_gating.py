"""Design sketch -- not used to produce reported results.

Stochastic gating between a cheap tactic generator and an expensive LLM.

Idea: during proof search, most states are "easy" -- a lightweight generator
trained on tactic-token statistics can propose a useful candidate. Only the
hard states justify a forward pass through a 7B-parameter prover. A small
classifier predicts the probability that the *cheap* path suffices for the
current context; the wrapper then routes the query stochastically:

    with probability p = P(proof success | context)  -> query the LLM
    otherwise                                        -> query the cheap generator

Sampling the route (rather than thresholding) keeps exploration alive during
search: even low-confidence states occasionally get the expensive model, and
high-confidence states occasionally exercise the cheap path, producing the
data needed to keep training the gate. The expected compute cost per query
interpolates between the two extremes and can be tuned by calibrating the
classifier.

This is a repaired and documented version of an early prototype; the
classifier/generator training loops were never implemented, and none of the
numbers reported in the README involve this module.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

BFS_PROVER_EMBEDDING_DIM = 3584
BFS_PROVER_N_TOKENS = 151642
N_TACTICS_CLASSIFY = 3
N_TOKENS_PER_TACTIC = 3
N_TOKENS_CONTEXT_CLASSIFY = N_TACTICS_CLASSIFY * N_TOKENS_PER_TACTIC
N_FEATURES_CLASSIFY = 10
N_TACTICS_GENERATE = 10
N_TOKENS_CONTEXT_GENERATE = N_TACTICS_GENERATE * N_TOKENS_PER_TACTIC
N_NON_PSEUDO_TOKENS = 1000


class ProofSuccessClassifier(nn.Module):
    """Predicts P(cheap path suffices) from a fixed-size tactic context.

    A two-layer MLP with a sigmoid head; the output is used as the gating
    probability in :class:`LLMWrapper`.
    """

    def __init__(
        self,
        context_size: int = N_TOKENS_CONTEXT_CLASSIFY,
        n_features: int = N_FEATURES_CLASSIFY,
    ) -> None:
        """Initialise the classifier.

        Args:
            context_size: Input dimensionality (tokens of preceding tactics).
            n_features: Hidden-layer width.
        """
        super().__init__()
        self.fc1 = nn.Linear(context_size, n_features)
        self.fc2 = nn.Linear(n_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the gating probability.

        Args:
            x: Context tensor of shape ``(..., context_size)``.

        Returns:
            Tensor of probabilities in (0, 1) with shape ``(..., 1)``.
        """
        x = F.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x))


class TacticGenerator(nn.Module):
    """Cheap tactic generator: context tokens -> distribution over frequent tokens.

    Restricted to the ``N_NON_PSEUDO_TOKENS`` most frequent tactic tokens
    (the same head/tail split as the pseudo-tactic mechanism in
    ``tactic_priors.ngram_models``).
    """

    def __init__(self, context_size: int = N_TOKENS_CONTEXT_GENERATE) -> None:
        """Initialise the generator MLP.

        Args:
            context_size: Input dimensionality (tokens of preceding tactics).
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(context_size, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, N_NON_PSEUDO_TOKENS),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Score the frequent-token vocabulary for the next tactic token.

        Args:
            x: Context tensor of shape ``(..., context_size)``.

        Returns:
            Logits over the frequent-token vocabulary,
            shape ``(..., N_NON_PSEUDO_TOKENS)``.
        """
        return self.net(x)


class LLMWrapper(nn.Module):
    """Routes tactic generation between an LLM and a cheap generator.

    The route is sampled per query with probability given by
    :class:`ProofSuccessClassifier`, trading average inference cost against
    tactic quality.
    """

    def __init__(self, llm: nn.Module, latent_dim: int = BFS_PROVER_EMBEDDING_DIM) -> None:
        """Wrap an LLM with a gated cheap generator.

        Args:
            llm: The expensive tactic generator (e.g. BFS-Prover).
            latent_dim: Embedding dimensionality of the LLM.
        """
        super().__init__()
        self.llm = llm
        self.latent_dim = latent_dim
        self.llm_embedding = llm.get_input_embeddings()
        self.proof_success_classifier = ProofSuccessClassifier()
        self.tactic_generator = TacticGenerator()

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Embed encoded tactic tokens with the LLM's input embedding.

        Args:
            x: Encoded tactic-token tensor.

        Returns:
            Embedding tensor of shape ``(..., latent_dim)``.
        """
        return self.llm_embedding(x)

    def forward(self, encoded_tactics: torch.Tensor) -> torch.Tensor:
        """Generate the next tactic tokens via the sampled route.

        Args:
            encoded_tactics: Encoded context tactic tokens.

        Returns:
            LLM output or cheap-generator logits, depending on the sampled
            route.
        """
        embeddings = self.embed(encoded_tactics)
        prob_success = self.proof_success_classifier(embeddings)
        if torch.rand(1)[0] < prob_success:
            return self.llm(encoded_tactics)
        return self.tactic_generator(embeddings)

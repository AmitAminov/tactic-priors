"""Best-first proof-search evaluation driver (cluster/GPU reference code).

This module contains the W x K x N evaluation procedure used for all reported
miniF2F results: ``K`` independent passes, each expanding up to ``N`` search
frontiers with ``W`` candidate tactics per state. Search state is a weighted
DAG of Lean proof states; the frontier is re-ranked by a best-first minimum
cumulative-cost path from the root, with edge cost ``-log P(tactic)``
(optionally length-normalised by ``(depth + 1) ** alpha``, ``ALPHA = 0.5``
for the neural prover runs).

The graph utilities (:func:`best_first_min_cost_path_to_leaf`,
:func:`dfs_find_path`) are pure and importable everywhere. The evaluation
driver additionally requires ``lean_dojo`` (and a working Lean toolchain) and
is guarded accordingly -- importing this module without ``lean_dojo``
installed works, but calling :func:`dataset_wkn_evaluation` raises.

Historically this file contained three near-identical drivers (one per
model family); they are unified here behind the :class:`TacticSource`
protocol, with one factory per model family.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from io import StringIO
from queue import PriorityQueue
from typing import Protocol

import networkx as nx
import numpy as np

try:  # pragma: no cover - exercised only on the cluster
    from lean_dojo import (
        Dojo,
        LeanError,
        ProofFinished,
        ProofGivenUp,
        TacticState,
        logger,
    )

    LEAN_DOJO_AVAILABLE = True
except ImportError:  # pragma: no cover
    LEAN_DOJO_AVAILABLE = False

#: Length-normalisation exponent used for the neural prover runs.
ALPHA: float = 0.5

#: Prompt separator expected by BFS-Prover-style tactic generators.
SEP: str = ":::"

#: Floor used when converting probabilities to log-probabilities.
MIN_PROB: float = 1e-12


# ---------------------------------------------------------------------------
# Pure graph utilities (no lean_dojo required)
# ---------------------------------------------------------------------------

def dfs_find_path(
    adjacency_matrix: Sequence[Sequence[int]],
    start_node: int,
    end_node: int,
    path: list[int] | None = None,
    visited: set[int] | None = None,
) -> list[int] | None:
    """Find a path between two nodes of an adjacency-matrix graph via DFS.

    Used to reconstruct the tactic sequence (proof) once a search edge
    reaches the ``ProofFinished`` node.

    Args:
        adjacency_matrix: Square matrix with ``matrix[i][j] == 1`` iff there
            is an edge from node ``i`` to node ``j``.
        start_node: Index of the start node.
        end_node: Index of the target node.
        path: Accumulator for the current path (recursion internal).
        visited: Set of visited nodes (recursion internal).

    Returns:
        The list of node indices from ``start_node`` to ``end_node``, or
        None if no path exists.
    """
    if path is None:
        path = []
    if visited is None:
        visited = set()

    path.append(start_node)
    visited.add(start_node)

    if start_node == end_node:
        return path

    for neighbor in range(len(adjacency_matrix)):
        if adjacency_matrix[start_node][neighbor] == 1 and neighbor not in visited:
            found_path = dfs_find_path(adjacency_matrix, neighbor, end_node, path, visited)
            if found_path:
                return found_path

    path.pop()
    return None


def best_first_min_cost_path_to_leaf(graph: nx.DiGraph) -> list[int] | None:
    """Return the minimum cumulative-cost path from node 0 to any leaf.

    Runs uniform-cost (best-first) search over a weighted DAG whose edge
    ``weight`` attribute stores a non-negative cost (here ``-log P``); the
    first leaf popped from the priority queue therefore closes the cheapest
    root-to-leaf path. That path becomes the next search frontier.

    Args:
        graph: Directed graph with a ``weight`` attribute per edge and the
            search root at node 0.

    Returns:
        List of node indices from the root to the selected leaf, or None if
        the graph has no reachable leaf.
    """
    start = 0
    non_leafs = {edge[0] for edge in graph.edges}

    frontier: PriorityQueue = PriorityQueue()
    frontier.put((0, start))
    came_from: dict[int, int] = {}
    cost_so_far: dict[int, float] = {start: 0}

    while not frontier.empty():
        current = frontier.get()[1]

        if current not in non_leafs:
            path = []
            while current != start:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path

        for neighbor in graph[current]:
            new_cost = cost_so_far[current] + graph[current][neighbor]["weight"]
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                frontier.put((new_cost, neighbor))
                came_from[neighbor] = current
    return None


# ---------------------------------------------------------------------------
# Tactic sources: one uniform interface for all three model families
# ---------------------------------------------------------------------------

class TacticSource(Protocol):
    """Anything that proposes ``w`` candidate tactics for a search state."""

    def __call__(
        self, state: object, preceding_tactics: tuple[str, str], width: int
    ) -> tuple[list[str], list[float]]:
        """Propose candidate tactics.

        Args:
            state: The current Lean proof state (``TacticState``).
            preceding_tactics: The two tactics applied on the path to this
                state (SOS-padded at the root).
            width: Number of candidates to propose (``W``).

        Returns:
            Tuple ``(tactics, log_probs)``.
        """
        ...


def _safe_log(probs: Sequence[float]) -> np.ndarray:
    """Convert probabilities to log-probabilities with a numerical floor.

    Args:
        probs: Probabilities in [0, 1].

    Returns:
        Array of log-probabilities.
    """
    return np.log(np.maximum(np.asarray(probs, dtype=float), MIN_PROB))


def unigram_source(model: object) -> TacticSource:
    """Wrap a :class:`~tactic_priors.ngram_models.UnigramTacticsModel`.

    Note:
        The original cluster driver passed raw probabilities (not logs) as
        edge weights for the unigram model. Per-expansion candidate ranking
        is unaffected (both are monotone), but cumulative path costs differ;
        this cleaned driver consistently uses ``-log P`` for all models.
        See ``docs/methodology.md``.

    Args:
        model: A fitted unigram tactics model.

    Returns:
        A :class:`TacticSource` sampling W tactics from the empirical
        distribution, independent of context.
    """

    def source(
        state: object, preceding_tactics: tuple[str, str], width: int
    ) -> tuple[list[str], list[float]]:
        tactics = model.sample_tactics(width)
        probs = [model.get_tactic_probability(t) for t in tactics]
        return tactics, list(_safe_log(probs))

    return source


def trigram_source(model: object) -> TacticSource:
    """Wrap a :class:`~tactic_priors.ngram_models.TrigramTacticsModel`.

    Args:
        model: A fitted trigram tactics model.

    Returns:
        A :class:`TacticSource` sampling W tactics conditioned on the two
        preceding tactics.
    """

    def source(
        state: object, preceding_tactics: tuple[str, str], width: int
    ) -> tuple[list[str], list[float]]:
        tactics, probs = model.get_k_sampled_tactics_and_probabilities(
            *preceding_tactics, width
        )
        return tactics, list(_safe_log(probs))

    return source


def hf_generator_source(
    model: object,
    tokenizer: object,
    device: str = "cpu",
    temperature: float = 1.0,
    top_p: float = 1.0,
    max_new_tokens: int = 1000,
) -> TacticSource:
    """Wrap a HuggingFace causal-LM tactic generator (e.g. BFS-Prover).

    The state pretty-print is used as the prompt (``state.pp + ":::"``);
    candidate tactics are sampled with per-sequence log-probabilities from
    ``compute_transition_scores``.

    Args:
        model: A ``transformers`` causal LM.
        tokenizer: The matching tokenizer.
        device: Torch device for inference.
        temperature: Sampling temperature.
        top_p: Nucleus-sampling threshold.
        max_new_tokens: Generation cap per tactic.

    Returns:
        A :class:`TacticSource` producing W sampled tactics per state.
    """

    def source(
        state: object, preceding_tactics: tuple[str, str], width: int
    ) -> tuple[list[str], list[float]]:
        prompt = state.pp + SEP
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = model.generate(
            **inputs,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=width,
            output_logits=True,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_new_tokens,
        )
        log_probs = model.compute_transition_scores(
            outputs.sequences, outputs.scores, normalize_logits=True
        )
        input_length = inputs.input_ids.shape[1]
        generated_tokens = outputs.sequences[:, input_length:]
        tactics = [
            "".join(tokenizer.decode(token, skip_special_tokens=True) for token in seq)
            for seq in generated_tokens
        ]
        tactic_log_probs = list(log_probs.sum(axis=1).to("cpu").numpy().astype(float))
        return tactics, tactic_log_probs

    return source


# ---------------------------------------------------------------------------
# Unified W x K x N evaluation driver (requires lean_dojo)
# ---------------------------------------------------------------------------

def _is_tactic_timing_out_repeatedly(output_string: str, tactic: str | None = None) -> bool:
    """Detect repeated tactic timeouts from the captured lean_dojo log.

    Args:
        output_string: Captured lean_dojo logger output so far.
        tactic: If given, only trigger when the last logged command matches.

    Returns:
        True if the most recent logged command timed out.
    """
    lines = [line for line in output_string.split("\n") if line]
    if not lines:
        return False
    last_tactic_match = re.search('"cmd": "(.*)"', lines[-1])
    if last_tactic_match is None:
        return False
    if tactic is not None and last_tactic_match.group(1) != tactic:
        return False
    return any("Tactic timed out" in line for line in lines[::-1][:3])


def _run_tactic(dojo: Dojo, state: object, tactic: str) -> tuple[object | None, bool]:
    """Apply a tactic in a Dojo and classify the outcome.

    Args:
        dojo: The active lean_dojo session.
        state: The proof state to apply the tactic to.
        tactic: The tactic string.

    Returns:
        Tuple ``(result_state, proof_finished)``; ``(None, False)`` on
        errors or abandoned proofs.
    """
    try:
        result = dojo.run_tac(state, tactic)
    except Exception:
        return None, False
    if isinstance(result, ProofFinished):
        return result, True
    if isinstance(result, TacticState):
        if hasattr(result, "goals"):
            return result, len(result.goals) == 0
        return result, True
    if isinstance(result, (LeanError, ProofGivenUp)):
        return None, False
    return result, False


def theorem_wkn_evaluation(
    theorem: object,
    tactic_source: TacticSource,
    expansion_width: int,
    num_expansions: int,
    alpha: float = 0.0,
    sos_token: str = "@PROOF_START@",
    verbose: bool = False,
) -> tuple[list[str] | None, bool]:
    """Run best-first W x N proof search on a single theorem.

    Starting from the root state, the current frontier is expanded with
    ``expansion_width`` candidate tactics per state; results are added as
    weighted edges (cost ``-log P``, optionally length-normalised by
    ``(depth + 1) ** alpha``). After each round the frontier is re-selected
    as the minimum-cost root-to-leaf path. If an edge reaches
    ``ProofFinished`` the proof is reconstructed with DFS over the state
    graph and returned.

    Args:
        theorem: A ``lean_dojo.Theorem``.
        tactic_source: Model-specific candidate proposer.
        expansion_width: ``W``, candidates proposed per expanded state.
        num_expansions: ``N``, maximum number of frontier expansions.
        alpha: Length-normalisation exponent (0 disables; the neural prover
            runs used :data:`ALPHA` = 0.5).
        sos_token: Start-of-proof marker used as context padding.
        verbose: If True, print search-graph diagnostics.

    Returns:
        Tuple ``(proof, succeeded)`` where ``proof`` is the tactic list, or
        ``(None, False)`` when the budget is exhausted.

    Raises:
        RuntimeError: If ``lean_dojo`` is not installed.
    """
    if not LEAN_DOJO_AVAILABLE:
        raise RuntimeError(
            "lean_dojo is required for proof-search evaluation; "
            "install the 'cluster' extra on a machine with a Lean toolchain."
        )

    captured_output = StringIO()
    logger.add(captured_output)

    proof_expansions = 0
    states_list: list[object] = []
    d_state_id_to_node_index: dict[str, int] = {}
    d_state_id_to_preceding_tactics: dict[str, tuple[str, str]] = {}
    n_states = 0

    edges_u: list[int] = []
    edges_v: list[int] = []
    weight_costs: list[float] = []
    n_states_max = int(2 + num_expansions * expansion_width**2)
    proof_finished_state_id = str(hash(ProofFinished))
    d_state_id_to_node_index[proof_finished_state_id] = n_states_max
    tactic_by_state_pair = np.zeros((n_states_max + 1, n_states_max + 1)).astype(object)

    dojo, state = Dojo(theorem).__enter__()

    states_list.append(state)
    d_state_id_to_node_index[str(state.id)] = n_states
    d_state_id_to_preceding_tactics[str(state.id)] = (sos_token, sos_token)
    n_states += 1

    top_states = [state]
    top_states_depth = [0]

    def reconstruct_proof(result_node_index: int) -> list[str]:
        adjacency = np.zeros_like(tactic_by_state_pair)
        nonzero = tactic_by_state_pair.nonzero()
        adjacency[nonzero[0], nonzero[1]] = 1
        adjacency = adjacency.astype(int)
        path = dfs_find_path(adjacency, 0, result_node_index)
        return [
            tactic_by_state_pair[path[i]][path[i + 1]] for i in range(len(path) - 1)
        ]

    while proof_expansions < num_expansions:
        for top_state, top_state_depth in zip(top_states, top_states_depth):
            top_state_id = str(top_state.id)
            preceding = d_state_id_to_preceding_tactics[top_state_id]
            tactics, log_probs = tactic_source(top_state, preceding, expansion_width)
            for tactic, log_prob in zip(tactics, log_probs):
                if _is_tactic_timing_out_repeatedly(captured_output.getvalue(), tactic):
                    continue
                result, success = _run_tactic(dojo, top_state, tactic)
                if result is None:
                    continue
                states_list.append(result)

                top_state_node_index = d_state_id_to_node_index[top_state_id]

                if success:
                    result_id = proof_finished_state_id
                else:
                    result_id = str(result.id)
                    d_state_id_to_node_index[result_id] = n_states
                    d_state_id_to_preceding_tactics[result_id] = (preceding[1], tactic)
                result_node_index = d_state_id_to_node_index[result_id]
                n_states += 1

                edges_u.append(top_state_node_index)
                edges_v.append(result_node_index)
                weight = log_prob / np.power(top_state_depth + 1, alpha) if alpha else log_prob
                weight_costs.append(-weight)

                tactic_by_state_pair[top_state_node_index][result_node_index] = tactic

                if success:
                    return reconstruct_proof(result_node_index), True
            proof_expansions += 1

        states_graph = nx.DiGraph()
        states_graph.add_nodes_from(range(n_states))
        states_graph.add_weighted_edges_from(zip(edges_u, edges_v, weight_costs))
        top_states_indexes = best_first_min_cost_path_to_leaf(states_graph)
        if top_states_indexes is None:
            break
        top_states = [states_list[index] for index in top_states_indexes]
        top_states_depth = [
            len(nx.shortest_path(states_graph, source=0, target=index)) - 1
            for index in top_states_indexes
        ]
        if verbose:
            print(f"Frontier after expansion {proof_expansions}: {top_states_indexes}")
    return None, False


def dataset_wkn_evaluation(
    theorems: Sequence[object],
    tactic_source: TacticSource,
    expansion_width: int,
    num_expansions: int,
    results_sink: list | None = None,
    alpha: float = 0.0,
) -> list[tuple[list[str] | None, bool]]:
    """Evaluate one W x N pass over a dataset of theorems.

    Run this function ``K`` times (e.g. via ``torch.multiprocessing``) and
    take the union of solved theorems for the aggregated W x K x N metric
    reported in the README.

    Args:
        theorems: ``lean_dojo.Theorem`` objects to attempt.
        tactic_source: Model-specific candidate proposer (see
            :func:`unigram_source`, :func:`trigram_source`,
            :func:`hf_generator_source`).
        expansion_width: ``W``, candidates per expanded state.
        num_expansions: ``N``, frontier expansions per theorem.
        results_sink: Optional shared list (e.g. ``multiprocessing.Manager``
            list) to which the pass results are appended.
        alpha: Length-normalisation exponent forwarded to
            :func:`theorem_wkn_evaluation`.

    Returns:
        List of ``(proof, succeeded)`` tuples, one per theorem.
    """
    pass_results = [
        theorem_wkn_evaluation(
            theorem, tactic_source, expansion_width, num_expansions, alpha=alpha
        )
        for theorem in theorems
    ]
    if results_sink is not None:
        results_sink.append(pass_results)
    return pass_results


MakeSource = Callable[..., TacticSource]

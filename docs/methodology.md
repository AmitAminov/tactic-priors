# Methodology

This document describes how the models were built and how the miniF2F
numbers in the README were produced. It is adapted from the internal
research write-ups; only facts that can be traced to code or measured
outputs are reported.

## 1. Data

- **Training corpus:** LeanDojo Benchmark 4, `random/train.json` split
  (traced Mathlib4 theorems; benchmark generated from Mathlib4 commit
  `f5c3f06`, traced with LeanDojo 2.2.0). The JSON dumps are several GB and
  are not committed; see the README for how to download them.
- **Evaluation set:** the miniF2F benchmark (Lean 4 port), Test split,
  244 theorems as traced by LeanDojo.

## 2. Empirical tactic distribution (unigram model)

Every traced tactic invocation in the training split is counted verbatim
(the tactic string as it appears in the proof, arguments included). Tactics
observed exactly once are dropped and the distribution is renormalised over
the remaining 16,850 distinct tactic strings (454 of them span multiple
lines, so the raw CSV has 17,581 physical lines), giving
`artifacts/empirical_tactics_probability_mathlib4.csv` (~1.3 MB as a pickled
model). The unigram model proposes tactics by sampling this distribution
without replacement, ignoring the proof state entirely.

## 3. Trigram model with pseudo-tactic mechanism

Proof tactic sequences are padded with two `@PROOF_START@` / `@PROOF_END@`
markers and turned into trigrams `(w1, w2) -> w3`.

**Pseudo-word design.** Tactic vocabulary is extremely long-tailed. Tactics
whose empirical probability falls below the threshold `1e-5`
(`NGRAM_PROB_TH`) are collapsed into a single `@PSEUDO_TACTIC@` token before
counting. This keeps the context space tractable and reserves explicit
probability mass for the tail. At sampling time, a drawn pseudo-tactic is
materialised as a uniformly random tactic from the rare-tactic pool, with
its probability floored at the threshold value. The trained model
(`artifacts/trigram_mathlib4.pkl`) is ~13 MB.

**Smoothing.** The cleaned package implements Witten-Bell interpolation
between the trigram and bigram estimates,
`P_WB(w3|w1,w2) = lambda * P_MLE(w3|w1,w2) + (1 - lambda) * P_bigram(w3|w2)`
with `lambda = n / (n + t)` (`n` = context count, `t` = distinct
continuation types), with the backoff distribution restricted to the
observed trigram support and renormalised. Note an honesty caveat: the
**released artifact** `trigram_mathlib4.pkl`, which produced the reported
miniF2F numbers, was trained with per-context maximum-likelihood estimates
plus the pseudo-tactic mechanism (the `smoothing="mle"` path in
`tactic_priors.ngram_models`); unseen events are handled by the pseudo-word
and by backing off unseen contexts to `(SOS, w1)`, not by Witten-Bell
discounting.

**Unseen contexts.** When `(w1, w2)` was never observed, sampling backs off
to the `(SOS, w1)` context and finally to `(SOS, SOS)`.

## 4. Search procedure (W x K x N best-first)

All three systems -- unigram, trigram, and the neural prover -- are
evaluated with the same search driver (`tactic_priors.search`):

1. Open the theorem in LeanDojo (`Dojo`), yielding the root proof state.
2. **Expand:** for each state on the current frontier, ask the model for
   `W` candidate tactics with log-probabilities and execute them in the
   Dojo. Failed tactics (`LeanError`, `ProofGivenUp`, exceptions) are
   discarded; repeated tactic timeouts detected in the LeanDojo log are
   skipped.
3. Each successful tactic adds an edge to a weighted state DAG with cost
   `-log P(tactic)`. For the neural prover runs the log-probability is
   length-normalised: `weight = log P / (depth + 1) ** ALPHA` with
   `ALPHA = 0.5`.
4. **Select:** after each expansion round, the frontier is re-selected as
   the minimum cumulative-cost root-to-leaf path (uniform-cost best-first
   search over the DAG).
5. If a tactic reaches `ProofFinished`, the proof is reconstructed by DFS
   over the state graph and the theorem counts as solved.
6. A pass ends after `N` frontier expansions. `K` independent passes are
   run (as separate processes); a theorem counts as solved if any pass
   solves it.

**Budget used for all reported numbers:** `W = 3`, `K = 10`, `N = 10` on
the miniF2F Test split (244 theorems).

**Tactic proposal per model:**

- *Unigram:* sample `W` distinct tactics from the empirical distribution
  (state-independent).
- *Trigram:* sample `W` tactics conditioned on the two preceding tactics on
  the search path.
- *BFS-Prover:* sample `W` sequences from the model with prompt
  `state.pp + ":::"`, temperature 1.0, `top_p = 1.0`; sequence
  log-probability from `compute_transition_scores`.

## 5. Results and external reference

| System | miniF2F Test (W=3, K=10, N=10) |
|---|---|
| BFS-Prover (reproduction, same budget) | 49.6% (121/244) |
| Unigram prior | 26.2% (64/244) |
| Trigram prior | 22.1% (54/244) |

The BFS-Prover paper (arXiv:2502.03438) reports **72.95%** on miniF2F Test
under a much larger search budget (2048 x 2 x 600 = 2,457,600 in W x K x N
product terms, vs 3 x 10 x 10 = 300 here -- about 8,000x more). The 49.6%
reproduction under the shared small budget is the apples-to-apples
reference for the baselines above.

## 6. Limitations (honest)

- **Cluster dependency.** The evaluation requires LeanDojo tracing, a Lean
  toolchain, and (for BFS-Prover) GPU inference; it was run on a SLURM
  cluster and cannot be reproduced from this repository alone. The search
  driver is published as reference code with guarded imports.
- **Modest search budget.** `W=3, K=10, N=10` is small compared to the
  BFS-Prover paper's `2048 x 2 x 600`. Baseline/prover gaps may change at
  larger budgets; conclusions here are strictly about the shared-budget
  comparison.
- **Driver asymmetry in the original runs.** The original unigram driver
  used raw probabilities (not `-log P`) as edge weights, and length
  normalisation (`ALPHA = 0.5`) was applied only in the neural-prover
  driver. Per-expansion candidate ranking is unaffected, but cumulative
  path costs -- and hence frontier selection -- were not perfectly
  identical across model families. The unified driver in
  `tactic_priors.search` fixes this going forward; the reported numbers
  predate the fix.
- **Smoothing mismatch.** The released trigram artifact is MLE + pseudo-word,
  not Witten-Bell (see section 3).
- **Solved-index snapshot.** `artifacts/solved_indexes.json` is an
  intermediate comparison snapshot exported from the cluster; its set sizes
  (98/44/50) are smaller than the final aggregated counts (121/64/54)
  because it predates the last passes of each run. Overlap conclusions in
  the notebook are drawn from the snapshot and labelled as such.
- **Statement-level contamination is not controlled.** Mathlib4 tactic
  frequencies are used to prove miniF2F theorems; the domains differ, but
  no de-duplication analysis was performed.

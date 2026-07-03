# tactic-priors

**How much of neural theorem-proving performance is explained by simple
statistical priors?**

Neural provers for Lean 4 are typically compared against each other, rarely
against the dumbest possible baseline. This repository asks what a
*context-free frequency table* of Mathlib4 tactics -- and its trigram
extension -- can prove on miniF2F when plugged into exactly the same
best-first proof search as a 7B-parameter neural prover, under exactly the
same search budget.

## Approach

1. Count every traced tactic invocation in LeanDojo Benchmark 4
   (`random/train.json`, Mathlib4) to obtain an empirical tactic
   distribution (16,850 distinct tactic strings occurring more than once).
2. Build two priors: a **unigram** model (sample tactics by frequency,
   ignore the proof state) and a **trigram** model (condition on the two
   preceding tactics, with rare tactics collapsed into a pseudo-word below
   an empirical-probability threshold of 1e-5).
3. Evaluate all systems with the same W x K x N best-first search driver
   over LeanDojo: W candidate tactics per expanded state, N frontier
   expansions per pass, K independent passes, frontier re-selected by
   minimum cumulative `-log P` path (length-normalised with ALPHA = 0.5 for
   the neural prover). Details in [docs/methodology.md](docs/methodology.md).

## Results (miniF2F Test, 244 theorems, W=3 K=10 N=10)

| System | Params / size | Pass rate |
|---|---|---|
| BFS-Prover (reproduction, same budget) | ~7B | **49.6%** (121/244) |
| Unigram prior | ~1.3 MB pickle | **26.2%** (64/244) |
| Trigram prior | ~13 MB pickle | **22.1%** (54/244) |
| BFS-Prover paper (arXiv:2502.03438), budget 2048x2x600 | ~7B | 72.95% (external reference) |

## Key findings

- **Priors get you about half.** A 1.3 MB state-blind frequency table
  proves 64 of the 121 theorems the 7B prover proves under the same budget
  (26.2% vs 49.6% pass rate, a ratio of 0.53). Roughly half of
  small-budget neural proving performance on miniF2F is explained by
  "guess popular tactics".
- **More context, fewer proofs: trigram < unigram.** Conditioning on the
  two preceding tactics *reduces* search performance (22.1% vs 26.2%). The
  notebook's held-out analysis shows why this is a search effect, not a
  modelling failure -- see `notebooks/analysis.ipynb` for the measured
  proposal-diversity numbers.
- **The baselines are not a strict subset of the prover.** In the solved-set
  snapshot (`artifacts/solved_indexes.json`), several theorems are solved
  by the n-gram baselines but missed by BFS-Prover; exact overlap counts
  are computed in the notebook.

## Figures

![Tactic usage concentration in Mathlib4](figures/fig1_tactic_concentration.png)

![Headline comparison](figures/fig2_headline_comparison.png)

![Solved-set overlap](figures/fig3_solved_overlap.png)

![Held-out next-tactic prediction](figures/fig4_heldout_prediction.png)

## Repository map -- three reproducibility tiers

| Tier | What | Where |
|---|---|---|
| 1. Runnable offline (CPU, no Lean) | Models, artifacts, notebook, tests | `src/tactic_priors/ngram_models.py`, `src/tactic_priors/build_distribution.py`, `artifacts/`, `notebooks/`, `tests/` |
| 2. Reference cluster code (needs lean_dojo + GPU) | Unified W x K x N search driver | `src/tactic_priors/search.py` (guarded imports), design sketch in `sketches/llm_gating.py` |
| 3. Lean solutions (human-written) | miniF2F proofs replacing upstream `sorry` | `minif2f_solutions/` (79 Valid + 2 Test; see its README for licensing) |

```
tactic-priors/
├── src/tactic_priors/      # ngram_models, build_distribution, search
├── artifacts/              # tactic CSV, trained pickles, solved indexes
├── notebooks/analysis.ipynb# executed analysis (figures reproducible offline)
├── figures/                # publication figures (PNG 300dpi + SVG)
├── minif2f_solutions/      # Lean 4 proofs (statements: Apache-2.0 upstream)
├── sketches/               # design sketch, not used for results
├── docs/methodology.md     # search procedure, smoothing, limitations
└── tests/                  # pytest (CPU-only)
```

## Reproduce tier 1

```bash
git clone https://github.com/amitaminov/tactic-priors.git && cd tactic-priors
python -m venv .venv && . .venv/bin/activate   # or .venv\Scripts\activate
pip install -e ".[dev]"
pytest -q                                       # all CPU-only tests

# Rebuild the distribution + trigram model from scratch:
# download LeanDojo Benchmark 4 (generated from Mathlib4 commit f5c3f06,
# traced with LeanDojo 2.2.0) and point the builder at it:
python -m tactic_priors.build_distribution --benchmark-dir /path/to/leandojo_benchmark_4

# Re-execute the analysis notebook (uses committed artifacts; the held-out
# section additionally wants LEANDOJO_BENCHMARK_4_DIR for random/val.json):
jupyter nbconvert --to notebook --execute --inplace notebooks/analysis.ipynb
```

Note that the model files under `artifacts/*.pkl` are Python pickle files, so only load them from a source you trust; `artifacts/empirical_tactics_probability_mathlib4.csv` is the inspectable plain-text equivalent of the underlying tactic distribution.

Tier 2 (proof-search evaluation) additionally requires `lean_dojo==2.2.0`,
a Lean 4 toolchain, a traced miniF2F repository, and for BFS-Prover a GPU;
it was run on a university SLURM cluster.

## Context

This is thesis-adjacent research from my M.Sc. in Computer Science at the
Hebrew University of Jerusalem, studying what statistical structure neural
tactic generators actually add over corpus priors in automated theorem
proving. Ongoing work on the cluster: RL fine-tuning of tactic generators
(GRPO with verifier rewards).

## Contact

Amit Aminov -- amitaminov1@gmail.com

## License

MIT for original code and proofs; miniF2F theorem statements under
`minif2f_solutions/` are Apache-2.0 (OpenAI, Lean 4 port by Kaiyu Yang).
See [LICENSE](LICENSE) and [minif2f_solutions/README.md](minif2f_solutions/README.md).

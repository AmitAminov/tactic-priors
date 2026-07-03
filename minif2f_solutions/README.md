# miniF2F solutions (Lean 4)

Manually completed proofs for miniF2F benchmark theorems in Lean 4.

## Contents

| File | Theorems | Proved (no `sorry`) |
|---|---|---|
| `Valid.lean` | 244 | **79** |
| `Test.lean` | 244 | **2** |
| `Minif2fImport.lean` | Mathlib import list | - |

Counts are exact: a theorem is counted as proved iff its statement block
contains no `sorry`.

## Attribution and licensing

- **Theorem statements** are upstream work: the miniF2F benchmark,
  Copyright (c) 2021 OpenAI, authors Kunhao Zheng, Stanislas Polu,
  David Renshaw, OpenAI GPT-f; ported from Lean 3 to Lean 4 by Kaiyu Yang.
  Released under the Apache License 2.0. The original copyright headers are
  preserved at the top of `Valid.lean` and `Test.lean`.
- **Proof bodies** -- every tactic script that replaces an upstream
  `sorry` -- are original work by Amit Aminov and are released under this
  repository's MIT license.

## Notes

- These are human-written proofs, produced while studying the benchmark and
  building intuition for tactic selection. They are separate from, and not
  used by, the automated evaluation results reported in the repository
  README (those come from the W x K x N proof-search runs on the Test
  split).

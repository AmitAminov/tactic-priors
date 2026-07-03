"""Build the empirical Mathlib4 tactic distribution and the trigram model.

Reads the ``random/train.json`` split of LeanDojo Benchmark 4 and produces:

    1. ``empirical_tactics_probability_mathlib4.csv`` -- the tactic
       frequency table (tactics that occur more than once, with empirical
       probabilities renormalised over that subset), which backs
       :class:`tactic_priors.ngram_models.UnigramTacticsModel`.
    2. ``trigram_mathlib4.pkl`` -- a pickled
       :class:`tactic_priors.ngram_models.TrigramTacticsModel`.
    3. ``tactics_frequency_analysis_mathlib4.png`` -- an ECDF plot of tactic
       usage counts.

Usage::

    python -m tactic_priors.build_distribution --benchmark-dir /path/to/leandojo_benchmark_4 \
        --output-dir artifacts

The benchmark directory can also be given via the ``LEANDOJO_BENCHMARK_4_DIR``
environment variable. Download LeanDojo Benchmark 4 separately (see README);
the JSON dumps are too large to commit.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tactic_priors.ngram_models import TrigramTacticsModel  # noqa: E402

TITLE_FONT_SIZE = 22
AXIS_TITLE_FONT_SIZE = 20
AXIS_TICKS_FONT_SIZE = 16
YTICKS = np.arange(0, 1.1, 0.1).round(1)


def count_tactics(proofs: list[dict]) -> pd.DataFrame:
    """Count tactic invocations across traced proofs.

    Args:
        proofs: Traced proofs from a LeanDojo Benchmark 4 JSON split.

    Returns:
        DataFrame with ``tactic`` and ``counter`` columns, sorted by
        descending count.
    """
    d_tactics_counter: dict[str, int] = {}
    for proof in proofs:
        for traced_tactic in proof["traced_tactics"]:
            tactic = traced_tactic["tactic"]
            d_tactics_counter[tactic] = d_tactics_counter.get(tactic, 0) + 1
    df = pd.DataFrame(
        {"tactic": list(d_tactics_counter.keys()), "counter": list(d_tactics_counter.values())}
    )
    return df.sort_values(by="counter", ascending=False)


def build_frequency_csv(df: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    """Write the empirical tactic-probability CSV (singleton tactics dropped).

    Tactics observed exactly once are removed and probabilities are
    renormalised over the remaining tactics, matching the released artifact.

    Args:
        df: Tactic count table from :func:`count_tactics`.
        output_path: Destination CSV path.

    Returns:
        The filtered DataFrame with an added ``prob`` column.
    """
    df_non_one = df[df["counter"] != 1].copy()
    n_sum = df_non_one["counter"].sum()
    df_non_one["prob"] = df_non_one["counter"] / n_sum
    df_non_one.to_csv(output_path, index=False)
    return df_non_one


def plot_usage_ecdf(counts: np.ndarray, output_path: Path) -> None:
    """Plot the ECDF of tactic usage counts.

    Args:
        counts: Array of per-tactic usage counts.
        output_path: Destination PNG path.
    """
    sorted_counts = np.sort(counts)
    ecdf = np.arange(1, len(sorted_counts) + 1) / len(sorted_counts)
    xticks = np.linspace(0, np.round(sorted_counts.max(), -3), 10).round(-2).astype(int)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(np.concatenate([[0], sorted_counts]), np.concatenate([[0], ecdf]))
    ax.set_xticks(xticks)
    ax.set_yticks(YTICKS)
    ax.tick_params(labelsize=AXIS_TICKS_FONT_SIZE)
    ax.set_title("ECDF of tactic usage frequency in Mathlib4", fontsize=TITLE_FONT_SIZE)
    ax.set_ylabel("Probability", fontsize=AXIS_TITLE_FONT_SIZE)
    ax.set_xlabel("#Times used", fontsize=AXIS_TITLE_FONT_SIZE)
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    """Build all artifacts from the LeanDojo Benchmark 4 training split.

    Args:
        argv: Optional argument list (defaults to ``sys.argv[1:]``).
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        default=os.environ.get("LEANDOJO_BENCHMARK_4_DIR"),
        help="Path to leandojo_benchmark_4 (or set LEANDOJO_BENCHMARK_4_DIR)",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument(
        "--smoothing",
        choices=["witten_bell", "mle"],
        default="mle",
        help="Trigram estimator ('mle' reproduces the released artifact)",
    )
    args = parser.parse_args(argv)
    if args.benchmark_dir is None:
        parser.error("--benchmark-dir is required (or set LEANDOJO_BENCHMARK_4_DIR)")

    train_path = Path(args.benchmark_dir) / "random" / "train.json"
    with train_path.open(encoding="utf-8") as file:
        proofs = json.load(file)
    print(f"Loaded {len(proofs)} traced theorems from random/train.json")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = count_tactics(proofs)
    build_frequency_csv(df, args.output_dir / "empirical_tactics_probability_mathlib4.csv")
    plot_usage_ecdf(
        df["counter"].to_numpy(),
        args.output_dir / "tactics_frequency_analysis_mathlib4.png",
    )

    trigram = TrigramTacticsModel.from_traced_proofs(
        proofs, smoothing=args.smoothing, show_progress=True
    )
    sos = TrigramTacticsModel.SOS
    tactics, probs = trigram.get_top_k_tactics_and_probabilities(sos, sos, 5)
    print("Trigram top-5 proof-starting tactics:")
    for tactic, prob in zip(tactics, probs):
        print(f"  {100 * prob:6.2f}%  {tactic}")

    with (args.output_dir / "trigram_mathlib4.pkl").open("wb") as file:
        pickle.dump(trigram, file)
    print(f"Artifacts written to {args.output_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

"""Plot the performance of self-play training."""

import json
from pathlib import Path
from typing import Any, Dict, Final

import matplotlib.pyplot as plt
import numpy as np


def read_eval_file(eval_file_path: Path) -> Dict[str, Any]:
    """Read the stats for an evaluation.

    Parameters:
        eval_file_path: Path
            Path to the eval.json file

    Returns:
        stats: Dict[str, Any]
            Dictionary containing the stats inside the eval.json file.
    """
    with eval_file_path.open("rt") as f:
        stats = json.load(f)
    return stats


def get_average_winnings(model_dir: Path) -> np.ndarray:
    """Get the average winnings against baseline for each epoch.

    Parameters:
        model_dir: Path
            Path to the model directory

    Returns:
        avg_winnings: np.ndarray
            Array containing the average winnings against baseline for each epoch.
    """
    num_iterations: Final[int] = 6
    num_epochs: Final[int] = 10

    print(f"Model dir      : {model_dir}")
    print(f"    Iterations : {num_iterations}")
    print(f"    Epochs     : {num_epochs}")

    # Get average winnings against baseline for each epoch
    avg_winnings: np.ndarray = np.full((num_iterations, num_epochs), np.nan)

    for iteration in range(num_iterations):
        iteration_dir = model_dir / f"iteration_{iteration + 1}"
        iteration_eval_file = iteration_dir / "eval.json"

        if not iteration_dir.exists():
            continue
        if not iteration_eval_file.exists():
            continue

        eval_stats: Dict[str, Any] = read_eval_file(iteration_eval_file)

        for epoch in range(len(eval_stats)):
            avg_winnings[iteration, epoch] = eval_stats[f"epoch_{epoch + 1}"][
                "average_winnings"
            ]

    return avg_winnings


def main():
    """Run the script."""

    model_dirs: Final[Dict[str, Path]] = {
        "linear": Path("models/3p_3s/sb_bootstrap"),
        "cnn": Path("models/3p_3s/sb_bootstrap_cnn"),
    }

    # Get average winnings against baseline for each epoch
    model_avg_winnings: Final[Dict[str, np.ndarray]] = {
        model_name: get_average_winnings(model_dir)
        for model_name, model_dir in model_dirs.items()
    }

    fig, ax = plt.subplots(figsize=(5, 4), dpi=192)

    for m_index, (m_name, m_avg_winnings) in enumerate(model_avg_winnings.items()):
        num_iterations: int = m_avg_winnings.shape[0]
        num_epochs: int = m_avg_winnings.shape[1]

        for iteration in range(num_iterations):
            X: np.ndarray = np.arange(
                iteration * num_epochs,
                (iteration + 1) * num_epochs,
            )
            y: np.ndarray = m_avg_winnings[iteration, :]
            # print(iteration)
            # print(X)
            # print(y)

            ax.plot(X, y, label=m_name if iteration == 0 else None, color=f"C{m_index}")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Average winnings")

    # x-axis
    #
    # Primary: Epoch
    # Secondary: Iteration
    num_iterations = 6
    num_epochs = 10

    ax.set_xlim(0, num_iterations * num_epochs)
    ax.set_xticks(np.arange(0, (num_iterations * num_epochs) + 1, num_epochs / 2))
    ax.set_xlabel("Epoch")

    ax.axhline(0, color="black", linewidth=1)

    ax2 = ax.twiny()
    ax2.set_xlim(0, num_iterations * num_epochs)
    ax2.set_xticks(np.arange(0, (num_iterations * num_epochs) + 1, num_epochs))
    ax2.set_xticklabels(np.arange(0, num_iterations + 1))
    ax2.set_xlabel("Iteration")

    ax.grid(which="both", linestyle="-", color="lightgray", alpha=0.5)
    ax.set_axisbelow(True)

    ax.legend(
        bbox_to_anchor=(0.5, -0.175),
        loc="upper center",
        borderaxespad=0.0,
        frameon=False,
        ncols=2,
    )

    fig.tight_layout()
    fig.savefig("plots/ppo-selfplay.png", dpi=192)
    plt.show()


if __name__ == "__main__":
    main()

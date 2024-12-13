#!/usr/bin/env python3

"""Plot convergence of the NEAT model as it trains."""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Final, List

import matplotlib.pyplot as plt
import numpy as np

# Import neuropoker
sys.path.append(os.getcwd())
from neuropoker.models.neat.neat import NEATModel
from neuropoker.models.utils import get_model_from_pickle


def get_args() -> argparse.Namespace:
    """Get command-line arguments.

    Returns:
        argparse.Namespace: The command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Plot the convergence of the NEAT model."
    )
    parser.add_argument(
        "-m",
        "--model-files",
        type=Path,
        nargs="+",
        help="Path to the model files",
        default=[Path("./models/model_0.pkl")],
    )
    parser.add_argument(
        "-n",
        "--model-names",
        type=str,
        nargs="+",
        help="Names of the models",
        default=["model_0"],
    )
    return parser.parse_args()


def get_model_fitness(model: NEATModel) -> np.ndarray:
    """Get the per-generation fitness of the best genome in each generation.

    Parameters:
        model: NEATEvolution
            The NEAT model.

    Returns:
        fitness: np.ndarray
            The fitness of each generation.
    """
    return np.array([genome.fitness for genome in model.stats.most_fit_genomes])


def main():
    """Plot the convergence of the NEAT model."""

    args: Final[argparse.Namespace] = get_args()
    model_files: Final[List[Path]] = args.model_files
    model_names: Final[List[str]] = args.model_names
    fitnesses: Dict[str, np.ndarray] = {}

    # Load the models
    for model_name, model_file in zip(model_names, model_files):
        model: NEATModel = get_model_from_pickle(model_file)  # type: ignore
        fitnesses[model_name] = get_model_fitness(model)

    # print(model)
    # print(f"Generations: {model.population.generation}")
    # print(len(model.stats.most_fit_genomes))
    # print(model.population.species)
    # print(fitness)
    # print(fitness.shape)

    # Plot the fitness
    fig, ax = plt.subplots(dpi=192)

    for model_name, fitness in fitnesses.items():
        ax.plot(fitness, label=model_name)

    ax.set_title("Training Fitness")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness")
    ax.legend()

    fig.tight_layout()

    plot_dir: Final[Path] = Path("plots")
    plot_dir.mkdir(exist_ok=True)

    plot_file: Final[Path] = plot_dir / "fitness.png"
    print(f"Saving plot to {plot_file}...")
    plt.savefig(plot_file)


if __name__ == "__main__":
    main()

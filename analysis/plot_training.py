#!/usr/bin/env python3

"""Plot convergence of the NEAT model as it trains.
"""
import argparse
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, Final, List

import matplotlib.pyplot as plt
import numpy as np

# Import neuropoker
sys.path.append(os.getcwd())
from neuropoker.models.neat_model import NEATEvolution


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
        help="Path to the NEAT model files",
        default=[Path("./models/model_0.pkl")],
    )
    return parser.parse_args()


def get_model_fitness(model: NEATEvolution) -> np.ndarray:
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
    """Plot the convergence of the NEAT model.

    TODO: Validate results against an actual run.
    """

    args: Final[argparse.Namespace] = get_args()
    model_files: Final[List[Path]] = args.model_files
    fitnesses: Dict[str, np.ndarray] = {}

    # Load the models
    for model_file in model_files:
        print(f"Loading model from {model_file}...")
        with model_file.open("rb") as mf:
            model: NEATEvolution = pickle.load(mf)
            model_name: str = model_file.name.replace(".pkl", "")
            fitnesses[model_name] = get_model_fitness(model)

    # print(model)
    # print(f"Generations: {model.population.generation}")
    # print(len(model.stats.most_fit_genomes))
    # print(model.population.species)
    # print(fitness)
    # print(fitness.shape)

    # Plot the fitness
    fig, ax = plt.subplots()

    for model_name, fitness in fitnesses.items():
        ax.plot(fitness, label=model_name)

    ax.set_title("Training Fitness")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness")
    ax.legend()

    fig.tight_layout()

    plot_dir: Final[Path] = Path("plots")
    plot_dir.mkdir(exist_ok=True)

    plot_file: Final[Path] = plot_dir / f"fitness.png"
    print(f"Saving plot to {plot_file}...")
    plt.savefig(plot_file)


if __name__ == "__main__":
    main()

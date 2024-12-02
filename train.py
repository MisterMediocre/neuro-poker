#!/usr/bin/env python3

"""Run NEAT to neuro-evolve poker players using self-play.
"""

import argparse
import os
import pickle
from pathlib import Path
from typing import Final, List

from neuropoker.model import NEATEvolution, run_neat

DEFAULT_EVOLUTION_FILE: Final[Path] = Path("models/neat_poker.pkl")
DEFAULT_CONFIG_FILE: Final[Path] = Path("configs/3p_3s_neat")
DEFAULT_NUM_GENERATIONS: Final[int] = 50
DEFAULT_NUM_CORES: Final[int] = os.cpu_count() or 1
DEFAULT_BATCH_SIZE: Final[int] = 5


def get_args() -> argparse.Namespace:
    """Get arguments for the script.

    Returns:
        args: argparse.Namespace
            Arguments.
    """
    parser = argparse.ArgumentParser(description="Train a poker player using NEAT.")

    parser.add_argument(
        "-f",
        "--evolution-file",
        type=Path,
        default=DEFAULT_EVOLUTION_FILE,
        help=(
            "File to read/save the evolution to/from. "
            f"(default: {DEFAULT_EVOLUTION_FILE})"
        ),
    )
    parser.add_argument(
        "-m",
        "--config-file",
        type=Path,
        default=DEFAULT_CONFIG_FILE,
        help=(
            "File to read the NEAT model configuration from. "
            f"(default: {DEFAULT_CONFIG_FILE})"
        ),
    )
    parser.add_argument(
        "-g",
        "--num-generations",
        type=int,
        default=DEFAULT_NUM_GENERATIONS,
        help=(f"Number of generations to run. (default: {DEFAULT_NUM_GENERATIONS})"),
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=(f"The size of each batch. (default: {DEFAULT_BATCH_SIZE})"),
    )
    parser.add_argument(
        "-c",
        "--num-cores",
        type=int,
        default=DEFAULT_NUM_CORES,
        help=(
            "Number of cores to use for evaluation. " f"(default: {DEFAULT_NUM_CORES})"
        ),
    )
    parser.add_argument(
        "-o",
        "--opponents",
        type=str,
        nargs="+",
        default=["CallPlayer"],
        help=("Name of the opponents from which to sample from"),
    )

    return parser.parse_args()


def main() -> None:
    """Run the script."""

    # Read args
    args: Final[argparse.Namespace] = get_args()
    evolution_file: Final[Path] = args.evolution_file
    config_file: Final[Path] = args.config_file
    num_cores: Final[int] = args.num_cores
    num_generations: Final[int] = args.num_generations
    batch_size: Final[int] = args.batch_size
    opponents: Final[List[str]] = args.opponents

    print("Running with:")
    print(f"    Evolution file : {evolution_file}")
    print(f"    Config file    : {config_file}")
    print(f"    Generations    : {num_generations}")
    print(f"    Cores          : {num_cores}")
    print(f"    Batch size     : {batch_size}")
    print(f"    Opponents      : {opponents}")
    print()

    # Initialize variables
    evolution = None  # Evolved population

    # Load previous evolution, if it exists
    if evolution_file.exists():
        # Load evolution
        print(f"Loading evolution file from {evolution_file}...")
        with evolution_file.open("rb") as ef:
            evolution = pickle.load(ef)

            if not isinstance(evolution, NEATEvolution):
                raise ValueError("Evolution file is invalid")

            print(
                f"Running until generation {num_generations + evolution.population.generation}..."
            )
    else:
        # Start from scratch
        print("Starting new evolution")
        print(f"Running until generation {num_generations}...")

    # Run NEAT

    # Save model after every <batch_size> generations
    # Reset fitness values as well, so that bad history is not carried over
    num_batches = num_generations // batch_size

    for i in range(num_batches):
        print(f"Running batch {i+1}/{num_batches}")
        evolution = run_neat(
            opponents,
            evolution=evolution,
            config_file=config_file,
            num_generations=batch_size,
            num_cores=num_cores,
        )

        # Save model
        with evolution_file.open("wb") as ef:
            pickle.dump(evolution, ef)
            print(f"Saved evolution at {evolution_file}")


if __name__ == "__main__":
    main()

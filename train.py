#!/usr/bin/env python3

"""Run NEAT to neuro-evolve poker players using self-play.
"""

import argparse
import os
from pathlib import Path
from typing import Final, List

from termcolor import colored

from neuropoker.config import Config
from neuropoker.model_utils import (
    get_model_from_config,
    get_model_from_pickle,
    save_model_to_pickle,
)
from neuropoker.models.neat_model import NEATModel
from neuropoker.player_utils import load_player, player_type_from_string
from neuropoker.players.base_player import BasePlayer

DEFAULT_CONFIG_FILE: Final[Path] = Path("configs/3p_4s_neat.toml")
DEFAULT_MODEL_FILE: Final[Path] = Path("models/neat_poker.pkl")
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
        "-t",
        "--config-file",
        type=Path,
        default=DEFAULT_CONFIG_FILE,
        help=(
            "The configuration file that defines the model and game."
            f"(default: {DEFAULT_CONFIG_FILE})"
        ),
    )
    parser.add_argument(
        "-f",
        "--model-file",
        type=Path,
        default=DEFAULT_MODEL_FILE,
        help=(
            f"The file to read and/or save the model to. (default: {DEFAULT_MODEL_FILE})"
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
        help=("Names of the opponents from which to sample from"),
    )

    return parser.parse_args()


def main() -> None:
    """Run the script."""

    # Read args
    args: Final[argparse.Namespace] = get_args()

    config_file: Final[Path] = args.config_file
    model_file: Final[Path] = args.model_file

    num_cores: Final[int] = args.num_cores
    num_generations: Final[int] = args.num_generations
    batch_size: Final[int] = args.batch_size
    opponents: Final[List[str]] = args.opponents

    print(colored("------------ train ------------", color="blue", attrs=["bold"]))
    print(
        colored(f'{"Config file":<12}', color="blue", attrs=["bold"])
        + colored(f": {config_file}", color="blue")
    )
    print(
        colored(f'{"Model file":<12}', color="blue", attrs=["bold"])
        + colored(f": {model_file}", color="blue")
    )
    print(
        colored(f'{"Generations":<12}', color="blue", attrs=["bold"])
        + colored(f": {num_generations}", color="blue")
    )
    print(
        colored(f'{"Cores":<12}', color="blue", attrs=["bold"])
        + colored(f": {num_cores}", color="blue")
    )
    print(
        colored(f'{"Batch size":<12}', color="blue", attrs=["bold"])
        + colored(f": {batch_size}", color="blue")
    )
    print(
        colored(f'{"Opponents":<12}', color="blue", attrs=["bold"])
        + colored(f": {opponents}", color="blue")
    )
    print()

    # Read config file
    print(colored(f"Reading config file {config_file}...", color="blue"))
    print()

    config: Final[Config] = Config(config_file)

    model_type: Final[str] = config["model"]["type"]
    game_suits: Final[List[str]] = config["game"]["suits"]
    game_ranks: Final[List[str]] = config["game"]["ranks"]
    game_players: Final[int] = config["game"]["players"]

    print(colored("Model:", color="blue", attrs=["bold"]))
    print(
        colored(f"{'    Type':<12}", color="blue", attrs=["bold"])
        + colored(f": {model_type}", color="blue")
    )
    print(colored("Game:", color="blue", attrs=["bold"]))
    print(
        colored(f"{'    Suits':<12}", color="blue", attrs=["bold"])
        + colored(f": {len(game_suits)} {game_suits}", color="blue")
    )
    print(
        colored(f"{'    Ranks':<12}", color="blue", attrs=["bold"])
        + colored(f": {len(game_ranks)} {game_ranks}", color="blue")
    )
    print(
        colored(f"{'    Players':<12}", color="blue", attrs=["bold"])
        + colored(f": {game_players}", color="blue")
    )
    print()

    # Set up the model
    #
    # Load previous model, if it exists
    # Otherwise, start from scratch
    model: NEATModel
    last_generation: int = num_generations

    if model_file.exists():
        model = get_model_from_pickle(model_file)  # type: ignore
        last_generation += model.population.generation
    else:
        model = get_model_from_config(config)  # type: ignore

    model.print_config()

    # Set up the opponents
    opponent_players: List[BasePlayer] = [
        load_player(
            player_type_from_string(opponent),
            f"opponent-{i}",
            model_type=None,
            model_file=None,
        )
        for i, opponent in enumerate(opponents)
    ]

    # print(opponent_players)

    # Train the model
    #
    # Checkpoint the model after every <batch_size> generations
    # Reset fitness values as well, so that bad history is not carried over
    initial_generation = last_generation - num_generations

    print(
        colored(
            f"Running for {num_generations} generations", color="blue", attrs=["bold"]
        ),
    )

    print(
        colored(f"{'First generation':<20}", color="blue", attrs=["bold"])
        + colored(f": {initial_generation}", color="blue")
    )
    print(
        colored(f"{'Last generation':<20}", color="blue", attrs=["bold"])
        + colored(f": {last_generation}", color="blue")
    )
    print(
        colored(f"{'Checkpoint interval':<20}", color="blue", attrs=["bold"])
        + colored(f": Every {batch_size} generations", color="blue")
    )

    for gen_i in range(initial_generation, last_generation):
        model.run(
            opponent_players,
            config,
            num_generations=1,
            num_cores=num_cores,
        )

        # Save model
        if (gen_i + 1) % batch_size == 0 or (gen_i == last_generation - 1):
            save_model_to_pickle(model, model_file)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

"""Run NEAT to neuro-evolve poker players using self-play.
"""

import argparse
import os
from pathlib import Path
from typing import Final, List

from neuropoker.cards import get_card_list
from neuropoker.config import Config
from neuropoker.models.neat_model import NEATModel
from neuropoker.player_utils import load_player, player_type_from_string
from neuropoker.players.base_player import BasePlayer

DEFAULT_CONFIG_FILE: Final[Path] = Path("configs/3p_3s_neat.toml")
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

    print("Running with:")
    print(f"    Config file    : {config_file}")
    print(f"    Model file     : {model_file}")
    print()
    print(f"    Generations    : {num_generations}")
    print(f"    Cores          : {num_cores}")
    print(f"    Batch size     : {batch_size}")
    print(f"    Opponents      : {opponents}")
    print()

    # Read config file
    print(f"Reading config file {config_file}...")
    config: Final[Config] = Config(config_file)

    model_type: Final[str] = config["model"]["type"]
    game_suits: Final[List[str]] = config["game"]["suits"]
    game_ranks: Final[List[str]] = config["game"]["ranks"]
    game_cards: Final[List[str]] = get_card_list(game_suits, game_ranks)
    game_players: Final[int] = config["game"]["players"]

    print("    Model:")
    print(f"        Model type   : {model_type}")
    print("    Game:")
    print(f"        Suits        : {len(game_suits)} {game_suits}")
    print(f"        Ranks        : {len(game_ranks)} {game_ranks}")
    print(f"        Deck size    : {len(game_cards)}")
    print(f"        Players      : {game_players}")
    print(f"    Model type       : {model_type}")
    print()

    # Set up the model
    #
    # Load previous model, if it exists
    # Otherwise, start from scratch
    if model_file.exists():
        print(f"Loading model file from {model_file}...")
        model: NEATModel = NEATModel.from_pickle(model_file)
        print(
            f"Running until generation {num_generations + model.population.generation}..."
        )
    else:
        neat_config_file: Final[Path] = config["model"]["neat_config_file"]

        print("Starting new model")
        print(f"Using NEAT config file {neat_config_file}")
        print(f"Running until generation {num_generations}...")
        model = NEATModel(neat_config_file=neat_config_file)

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

    # Train the model
    #
    # Checkpoint the model after every <batch_size> generations
    # Reset fitness values as well, so that bad history is not carried over
    num_batches = num_generations // batch_size

    for i in range(num_batches):
        print(f"Running batch {i+1}/{num_batches}")

        model.run(
            opponent_players,
            config,
            num_generations=batch_size,
            num_cores=num_cores,
        )

        # Save model
        model.to_pickle(model_file)


if __name__ == "__main__":
    main()

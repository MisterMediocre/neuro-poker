#!/usr/bin/env python3

"""Run NEAT to neuro-evolve poker players using self-play."""

import argparse
import os
from pathlib import Path
from typing import Final, List

from termcolor import colored

from neuropoker.extra.config import Config
from neuropoker.models.neat.neat import NEATModel
from neuropoker.models.utils import (
    get_model_from_config,
    get_model_from_pickle,
    save_model_to_pickle,
)
from neuropoker.players.utils import (
    PlayerDefinition,
    player_type_from_string,
)

DEFAULT_CONFIG_FILE: Final[Path] = Path("configs/3p_4s_neat.toml")
DEFAULT_MODEL_FILE: Final[Path] = Path("models/neat_poker.pkl")
DEFAULT_NUM_GENERATIONS: Final[int] = 50
DEFAULT_NUM_GAMES: Final[int] = 400
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
        "--model-file",
        type=Path,
        default=DEFAULT_MODEL_FILE,
        help=(
            f"The file to read and/or save the model to. (default: {DEFAULT_MODEL_FILE})"
        ),
    )
    parser.add_argument(
        "-m",
        "--config-file",
        type=Path,
        default=DEFAULT_CONFIG_FILE,
        help=(
            "The configuration file that defines the model and game."
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
        "--num-games",
        type=int,
        default=DEFAULT_NUM_GAMES,
        help=(
            "Number of games to play for each genome in each generation. "
            f"(default: {DEFAULT_NUM_GAMES})"
        ),
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
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=None,
        help=("Seed for the random number generator."),
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
    num_games: Final[int] = 400
    batch_size: Final[int] = args.batch_size
    opponents: Final[List[str]] = args.opponents
    seed: Final[int] = args.seed

    print(
        colored("---------------- Args ----------------", color="blue", attrs=["bold"])
    )
    print(colored(f'{"Config file":<12}: ', color="blue") + f"{config_file}")
    print(colored(f'{"Model file":<12}: ', color="blue") + f"{model_file}")
    print(colored(f'{"Generations":<12}: ', color="blue") + f"{num_generations}")
    print(colored(f'{"Games":<12}: ', color="blue") + f"{num_games}")
    print(colored(f'{"Cores":<12}: ', color="blue") + f"{num_cores}")
    print(colored(f'{"Batch size":<12}: ', color="blue") + f"{batch_size}")
    print(colored(f'{"Opponents":<12}: ', color="blue") + f"{opponents}")
    print(colored(f'{"Seed":<12}: ', color="blue") + f"{seed}")
    print()

    # Read config file
    print(f"Reading config file {config_file}...")
    print()

    config: Final[Config] = Config(config_file)

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
    opponent_players: List[PlayerDefinition] = [
        PlayerDefinition(
            player_type_from_string(opponent),
            model_type=None,
            model_file=None,
        )
        for opponent in opponents
    ]

    # Print game config
    #
    game_suits: Final[List[str]] = config["game"]["suits"]
    game_ranks: Final[List[str]] = config["game"]["ranks"]
    game_players: Final[int] = config["game"]["players"]

    print(
        colored("---------------- Game ----------------", color="blue", attrs=["bold"])
    )
    print(colored(f"{'Suits':<12}: ", color="blue") + f"{len(game_suits)} {game_suits}")
    print(colored(f"{'Ranks':<12}: ", color="blue") + f"{len(game_ranks)} {game_ranks}")
    print(colored(f"{'Players':<12}: ", color="blue") + f"{game_players}")
    print()

    # print(opponent_players)

    # Train the model
    #
    # Checkpoint the model after every <batch_size> generations
    # Reset fitness values as well, so that bad history is not carried over
    initial_generation: Final[int] = last_generation - num_generations
    num_batches: Final[int] = num_generations // batch_size

    print(
        colored("-------------- Training --------------", color="blue", attrs=["bold"])
    )
    print(
        colored(f"{'Generations':<12}: ", color="blue")
        + f"{initial_generation} to {last_generation} ({num_generations} total)"
    )
    print(
        colored(f"{'Checkpoints':<12}: ", color="blue")
        + f"Every {batch_size} generations"
    )

    for _ in range(num_batches):
        # Run NEAT
        model.run(
            opponent_players,
            config,
            seed=seed,
            num_generations=batch_size,
            num_games=num_games,
            num_cores=num_cores,
        )

        # Save model
        save_model_to_pickle(model, model_file)


if __name__ == "__main__":
    main()

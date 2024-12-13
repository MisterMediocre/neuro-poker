#!/usr/bin/env python3

"""Plot the actions of a model based on its hand at the dealer position."""

import argparse
import itertools
import os
import sys
from pathlib import Path
from typing import Final, Generator, List

import numpy as np

# Import neuropoker
sys.path.append(os.getcwd())
from neuropoker.extra.config import Config
from neuropoker.game.cards import get_card_list
from neuropoker.models.neat.neat import NEATModel
from neuropoker.models.utils import get_model_from_pickle
from neuropoker.players.neat import NEATPlayer


def get_args() -> argparse.Namespace:
    """Get command-line arguments.

    Returns:
        argparse.Namespace: The command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Plot the actions of a model based on its private hand, "
            "when it is playing at the dealer position."
        )
    )
    parser.add_argument(
        "-c",
        "--config-file",
        type=Path,
        help="Path to the configuration file",
        default=Path("./configs/3p_3s_neat.toml"),
    )
    parser.add_argument(
        "-m",
        "--model-file",
        type=Path,
        help="Path to the model file",
        default=Path("./models/model_0.pkl"),
    )
    parser.add_argument(
        "-n",
        "--model-name",
        type=str,
        help="Name of the model",
        default="model_0",
    )
    return parser.parse_args()


def iterate_hole_cards(config: Config) -> Generator[List[str], None, None]:
    """Generate an iterator over all possible hole cards.

    Parameters:
        config: Config
            The game configuration.

    Yields:
        hand: List[str]
            A list of two cards representing a possible private hand.
    """
    print(config["game"])
    suits: Final[List[str]] = config["game"]["suits"]
    ranks: Final[List[str]] = config["game"]["ranks"]
    cards: Final[List[str]] = get_card_list(suits, ranks)
    hand_size: Final[int] = 2  # TODO: Un-hardcode?

    # Generate all possible samples of two cards, without replacement
    for hand in itertools.combinations(cards, hand_size):
        yield list(hand)


def is_hand_suited(hand: List[str]) -> bool:
    """Determine if a hand is suited or unsuited.

    Parameters:
        hand: List[str]
            The hand to evaluate.

    Returns:
        suited: bool
            Whether both cards in the hand have the same suit.
    """
    if len(hand) != 2:
        raise ValueError("The hand must contain exactly two cards.")

    return hand[0][0] == hand[1][0]


def get_model_actions(model: NEATModel) -> np.ndarray:
    """Get the actions of the model based on its hand.

    Parameters:
        model: NEATModel
            The model to evaluate.

    Returns:
        actions: np.ndarray
            A 3D array representing the (probabilistic) actions
            of the model based on its hand.

            Each cell represents one of the possible hands, given
            two private cards.

            Upper-right diagonal: Cards with the same suit ("suited")
            Lower-left diagonal: Cards with different suits ("unsuited")
            Values: 3-tuples of probabilities (call, fold, raise)
    """
    print(model)

    player = NEATPlayer("player", model.get_best_genome_network(), training=False)
    print(player)

    # action, amount = player.declare_action()

    return np.zeros((13, 13, 3))


def main():
    """Plot the actions taken by the model."""

    args: Final[argparse.Namespace] = get_args()
    config_file: Final[Path] = args.config_file
    model_file: Final[Path] = args.model_file
    model_name: Final[str] = args.model_name

    # Load the configuration
    config: Final[Config] = Config(config_file)

    for hand in iterate_hole_cards(config):
        print(is_hand_suited(hand), hand)

    # Load the model
    model: Final[NEATModel] = get_model_from_pickle(model_file)  # type: ignore
    actions: Final[np.ndarray] = get_model_actions(model)
    # print(actions)


if __name__ == "__main__":
    main()

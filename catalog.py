#!/usr/bin/env python3

"""Run a catalog of poker games with different players.
"""
from typing import Dict, Final, List
import random
import time

from neuropoker.game import (
    PlayerStats,
    default_player_stats,
    evaluate_performance,
    merge,
)
from neuropoker.model import load_player
from neuropoker.players.base import BasePlayer

CATALOG: Final[Dict[str, BasePlayer]] = {
    # Random players
    "random": load_player("RandomPlayer", "random"),
    "random2": load_player("RandomPlayer", "random2"),
    "random3": load_player("RandomPlayer", "random3"),
    # Fold players
    "fold": load_player("FoldPlayer", "fold"),
    "fold2": load_player("FoldPlayer", "fold2"),
    "fold3": load_player("FoldPlayer", "fold3"),
    # Call players
    "call": load_player("CallPlayer", "call"),
    "call2": load_player("CallPlayer", "call2"),
    "call3": load_player("CallPlayer", "call3"),
    # model_0 has been trained against the fold player, for playing 4-suit 3-player
    "model_0": load_player("models/3p_4s/model_0.pkl", "model_0"),
    # model_1 has been trained against the call player, for playing 4-suit 3-player
    # "model_1": load_player("models/3p_4s/model_1.pkl", "model_1"),
    "model_1": load_player("models/3p_3s/model_0.pkl", "model_1"),
}


def compete(player_1: str, player_2: str, player_3: str, num_games: int = 100) -> None:
    """Run a multi-player poker game.

    Parameters:
        player_1: str
            The name of the first player, from the catalog.
        player_2: str
            The name of the second player, from the catalog.
        player_3: str
            The name of the third player, from the catalog.
        num_games: int
            The number of games to play.
    """

    player_names: Final[List[str]] = [player_1, player_2, player_3]

    print("\n\n")
    print(f"In the competition, the players are: {player_names}")
    print(f"They play {num_games} games from each position.")

    player_models: Final[List[BasePlayer]] = [
        CATALOG[player_name] for player_name in player_names
    ]

    performances: Dict[str, PlayerStats]
    performances = {}
    for i in range(0, 3):
        default = default_player_stats()
        default["uuid"] = player_names[i]
        performances[player_names[i]] = default


    random.seed(time.time())
    seed = random.randint(0, 1000)
    for i in range(0, 3):
        # Shift the players to the left
        player_names_i = player_names[i:] + player_names[:i]
        player_models_i = player_models[i:] + player_models[:i]

        performance = evaluate_performance(
            player_names_i, player_models_i, num_games, seed=seed
        )
        for player_name, stats in performance.items():
            performances[player_name] = merge(performances[player_name], stats)

    for player_name, stats in performances.items():
        print(player_name)
        print(stats)
        print("Average winning:", stats["winnings"] / stats["num_games"])

    print("\n\n")


def main():
    """Run the script."""
    # Expect each to win 0 on average
    # compete("fold", "fold2", "fold3", 30)

    # Expect call player to win every single round, on average 50
    # When small blind, will win 50
    # When big blind, will win 25
    # When neither, will win 75
    # compete("call", "fold", "fold2", 1)

    # Expect to match call player's performance, ie average 50
    # Success
    # compete("model_0", "fold", "fold2", 30)

    # Expect same as before, since we're invariant to the order of the players
    # Success
    # compete("fold", "model_0", "fold2", 30)

    # compete("call", "call2", "call3", 300)

    # compete("call", "model_1", "fold", 100)

    # Expect the model to fully exploit the folders
    # compete("model_1", "fold", "fold2", 300)

    # Expect the model to fully exploit the call players
    compete("call", "call2", "call3", 100)
    compete("model_1", "call", "call2", 500)

    # Expect the models to tie, while taking advantage of the fold player
    # compete("model_1", "model_1_2", "fold", 30)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

"""Run a catalog of poker games with different players.
"""
from typing import Dict, Final, List

from neuropoker.game import PlayerStats, evaluate_performance
from neuropoker.model import load_player
from neuropoker.player import BasePlayer

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
    # model_0 has been trained against the fold player
    "model_0": load_player("models/model_0.pkl", "model_0"),
    # model_1 has been trained against the call player
    "model_1": load_player("models/model_1.pkl", "model_1"),
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
    print(f"They play {num_games} games.")

    player_models: Final[List[BasePlayer]] = [
        CATALOG[player_name] for player_name in player_names
    ]

    performances = evaluate_performance(player_names, player_models, num_games)
    performances: Dict[str, PlayerStats]

    for player_name, stats in performances.items():
        print(player_name)
        print(stats)
        print("Average winning:", stats["winnings"] / stats["num_games"])
        print("\n")

    print("\n\n")


def main():
    """Run the script."""
    # Expect each to win 0 on average
    compete("fold", "fold2", "fold3", 3)

    # Expect call player to win every single round, on average 50
    # When small blind, will win 50
    # When big blind, will win 25
    # When neither, will win 75
    compete("call", "fold", "fold2", 3)

    # Random is an absolutely horrible player, since he will often
    # raise first and fold later.
    # compete("random", "fold", "call", 30)

    # Expect to match call player's performance, ie average 50
    compete("model_0", "fold", "fold2", 30)

    # Expect to win the rounds with good cards, and be conservative
    # with bad cards
    # compete("model_0", "call", "fold", 30)

    # compete("call", "call2", "call3", 10)
    # compete("random", "fold", "call", 10)


if __name__ == "__main__":
    main()

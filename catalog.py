#!/usr/bin/env python3

"""Run a catalog of poker games with different players.
"""
import random
import time
from typing import Dict, Final, List

from neuropoker.config import Config
from neuropoker.game import Game, PlayerStats, default_player_stats, merge
from neuropoker.player_utils import load_player
from neuropoker.players.base_player import BasePlayer
from neuropoker.players.naive_player import CallPlayer, FoldPlayer, RandomPlayer

CATALOG: Final[Dict[str, BasePlayer]] = {
    # Random players
    "random": load_player(RandomPlayer, "random"),
    "random2": load_player(RandomPlayer, "random2"),
    "random3": load_player(RandomPlayer, "random3"),
    # Fold players
    "fold": load_player(FoldPlayer, "fold"),
    "fold2": load_player(FoldPlayer, "fold2"),
    "fold3": load_player(FoldPlayer, "fold3"),
    # Call players
    "call": load_player(CallPlayer, "call"),
    "call2": load_player(CallPlayer, "call2"),
    "call3": load_player(CallPlayer, "call3"),
    # model_0 has been trained against the fold player, for playing 4-suit 3-player
    # "model_0": load_player(
    #     NEATPlayer, "model_0", NEATModel, Path("models/3p_4s/model_0.pkl")
    # ),
    # # model_1 has been trained against the call player, for playing 4-suit 3-player
    # "model_1": load_player(
    #     NEATPlayer, "model_1", NEATModel, Path("models/3p_3s/model_0.pkl")
    # ),
}


def compete(
    player_names: List[str],
    config: Config,
    num_games: int = 100,
) -> None:
    """Run a multi-player poker game.

    Parameters:
        player_names: List[str]
            The name of each player, taken from the catalog.
        config: Config
            The game configuration.
        num_games: int
            The number of games to play.
    """

    if len(player_names) != config["game"]["players"]:
        raise ValueError(
            "The number of players in the game configuration "
            f"({config['game']['players']})"
            "does not match the number of players provided "
            f"({len(player_names)})"
        )

    print("\n\n")
    print(f"In the competition, the players are: {player_names}")
    print(f"They play {num_games} games from each position.")

    players: Final[List[BasePlayer]] = [
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

    for i in range(0, len(players)):
        # Shift the players
        players_: List[BasePlayer] = players[i:] + players[:i]

        # Run the game
        game: Game = Game.from_config(players_, config)
        game_stats: Dict[str, PlayerStats] = game.play_multiple(
            num_games=num_games, seed=seed
        )

        for player_name, stats in game_stats.items():
            performances[player_name] = merge(performances[player_name], stats)

    for player_name, stats in performances.items():
        print(player_name)
        print(stats)
        print("Average winning:", stats["winnings"] / stats["num_games"])

    print("\n\n")


def main():
    """Run the script."""

    # TODO: Configure config at runtime
    config: Final[Config] = Config("configs/3p_4s_neat.toml")

    # 3 fold:
    # Expect each player to win 0 on average
    print("~~~ 3 fold ~~~")
    compete(["fold", "fold2", "fold3"], config)

    # 3 call:
    # Expect each player to win 0 on average
    print("~~~ 3 call ~~~")
    compete(["call", "call2", "call3"], config)

    # 1 call + 2 fold:
    #
    # Expect call player to win every single round, on average 50
    # - When small blind, will win 50
    # - When big blind, will win 25
    # - When neither, will win 75
    print("~~~ 1 call + 2 fold ~~~")
    compete(["call", "fold", "fold2"], config)

    # 2 call + 1 fold:
    #
    # Expect each call player to win ~50% of the rounds on average.
    # - Each caller wins ~12.5 per game
    # - The fold player loses ~25.0 per game
    print("~~~ 2 call + 1 fold ~~~")
    compete(["call", "call2", "fold"], config)

    # Expect to match call player's performance, ie average 50
    # Success
    # compete("model_0", "fold", "fold2", 30)q.q

    # Expect same as before, since we're invariant to the order of the players
    # Success
    # compete("fold", "model_0", "fold2", 30)

    # compete("call", "call2", "call3", 300)

    # compete("call", "model_1", "fold", 100)

    # Expect the model to fully exploit the folders
    # compete("model_1", "fold", "fold2", 300)

    # Expect the model to fully exploit the call players
    # compete("model_1", "call", "call2", 500)

    # Expect the models to tie, while taking advantage of the fold player
    # compete("model_1", "model_1_2", "fold", 30)


if __name__ == "__main__":
    main()

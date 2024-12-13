#!/usr/bin/env python3

"""Run a catalog of poker games with different players."""

import random
import time
from pathlib import Path
from typing import Dict, Final, List

from termcolor import colored

from gym_env import load_model_player
from neuropoker.config import Config
from neuropoker.game import (
    Game,
    PlayerStats,
    default_player_stats,
    merge,
)
from neuropoker.models.es_hyperneat_model import ESHyperNEATModel
from neuropoker.models.hyperneat_model import HyperNEATModel
from neuropoker.models.neat_model import NEATModel
from neuropoker.player_utils import PlayerDefinition
from neuropoker.players.base_player import BasePlayer
from neuropoker.players.naive_player import CallPlayer, FoldPlayer, RandomPlayer
from neuropoker.players.neat_player import NEATPlayer

CATALOG: Final[Dict[str, PlayerDefinition]] = {
    # Naive players
    "random": PlayerDefinition(RandomPlayer),
    "fold": PlayerDefinition(FoldPlayer),
    "call": PlayerDefinition(CallPlayer),
    # SB players
    "sb": load_model_player("models/3p_3s/sb", "sb"),
    "sb2": load_model_player("models/3p_3s/sb2", "sb2"),
    "sb3": load_model_player("models/3p_3s/sb3", "sb3"),
    "sb4": load_model_player("models/3p_3s/sb4", "sb4"),
    "sb5": load_model_player("models/3p_3s/sb5", "sb5"),
    "sb6": load_model_player("models/3p_3s/sb6", "sb6"),
    "sb_backup": load_model_player("models/3p_3s/sb_backup", "sb_backup"),
    # Neuroevolution players trained against Call player
    "3p_3s_neat": PlayerDefinition(
        NEATPlayer, NEATModel, Path("models/3p_3s/3p_3s_neat__call__1000g.pkl")
    ),
    "3p_3s_hyperneat": PlayerDefinition(
        NEATPlayer,
        HyperNEATModel,
        Path("models/3p_3s/3p_3s_hyperneat__call__1000g.pkl"),
    ),
    "3p_3s_es-hyperneat": PlayerDefinition(
        NEATPlayer,
        ESHyperNEATModel,
        Path("models/3p_3s/3p_3s_es-hyperneat__call__1000g.pkl"),
    ),
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

    player_names_: Final[List[str]] = [
        f"{player_name}-{i}" for i, player_name in enumerate(player_names)
    ]

    print(f"In the competition, the players are: {player_names_}")
    print(f"They play {num_games} games from each position.")
    print()

    players: Final[List[BasePlayer]] = [
        CATALOG[name_orig].load(name)
        for name_orig, name in zip(player_names, player_names_)
    ]

    performances: Dict[str, PlayerStats]
    performances = {}
    for player_name in player_names_:
        default = default_player_stats()
        default["uuid"] = player_name
        performances[player_name] = default

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
        print(stats)
    print()

    for player_name, stats in performances.items():
        print(
            f"{player_name:20} avg winnings: {stats["winnings"] / stats["num_games"]:>10.4f}"
        )
    print()
    print()


def main():
    """Run the script."""

    # TODO: Configure config at runtime
    config: Final[Config] = Config("configs/3p_3s_neat.toml")

    # 3 fold:
    # Expect each player to win 0 on average
    print(colored("-------- 3 fold --------", color="blue", attrs=["bold"]))
    compete(["fold", "fold", "fold"], config)

    # 3 call:
    # Expect each player to win 0 on average
    print(colored("-------- 3 call --------", color="blue", attrs=["bold"]))
    compete(["call", "call", "call"], config)

    # 1 call + 2 fold:
    #
    # Expect call player to win every single round, on average 50
    # - When small blind, will win 50
    # - When big blind, will win 25
    # - When neither, will win 75
    print(colored("-------- 1 call + 2 fold --------", color="blue", attrs=["bold"]))
    compete(["call", "fold", "fold"], config)

    # 2 call + 1 fold:
    #
    # Expect each call player to win ~50% of the rounds on average.
    # - Each caller wins ~12.5 per game
    # - The fold player loses ~25.0 per game
    print(colored("-------- 2 call + 1 fold --------", color="blue", attrs=["bold"]))
    compete(["call", "call", "fold"], config)

    # NEAT + 2 call:
    #
    print(colored("-------- NEAT + 2 call --------", color="blue", attrs=["bold"]))
    compete(["3p_3s_neat", "call", "call"], config)

    # HyperNEAT + 2call:
    print(colored("-------- HyperNEAT + 2 call --------", color="blue", attrs=["bold"]))
    compete(["3p_3s_hyperneat", "call", "call"], config)

    # ES-HyperNEAT + 2call:
    print(
        colored("-------- ES-HyperNEAT + 2 call --------", color="blue", attrs=["bold"])
    )
    compete(["3p_3s_es-hyperneat", "call", "call"], config)

    # Expect same as before, since we're invariant to the order of the players
    # Success
    compete(["fold", "model_0", "fold"], config, num_games=30)
    compete(["call", "call", "call"], config, num_games=300)
    compete(["call", "model_1", "fold"], config, num_games=100)

    # Expect the model to fully exploit the folders
    compete(["model_1", "fold", "fold"], config, num_games=300)

    # Expect the model to fully exploit the call players
    compete(["model_1", "call", "call"], config, num_games=500)
    compete(["sb", "call", "call"], config, num_games=1000)
    compete(["sb_backup", "call", "call"], config, num_games=1000)
    compete(["sb2", "call", "call"], config, num_games=3000)
    compete(["sb3", "call", "call"], config, num_games=3000)
    compete(["sb3", "sb2", "call"], config, num_games=3000)
    compete(["sb2", "sb3", "call"], config, num_games=3000)
    compete(["sb4", "call", "call"], config, num_games=3000)
    compete(["sb4", "sb3", "sb2"], config, num_games=3000)
    compete(["sb4", "sb3", "sb2"], config, num_games=3000)
    compete(["sb5", "call", "call"], config, num_games=3000)
    compete(["sb5", "sb4", "sb3"], config, num_games=3000)
    compete(["sb5", "sb3", "sb2"], config, num_games=3000)
    compete(["sb5", "sb2", "call"], config, num_games=3000)
    compete(["sb6", "call", "call"], config, num_games=3000)
    compete(["sb6", "sb5", "sb4"], config, num_games=3000)
    compete(["sb6", "sb3", "sb2"], config, num_games=3000)

    # Expect the models to tie, while taking advantage of the fold player
    compete(["model_1", "model_1", "fold"], config, num_games=30)


if __name__ == "__main__":
    main()

"""Catalog of trained players."""

import random
import time
from pathlib import Path
from typing import Dict, Final, List

from neuropoker.extra.config import Config
from neuropoker.game.game import Game, PlayerStats, default_player_stats, merge
from neuropoker.models.neat.es_hyperneat import ESHyperNEATModel
from neuropoker.models.neat.hyperneat import HyperNEATModel
from neuropoker.models.neat.neat import NEATModel
from neuropoker.players.base import BasePlayer
from neuropoker.players.naive import CallPlayer, FoldPlayer, RandomPlayer
from neuropoker.players.neat import NEATPlayer
from neuropoker.players.ppo import PPOPlayer
from neuropoker.players.utils import PlayerDefinition

CATALOG: Final[Dict[str, PlayerDefinition]] = {
    #
    # Naive players
    #
    "random": PlayerDefinition(RandomPlayer),
    "fold": PlayerDefinition(FoldPlayer),
    "call": PlayerDefinition(CallPlayer),
    #
    # SB players
    #
    # "sb": PlayerDefinition(PPOPlayer, None, Path("models/3p_3s/sb")),
    # "sb2": PlayerDefinition(PPOPlayer, None, Path("models/3p_3s/sb2")),
    # "sb3": PlayerDefinition(PPOPlayer, None, Path("models/3p_3s/sb3")),
    # "sb4": PlayerDefinition(PPOPlayer, None, Path("models/3p_3s/sb4")),
    # "sb5": PlayerDefinition(PPOPlayer, None, Path("models/3p_3s/sb5")),
    "sb6": PlayerDefinition(PPOPlayer, None, Path("models/3p_3s/sb6")),
    "sb_backup": PlayerDefinition(PPOPlayer, None, Path("models/3p_3s/sb_backup")),
    # NEAT players
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
    #
    # Opposition players
    #
    # "op115": load_opposition_player("models/3p_3s/op1", "op115", "call"),
    # "op116": load_opposition_player("models/3p_3s/op1", "op116", "call"),
    # "op110": load_opposition_player("models/3p_3s/op1", "op110", "sb5"),
    # "op111": load_opposition_player("models/3p_3s/op1", "op111", "sb5"),
    # "op11": load_opposition_player("models/3p_3s/op1", "op11", "sb6"),
    # "op12": load_opposition_player("models/3p_3s/op1", "op12", "sb6"),
}


def compete(
    player_names: List[str],
    config: Config,
    num_games: int = 100,
) -> List[BasePlayer]:
    """Run a multi-player poker game.

    Parameters:
        player_names: List[str]
            The name of each player, taken from the catalog.
        config: Config
            The game configuration.
        num_games: int
            The number of games to play.

    Returns:
        players: List[BasePlayer]
            The players that were run, with their updated statistics.
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

    return players

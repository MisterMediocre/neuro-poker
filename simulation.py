#!/usr/bin/env python3

"""Simulate a poker game with multiple players.

TODO: Fill out script, add more options, etc.
"""

from typing import Final

from neuropoker.extra.config import Config
from neuropoker.game.game import Game
from neuropoker.players.naive import CallPlayer, RandomPlayer


def main() -> None:
    """Run the script."""
    config: Final[Config] = Config("configs/3p_4s_neat.toml")

    game_1 = Game.from_config(
        [RandomPlayer("random"), RandomPlayer("random2"), RandomPlayer("random3")],
        config,
    )
    fitness_1 = game_1.play_multiple(num_games=100, seed=1337)
    print(fitness_1)

    game_2 = Game.from_config(
        [CallPlayer("call"), RandomPlayer("random"), RandomPlayer("random2")],
        config,
    )
    fitness_2 = game_2.play_multiple(num_games=100, seed=1337)
    print(fitness_2)


if __name__ == "__main__":
    main()

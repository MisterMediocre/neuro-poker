#!/usr/bin/env python3

"""Simulate a poker game with multiple players.

TODO: Fill out script, add more options, etc.
"""
from typing import Final

from neuropoker.config import Config
from neuropoker.game import evaluate_performance
from neuropoker.players.naive_player import CallPlayer, RandomPlayer


def main() -> None:
    """Run the script."""
    config: Final[Config] = Config("configs/3p_4s_neat.toml")

    f1 = evaluate_performance(
        [RandomPlayer("random"), RandomPlayer("random2"), RandomPlayer("random3")],
        config,
    )
    print(f1)

    f2 = evaluate_performance(
        [CallPlayer("call"), RandomPlayer("random"), RandomPlayer("random2")],
        config,
    )
    print(f2)


if __name__ == "__main__":
    main()

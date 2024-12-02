#!/usr/bin/env python3

"""Simulate a poker game with multiple players.

TODO: Fill out script, add more options, etc.
"""

from neuropoker.game import evaluate_fitness
from neuropoker.player import CallPlayer, RandomPlayer


def main() -> None:
    """Run the script."""
    f1 = evaluate_fitness(
        ["random", "random2", "random3"],
        [RandomPlayer(), RandomPlayer(), RandomPlayer()],
    )
    print(f1)

    f2 = evaluate_fitness(
        ["call", "random", "random2"], [CallPlayer(), RandomPlayer(), RandomPlayer()]
    )
    print(f2)


if __name__ == "__main__":
    main()

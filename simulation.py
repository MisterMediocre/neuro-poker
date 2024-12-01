#!/usr/bin/env python3

"""Simulate a poker game with multiple players.

TODO: Fill out script, add more options, etc.
"""

from neuropoker.player import RandomPlayer, CallPlayer
from neuropoker.game import evaluate_fitness


def main() -> None:
    """Run the script."""
    f1 = evaluate_fitness(RandomPlayer(), [RandomPlayer(), RandomPlayer()])
    f2 = evaluate_fitness(CallPlayer(), [RandomPlayer(), RandomPlayer()])
    print(f1, f2)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

"""Run a catalog of poker games with different players."""

from typing import Final

from termcolor import colored

from neuropoker.extra.catalog import compete
from neuropoker.extra.config import Config


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

    # Expect the models to tie, while taking advantage of the fold player
    compete(["model_1", "model_1", "fold"], config, num_games=30)

    # Expect the model to fully exploit the call players
    compete(["model_1", "call", "call"], config, num_games=500)
    # compete(["sb", "call", "call"], config, num_games=1000)
    compete(["sb_backup", "call", "call"], config, num_games=1000)
    # compete(["sb2", "call", "call"], config, num_games=3000)
    # compete(["sb3", "call", "call"], config, num_games=3000)
    # compete(["sb3", "sb2", "call"], config, num_games=3000)
    # compete(["sb2", "sb3", "call"], config, num_games=3000)
    # compete(["sb4", "call", "call"], config, num_games=3000)
    # compete(["sb4", "sb3", "sb2"], config, num_games=3000)
    # compete(["sb4", "sb3", "sb2"], config, num_games=3000)
    # compete(["sb5", "call", "call"], config, num_games=3000)
    # compete(["sb5", "sb4", "sb3"], config, num_games=3000)
    # compete(["sb5", "sb3", "sb2"], config, num_games=3000)
    # compete(["sb5", "sb2", "call"], config, num_games=3000)
    # compete(["sb6", "call", "call"], config, num_games=3000)
    # compete(["sb6", "sb5", "sb4"], config, num_games=3000)
    # compete(["sb6", "sb3", "sb2"], config, num_games=3000)
    # compete(["sb6", "op11", "op12"], config, num_games=3000)
    # compete(["sb5", "op110", "op111"], config, num_games=3000)
    # compete(["call", "op115", "op116"], config, num_games=3000)
    compete(["call", "call", "call"], config, num_games=100)


if __name__ == "__main__":
    main()

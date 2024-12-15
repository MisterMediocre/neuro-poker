#!/usr/bin/env python3

"""Run a catalog of poker games with different players.
"""
from typing import Dict, Final, List
import random
from matplotlib.colors import ListedColormap
import numpy as np
import time
from matplotlib import pyplot as plt
from matplotlib.patches import Patch

from neuropoker.cards import ALL_RANKS
from neuropoker.game import (
    PlayerStats,
    default_player_stats,
    evaluate_performance,
    merge,
)
from neuropoker.model import load_player
from neuropoker.players.base import BasePlayer
from gym_env import load_model_player

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

    "sb_backup": load_model_player("models/3p_3s/sb_backup", "sb_backup"),
    "sb": load_model_player("models/3p_3s/sb", "sb"),
    "sb2": load_model_player("models/3p_3s/sb2", "sb2"),
    "sb3": load_model_player("models/3p_3s/sb3", "sb3"),
    "sb4": load_model_player("models/3p_3s/sb4", "sb4"),
    "sb5": load_model_player("models/3p_3s/sb5", "sb5"),
    "sb6": load_model_player("models/3p_3s/sb6", "sb6"),

    # "op115": load_opposition_player("models/3p_3s/op1", "op115", "call"),
    # "op116": load_opposition_player("models/3p_3s/op1", "op116", "call"),

    # "op110": load_opposition_player("models/3p_3s/op1", "op110", "sb5"),
    # "op111": load_opposition_player("models/3p_3s/op1", "op111", "sb5"),

    # "op11": load_opposition_player("models/3p_3s/op1", "op11", "sb6"),
    # "op12": load_opposition_player("models/3p_3s/op1", "op12", "sb6")
}

def produce_heatmap(actions):
    grid = np.zeros((len(ALL_RANKS), len(ALL_RANKS)))

    for hole_cards, stats in actions.items():
        same_suit = hole_cards[0][0] == hole_cards[1][0]

        h1_rank = ALL_RANKS.index(hole_cards[0][1])
        h2_rank = ALL_RANKS.index(hole_cards[1][1])

        lower_rank = min(h1_rank, h2_rank)
        higher_rank = max(h1_rank, h2_rank)

        if same_suit:
            assert h1_rank != h2_rank
            x = lower_rank
            y = higher_rank
        else:
            x = higher_rank
            y = lower_rank

        num_calls = stats.get("call", 0)
        num_folds = stats.get("fold", 0)
        num_raises = stats.get("raise", 0)
        max_action = max(num_calls, num_folds, num_raises)
        num_actions = num_calls + num_folds + num_raises

        if num_actions == 0:
            raise ValueError("No actions recorded for " + str(hole_cards))
        elif num_raises == max_action:
            grid_value = 3
        elif num_calls == max_action:
            grid_value = 2
        elif num_folds == max_action:
            grid_value = 1
        else:
            raise ValueError("One action must be the most common.")

        grid[x][y] = grid_value

    cmap = ListedColormap(["yellow", "red", "blue", "green"])
    plt.figure(figsize=(13, 9))
    plt.imshow(grid, cmap=cmap, origin="lower")
    plt.xticks(range(len(ALL_RANKS)), ALL_RANKS)
    plt.yticks(range(len(ALL_RANKS)), ALL_RANKS)
    plt.gca().set_xticks([x - 0.5 for x in range(1, len(ALL_RANKS))], minor=True)
    plt.gca().set_yticks([y - 0.5 for y in range(1, len(ALL_RANKS))], minor=True)
    plt.grid(which="minor", color="black", linestyle="-", linewidth=1)
    plt.xlabel("Same suit below the diagonal")
    plt.ylabel("Different suit above and including the diagonal")
    plt.plot(range(len(ALL_RANKS)), range(len(ALL_RANKS)), color="black", linestyle="--", linewidth=1)
    plt.title("Dealer Action Heatmap")

    legend_handles = [
        Patch(facecolor='yellow', edgecolor='black', label='No data'),
        Patch(facecolor='red', edgecolor='black', label='Fold'),
        Patch(facecolor='blue', edgecolor='black', label='Call'),
        Patch(facecolor='green', edgecolor='black', label='Raise'),
    ]
    plt.legend(handles=legend_handles, title="Actions", loc="upper right", bbox_to_anchor=(1.2, 1.0))

    plt.show()


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
    seed = -2
    # We never train on negative seeds
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

    # Expect the model to fully exploit the call players
    compete("sb6", "call2", "call3", 1000)
    produce_heatmap(CATALOG["sb6"].dealer_action)


if __name__ == "__main__":
    main()

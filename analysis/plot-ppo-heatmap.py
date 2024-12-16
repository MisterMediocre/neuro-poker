#!/usr/bin/env python3

"""Run a catalog of poker games with different players."""

import os
import sys
from typing import Final

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

# Import neuropoker
sys.path.append(os.getcwd())
from neuropoker.extra.catalog import compete
from neuropoker.extra.config import Config
from neuropoker.game.cards import ALL_RANKS


def produce_heatmap(actions) -> None:
    """Plot the action heatmap for a particular player's set of actions.

    Parameters:
        actions: ???
            TODO
    """
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

    fig, ax = plt.subplots(figsize=(5, 5), dpi=192)
    ax.imshow(grid, cmap=cmap, origin="lower")
    ax.set_xticks(range(len(ALL_RANKS)))
    ax.set_yticks(range(len(ALL_RANKS)))
    ax.set_xticks([x - 0.5 for x in range(1, len(ALL_RANKS))], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, len(ALL_RANKS))], minor=True)
    ax.set_xticklabels(ALL_RANKS)
    ax.set_yticklabels(ALL_RANKS)
    ax.grid(which="minor", color="black", linestyle="-", linewidth=1)
    ax.set_xlabel("Same suit\n(Below the diagonal)")
    ax.set_ylabel("Different suit\n(Above and including the diagonal)")
    ax.plot(
        [-0.5, len(ALL_RANKS) - 0.5],
        [-0.5, len(ALL_RANKS) - 0.5],
        color="black",
        linestyle="--",
        linewidth=1,
    )
    # fig.suptitle("Dealer Action Heatmap")

    # plt.figure(figsize=(13, 9))
    # plt.imshow(grid, cmap=cmap, origin="lower")
    # plt.xticks(range(len(ALL_RANKS)), ALL_RANKS)
    # plt.yticks(range(len(ALL_RANKS)), ALL_RANKS)
    # plt.gca().set_xticks([x - 0.5 for x in range(1, len(ALL_RANKS))], minor=True)
    # plt.gca().set_yticks([y - 0.5 for y in range(1, len(ALL_RANKS))], minor=True)
    # plt.grid(which="minor", color="black", linestyle="-", linewidth=1)
    # plt.xlabel("Same suit below the diagonal")
    # plt.ylabel("Different suit above and including the diagonal")
    # plt.plot(
    #     range(len(ALL_RANKS)),
    #     range(len(ALL_RANKS)),
    #     color="black",
    #     linestyle="--",
    #     linewidth=1,
    # )
    # plt.title("Dealer Action Heatmap")

    legend_handles = [
        Patch(facecolor="yellow", edgecolor="black", label="No data"),
        Patch(facecolor="red", edgecolor="black", label="Fold"),
        Patch(facecolor="blue", edgecolor="black", label="Call"),
        Patch(facecolor="green", edgecolor="black", label="Raise"),
    ]
    ax.legend(
        handles=legend_handles,
        bbox_to_anchor=(0.5, -0.175),
        # title="Actions",
        # loc="upper right",
        loc="upper center",
        borderaxespad=0.0,
        frameon=False,
        ncols=4,
        # bbox_to_anchor=(1.2, 1.0),
    )

    fig.tight_layout()
    fig.savefig("plots/ppo-heatmap.png", dpi=192)
    plt.show()


def main():
    """Run the script."""
    # TODO: Configure config at runtime
    config: Final[Config] = Config("configs/3p_3s_neat.toml")

    # Expect the model to fully exploit the call players
    # players = compete(["sb6", "call", "call"], config, num_games=1000)
    players = compete(["sb_linear__4", "call", "call"], config, num_games=1000)

    produce_heatmap(players[0].dealer_action)
    # produce_heatmap(CATALOG["sb6"].dealer_action)


if __name__ == "__main__":
    main()

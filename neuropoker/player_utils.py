"""Utility functions for players."""

import pickle

from neuropoker.player.base import BasePlayer
from neuropoker.player.naive import CallPlayer, FoldPlayer, RandomPlayer
from neuropoker.player.neat import NEATPlayer


def load_player(definition: str, name: str) -> BasePlayer:
    """Load a player, based on a string definition.

    Parameters:
        definition: str
            The player to load, or a path to an evolution file.
        name: str
            The name to assign to the player.

    Returns:
        player: BasePlayer
            The loaded player.
    """
    if definition == "RandomPlayer":
        return RandomPlayer()
    if definition == "CallPlayer":
        return CallPlayer()
    if definition == "FoldPlayer":
        return FoldPlayer()

    # Else, the definition is a path to a NEAT file
    with open(definition, "rb") as f:
        evolution = pickle.load(f)
        net = get_network(evolution)
        return NEATPlayer(net, name)

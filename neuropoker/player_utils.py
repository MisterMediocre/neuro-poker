"""Utility functions for players."""

import pickle

from neuropoker.models.neat_model import get_network
from neuropoker.players.base_player import BasePlayer
from neuropoker.players.naive_player import CallPlayer, FoldPlayer, RandomPlayer
from neuropoker.players.neat_player import NEATPlayer


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

"""Utility values and functions for the poker game."""

from typing import Dict, Final

# TODO: Remove hardcoding
STACK: Final[int] = 1000
NUM_PLAYERS: Final[int] = 3

# Street name -> index
STREET_MAPPING: Final[Dict[str, int]] = {
    "preflop": 1,
    "flop": 2,
    "turn": 3,
    "river": 4,
}

# Community card index -> street in which the card was dealt
COMMUNITY_CARD_MAPPING: Final[Dict[int, str]] = {
    0: "preflop",
    1: "preflop",
    2: "preflop",
    3: "flop",
    4: "turn",
}

# State name -> index
STATE_MAPPING: Final[Dict[str, int]] = {
    "folded": 0,
    "participating": 1,
    "allin": 2,
}

# Action name -> index
ACTION_MAPPING: Final[Dict[str, int]] = {
    "fold": 0,
    "call": 1,
    "raise": 2,
}

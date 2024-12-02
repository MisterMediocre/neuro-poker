"""Naive poker players with simple policies (e.g. always fold/call).
"""

import random

from neuropoker.players.base import BasePlayer


class RandomPlayer(BasePlayer):
    """A player which takes random actions."""

    def declare_action(self, valid_actions, hole_card, round_state):
        """Declare an action, which is selected at random."""
        valid_actions.pop(0) # Never fold

        action = random.choice(valid_actions)

        if action["action"] == "raise":
            # Some multiples of min
            multiple = random.choice([1, 2, 3])
            return action["action"], multiple*action["amount"]["min"]

        return action["action"], action["amount"]


class CallPlayer(BasePlayer):
    """A player which always calls."""

    def declare_action(self, valid_actions, hole_card, round_state):
        """Declare an action, which is to always call."""
        action = valid_actions[1]
        return action["action"], action["amount"]


class FoldPlayer(RandomPlayer):
    """A player which always folds."""

    def declare_action(self, valid_actions, hole_card, round_state):
        """Declare an action, which is to always fold."""
        action = valid_actions[0]
        # print(action['action'])
        return action["action"], action["amount"]

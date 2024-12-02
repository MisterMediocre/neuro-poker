"""Naive poker players with simple policies (e.g. always fold/call).
"""

import random

from neuropoker.player.base import BasePlayer


class RandomPlayer(BasePlayer):
    """A player which takes random actions."""

    def declare_action(self, valid_actions, hole_card, round_state):
        """Declare an action, which is selected at random."""
        action = random.choice(valid_actions)

        if action["action"] == "raise":
            return action["action"], random.randint(
                action["amount"]["min"], action["amount"]["max"]
            )

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

"""Naive poker players with simple policies (e.g. always fold/call)."""

import random
from typing import Tuple

from neuropoker.players.base import BasePlayer


class RandomPlayer(BasePlayer):
    """A player which takes random actions."""

    def declare_action(self, valid_actions, hole_card, round_state) -> Tuple[str, int]:
        """Select an action at random.

        Parameters:
            valid_actions: List[Dict[str, Union[str, int]]]
                The valid actions the player can take.
            hole_card: List[str]
                The player's hole cards.
            round_state: Dict[str, Any]
                The state of the round.

        Returns:
            action: str
                The action to take.
            amount: int
                The amount to bet or raise.
        """
        valid_actions.pop(0)  # Never fold

        action = random.choice(valid_actions)

        if action["action"] == "raise":
            # Some multiples of min
            multiple = random.choice([1, 2, 3])
            res = (action["action"], multiple * action["amount"]["min"])
        else:
            res = action["action"], action["amount"]

        self.report_action(res, hole_card, round_state)
        return res


class CallPlayer(BasePlayer):
    """A player which always calls."""

    def declare_action(self, valid_actions, hole_card, round_state) -> Tuple[str, int]:
        """Declare an action, which is to always call.

        Parameters:
            valid_actions: List[Dict[str, Union[str, int]]]
                The valid actions the player can take.
            hole_card: List[str]
                The player's hole cards.
            round_state: Dict[str, Any]
                The state of the round.

        Returns:
            action: str
                The action to take.
            amount: int
                The amount to bet or raise.
        """
        action = valid_actions[1]
        res = (action["action"], action["amount"])
        self.report_action(res, hole_card, round_state)
        return res


class FoldPlayer(RandomPlayer):
    """A player which always folds."""

    def declare_action(self, valid_actions, hole_card, round_state) -> Tuple[str, int]:
        """Declare an action, which is to always fold.

        Parameters:
            valid_actions: List[Dict[str, Union[str, int]]]
                The valid actions the player can take.
            hole_card: List[str]
                The player's hole cards.
            round_state: Dict[str, Any]
                The state of the round.

        Returns:
            action: str
                The action to take.
            amount: int
                The amount to bet or raise.
        """
        action = valid_actions[0]
        res = (action["action"], action["amount"])
        self.report_action(res, hole_card, round_state)
        return res

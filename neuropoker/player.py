"""Classes and functions for poker players.
"""

import random
from typing import Tuple

import numpy as np
from pypokerengine.players import BasePokerPlayer

from neuropoker.game_utils import extract_features


class BasePlayer(BasePokerPlayer):
    """Base class for poker players."""

    def declare_action(self, valid_actions, hole_card, round_state):
        raise NotImplementedError

    def receive_game_start_message(self, game_info):
        pass

    def receive_round_start_message(self, round_count, hole_card, seats):
        pass

    def receive_street_start_message(self, street, round_state):
        pass


class RandomPlayer(BasePlayer):
    """A player which takes random actions."""

    def declare_action(self, valid_actions, hole_card, round_state):
        action = random.choice(valid_actions)

        if action["action"] == "raise":
            return action["action"], random.randint(
                action["amount"]["min"], action["amount"]["max"]
            )

        return action["action"], action["amount"]


class NEATPlayer(BasePlayer):
    """A player which uses a NEAT neuro-evolved network to take actions."""

    def __init__(self, net, uuid) -> None:
        self.net = net
        self.uuid = uuid

    def declare_action(self, valid_actions, hole_card, round_state) -> Tuple[str, int]:
        # print(hole_card)
        # print(round_state["community_card"])
        features = extract_features(hole_card, round_state, self.uuid)
        output = self.net.activate(features)  # Neural network output
        chosen_action_idx = np.argmax(output)

        # Following are the allowed actions
        # 0: fold
        # 1: call
        # 2: raise min
        # 3: raise 2x min
        # 4: raise 3x min
        # 5: raise max

        if chosen_action_idx == 0:  # Fold
            # print(hole_card)
            return "fold", 0

        if chosen_action_idx == 1:  # Call
            return "call", valid_actions[1]["amount"]

        raise_action = valid_actions[2]
        min_raise = raise_action["amount"]["min"]
        max_raise = raise_action["amount"]["max"]

        if chosen_action_idx == 2:  # Raise min
            return "raise", min_raise

        if chosen_action_idx == 3:  # Raise 2x min
            return "raise", min_raise * 2

        if chosen_action_idx == 4 or chosen_action_idx == 5:  # Raise 3x min
            return "raise", min_raise * 3

        # if chosen_action_idx == 5:  # Raise max
            # return "raise", max_raise  # All-in
            # Makes training unstable

        raise ValueError(f"Invalid action index: {chosen_action_idx}")


class CallPlayer(BasePlayer):
    """A player which always calls."""

    def declare_action(self, valid_actions, hole_card, round_state):
        action = valid_actions[1]
        return action["action"], action["amount"]


class FoldPlayer(RandomPlayer):
    """A player which always folds."""

    def declare_action(self, valid_actions, hole_card, round_state):
        action = valid_actions[0]
        # print(action['action'])
        return action["action"], action["amount"]

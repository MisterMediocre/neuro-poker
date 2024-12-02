"""Poker player which is learned through NEAT neuroevolution.
"""

from typing import Tuple

import numpy as np

from neuropoker.game_utils import extract_features
from neuropoker.player.base import BasePlayer
from neuropoker.player.naive import RandomPlayer


class NEATPlayer(BasePlayer):
    """A player which uses a NEAT neuro-evolved network to take actions."""

    def __init__(self, net, uuid, training=False) -> None:
        self.training = training
        self.net = net
        self.uuid = uuid

    def declare_action(self, valid_actions, hole_card, round_state) -> Tuple[str, int]:

        random_player = RandomPlayer()

        ## Bootstrap the model by ensuring it sees a variety of situations
        if self.training:
            if len(round_state["community_card"]) == 0 and np.random.rand() < 0.30:
                return random_player.declare_action(
                    valid_actions, hole_card, round_state
                )
            if len(round_state["community_card"]) == 3 and np.random.rand() < 0.2:
                return random_player.declare_action(
                    valid_actions, hole_card, round_state
                )
            if len(round_state["community_card"]) == 4 and np.random.rand() < 0.15:
                return random_player.declare_action(
                    valid_actions, hole_card, round_state
                )

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
        _max_raise = raise_action["amount"]["max"]

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

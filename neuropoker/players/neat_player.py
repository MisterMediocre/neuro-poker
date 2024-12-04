"""Poker player which is learned through NEAT neuroevolution.
"""

from typing import Final, Optional, Tuple, Union

import numpy as np
from neat.nn import FeedForwardNetwork, RecurrentNetwork

from neuropoker.game_utils import extract_features
from neuropoker.players.base_player import BasePlayer
from neuropoker.players.naive_player import RandomPlayer

NEATNetwork = Union[FeedForwardNetwork, RecurrentNetwork]


class NEATPlayer(BasePlayer):
    """A player which uses a NEAT neuro-evolved network to take actions."""

    def __init__(
        self,
        uuid: str,
        net: NEATNetwork,
        training: bool = False,
    ) -> None:
        """Initialize the model.

        Parameters:
            uuid: str
                A unique name for the player.
            net: NEATNetwork
                The NEAT network to use for decision making.
            training: bool
                Whether the player is training or not.
        """
        super().__init__(uuid)

        self.net: Final[NEATNetwork] = net
        self.training: Final[bool] = training

        # Helps bootstrap the model by ensuring it sees a variety of
        # situations.
        self.random_surrogate_player: Final[Optional[RandomPlayer]] = (
            RandomPlayer("surrogate") if self.training else None
        )

    def declare_action(self, valid_actions, hole_card, round_state) -> Tuple[str, int]:
        """Select an action, using the model.

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

        # Bootstrap the model by ensuring it sees a variety of situations
        if self.training:
            # Validate that the surrogate player exists
            if self.random_surrogate_player is None:
                raise ValueError("Random surrogate player not initialized.")

            # With some random probability, select a random action
            #
            # (Similar to e-greedy exploration in RL)
            if (
                (len(round_state["community_card"]) == 0 and np.random.rand() < 0.30)
                or (len(round_state["community_card"]) == 3 and np.random.rand() < 0.2)
                or (len(round_state["community_card"]) == 4 and np.random.rand() < 0.15)
            ):
                return self.random_surrogate_player.declare_action(
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

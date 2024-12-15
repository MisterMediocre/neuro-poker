"""Class for Stable Baselines 3 PPO-based poker players."""

from pathlib import Path
from typing import Final, Tuple

import numpy as np
from stable_baselines3.ppo.ppo import PPO

from neuropoker.game.utils import extract_features
from neuropoker.players.base import BasePlayer


class PPOPlayer(BasePlayer):
    """A player which uses a PPO model to make decisions."""

    def __init__(self, model: PPO, uuid: str) -> None:
        """Initialize the player.

        Parameters:
            model: PPO
                The model to use for decision making.
            uuid: str
                The uuid of this player.
        """
        super().__init__(uuid)
        self.model: Final[PPO] = model

    @staticmethod
    def from_model_file(model_file: Path | str, uuid: str) -> "PPOPlayer":
        """Load a model from a file.

        Parameters:
            model_file: Path | str
                The path to the model file.
            uuid: str
                The uuid of this player

        Returns:
            player: PPOPlayer
                The loaded player.
        """
        model: Final[PPO] = PPO.load(model_file)
        return PPOPlayer(model, uuid)

    @staticmethod
    def int_to_action(output: int, valid_actions) -> Tuple[str, int]:
        """Map an integer output to an action

        Parameters:
            output: int
                The integer output of the model.
            valid_actions: List[Dict[str, Union[str, int]]]
                The valid actions the player can take.

        Returns:
            action: str
                The action to take.
            amount: int
                The amount to bet or raise.

        Raises:
            ValueError: If the integer output is invalid.
        """
        match output:
            case 0:
                return "fold", 0
            case 1:
                return "call", valid_actions[1]["amount"]
            case 2:
                return "raise", valid_actions[2]["amount"]["min"]
            case 3 | 4:
                return "raise", valid_actions[2]["amount"]["min"] * 2
            case _:
                raise ValueError("Invalid action")

    def declare_action(self, valid_actions, hole_card, round_state) -> Tuple[str, int]:
        """Select an action.

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
        features: Final[np.ndarray] = extract_features(
            hole_card, round_state, self.uuid
        )

        # TODO: Fix type error
        action: Final[int] = self.model.predict(features[np.newaxis, :])[0]  # type: ignore

        action_: Final[Tuple[str, int]] = self.int_to_action(action, valid_actions)
        self.report_action(action_, hole_card, round_state)
        return action_

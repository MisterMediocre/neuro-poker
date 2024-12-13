"""Base class for poker players."""

from abc import ABC, abstractmethod
from typing import Final, Tuple, TypeVar

from pypokerengine.players import BasePokerPlayer

PlayerT = TypeVar("PlayerT", bound="BasePlayer")


class BasePlayer(ABC, BasePokerPlayer):
    """Base class for poker players."""

    def __init__(self, uuid: str) -> None:
        """Initialize the player.

        Parameters:
            uuid: str
                The uuid of this player.
        """
        self.uuid: Final[str] = uuid

    @abstractmethod
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

    def receive_game_start_message(self, game_info):
        pass

    def receive_round_start_message(self, round_count, hole_card, seats):
        pass

    def receive_street_start_message(self, street, round_state):
        pass

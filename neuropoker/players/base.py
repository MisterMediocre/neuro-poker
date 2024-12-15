"""Base class for poker players."""

from typing import Final, Tuple, TypeVar

from pypokerengine.players import BasePokerPlayer

PlayerT = TypeVar("PlayerT", bound="BasePlayer")


class BasePlayer(BasePokerPlayer):
    """Base class for poker players."""

    def __init__(self, uuid: str):
        """Initialize the player.

        Parameters:
            uuid: str
                The uuid of this player.
        """
        self.dealer_action = {}
        self.uuid: Final[str] = uuid
        super(BasePlayer, self).__init__()

    def report_action(self, action, hole_card, round_state):
        next_player = round_state["next_player"]
        dealer = round_state["dealer_btn"]
        street = round_state["street"]

        if next_player != dealer or street != "preflop":
            return

        h1 = min(hole_card[0], hole_card[1])
        h2 = max(hole_card[0], hole_card[1])
        rep = (h1, h2)

        self.dealer_action[rep] = self.dealer_action.get(rep, {})
        self.dealer_action[rep][action[0]] = self.dealer_action[rep].get(action, 0) + 1

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
        raise NotImplementedError

    def receive_game_start_message(self, game_info):
        pass

    def receive_round_start_message(self, round_count, hole_card, seats):
        pass

    def receive_street_start_message(self, street, round_state):
        pass

"""Base class for poker players.
"""

from pypokerengine.players import BasePokerPlayer


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

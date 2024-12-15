"""Base class for poker players.
"""

from pypokerengine.players import BasePokerPlayer
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np



class BasePlayer(BasePokerPlayer):
    """Base class for poker players."""

    def __init__(self):
        self.dealer_action = {}
        super(BasePlayer, self).__init__()

    def report_action(self, action, hole_card, round_state):
        next_player = round_state['next_player']
        dealer = round_state['dealer_btn']
        street = round_state['street']

        if next_player != dealer or street != "preflop":
            return

        h1 = min(hole_card[0], hole_card[1])
        h2 = max(hole_card[0], hole_card[1])
        rep = (h1, h2)

        self.dealer_action[rep] = self.dealer_action.get(rep, {})
        self.dealer_action[rep][action[0]] = self.dealer_action[rep].get(action, 0) + 1

    def declare_action(self, valid_actions, hole_card, round_state):
        raise NotImplementedError

    def receive_game_start_message(self, game_info):
        pass

    def receive_round_start_message(self, round_count, hole_card, seats):
        pass

    def receive_street_start_message(self, street, round_state):
        pass

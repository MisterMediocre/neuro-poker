"""Classes and functions for poker games.
"""

import random
from random import shuffle

import numpy as np
from pypokerengine.engine.table import Table
from pypokerengine.utils.card_utils import gen_cards
from pypokerengine.engine import deck
from pypokerengine.engine.player import Player
from pypokerengine.engine.poker_constants import PokerConstants as Const
from pypokerengine.players import BasePokerPlayer
from pypokerengine.api.game import setup_config, start_poker
from pypokerengine.engine.card import Card
from pypokerengine.api.emulator import Emulator


# C = Clubs, D = Diamonds, H = Hearts, S = Spades
short_cards = [
    "C6", "D6", "H6", "S6",  # 6s
    "C7", "D7", "H7", "S7",  # 7s
    "C8", "D8", "H8", "S8",  # 8s
    "C9", "D9", "H9", "S9",  # 9s
    "CT", "DT", "HT", "ST",  # 10s
    "CJ", "DJ", "HJ", "SJ",  # Jacks
    "CQ", "DQ", "HQ", "SQ",  # Queens
    "CK", "DK", "HK", "SK",  # Kings
    "CA", "DA", "HA", "SA",  # Aces
]
short_card_ids = [Card.from_str(s).to_id() for s in short_cards]
def gen_deck(seed=None):
    if seed is not None:
        shuffle(short_card_ids)
        random.seed(seed)
    return deck.Deck(cheat=True, cheat_card_ids=short_card_ids)

NUM_PLAYERS = 3
SMALL_BLIND_AMOUNT = 25
BIG_BLIND_AMOUNT = 50
ANTE = 0
STACK = 1000

emulator = Emulator()
emulator.set_game_rule(NUM_PLAYERS, 5, SMALL_BLIND_AMOUNT, BIG_BLIND_AMOUNT)


players_info = {
    "uuid-1": { "name": "player1", "stack": STACK },
    "uuid-2": { "name": "player2", "stack": STACK },
    "uuid-3": { "name": "player3", "stack": STACK }
}

ranks = {'6': 0, '7': 1, '8': 2, '9': 3, 'T': 4, 'J': 5, 'Q': 6, 'K': 7, 'A': 8}
suits = {'C': 0, 'D': 1, 'H': 2, 'S': 3}
street_mapping = {"flop": 1, "turn": 2, "river": 3}

def card_to_index(card):
    return ranks[card[1]] + suits[card[0]] * 9

def extract_features(hole_card, round_state, player_uuid):
    public_cards = np.zeros(36)
    private_cards = np.zeros(36)
    player_bets = {street: [0] * NUM_PLAYERS for street in ["preflop", "flop", "turn", "river"]}  # Bets per street

    street = round_state["street"]
    community_cards = round_state["community_card"]

    stack_sizes = [player["stack"] for player in round_state["seats"]]

    for card in community_cards:
        idx = card_to_index(card)
        public_cards[idx] = street_mapping[street]

    for card in hole_card:
        idx = card_to_index(card)

    for street, actions in round_state["action_histories"].items():
        for action in actions:
            if action["action"] in ["call", "raise"]:
                player_pos = next(i for i, p in enumerate(round_state["seats"]) if p["uuid"] == action["uuid"])
                player_bets[street][player_pos] += action["amount"]

    player_positions = [p["uuid"] for p in round_state["seats"]]
    player_position_index = player_positions.index(player_uuid)
    normalized_position = player_position_index / (NUM_PLAYERS - 1)

    flattened_bets = np.concatenate([player_bets[street] for street in ["preflop", "flop", "turn", "river"]])
    features = np.concatenate([public_cards, private_cards, flattened_bets, stack_sizes, [normalized_position]])
    return features


def evaluate_fitness(
    player_model, 
    opponent_models, 
    num_games: int = 100, 
    seed: int = 1,
) -> float:
    """Evaluate the fitness of a player against other opponents.

    Parameters:
        player_model:
            The player model to evaluate.
        opponent_models:
            The list of opponent models to play against.
        num_games:
            The number of games to play.
        seed:
            The seed to use for the evaluation.
    
    Returns:
        fitness: float
            The fitness of the player model.

    Fitness is defined by the average winnings per game, but can
    be adjusted to something else (TODO).
    """
    emulator.register_player("uuid-1", player_model)
    emulator.register_player("uuid-2", opponent_models[0])
    emulator.register_player("uuid-3", opponent_models[1])

    sum_winnings = 0

    for i in range(num_games):
        initial_state = emulator.generate_initial_game_state(players_info)

        # Same seed means same cards dealt
        initial_state['table'].deck = gen_deck(seed=seed*100 + i)
        initial_state['table'].dealer_btn = random.randint(0, 2)

        game_state, events = emulator.start_new_round(initial_state)
        game_state, events = emulator.run_until_game_finish(game_state)
        stack = game_state['table'].seats.players[0].stack
        sum_winnings += (stack - STACK)

    return sum_winnings/num_games

# if __name__ == "__main__":
#     f1 = evaluate_fitness(RandomPlayer(), [RandomPlayer(), RandomPlayer()])
#     f2 = evaluate_fitness(CallPlayer(), [RandomPlayer(), RandomPlayer()])
#     print(f1, f2)

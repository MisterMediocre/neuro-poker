"""Classes and functions for poker games.
"""

import random
from typing import Dict, List, Optional

import numpy as np
from pypokerengine.api.emulator import Emulator
from pypokerengine.engine.card import Card
from pypokerengine.engine.deck import Deck

# from pypokerengine.api.game import setup_config, start_poker
# from pypokerengine.engine.player import Player
# from pypokerengine.engine.poker_constants import PokerConstants as Const
# from pypokerengine.engine.table import Table
# from pypokerengine.players import BasePokerPlayer
# from pypokerengine.utils.card_utils import gen_cards

# C = Clubs, D = Diamonds, H = Hearts, S = Spades
short_cards = [
    "C6",
    "D6",
    "H6",
    "S6",  # 6s
    "C7",
    "D7",
    "H7",
    "S7",  # 7s
    "C8",
    "D8",
    "H8",
    "S8",  # 8s
    "C9",
    "D9",
    "H9",
    "S9",  # 9s
    "CT",
    "DT",
    "HT",
    "ST",  # 10s
    "CJ",
    "DJ",
    "HJ",
    "SJ",  # Jacks
    "CQ",
    "DQ",
    "HQ",
    "SQ",  # Queens
    "CK",
    "DK",
    "HK",
    "SK",  # Kings
    "CA",
    "DA",
    "HA",
    "SA",  # Aces
]
short_card_ids = [Card.from_str(s).to_id() for s in short_cards]


NUM_PLAYERS = 3
SMALL_BLIND_AMOUNT = 25
BIG_BLIND_AMOUNT = 50
ANTE = 0
STACK = 1000

emulator = Emulator()
emulator.set_game_rule(NUM_PLAYERS, 5, SMALL_BLIND_AMOUNT, ANTE)

def gen_deck(seed: Optional[int] = None) -> Deck:
    """Generate a deck.

    Parameters:
        seed: int | None
            The random seed to use.

    Returns:
        deck: Deck
            The deck of cards.
    """
    if seed is not None:
        random.shuffle(short_card_ids)
        random.seed(seed)
    return Deck(cheat=True, cheat_card_ids=short_card_ids)



ranks = {"6": 0, "7": 1, "8": 2, "9": 3, "T": 4, "J": 5, "Q": 6, "K": 7, "A": 8}
suits = {"C": 0, "D": 1, "H": 2, "S": 3}
street_mapping = {"flop": 1, "turn": 2, "river": 3}


def card_to_index(card) -> int:
    """Convert a card to an index.

    Parameters:
        card: ??? (TODO: Determine type)
            The card to convert.

    Returns:
        index: int
            The index of the card.
    """
    return ranks[card[1]] + suits[card[0]] * 9

state_mapping = {"participating": 1, "folded": 0, "allin": 2}




# TODO: Add state of each player (Folded, all-in etc)
def extract_features(hole_card, round_state, player_uuid) -> np.ndarray:
    """Extract features for a poker agent from the current game state.

    Parameters:
        hole_card: ???
            The private cards of the player.
        round_state: ???
            The state of the current round.
        player_uuid: ???
            The player's UUID

    Returns:
        features: np.ndarray
            The features extracted from the game state.
    """
    public_cards: np.ndarray = np.zeros(36)
    private_cards: np.ndarray = np.zeros(36)
    player_bets: Dict[str, List[int]] = {
        street: [0] * NUM_PLAYERS for street in ["preflop", "flop", "turn", "river"]
    }  # Bets per street

    street = round_state["street"]
    community_cards = round_state["community_card"]

    for card in community_cards:
        idx = card_to_index(card)
        public_cards[idx] = street_mapping[street]

    for card in hole_card:
        idx = card_to_index(card)
        private_cards[idx] = 1

    # Dealer position is 0, then 1, then 2 is the guy before the dealer
    dealer_index = round_state["dealer_btn"]
    rotated_seats = round_state["seats"][dealer_index:] + round_state["seats"][:dealer_index]

    player_positions = {p["uuid"]: i for i, p in enumerate(rotated_seats)}
    normalized_position = player_positions[player_uuid] / (NUM_PLAYERS - 1)

    stack_sizes = [p["stack"] for p in rotated_seats]

    # Store the bet made by each player, relative to the dealer position
    # The sum of all bets is the pot size, which the model can figure out
    for street_name, actions in round_state["action_histories"].items():
        for action in actions:
            if action["action"].lower() in ["call", "raise"]:
                bet_amount = action["amount"]
                relative_pos = player_positions[action["uuid"]]
                if relative_pos != -1:
                    player_bets[street_name][relative_pos] += bet_amount

    # Self-state is redundant, but included for consistency
    player_states = [
        state_mapping[p["state"]] for p in rotated_seats
    ]

    flattened_bets = np.concatenate(
        [player_bets[street] for street in ["preflop", "flop", "turn", "river"]]
    )
    features = np.concatenate(
        [
            public_cards,
            private_cards,
            flattened_bets,
            stack_sizes,
            player_states,
            [normalized_position],
        ]
    )
    return features

def evaluate_fitness(
    player_names,
    player_models,
    num_games: int = 100,
    seed: int = 1,
) -> List[float]:
    """Evaluate the fitness of a player against other opponents.

    Parameters:
        player_name:
            The name of the player models.
        player_model:
            The player models to simulate.
        num_games:
            The number of games to play.
        seed:
            The seed to use for the evaluation.

    Returns:
        fitnesses: List[float]
            The fitness of the players

    Fitness is defined by the average winnings per game, but can
    be adjusted to something else (TODO).
    """

    assert len(player_names) == len(player_models)
    assert len(player_names) ==  NUM_PLAYERS
    assert num_games > 0


    for name, model in zip(player_names, player_models):
        emulator.register_player(name, model)

    players_info = {
        name: {"name": name, "stack": STACK, "uuid": name} for name in player_names
    }

    sum_winnings = [0.0] * NUM_PLAYERS

    dealer_btn = 0

    for i in range(num_games):
        initial_state = emulator.generate_initial_game_state(players_info)

        # Same seed means same cards dealt
        initial_state["table"].deck = gen_deck(seed=seed * 100 + i)
        initial_state["table"].dealer_btn = dealer_btn

        game_state, _ = emulator.start_new_round(initial_state)  # game_state, events
        game_state, _ = emulator.run_until_round_finish(game_state)  # game_state, events

        for j, player in enumerate(game_state["table"].seats.players):
            # print(j, player.name, player.stack - STACK)
            sum_winnings[j] += player.stack - STACK

        dealer_btn = (dealer_btn + 1) % NUM_PLAYERS

    # divide each by number of games
    for i in range(NUM_PLAYERS):
        sum_winnings[i] = sum_winnings[i] / num_games


    return sum_winnings

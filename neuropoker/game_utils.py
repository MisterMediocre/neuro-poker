"""Utility values and functions for the poker game."""

from typing import Any, Dict, Final, List

import numpy as np

from neuropoker.cards import SHORT_RANKS, SHORT_SUITS, get_card_index, SHORTER_SUITS

STACK: Final[int] = 1000

NUM_PLAYERS: Final[int] = 3

STREET_MAPPING: Final[Dict[int, int]] = {
    0: 1,  # Preflop
    1: 1,  # Preflop
    2: 1,  # Preflop
    3: 2,  # Flop
    4: 3,  # Turn
}
STATE_MAPPING: Final[Dict[str, int]] = {
    "folded": 0,
    "participating": 1,
    "allin": 2,
}


def extract_features(
    hole_card: List[str], round_state: Dict[str, Any], player_uuid: str
) -> np.ndarray:
    """Extract features for a poker agent from the current game state.

    Parameters:
        hole_card: List[str]
            The private cards of the player.
        round_state: Dict[str, Any]
            The state of the current round.
        player_uuid: str
            The player's UUID

    Returns:
        features: np.ndarray
            The features extracted from the game state.

    TODO: Un-hardcode the following:
        Deck size
        Number of players
        Game ranks
        Game suits
    """
    public_cards: np.ndarray = np.zeros(len(SHORT_RANKS) * len(SHORTER_SUITS))
    private_cards: np.ndarray = np.zeros(len(SHORT_RANKS) * len(SHORTER_SUITS))

    player_bets: Dict[str, List[float]] = {
        street: [0] * NUM_PLAYERS for street in ["preflop", "flop", "turn", "river"]
    }  # Normalized bets per street

    community_cards: Final[List[str]] = round_state["community_card"]

    # for card in community_cards:
    for i, card in enumerate(community_cards):
        idx: int = get_card_index(card, ranks=SHORT_RANKS, suits=SHORTER_SUITS)
        public_cards[idx] = STREET_MAPPING[i]

    for card in hole_card:
        idx: int = get_card_index(card, ranks=SHORT_RANKS, suits=SHORTER_SUITS)
        private_cards[idx] = 1

    # Dealer position is 0, then 1, then 2 is the guy before the dealer
    dealer_index: Final[int] = round_state["dealer_btn"]
    rotated_seats: Final[List[Dict[str, Any]]] = (
        round_state["seats"][dealer_index:] + round_state["seats"][:dealer_index]
    )

    player_positions: Final[Dict[str, int]] = {
        p["uuid"]: i for i, p in enumerate(rotated_seats)
    }
    normalized_position: Final[float] = player_positions[player_uuid] / (
        NUM_PLAYERS - 1
    )

    stack_sizes: Final[List[int]] = [p["stack"] for p in rotated_seats]

    # Store the bet made by each player, relative to the dealer position
    # The sum of all bets is the pot size, which the model can figure out
    for street_name, actions in round_state["action_histories"].items():
        for action in actions:
            if action["action"].lower() in ["call", "raise"]:
                bet_amount = action["amount"]
                normalized_bet_amount: float = bet_amount/STACK
                relative_pos = player_positions[action["uuid"]]
                if relative_pos != -1:
                    player_bets[street_name][relative_pos] += normalized_bet_amount

    # Self-state is redundant, but included for consistency
    player_states: List[int] = [STATE_MAPPING[p["state"]] for p in rotated_seats]

    flattened_bets: np.ndarray = np.concatenate(
        [player_bets[street] for street in ["preflop", "flop", "turn", "river"]]
    )

    features: np.ndarray = np.concatenate(
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

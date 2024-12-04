"""Utility values and functions for the poker game."""

from typing import Any, Dict, Final, List

import numpy as np

from neuropoker.cards import SHORT_RANKS, SHORT_SUITS, get_card_index, SHORTER_SUITS, get_card_indices

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
ACTION_MAPPING: Final[Dict[str, int]] = {
        "fold": 0,
        "call": 1,
        "raise": 2,
}

def encode_cards_tensor(hole_card, community_cards):
    card_tensor = np.zeros((8, len(SHORT_RANKS)*len(SHORTER_SUITS)),  dtype=np.float32)  # Rows for community and private cards
    for i, card in enumerate(community_cards):
        rank, suit = get_card_indices(card, SHORT_RANKS, SHORTER_SUITS)
        card_tensor[i, suit * len(SHORT_RANKS) + rank] = 1  # Map (suit, rank) to index
    for card in hole_card:
        rank, suit = get_card_indices(card, SHORT_RANKS, SHORTER_SUITS)
        card_tensor[7, suit * len(SHORT_RANKS) + rank] = 1  # Hole cards at row 7
    return card_tensor

def encode_player_bets_tensor(round_state, player_positions):
    """
    Encode bets separately for each street (preflop, flop, turn, river).
    """
    num_streets = 4  # Preflop, Flop, Turn, River
    bet_tensor = np.zeros((num_streets * NUM_PLAYERS, len(SHORT_RANKS) * len(SHORTER_SUITS)), dtype=np.float32)

    for street, actions in round_state["action_histories"].items():
        street_index = STREET_MAPPING[street] - 1  # Map street to an index (0-based)
        print(round_state["action_histories"])
        for action in actions:
            if action["action"].lower() in ["call", "raise"]:
                bet = action["amount"]
                normalized_bet = bet / STACK  # Normalize
                player_idx = player_positions[action["uuid"]]
                row_idx = street_index * NUM_PLAYERS + player_idx
                bet_tensor[row_idx, :] += normalized_bet  # Add normalized bet for the specific street

    return bet_tensor

def encode_stack_sizes_tensor(round_state, player_positions):
    stack_tensor = np.zeros((3, len(SHORT_RANKS) * len(SHORTER_SUITS)),  dtype=np.float32)  # 3 rows for players, stack sizes

    for i, player in enumerate(round_state["seats"]):
        normalized_stack = player["stack"] / STACK  # Normalize
        stack_tensor[player_positions[player["uuid"]], :] = normalized_stack

    return stack_tensor

def encode_player_states_tensor(round_state, player_positions):
    state_tensor = np.zeros((3, len(SHORT_RANKS) * len(SHORTER_SUITS)),  dtype=np.float32)  # 3 rows for player states

    for i, player in enumerate(round_state["seats"]):
        state_index = STATE_MAPPING[player["state"]]  # Map state to index (e.g., active, folded)
        state_tensor[player_positions[player["uuid"]], :] = state_index

    return state_tensor

def encode_legal_actions_tensor(legal_actions):
    legal_actions_tensor = np.zeros((1, len(SHORT_RANKS) * len(SHORTER_SUITS)),  dtype=np.float32)  # 1 row for legal actions

    for action in legal_actions:
        action_index = ACTION_MAPPING[action]  # Map actions (e.g., fold, call, raise) to indices
        legal_actions_tensor[0, action_index] = 1

    return legal_actions_tensor


def extract_features_tensor(
    hole_card: list[str], round_state: dict, player_uuid: str
) -> np.ndarray:
    # Initialize tensor: Channels x Height x Width
    tensor = np.zeros((4, 8, len(SHORT_RANKS) * len(SHORTER_SUITS)),  dtype=np.float32)

    # Encode cards
    community_cards = round_state["community_card"]
    tensor[0, :, :] = encode_cards_tensor(hole_card, community_cards)

    # Player positions
    dealer_index = round_state["dealer_btn"]
    rotated_seats = round_state["seats"][dealer_index:] + round_state["seats"][:dealer_index]
    player_positions = {p["uuid"]: i for i, p in enumerate(rotated_seats)}

    # Encode player bets
    tensor[1, :NUM_PLAYERS, :] = encode_player_bets_tensor(round_state, player_positions)

    # Encode stack sizes
    tensor[2, :NUM_PLAYERS, :] = encode_stack_sizes_tensor(round_state, player_positions)

    # Encode player states
    tensor[3, :NUM_PLAYERS, :] = encode_player_states_tensor(round_state, player_positions)

    # Encode legal actions
    # legal_actions = round_state["legal_actions"]
    # tensor[4, :, :] = encode_legal_actions_tensor(legal_actions)

    return tensor


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
    normalized_stack_sizes: Final[List[float]] = [
        stack_size / STACK for stack_size in stack_sizes
    ]

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

    # print(private_cards)
    # print(normalized_position)
    # print(stack_sizes)
    # print(player_states)

    features: np.ndarray = np.concatenate(
        [
            public_cards,
            private_cards,
            flattened_bets,
            normalized_stack_sizes,
            player_states,
            [normalized_position],
        ]
    )
    return features

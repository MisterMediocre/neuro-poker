"""Utility values and functions for the poker game."""

from typing import Any, Dict, Final, List

import numpy as np

from neuropoker.game.cards import (
    SHORT_RANKS,
    SHORTER_SUITS,
    get_card_index,
    get_card_indices,
)

# TODO: Remove hardcoding
STACK: Final[int] = 1000
NUM_PLAYERS: Final[int] = 3

# Street name -> index
STREET_MAPPING: Final[Dict[str, int]] = {
    "preflop": 1,
    "flop": 2,
    "turn": 3,
    "river": 4,
}

# Community card index -> street in which the card was dealt
COMMUNITY_CARD_MAPPING: Final[Dict[int, str]] = {
    0: "preflop",
    1: "preflop",
    2: "preflop",
    3: "flop",
    4: "turn",
}

# State name -> index
STATE_MAPPING: Final[Dict[str, int]] = {
    "folded": 0,
    "participating": 1,
    "allin": 2,
}

# Action name -> index
ACTION_MAPPING: Final[Dict[str, int]] = {
    "fold": 0,
    "call": 1,
    "raise": 2,
}


def encode_cards_tensor(hole_card, community_cards) -> np.ndarray:
    """Encode cards into a one-hot tensor.

    Parameters:
        hole_card: List[str]
            The private cards of the player.
        community_cards: List[str]
            The public cards on the table.

    Returns:
        card_tensor: np.ndarray
            The tensor encoding the cards.
    """

    card_tensor: Final[np.ndarray] = np.zeros(
        (
            # x: rows
            8,
            # y: cards
            len(SHORT_RANKS) * len(SHORTER_SUITS),
        ),
        dtype=np.float32,
    )

    ranks: Final[List[str]] = SHORT_RANKS
    suits: Final[List[str]] = SHORTER_SUITS

    num_ranks: Final[int] = len(ranks)
    num_suits: Final[int] = len(suits)

    # Rows for community cards
    for i, card in enumerate(community_cards):
        rank, suit = get_card_indices(card, ranks, suits)
        card_tensor[i, suit * num_ranks + rank] = 1  # Map (suit, rank) to index

    # Last row for hole cards
    for card in hole_card:
        rank, suit = get_card_indices(card, ranks, suits)
        card_tensor[-1, suit * num_ranks + rank] = 1  # Hole cards at row 7

    return card_tensor


def encode_player_bets_tensor(
    round_state,
    player_positions,
    num_players: int = NUM_PLAYERS,
    num_streets: int = 4,
    num_cards: int = len(SHORT_RANKS) * len(SHORTER_SUITS),
) -> np.ndarray:
    """Encode bets for each street (preflop, flop, turn, river).

    Parameters:
        round_state: Dict[str, Any]
            The state of the current round.
        player_positions: Dict[str, int]
            The position of each players at the table.
        num_players: int
            The number of players at the table.
        num_streets: int
            The number of streets in the game.
        num_cards: int
            The number of cards in the game.

    Returns:
        bet_tensor: np.ndarray
            The tensor encoding the bets made by each
    """
    bet_tensor: Final[np.ndarray] = np.zeros(
        (
            # row / x: players
            num_players * num_streets,
            # col / y: cards
            num_cards,
        ),
        dtype=np.float32,
    )

    for street, actions in round_state["action_histories"].items():
        # Map street to an integer index
        #
        # Zero-indexed (0 to 4)
        street_index = STREET_MAPPING[street] - 1

        # Iterate over actions in the street
        for action in actions:
            if action["action"].lower() in ["call", "raise"]:
                # Encode the bet made by the player
                bet: float = action["amount"]
                normalized_bet: float = bet / STACK  # Normalize

                player_pos: int = player_positions[action["uuid"]]
                player_street_pos: int = street_index * NUM_PLAYERS + player_pos

                bet_tensor[player_street_pos, :] += (
                    normalized_bet  # Add normalized bet for the specific street
                )

    return bet_tensor


def encode_stack_sizes_tensor(round_state, player_positions):
    stack_tensor = np.zeros(
        (3, len(SHORT_RANKS) * len(SHORTER_SUITS)), dtype=np.float32
    )  # 3 rows for players, stack sizes

    for i, player in enumerate(round_state["seats"]):
        normalized_stack = player["stack"] / STACK  # Normalize
        stack_tensor[player_positions[player["uuid"]], :] = normalized_stack

    return stack_tensor


def encode_player_states_tensor(round_state, player_positions):
    state_tensor = np.zeros(
        (3, len(SHORT_RANKS) * len(SHORTER_SUITS)), dtype=np.float32
    )  # 3 rows for player states

    for i, player in enumerate(round_state["seats"]):
        state_index = STATE_MAPPING[
            player["state"]
        ]  # Map state to index (e.g., active, folded)
        state_tensor[player_positions[player["uuid"]], :] = state_index

    return state_tensor


def encode_legal_actions_tensor(legal_actions):
    legal_actions_tensor = np.zeros(
        (1, len(SHORT_RANKS) * len(SHORTER_SUITS)), dtype=np.float32
    )  # 1 row for legal actions

    for action in legal_actions:
        action_index = ACTION_MAPPING[
            action
        ]  # Map actions (e.g., fold, call, raise) to indices
        legal_actions_tensor[0, action_index] = 1

    return legal_actions_tensor


def extract_features_tensor(
    hole_card: list[str],
    round_state: dict,
    player_uuid: str,
    num_players: int = NUM_PLAYERS,
    num_streets: int = 4,
    num_cards: int = len(SHORT_RANKS) * len(SHORTER_SUITS),
) -> np.ndarray:
    """Extract features for a poker agent from the current game state.

    Parameters:
        hole_card: List[str]
            The private cards of the player.
        round_state: Dict[str, Any]
            The state of the current round.
        player_uuid: str
            The player's UUID
        num_players: int
            The number of players at the table.
        num_cards: int
            The number of cards in the game.

    Returns:
        tensor: np.ndarray
            The features extracted from the game state.
    """
    # Initialize tensor: Channels x Height x Width
    tensor: np.ndarray = np.zeros(
        (
            # x: channels (cards, bets, stack sizes, states)
            4,
            # y: ?
            8,
            # z: cards
            num_cards,
        ),
        dtype=np.float32,
    )

    #
    # Channel 0: Cards
    #
    community_cards: Final[List[str]] = round_state["community_card"]
    tensor[0, :, :] = encode_cards_tensor(hole_card, community_cards)

    #
    # Channel 1: Player bets
    #
    # Get player positions
    dealer_index: Final[int] = round_state["dealer_btn"]
    rotated_seats: Final[List[Dict[str, Any]]] = (
        round_state["seats"][dealer_index:] + round_state["seats"][:dealer_index]
    )
    player_positions: Final[Dict[str, int]] = {
        p["uuid"]: i for i, p in enumerate(rotated_seats)
    }

    # tensor[1, :num_players, :] = encode_player_bets_tensor(
    #     round_state,
    #     player_positions,
    #     num_players=num_players,
    #     num_streets=num_streets,
    #     num_cards=num_cards,
    # )

    #
    # Channel 2: Stack sizes
    #
    tensor[2, :num_players, :] = encode_stack_sizes_tensor(
        round_state, player_positions
    )

    #
    # Channel 3: Player states
    #
    tensor[3, :num_players, :] = encode_player_states_tensor(
        round_state, player_positions
    )

    # Encode legal actions
    # legal_actions = round_state["legal_actions"]
    # tensor[4, :, :] = encode_legal_actions_tensor(legal_actions)

    return tensor

def extract_features_opp(hole_card: List[str], round_state: Dict[str, Any], player_uuid: str, opp_uuid: str) -> np.ndarray:
    player_features = extract_features(hole_card, round_state, player_uuid)

    dealer_index: Final[int] = round_state["dealer_btn"]
    rotated_seats: Final[List[Dict[str, Any]]] = (
        round_state["seats"][dealer_index:] + round_state["seats"][:dealer_index]
    )
    player_positions: Final[Dict[str, int]] = {
        p["uuid"]: i for i, p in enumerate(rotated_seats)
    }
    normalized_position: Final[float] = player_positions[opp_uuid] / (
        NUM_PLAYERS - 1
    )

    return np.append(player_features, normalized_position)


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
        Game suits
        Game ranks
        Game stack
        Number of players

    """
    game_suits: Final[List[str]] = SHORTER_SUITS
    game_ranks: Final[List[str]] = SHORT_RANKS
    game_stack: Final[int] = STACK
    num_game_players: Final[int] = NUM_PLAYERS

    num_game_suits: Final[int] = len(game_suits)
    num_game_ranks: Final[int] = len(game_ranks)
    num_game_cards: Final[int] = num_game_suits * num_game_ranks

    public_cards: np.ndarray = np.zeros(num_game_cards)
    private_cards: np.ndarray = np.zeros(num_game_cards)

    player_bets: Dict[str, List[float]] = {
        street: [0] * num_game_players
        for street in ["preflop", "flop", "turn", "river"]
    }  # Normalized bets per street

    community_cards: Final[List[str]] = round_state["community_card"]

    # for card in community_cards:
    for i, card in enumerate(community_cards):
        idx: int = get_card_index(card, game_suits, game_ranks)
        public_cards[idx] = STREET_MAPPING[COMMUNITY_CARD_MAPPING[i]]

    for card in hole_card:
        idx: int = get_card_index(card, game_suits, game_ranks)
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
        num_game_players - 1
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
                normalized_bet_amount: float = bet_amount / game_stack
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

    # print("Game ranks           : ", game_ranks)
    # print("Game suits           : ", game_suits)
    # print(f"Num game cards      : {num_game_cards}")
    # print(f"Public cards shape  : {public_cards.shape}")
    # print(f"Private cards shape : {private_cards.shape}")
    # print(f"Features shape : {features.shape}")
    # print(f"Features       : {features}")
    # print(f"Public cards   : {public_cards}")
    # print(f"Private cards  : {private_cards}")
    # print(f"Flattened bets : {flattened_bets}")
    # print(f"Stack sizes    : {stack_sizes}")
    # print(f"Player states  : {player_states}")
    # print(f"Position       : {normalized_position}")

    return features

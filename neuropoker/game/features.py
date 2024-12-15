"""Classes and functions for collecting game features."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Final, List, Tuple, override

import gymnasium
import numpy as np

from neuropoker.game.cards import (
    ALL_RANKS,
    ALL_SUITS,
    SHORT_RANKS,
    SHORTER_SUITS,
    get_card_index,
    get_card_indices,
)
from neuropoker.game.utils import (
    ACTION_MAPPING,
    COMMUNITY_CARD_MAPPING,
    NUM_PLAYERS,
    STACK,
    STATE_MAPPING,
    STREET_MAPPING,
)


class FeaturesCollector(ABC):
    """Base class for feature collectors."""

    def __init__(
        self,
        shape: Tuple[int, ...] = (1,),
        low: float = 0.0,
        high: float = 1.0,
        dtype: np.dtype = np.float32,  # type: ignore
    ) -> None:
        """Initialize the FeaturesCollector.

        Parameters:
            shape: Tuple[int, ...]
                The shape of the observation space.
            low: float
                The lower bound of possible observation space values.
            high: float
                The upper bound of possible observation space values.
            dtype: np.dtype
                The data type of the observation space values.
        """
        self.shape: Tuple[int, ...] = shape
        self.low: float = low
        self.high: float = high
        self.dtype: Final[np.dtype] = dtype

    def space(self) -> gymnasium.spaces.Space:
        """Convert the observation space to a gymnasium Box space."""
        return gymnasium.spaces.Box(
            low=self.low,
            high=self.high,
            shape=self.shape,
            dtype=self.dtype,  # type: ignore
        )

    @abstractmethod
    def extract_features(
        self, hole_card: List[str], round_state: Dict[str, Any], player_uuid: str
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
        """

    def __call__(self, *args, **kwargs) -> np.ndarray:
        """Call the extract_features method.

        Parameters:
            *args: Any
                Positional arguments.
            **kwargs: Any
                Keyword arguments.

        Returns:
            features: np.ndarray
                The features extracted from the game state.
        """
        return self.extract_features(*args, **kwargs)


class LinearFeaturesCollector(FeaturesCollector):
    """Feature extractor which returns a linear feature vector."""

    def __init__(self) -> None:
        """Initialize the LinearFeaturesCollector.

        TODO: Un-hardcode the following:
            Game suits
            Game ranks
            Game stack
            Number of players
        """
        super().__init__(
            shape=(73,),
            low=0.0,
            high=1.0,
        )

        self.suits: Final[List[str]] = SHORTER_SUITS
        self.ranks: Final[List[str]] = SHORT_RANKS
        self.stack: Final[int] = STACK
        self.num_players: Final[int] = NUM_PLAYERS

    @override
    def extract_features(
        self, hole_card: List[str], round_state: Dict[str, Any], player_uuid: str
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
        """
        num_suits: Final[int] = len(self.suits)
        num_ranks: Final[int] = len(self.ranks)
        num_cards: Final[int] = num_suits * num_ranks

        public_cards: np.ndarray = np.zeros(num_cards)
        private_cards: np.ndarray = np.zeros(num_cards)

        player_bets: Dict[str, List[float]] = {
            street: [0] * self.num_players
            for street in ["preflop", "flop", "turn", "river"]
        }  # Normalized bets per street

        community_cards: Final[List[str]] = round_state["community_card"]

        # for card in community_cards:
        for i, card in enumerate(community_cards):
            idx: int = get_card_index(card, self.suits, self.ranks)
            public_cards[idx] = STREET_MAPPING[COMMUNITY_CARD_MAPPING[i]]

        for card in hole_card:
            idx: int = get_card_index(card, self.suits, self.ranks)
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
            self.num_players - 1
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
                    normalized_bet_amount: float = bet_amount / self.stack
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

        if features.shape != self.shape:
            raise ValueError(
                f"Expected features shape {self.shape}, got {features.shape}"
            )

        return features


class CNNFeaturesCollector(FeaturesCollector):
    """Feature extractor which returns a CNN feature tensor."""

    def __init__(self) -> None:
        """Initialize the CNNFeaturesCollector."""
        super().__init__(
            shape=(4, 8, 52),
            low=0.0,
            high=1.0,
        )

        self.num_players = NUM_PLAYERS
        self.num_streets = 4
        self.stack = STACK

    def _encode_card_tensor(self, hole_card, community_cards) -> np.ndarray:
        """Encode cards into a one-hot tensor.

        Parameters:
            hole_card: List[str]
                The private cards of the player.
            community_cards: List[str]
                The public cards on the table.

        Returns:
            card_tensor: np.ndarray
                The tensor encoding the cards.

        The tensor is structured as follows:
            (Height x Width)

            Height: Number of cards in hand (7)
            Width: Number of cards in deck (52)
        """
        num_ranks: Final[int] = len(ALL_RANKS)
        num_suits: Final[int] = len(ALL_SUITS)

        card_tensor: Final[np.ndarray] = np.zeros(
            (
                # Height   (x) : Deck
                # Community cards (5, one-hot) + hole cards (2, two-hot)
                6,
                # Width    (y) : Cards
                # Use full deck in feature encoding, even if using a shorter deck.
                num_suits * num_ranks,
            ),
            dtype=np.float32,
        )

        # First rows for community cards
        for card_i, card in enumerate(community_cards):
            rank, suit = get_card_indices(card, ALL_RANKS, ALL_SUITS)

            # Map (suit, rank) to index
            card_tensor[card_i, suit * num_ranks + rank] = 1

        # Last row for hole cards
        for card in hole_card:
            rank, suit = get_card_indices(card, ALL_RANKS, ALL_SUITS)

            # Hole cards at row 7
            card_tensor[-1, suit * num_ranks + rank] = 1

        return card_tensor

    def _encode_bets_tensor(
        self,
        round_state,
        player_positions,
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

        Returns:
            bet_tensor: np.ndarray
                The tensor encoding the bets made by each

        The tensor is structured as follows:
            (Height x Width)

            Height: Number of players
            Width: Number of streets
        """
        bet_tensor: Final[np.ndarray] = np.zeros(
            (
                # Height   (x) : Players
                self.num_players,
                # Width    (y) : Streets
                self.num_streets,
            ),
            dtype=np.float32,
        )

        for street, actions in round_state["action_histories"].items():
            # Map street to an integer index
            #
            # Zero-indexed (0 to 4)
            street_pos: int = STREET_MAPPING[street] - 1

            # Iterate over actions in the street
            for action in actions:
                if action["action"].lower() in ["call", "raise"]:
                    # Encode the bet made by the player
                    bet: float = action["amount"]
                    normalized_bet: float = bet / STACK  # Normalize

                    player_pos: int = player_positions[action["uuid"]]

                    bet_tensor[player_pos, street_pos] += (
                        normalized_bet  # Add normalized bet for the specific street
                    )

        return bet_tensor

    def _encode_stack_tensor(self, round_state, player_positions):
        """Encode stacks.

        TODO: Docstring
        """
        # 3 rows for players, stack sizes
        stack_tensor: Final[np.ndarray] = np.zeros(
            (3, len(SHORT_RANKS) * len(SHORTER_SUITS)), dtype=np.float32
        )

        for player in round_state["seats"]:
            # Normalize stack
            normalized_stack: float = player["stack"] / self.stack
            player_pos: int = player_positions[player["uuid"]]

            stack_tensor[player_pos, :] = normalized_stack

        return stack_tensor

    def _encode_state_tensor(self, round_state, player_positions):
        """Encode states.

        TODO: Docstring
        """
        # 3 rows for player states
        state_tensor: Final[np.ndarray] = np.zeros(
            (3, len(SHORT_RANKS) * len(SHORTER_SUITS)), dtype=np.float32
        )

        for i, player in enumerate(round_state["seats"]):
            # Map state to index (e.g., active, folded)
            state_index: float = STATE_MAPPING[player["state"]] / max(
                STATE_MAPPING.values()
            )
            player_pos: int = player_positions[player["uuid"]]

            state_tensor[player_pos, :] = state_index

        return state_tensor

    def _encode_legal_actions_tensor(self, legal_actions):
        """Encode legal actions.

        TODO: Docstring
        """
        # 1 row for legal actions
        legal_actions_tensor: Final[np.ndarray] = np.zeros(
            (1, len(SHORT_RANKS) * len(SHORTER_SUITS)), dtype=np.float32
        )

        for action in legal_actions:
            action_index = ACTION_MAPPING[
                action
            ]  # Map actions (e.g., fold, call, raise) to indices
            legal_actions_tensor[0, action_index] = 1

        return legal_actions_tensor

    @override
    def extract_features(
        self,
        hole_card: List[str],
        round_state: Dict[str, Any],
        player_uuid: str,
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

        The tensor is structured as follows:
            (Channels x Height x Width)

            Channels:
                0: Cards
                1: Player bets
                2: Stack sizes
                3: Player states

            Height:
                ?

            Width:
                Number of cards
        """
        # Initialize tensor: Channels x Height x Width
        tensor: np.ndarray = np.zeros(
            self.shape,
            dtype=self.dtype,
        )

        #
        # Channel 0: Cards
        #
        community_cards: Final[List[str]] = round_state["community_card"]
        card_tensor: Final[np.ndarray] = self._encode_card_tensor(
            hole_card, community_cards
        )
        tensor[0, : card_tensor.shape[0], : card_tensor.shape[1]] = card_tensor

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

        bet_tensor: Final[np.ndarray] = self._encode_bets_tensor(
            round_state,
            player_positions,
        )
        tensor[1, : bet_tensor.shape[0], : bet_tensor.shape[1]] = bet_tensor

        #
        # Channel 2: Stack sizes
        #
        stack_tensor: Final[np.ndarray] = self._encode_stack_tensor(
            round_state, player_positions
        )
        tensor[2, : stack_tensor.shape[0], : stack_tensor.shape[1]] = stack_tensor

        #
        # Channel 3: Player states
        #
        state_tensor: Final[np.ndarray] = self._encode_state_tensor(
            round_state, player_positions
        )
        tensor[3, : state_tensor.shape[0], : state_tensor.shape[1]] = state_tensor

        # Encode legal actions
        # legal_actions = round_state["legal_actions"]
        # tensor[4, :, :] = encode_legal_actions_tensor(legal_actions)

        if tensor.shape != self.shape:
            raise ValueError(
                f"Expected features shape {self.shape}, got {tensor.shape}"
            )

        return tensor

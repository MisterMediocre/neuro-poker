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
        self.shape: Final[Tuple[int, ...]] = shape
        self.low: Final[float] = low
        self.high: Final[float] = high
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
            shape=(
                1,
                73,
            ),
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
        player_states: List[float] = [
            STATE_MAPPING[p["state"]] / max(STATE_MAPPING.values())
            for p in rotated_seats
        ]

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
        features = features.reshape(1, -1)

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
            shape=(10, 4, 13),
            low=0.0,
            high=1.0,
        )

        self.num_players = NUM_PLAYERS
        self.num_streets = 4
        self.stack = STACK

    def _encode_card_set_tensor(self, card_set) -> np.ndarray:
        """Encode n cards into an n-hot, 4 x 13 tensor.

        Parameters:
            card_set: List[str]
                The list of cards to encode.

        Returns:
            card_set_tensor: np.ndarray
                The tensor encoding the cards.

        The tensor is structured as follows:
            (Width x Height)

            Width: Number of suits (4)
                0: Clubs
                1: Diamonds
                2: Hearts
                3: Spades
            Height: Number of ranks (13)
                0: 2
                1: 3
                ...
                11: K
                12: A

        This is used by AlphaHoldem.
        """
        card_set_tensor: Final[np.ndarray] = np.zeros((4, 13))

        for card in card_set:
            rank, suit = get_card_indices(card, ALL_RANKS, ALL_SUITS)
            card_set_tensor[suit, rank] = 1

        return card_set_tensor

    def _encode_card_tensor(
        self, hole_cards: List[str], community_cards: List[str]
    ) -> np.ndarray:
        """Encode cards information into a tensor.

        Parameters:
            hole_cards: List[str]
                The private cards of the player.
            community_cards: List[str]
                The public cards on the table, in order of appearance.

        Returns:
            card_tensor: np.ndarray
                The tensor encoding the cards.

        The tensor is structured as follows:
            (Channels x Width x Height)

            Channels: Number of sub-decks (6)
                0: Hole cards
                1: Cards revealed in flop
                2: Card revealed in turn
                3: Card revealed in river
                4: Public cards (flop, turn, river)
                5: All cards (flop, turn, river, hole)
            Width: Number of suits (4)
            Height: Number of ranks (13)
        """
        # num_ranks: Final[int] = len(ALL_RANKS)
        # num_suits: Final[int] = len(ALL_SUITS)

        # Get each set of cards
        street_cards: Dict[str, List[str]] = {
            street: [] for street in COMMUNITY_CARD_MAPPING.values()
        }
        street_cards["hole"] = hole_cards

        for card_i, card in enumerate(community_cards):
            street: str = COMMUNITY_CARD_MAPPING[card_i]
            street_cards[street].append(card)

        # Encode each set of cards as an n-hot array
        street_arrs: Dict[str, np.ndarray] = {
            street: self._encode_card_set_tensor(street_cards[street])
            for street in street_cards
        }

        # Add additional tensors (public, all)
        street_arrs["public"] = (
            street_arrs["flop"] + street_arrs["turn"] + street_arrs["river"]
        )
        street_arrs["all"] = street_arrs["public"] + street_arrs["hole"]

        card_arr = np.stack([street_arrs[street] for street in street_arrs])
        # print(card_tensor.shape)
        return card_arr

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
            (Width x Height)

            Width: Number of players
            Height: Number of streets
        """
        bet_tensor: Final[np.ndarray] = np.zeros(
            (self.num_players, self.num_streets),
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

    def _action_to_index(self, action: str) -> int:
        """Map an action back to its index.

        Parameters:
            action: Tuple[str, int]
                The action to map.

        Returns:
            index: int
                The index of the action.
        """
        match action.lower():
            case "fold":
                return 0
            case "call":
                return 1
            case "raise":
                return 2
            case "smallblind":
                return 3
            case "bigblind":
                return 4
            case "ante":
                return 5
            case _:
                raise ValueError("Invalid action")

    def _player_to_index(self, player: str) -> int:
        """Map a player back to its index.

        Parameters:
            player: str
                The player to map.

        Returns:
            index: int
                The index of the player.
        """
        if player == "me":
            return 0

        return int(player.replace("opponent", "").replace("_", ""))

    def _encode_action_tensor(
        self,
        round_state: Dict[str, Any],
        player_positions: Dict[str, int],
        # legal_actions: List[str],
    ) -> np.ndarray:
        """Encode action information into a tensor.

        Parameters:
            round_state: Dict[str, Any]
                The state of the current round.
            player_positions: Dict[str, int]
                The position of each players at the table.
            legal_actions: List[str]
                The list of legal actions for the player.

        Returns:
            action_arr: np.ndarray
                The array encoding the actions.


        The tensor is structured as follows:
            (Channels x Width x Height)

            Channels: Number of streets (4)
            Width: Number of players (3)
            Height: Number of actions (6)
        """
        action_tensor: Final[np.ndarray] = np.zeros((4, 3, 6))

        for street, actions in round_state["action_histories"].items():
            street_idx: int = STREET_MAPPING[street] - 1
            for action_dict in actions:
                action: str = action_dict["action"]
                action_idx: int = self._action_to_index(action)

                player: str = action_dict["uuid"]
                player_idx: int = self._player_to_index(player)

                # amount: int = action_dict["amount"]
                # print(
                #     f"{(street_idx, action_idx, player_idx)}",
                #     street,
                #     action,
                #     player,
                #     amount,
                # )
                action_tensor[street_idx, player_idx, action_idx] = 1

        return action_tensor

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
            arr: np.ndarray
                The feature array extracted from the game state.

        The tensor is structured as follows:
            (Channels x Width x Height)

            Channels: 8
                0-5: Cards
                6-9: Actions

            Width: ?
            Height: ?
        """
        # Initialize tensor: Channels x Height x Width
        arr: np.ndarray = np.zeros(
            self.shape,
            dtype=self.dtype,
        )

        #
        # Channels 0-5: Cards
        #
        community_cards: Final[List[str]] = round_state["community_card"]
        card_arr: Final[np.ndarray] = self._encode_card_tensor(
            hole_card, community_cards
        )
        arr[0:6, 0:4, 0:13] = card_arr

        #
        # Channels 6-?: Actions
        #
        dealer_index: Final[int] = round_state["dealer_btn"]
        rotated_seats: Final[List[Dict[str, Any]]] = (
            round_state["seats"][dealer_index:] + round_state["seats"][:dealer_index]
        )
        player_positions: Final[Dict[str, int]] = {
            p["uuid"]: i for i, p in enumerate(rotated_seats)
        }

        # legal_actions = round_state["legal_actions"]
        action_arr: Final[np.ndarray] = self._encode_action_tensor(
            round_state,
            player_positions,  # legal_actions
        )
        arr[6:10, 0:3, 0:6] = action_arr

        #
        # Channel 1: Player bets
        #
        # Get player positions
        # dealer_index: Final[int] = round_state["dealer_btn"]
        # rotated_seats: Final[List[Dict[str, Any]]] = (
        #     round_state["seats"][dealer_index:] + round_state["seats"][:dealer_index]
        # )
        # player_positions: Final[Dict[str, int]] = {
        #     p["uuid"]: i for i, p in enumerate(rotated_seats)
        # }

        # bet_tensor: Final[np.ndarray] = self._encode_bets_tensor(
        #     round_state,
        #     player_positions,
        # )
        # tensor[5, : bet_tensor.shape[0], : bet_tensor.shape[1]] = bet_tensor

        # #
        # # Channel 2: Stack sizes
        # #
        # stack_tensor: Final[np.ndarray] = self._encode_stack_tensor(
        #     round_state, player_positions
        # )
        # tensor[6, : stack_tensor.shape[0], : stack_tensor.shape[1]] = stack_tensor

        # #
        # # Channel 3: Player states
        # #
        # state_tensor: Final[np.ndarray] = self._encode_state_tensor(
        #     round_state, player_positions
        # )
        # tensor[7, : state_tensor.shape[0], : state_tensor.shape[1]] = state_tensor

        # Encode legal actions
        # legal_actions = round_state["legal_actions"]
        # tensor[4, :, :] = encode_legal_actions_tensor(legal_actions)

        if arr.shape != self.shape:
            raise ValueError(f"Expected features shape {self.shape}, got {arr.shape}")

        return arr

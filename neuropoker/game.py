"""Classes and functions for poker games.
"""

import random
from typing import Any, Dict, Final, List

import numpy as np
from pypokerengine.api.emulator import Emulator
from pypokerengine.players import BasePokerPlayer

from cards import get_card_index, get_card_list, get_deck, SHORT_RANKS, SHORT_SUITS

# from pypokerengine.api.game import setup_config, start_poker
# from pypokerengine.engine.player import Player
# from pypokerengine.engine.poker_constants import PokerConstants as Const
# from pypokerengine.engine.table import Table
# from pypokerengine.utils.card_utils import gen_cards


STREET_MAPPING: Final[Dict[str, int]] = {
    "flop": 1,
    "turn": 2,
    "river": 3,
}
STATE_MAPPING: Final[Dict[str, int]] = {
    "folded": 0,
    "participating": 1,
    "allin": 2,
}

NUM_PLAYERS: Final[int] = 3


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

    TODO: Add state of each player (Folded, all-in, etc.)
    """
    public_cards: np.ndarray = np.zeros(36)
    private_cards: np.ndarray = np.zeros(36)

    player_bets: Dict[str, List[int]] = {
        street: [0] * NUM_PLAYERS for street in ["preflop", "flop", "turn", "river"]
    }  # Bets per street

    street: Final[str] = round_state["street"]
    community_cards: Final[List[str]] = round_state["community_card"]

    for card in community_cards:
        idx: int = get_card_index(card, ranks=SHORT_RANKS, suits=SHORT_SUITS)
        public_cards[idx] = STREET_MAPPING[street]

    for card in hole_card:
        idx: int = get_card_index(card, ranks=SHORT_RANKS, suits=SHORT_SUITS)
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
                relative_pos = player_positions[action["uuid"]]
                if relative_pos != -1:
                    player_bets[street_name][relative_pos] += bet_amount

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


class Game:
    """A poker game."""

    def __init__(
        self,
        players: List[BasePokerPlayer],
        cards: List[str] = get_card_list(),
        max_rounds: int = 5,
        small_blind_amount: int = 25,
        big_blind_amount: int = 50,
        stack: int = 1000,
    ) -> None:
        """Initialize the game.

        Parameters:
            players: List[BasePokerPlayer]
                The list of players.
            cards: List[str]
                The list of cards available to use in the deck.
            max_rounds: int
                The maximum number of rounds.
            small_blind_amount: int
                The amount of the small blind.
            big_blind_amount: int
                The amount of the big blind.
            stack: int
                The stack size for each player.
        """
        self.players: Final[List[BasePokerPlayer]] = players
        self.players_info: Final[Dict[str, Dict[str, Any]]] = {
            f"uuid-{i}": {"name": f"player{i}", "stack": stack}
            for i in range(len(self.players))
        }

        self.cards: Final[List[str]] = cards
        self.max_rounds: Final[int] = max_rounds
        self.small_blind_amount: Final[int] = small_blind_amount
        self.big_blind_amount: Final[int] = big_blind_amount
        self.stack: Final[int] = stack

        # Confiugre poker emulator
        self.emulator: Final[Emulator] = Emulator()
        self.emulator.set_game_rule(
            len(self.players),
            self.max_rounds,
            self.small_blind_amount,
            self.big_blind_amount,
        )
        for i, player in enumerate(self.players):
            self.emulator.register_player(f"uuid-{i}", player)

    def play(
        self,
        seed: int = 1,
        games_played: int = 0,
    ) -> List[float]:
        """Play a single round/game of Poker.

        Parameters:
            seed: int
                The seed to use for the game.
            games_played: int
                The number of games previously played.
                (Used to keep decks unique in each game)

        Returns:
            winnings: List[float]
                The winnings of each player.
        """
        winnings: List[float] = [0] * len(self.players)
        initial_state: Final[Dict[str, Any]] = (
            self.emulator.generate_initial_game_state(self.players_info)
        )

        # Same seed means same cards dealt
        initial_state["table"].deck = get_deck(
            cards=self.cards, seed=seed * 100 + games_played
        )
        initial_state["table"].dealer_btn = random.randint(0, len(self.players) - 1)

        game_state, _event = self.emulator.start_new_round(initial_state)
        game_state, _event = self.emulator.run_until_game_finish(game_state)

        for i, _ in enumerate(self.players):
            stack = game_state["table"].seats.players[i].stack
            winnings[i] = stack - self.stack

        return winnings

    def play_multiple(self, num_games: int = 100, seed: int = 1) -> List[float]:
        """Play multiple games of Poker.

        Parameters:
            num_games: int
                The number of games to play.
            seed: int
                The seed to use for the games.

        Returns:
            winnings: List[float]
                Cumulative winnings of each player.
        """
        winnings: List[float] = [0] * len(self.players)
        for i in range(num_games):
            game_winnings: List[float] = self.play(seed=seed, games_played=i)
            winnings = [w + game_winnings[j] for j, w in enumerate(winnings)]

        return winnings


def evaluate_fitness(
    player_model: BasePokerPlayer,
    opponent_models: List[BasePokerPlayer],
    num_games: int = 100,
    seed: int = 1,
) -> float:
    """Evaluate the fitness of a player against other opponents.

    Parameters:
        player_model: BasePokerPlayer
            The player model to evaluate.
        opponent_models: List[BasePokerPlayer]
            The list of opponent models to play against.
        num_games: int
            The number of games to play.
        seed: int
            The seed to use for the evaluation.

    Returns:
        fitness: float
            The fitness of the player model.

    Fitness is defined by the average winnings per game, but can
    be adjusted to something else (TODO).
    """
    cards: Final[List[str]] = get_card_list(suits=SHORT_SUITS, ranks=SHORT_RANKS)

    game: Final[Game] = Game(
        [player_model] + opponent_models,
        cards=cards,
    )

    # Play multiple games and average the winnings
    sum_winnings = game.play_multiple(num_games=num_games, seed=seed)

    # Return player's average winnings per game
    return sum_winnings[0] / num_games

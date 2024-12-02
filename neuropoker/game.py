"""Classes and functions for poker games.
"""

from typing import Any, Dict, Final, List

from pypokerengine.api.emulator import Emulator

from neuropoker.cards import SHORT_RANKS, SHORT_SUITS, get_card_list, get_deck
from neuropoker.game_utils import NUM_PLAYERS
from neuropoker.player import BasePlayer
    


class Game:
    """A poker game."""

    def __init__(
        self,
        player_names: List[str],
        player_models: List[BasePlayer],
        cards: List[str] = get_card_list(),
        max_rounds: int = 5,
        small_blind_amount: int = 25,
        big_blind_amount: int = 50,
        stack: int = 1000,
    ) -> None:
        """Initialize the game.

        Parameters:
            player_names: List[str]
                The names of the players.
            player_models: List[BasePlayer]
                The models of the players.
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
        assert len(player_names) == len(player_models)
        assert len(player_names) == NUM_PLAYERS

        self.players: Final[Dict[str, BasePlayer]] = {
            name: model for name, model in zip(player_names, player_models)
        }
        self.players_info: Final[Dict[str, Dict[str, Any]]] = {
            name: {"name": name, "stack": stack, "uuid": name} for name in self.players
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
        for name, model in self.players.items():
            self.emulator.register_player(name, model)

    def play(
        self,
        dealer_button: int = 0,
        seed: int = 1,
        games_played: int = 0,
    ) -> List[float]:
        """Play a single round/game of Poker.

        Parameters:
            dealer_button: int
                The position of the dealer button.
            seed: int
                The seed to use for the game.
            games_played: int
                The number of games previously played.
                (Used to keep decks unique in each game)

        Returns:
            winnings: List[float]
                The winnings of each player.
        """
        winnings: List[float] = [0.0] * len(self.players)
        initial_state: Final[Dict[str, Any]] = (
            self.emulator.generate_initial_game_state(self.players_info)
        )

        # Same seed means same cards dealt
        initial_state["table"].deck = get_deck(
            cards=self.cards, seed=seed * 100 + games_played
        )
        initial_state["table"].dealer_btn = dealer_button

        game_state, _event = self.emulator.start_new_round(initial_state)
        game_state, _event = self.emulator.run_until_round_finish(game_state)

        for j, player in enumerate(game_state["table"].seats.players):
            # print(j, player.name, player.stack - STACK)
            winnings[j] = player.stack - self.stack

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
        dealer_button: int = 0

        for i in range(num_games):
            # Play a game
            game_winnings: List[float] = self.play(
                dealer_button=dealer_button, seed=seed, games_played=i
            )

            # Update cumulative winnings
            winnings = [w + game_winnings[j] for j, w in enumerate(winnings)]

            # Update dealer button
            dealer_button = (dealer_button + 1) % len(self.players)

        return winnings


def evaluate_fitness(
    player_names: List[str],
    player_models: List[BasePlayer],
    num_games: int = 100,
    seed: int = 1,
) -> List[float]:
    """Evaluate the fitness of a player against other opponents.

    Parameters:
        player_names: List[str]
            The names of the players to simulate.
        player_models: List[BasePlayer]
            The models of the players to simulate.
        num_games: int
            The number of games to play.
        seed: int
            The seed to use for the evaluation.

    Returns:
        fitnesses: List[float]
            The fitness of the players

    Fitness is defined by the average winnings per game, but can
    be adjusted to something else (TODO).
    """
    assert len(player_names) == len(player_models)
    assert len(player_names) == NUM_PLAYERS
    assert num_games > 0

    # Create card list
    cards: Final[List[str]] = get_card_list(
        ranks=SHORT_RANKS, suits=SHORT_SUITS
    )  # Use a smaller deck

    # Create game
    game: Final[Game] = Game(player_names, player_models, cards=cards)

    # Play multiple games
    sum_winnings: Final[List[float]] = game.play_multiple(
        num_games=num_games, seed=seed
    )

    # Compute each player's fitness
    #
    # Divide each player's winnings by the number of games played.
    for i in range(len(player_names)):
        sum_winnings[i] = sum_winnings[i] / num_games

    # Return the fitnesses
    return sum_winnings

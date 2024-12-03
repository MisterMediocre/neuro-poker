"""Classes and functions for poker games.
"""

from typing import Any, Dict, Final, List, Optional, TypedDict

from pypokerengine.api.emulator import Emulator
from pypokerengine.engine.player import Player

from neuropoker.cards import get_card_list, get_deck
from neuropoker.config import Config
from neuropoker.game_utils import NUM_PLAYERS
from neuropoker.players.base_player import BasePlayer

# TODO: Un-hardcode these constants
SMALL_BLIND_AMOUNT: Final[int] = 25
BIG_BLIND_AMOUNT: Final[int] = 50
STACK: Final[int] = 1000


class PlayerStats(TypedDict):
    """A player's statistics for a game."""

    uuid: str
    winnings: float
    num_games: int
    folds: int
    calls: int
    raises: int
    allin: int
    big_blinds: int
    small_blinds: int
    # preflop_folds: int
    # preflop_calls: int
    # preflop_raises: int
    # flop_folds: int
    # flop_calls: int
    # flop_raises: int
    # turn_folds: int
    # turn_calls: int
    # turn_raises: int
    # river_folds: int
    # river_calls: int
    # river_raises: int


def default_player_stats() -> PlayerStats:
    """Get the default set of player stats.

    Returns:
        default_stats: PlayerStats
            The default set of player stats.
    """
    return {
        "uuid": "",
        "winnings": 0,
        "big_blinds": 0,
        "small_blinds": 0,
        "num_games": 0,
        "folds": 0,
        "calls": 0,
        "raises": 0,
        "allin": 0,
        # "preflop_folds": 0,
        # "preflop_calls": 0,
        # "preflop_raises": 0,
        # "flop_folds": 0,
        # "flop_calls": 0,
        # "flop_raises": 0,
        # "turn_folds": 0,
        # "turn_calls": 0,
        # "turn_raises": 0,
        # "river_folds": 0,
        # "river_calls": 0,
        # "river_raises": 0,
    }


def read_game(game_state, events) -> Dict[str, PlayerStats]:
    """Obtain each player's stats for a game.

    Parameters:
        game_state: Dict[str, Any]
            The game state.
        events: List[Dict[str, Any]]
            The events of the game.

    Returns:
        player_stats: Dict[str, PlayerStats]
            Each player's stats.
    """

    # print(events)

    player_stats: Dict[str, PlayerStats] = {}
    players: Final[List[Player]] = game_state["table"].seats.players
    for player in players:
        default = default_player_stats()
        default["uuid"] = player.uuid
        default["winnings"] = player.stack - STACK
        default["num_games"] = 1

        # print(player.action_histories)
        player_stats[player.uuid] = default

    for event in events:
        if event["type"] == "event_round_finish":
            # Process action histories to count actions
            for _street, actions in event["round_state"]["action_histories"].items():
                for action in actions:
                    uuid = action["uuid"]
                    # print(uuid, action["action"])
                    assert uuid in player_stats
                    if action["action"] == "FOLD":
                        player_stats[uuid]["folds"] += 1
                    elif action["action"] == "CALL":
                        player_stats[uuid]["calls"] += 1
                    elif action["action"] == "RAISE":
                        player_stats[uuid]["raises"] += 1
                    elif action["action"] == "BIGBLIND":
                        player_stats[uuid]["big_blinds"] += 1
                    elif action["action"] == "SMALLBLIND":
                        player_stats[uuid]["small_blinds"] += 1

            for p in event["round_state"]["seats"]:
                p: Dict[str, Any]
                uuid = p["uuid"]
                assert uuid in player_stats
                # all-in
                if p["state"] == "allin":
                    player_stats[uuid]["allin"] += 1

    return player_stats


def merge(stats1: PlayerStats, stats2: PlayerStats) -> PlayerStats:
    """Merge two players' stats into one.

    Parameters:
        stats1: PlayerStats
            The first player's stats.
        stats2: PlayerStats
            The second player's stats.

    Returns:
        stats_merged: PlayerStats
            The merged stats.
    """
    if stats1["uuid"] != stats2["uuid"]:
        raise ValueError(
            f"UUIDs do not match, found {stats1['uuid']} and {stats2['uuid']}"
        )
    return {
        "uuid": stats1["uuid"],
        "winnings": stats1["winnings"] + stats2["winnings"],
        "num_games": stats1["num_games"] + stats2["num_games"],
        "folds": stats1["folds"] + stats2["folds"],
        "calls": stats1["calls"] + stats2["calls"],
        "raises": stats1["raises"] + stats2["raises"],
        "allin": stats1["allin"] + stats2["allin"],
        "big_blinds": stats1["big_blinds"] + stats2["big_blinds"],
        "small_blinds": stats1["small_blinds"] + stats2["small_blinds"],
        # "preflop_folds": stats1["preflop_folds"] + stats2["preflop_folds"],
        # "preflop_calls": stats1["preflop_calls"] + stats2["preflop_calls"],
        # "preflop_raises": stats1["preflop_raises"] + stats2["preflop_raises"],
        # "flop_folds": stats1["flop_folds"] + stats2["flop_folds"],
        # "flop_calls": stats1["flop_calls"] + stats2["flop_calls"],
        # "flop_raises": stats1["flop_raises"] + stats2["flop_raises"],
        # "turn_folds": stats1["turn_folds"] + stats2["turn_folds"],
        # "turn_calls": stats1["turn_calls"] + stats2["turn_calls"],
        # "turn_raises": stats1["turn_raises"] + stats2["turn_raises"],
        # "river_folds": stats1["river_folds"] + stats2["river_folds"],
        # "river_calls": stats1["river_calls"] + stats2["river_calls"],
        # "river_raises": stats1["river_raises"] + stats2["river_raises"],
    }


def bb_per_hand(stats: PlayerStats) -> float:
    """Compute the big blind per hand."""
    return stats["winnings"] / (stats["num_games"] * BIG_BLIND_AMOUNT)


class Game:
    """A poker game."""

    def __init__(
        self,
        players: List[BasePlayer],
        cards: List[str],
        max_rounds: int = 1,
        small_blind_amount: int = SMALL_BLIND_AMOUNT,
        stack: int = STACK,
    ) -> None:
        """Initialize the game.

        Parameters:
            players: List[str]
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
        if len(players) != NUM_PLAYERS:
            raise ValueError(
                f"There must be 3 players, instead received {len(players)} players"
            )
        if len(cards) <= 0:
            raise ValueError("No cards provided")

        self.players: Final[List[BasePlayer]] = players
        self.players_info: Final[Dict[str, Dict[str, Any]]] = {
            model.uuid: {"name": model.uuid, "stack": stack, "uuid": model.uuid}
            for model in self.players
        }

        # Check that all players have unique UUIDs
        if len(self.players_info) != len(self.players):
            raise ValueError("Players must have unique UUIDs")

        self.cards: Final[List[str]] = cards
        self.max_rounds: Final[int] = max_rounds
        self.small_blind_amount: Final[int] = small_blind_amount
        self.stack: Final[int] = stack

        # Confiugre poker emulator
        self.emulator: Final[Emulator] = Emulator()
        self.emulator.set_game_rule(
            len(self.players),
            self.max_rounds,
            self.small_blind_amount,
            0,  # NO ANTE
        )
        for model in self.players:
            self.emulator.register_player(model.uuid, model)

    @staticmethod
    def from_config(
        players: List[BasePlayer],
        config: Config,
    ) -> "Game":
        """Initialize the game from a configuration file.

        Parameters:
            players: List[BasePlayer]
                The list of players.
            config: Config
                The configuration file.
        """
        config = config["game"]
        if len(players) != config["players"]:
            raise ValueError(
                f"Number of players in config ({config['players']}) "
                f"does not match length of provided players "
                f"({len(players)})."
            )

        cards: Final[List[str]] = get_card_list(config["suits"], config["ranks"])

        return Game(
            players,
            cards,
            max_rounds=config["rounds"],
            small_blind_amount=config["small_blind"],
            # big_blind_amount=config["big_blind"],
            stack=config["stack"],
        )

    def play(
        self,
        dealer_button: int = 0,
        games_played: int = 0,
        seed: Optional[int] = None,
    ) -> Dict[str, PlayerStats]:
        """Play a single round/game of Poker.

        Parameters:
            dealer_button: int
                The position of the dealer button.
            games_played: int
                The number of games previously played.
                (Used to keep decks unique in each game)
            seed: Optional[int]
                The seed to use for the game.

        Returns:
            winnings: List[float]
                The winnings of each player.
        """
        # winnings: List[float] = [0.0] * len(self.players)
        initial_state: Final[Dict[str, Any]] = (
            self.emulator.generate_initial_game_state(self.players_info)
        )

        # Same seed means same cards dealt
        initial_state["table"].deck = get_deck(
            cards=self.cards,
            seed=(seed * 100 + games_played if seed is not None else None),
        )
        initial_state["table"].dealer_btn = dealer_button

        game_state, _event = self.emulator.start_new_round(initial_state)
        game_state, events = self.emulator.run_until_round_finish(game_state)

        return read_game(game_state, events)

    def play_multiple(
        self, num_games: int = 100, seed: Optional[int] = None
    ) -> Dict[str, PlayerStats]:
        """Play multiple games of Poker.

        Parameters:
            num_games: int
                The number of games to play.
            seed: Optional[int]
                The seed to use for the games.

        Returns:
            winnings: List[float]
                Cumulative winnings of each player.
        """
        dealer_button: int = 0

        player_stats = {}
        for player in self.players:
            player_stats[player.uuid] = default_player_stats()
            player_stats[player.uuid]["uuid"] = player.uuid

        for i in range(num_games):
            # Play a game
            round_stats = self.play(
                dealer_button=dealer_button, seed=seed, games_played=i
            )

            for player_uuid, player_round_stats in round_stats.items():
                player_stats[player_uuid] = merge(
                    player_stats[player_uuid], player_round_stats
                )

            # print("Game", i, "done", round_stats['model_1']['winnings'])

            # Merge player stats
            # Update dealer button
            dealer_button = (dealer_button + 1) % len(self.players)

        return player_stats

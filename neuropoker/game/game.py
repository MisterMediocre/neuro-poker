"""Classes and functions for poker games."""

from typing import Any, Dict, Final, List, Optional, Tuple, TypedDict

from pypokerengine.api.emulator import Emulator
from pypokerengine.engine.player import Player

from neuropoker.extra.config import Config
from neuropoker.game.cards import get_card_list, get_deck
from neuropoker.game.utils import NUM_PLAYERS
from neuropoker.players.base import BasePlayer

# TODO: Un-hardcode these constants
SMALL_BLIND_AMOUNT: Final[int] = 25
BIG_BLIND_AMOUNT: Final[int] = 50
STACK: Final[int] = 1000

# Type for game state
GameState = Dict[str, Any]

# Type for event state
EventState = List[Dict[str, Any] | None] | List[Dict[str, Any]]


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
    plays: Dict[Tuple[int, str, str, int], int]
    dealer_fold: Dict[Tuple[str, str], int]
    dealer_call: Dict[Tuple[str, str], int]
    dealer_raise: Dict[Tuple[str, str], int]

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
        "plays": {},
        "dealer_fold": {},
        "dealer_call": {},
        "dealer_raise": {},
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

    player_stats: Dict[str, PlayerStats] = {}
    players: Final[List[Player]] = game_state["table"].seats.players
    for player in players:
        default = default_player_stats()
        default["uuid"] = player.uuid
        default["winnings"] = player.stack - STACK
        default["num_games"] = 1

        player_stats[player.uuid] = default

    for event in events:
        if event["type"] == "event_round_finish":
            # Process action histories to count actions
            for street, actions in event["round_state"]["action_histories"].items():
                for action in actions:
                    uuid = action["uuid"]
                    # print(uuid, action["action"])

                    # hole_card = tuple(event["round_state"]["seats"][uuid]["hole_card"])
                    # print(hole_card)

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


def dict_add(
    d1: Dict[Tuple[str, str], int], d2: Dict[Tuple[str, str], int]
) -> Dict[Tuple[str, str], int]:
    """Add two dictionaries of tuples."""
    for k, v in d2.items():
        if k in d1:
            d1[k] += v
        else:
            d1[k] = v
    return d1


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
        "plays": stats1["plays"],
        "dealer_fold": dict_add(stats1["dealer_fold"], stats2["dealer_fold"]),
        "dealer_call": dict_add(stats1["dealer_call"], stats2["dealer_call"]),
        "dealer_raise": dict_add(stats1["dealer_raise"], stats2["dealer_raise"]),
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
            player.uuid: {"name": player.uuid, "stack": stack, "uuid": player.uuid}
            for player in self.players
        }

        # Check that all players have unique UUIDs
        if len(self.players_info) != len(self.players):
            raise ValueError(
                f"Players must have unique UUIDs, received {[p.uuid for p in self.players]}"
            )

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

    def read_game(
        self,
        game_state: GameState,
        events: EventState,
    ) -> Dict[str, PlayerStats]:
        """Obtain each player's stats for a game.

        Parameters:
            game_state: GameState
                The game state.
            events: EventState
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
            default["winnings"] = player.stack - self.stack
            default["num_games"] = 1

            # print(player.action_histories)
            player_stats[player.uuid] = default

        for event in events:
            if event is None:
                continue

            if event["type"] == "event_round_finish":
                # Process action histories to count actions
                for _street, actions in event["round_state"][
                    "action_histories"
                ].items():
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

    def get_initial_state(
        self, dealer_button: int = 0, seed: int | None = None
    ) -> GameState:
        """Get the initial state of the game.

        Returns:
            initial_state: Dict[str, Any]
                The initial state of the game.
        """
        initial_state: Final[GameState] = self.emulator.generate_initial_game_state(
            self.players_info
        )

        # Same seed means same cards dealt
        initial_state["table"].deck = get_deck(
            cards=self.cards,
            seed=seed,
        )
        initial_state["table"].dealer_btn = dealer_button

        return initial_state

    def start_round(
        self, dealer_button: int = 0, seed: int | None = None
    ) -> Tuple[GameState, EventState]:
        """Start a round of poker.

        Parameters:
            dealer_button: int
                The position of the dealer button.
            seed: Optional[int]
                The seed to use to shuffle the deck.

        Returns:
            game_state: GameState
                The game state.
            events: EventState
                The event.
        """
        initial_state: Final[GameState] = self.get_initial_state(
            dealer_button=dealer_button, seed=seed
        )

        game_state, events = self.emulator.start_new_round(initial_state)
        return game_state, events

    def play(
        self,
        dealer_button: int = 0,
        seed: int | None = None,
    ) -> Dict[str, PlayerStats]:
        """Play a single round/game of Poker.

        Parameters:
            dealer_button: int
                The position of the dealer button.
            seed: Optional[int]
                The seed to use to shuffle the deck.

        Returns:
            winnings: List[float]
                The winnings of each player.
        """
        game_state, _ = self.start_round(dealer_button=dealer_button, seed=seed)
        game_state, events = self.emulator.run_until_round_finish(game_state)
        return self.read_game(game_state, events)

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
        if num_games <= 0:
            raise ValueError(f"Number of games must be positive, found {num_games}")

        player_stats: Dict[str, PlayerStats] = {}
        for player in self.players:
            player_stats[player.uuid] = default_player_stats()
            player_stats[player.uuid]["uuid"] = player.uuid

        for game_i in range(num_games):
            # Play a game
            round_stats: Dict[str, PlayerStats] = self.play(
                dealer_button=(game_i % len(self.players)),
                seed=(seed + game_i if seed is not None else None),
            )

            for uuid, stats in round_stats.items():
                player_stats[uuid] = merge(player_stats[uuid], stats)

            # print("Game", i, "done", round_stats['model_1']['winnings'])

        return player_stats

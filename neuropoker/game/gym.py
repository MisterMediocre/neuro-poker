"""Classes for modeling a poker game in a Gym environment."""

import random
import time
from typing import Any, Dict, Final, List, Tuple, override

import gymnasium
import numpy as np
from gymnasium import spaces
from pypokerengine.engine.data_encoder import DataEncoder
from pypokerengine.engine.poker_constants import PokerConstants as Const

from neuropoker.game.game import Game, GameState
from neuropoker.game.utils import STACK, extract_features
from neuropoker.players.base import BasePlayer
from neuropoker.players.ppo import PPOPlayer

RESET_THRESHOLD: Final[int] = 30000


class PokerEnv(gymnasium.Env):
    """Class for modeling a poker game in a Gym environment."""

    def __init__(self, game: Game) -> None:
        """Initialize the poker gym environment.

        Parameters:
            game: Game
                The underlying poker game.
        """
        num_features: Final[int] = 73

        # Gym options
        self.observation_space: spaces.Space = spaces.Box(
            low=0, high=3, shape=(1, num_features), dtype=np.float32
        )
        self.action_space: spaces.Space = spaces.Discrete(5)

        # Game
        self.game: Game = game

        random.seed(time.time())
        self.seed: Final[int] = random.randint(0, 10000)
        print("SEED:", self.seed)

        # Stats
        self.game_state: GameState = {}
        self.num_games: int = 0
        self.bad_games: int = 0
        self.total_reward: float = 0
        self.cumulative_reward: float = 0
        self.statistics: Dict[Tuple[str, str], Any] = {}

        self.reset()

    def keep_playing(self, break_me: bool) -> None:
        """Play a single round/game of Poker.

        Parameters:
            break_me: bool
                Whether our player should break after the player's turn.
        """
        game_state: GameState = self.game_state

        while game_state["street"] != Const.Street.FINISHED:
            next_player: int = game_state["next_player"]
            if next_player == self.player_pos and break_me:
                break

            table = game_state["table"]
            round_state = DataEncoder.encode_round_state(game_state)
            player = table.seats.players[next_player]
            valid_actions = self.game.emulator.generate_possible_actions(game_state)
            hole_card = DataEncoder.encode_player(player, holecard=True)["hole_card"]

            if len(hole_card) != 2:
                raise ValueError(f"Invalid hole card: {hole_card}")

            action, bet = self.game.players[next_player].declare_action(
                valid_actions, hole_card, round_state
            )
            game_state, _event = self.game.emulator.apply_action(
                game_state, action, bet
            )

        self.game_state = game_state

    @override
    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment.

        Parameters:
            **kwargs
                Unused

        Returns:
            observation: np.ndarray
                The observation after the reset.
            info: Dict[str, Any]
                Additional information.
        """
        self.num_games += 1

        seed = (self.num_games // len(self.game.players)) % 10000
        seed += self.seed

        # For the same cards dealt, the player should try each position
        # at the table.
        #
        # Dealer is always 1 (arbitrary)
        dealer_pos: Final[int] = 1
        self.player_pos: int = self.num_games % 3

        new_players: Final[List[BasePlayer]] = (
            self.game.players[self.player_pos :] + self.game.players[: self.player_pos]
        )
        self.game = Game(
            # Create an identical game with re-ordered positions
            new_players,
            self.game.cards,
            max_rounds=self.game.max_rounds,
            small_blind_amount=self.game.small_blind_amount,
            stack=self.game.stack,
        )

        self.game_state, _events = self.game.start_round(
            dealer_button=dealer_pos, seed=seed
        )
        self.keep_playing(break_me=True)

        round_state: Final[Dict[str, Any]] = DataEncoder.encode_round_state(
            self.game_state
        )
        if round_state["street"] is None:
            # Both opponents folded already...
            self.bad_games += 1
            return self.reset()

        encoded_player: Final[Dict[str, Any]] = DataEncoder.encode_player(
            self.game_state["table"].seats.players[self.player_pos], holecard=True
        )
        hole_card: Final[List[str]] = encoded_player["hole_card"]

        extracted_features: np.ndarray = extract_features(hole_card, round_state, "me")
        extracted_features = extracted_features[np.newaxis, :]
        return extracted_features, {}

    @override
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Perform one step in the environment.

        Parameters:
            action: int
                The action to take.

        Returns:
            observation: np.ndarray
                The observation after the action.
            reward: float
                The reward after the action.
            done: bool
                Whether the episode is done.
            truncated: bool
                Whether the episode was truncated.
            info: Dict[str, Any]
                Additional information.
        """
        game_state: Final[GameState] = self.game_state
        hole_card: Final[List[str]] = DataEncoder.encode_player(
            self.game_state["table"].seats.players[self.player_pos], holecard=True
        )["hole_card"]

        if len(hole_card) != 2:
            raise ValueError(f"Invalid hole card: {hole_card}")

        valid_actions: List[Dict[str, Any]] = (
            self.game.emulator.generate_possible_actions(game_state)
        )
        street: Final[str] = game_state["street"]
        my_action, bet_amount = PPOPlayer.int_to_action(action, valid_actions)
        self.statistics[(street, my_action)] = (
            self.statistics.get((street, my_action), 0) + 1
        )

        # print(my_action, bet_amount)
        self.game_state, _event = self.game.emulator.apply_action(
            game_state, my_action, bet_amount
        )
        self.keep_playing(break_me=True)

        round_state = DataEncoder.encode_round_state(self.game_state)
        extracted_features = extract_features(hole_card, round_state, "me")[
            np.newaxis, :
        ]

        if self.game_state["street"] == Const.Street.FINISHED:
            stack: float = self.game_state["table"].seats.players[self.player_pos].stack
            reward: float = stack - STACK
            self.total_reward += reward
            self.cumulative_reward += reward

            if self.num_games % RESET_THRESHOLD == 0:
                # print("opponent", self.opponent_path)
                # print("opponent2:", self.opponent2_path)
                # print("Total games:", self.num_games)
                # print("Bad games:", self.bad_games)
                # print("Cumulative reward:", self.cumulative_reward)
                # print("Total reward:", self.total_reward)
                # print("Average reward:", self.total_reward / RESET_THRESHOLD)
                # print("Average rewards since beginning:", self.cumulative_reward / self.num_games)
                # print("bb per 100g:", self.total_reward * 100 / (RESET_THRESHOLD * BIG_BLIND_AMOUNT))
                # print("Statistics:", self.statistics)
                # for i, street in enumerate(["preflop", "flop", "turn", "river"]):
                # for act in ["fold", "call", "raise"]:
                # print(f"{street} {act}: {self.statistics.get((i, act), 0)}")
                # print("")

                self.total_reward = 0
                self.statistics = {}

                # self.set_players()

            # fresh_observation, _ = self.reset()
            return extracted_features, reward / STACK, True, False, {}

        return extracted_features, 0, False, False, {}

    @override
    def render(self, mode: str = "human") -> None:
        """Render the environment.

        Parameters:
            mode: str
                The rendering mode.
        """
        raise NotImplementedError("'render' is not implemented for PokerEnv.")

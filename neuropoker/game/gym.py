"""Classes for modeling a poker game in a Gym environment."""

import random
import time
from pathlib import Path
from typing import Any, Callable, Dict, Final, List, Tuple, override

import gymnasium
import numpy as np
import torch
from gymnasium import spaces
from pypokerengine.engine.data_encoder import DataEncoder
from pypokerengine.engine.poker_constants import PokerConstants as Const
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from neuropoker.game.cards import SHORT_RANKS, SHORTER_SUITS, get_card_list
from neuropoker.game.features import FeaturesCollector, LinearFeaturesCollector
from neuropoker.game.game import Game, GameState
from neuropoker.game.utils import STACK
from neuropoker.players.base import BasePlayer
from neuropoker.players.ppo import PPOPlayer
from neuropoker.players.utils import load_ppo_player

DEFAULT_RESET_THRESHOLD: Final[int] = 30000


class PokerEnv(gymnasium.Env):
    """Class for modeling a poker game in a Gym environment."""

    def __init__(
        self,
        game: Game,
        feature_collector: FeaturesCollector | None = None,
        reset_threshold: int = DEFAULT_RESET_THRESHOLD,
    ) -> None:
        """Initialize the poker gym environment.

        Parameters:
            game: Game
                The underlying poker game.
            feature_collector: FeaturesCollector | None
                The feature extractor to use.
            reset_threshold: int
                The number of games after which to reset the environment.
        """
        # Game
        self.game: Game = game
        self.reset_threshold: Final[int] = reset_threshold

        # Feature extractor
        self.feature_collector: Final[FeaturesCollector] = (
            feature_collector
            if feature_collector is not None
            else LinearFeaturesCollector()
        )

        # Gym options
        self.action_space: spaces.Space = spaces.Discrete(5)
        self.observation_space: spaces.Space = self.feature_collector.space()

        # Stats
        self.game_state: GameState = {}
        self.num_games: int = 0
        self.bad_games: int = 0
        self.total_reward: float = 0
        self.cumulative_reward: float = 0
        self.statistics: Dict[Tuple[str, str], Any] = {}

        # Seed
        random.seed(time.time())
        self.seed: Final[int] = random.randint(0, 10000)
        print("SEED:", self.seed)

        # Reset
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

        extracted_features: np.ndarray = self.feature_collector(
            hole_card, round_state, "me"
        )
        # extracted_features = extracted_features[np.newaxis, :]
        # print(extracted_features.shape)
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
        extracted_features = self.feature_collector(hole_card, round_state, "me")
        # extracted_features = extracted_features[np.newaxis, :]
        # print(extracted_features.shape)

        if self.game_state["street"] == Const.Street.FINISHED:
            stack: float = self.game_state["table"].seats.players[self.player_pos].stack
            reward: float = stack - STACK
            self.total_reward += reward
            self.cumulative_reward += reward

            if self.num_games % self.reset_threshold == 0:
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


class PokerCNNExtractor(BaseFeaturesExtractor):
    """CNN-based feature extractor for poker environments."""

    def __init__(
        self, observation_space: gymnasium.spaces.Space, features_dim: int = 256
    ) -> None:
        """Initialize the PokerCNNExtractor.

        Parameters:
            observation_space: gym.spaces.Space
                The observation space.
            features_dim: int
                The size of the generated feature vector.
        """
        super().__init__(observation_space, features_dim)

        if observation_space.shape is None:
            raise ValueError("The observation space shape should not be None")

        # Observation space dimensions (channels, height, width)
        n_input_channels = observation_space.shape[0]

        # Define the CNN layers
        self.cnn = torch.nn.Sequential(
            # Conv Layer 1
            torch.nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            # Conv Layer 2
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            # Flatten to 1D vector
            torch.nn.Flatten(),
        )

        # Compute the size of the output after CNN layers
        with torch.no_grad():
            sample_input = torch.zeros((1,) + observation_space.shape)
            sample_output = self.cnn(sample_input)
            n_flatten = sample_output.shape[1]

        # Fully connected layer to produce feature vector
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(n_flatten, features_dim),
            torch.nn.ReLU(),
        )

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Make a forward pass through the network.

        Parameters:
            x: torch.Tensor
                The input tensor.

        Returns:
            y: torch.Tensor
                The output tensor.
        """
        # print(f"x: {x.shape}")

        # Input x -> Hidden representation h
        h: torch.Tensor = self.cnn(x)
        # print(f"h: {h.shape}")

        # Hidden representation h -> Output y
        y: torch.Tensor = self.fc(h)
        # print(f"y: {y.shape}")
        return y


def make_env(
    starting_model_path: str | Path | None = None,
    opponent_model_path: str | Path | None = None,
    feature_collector: FeaturesCollector | None = None,
    reset_threshold: int = DEFAULT_RESET_THRESHOLD,
    suits: List[str] | None = None,
    ranks: List[str] | None = None,
) -> Callable[[], PokerEnv]:
    """Create a poker environment.

    Parameters:
        starting_model_path_file: str | Path | None
            The path to the starting model file to use.
        opponent_model_path: str | Path | None
            The path to the opponent model file to use.
        feature_collector: FeaturesCollector | None
            The feature extractor to use.
        reset_threshold: int
            The number of games after which to reset the environment.
        suits: List[str] | None
            The suits to use in the game.
        ranks: List[str] | None
            The ranks to use in the game.

    Returns:
        env: () -> PokerEnv
            A function that creates a poker environment.
    """
    suits_: Final[List[str]] = suits or SHORTER_SUITS
    ranks_: Final[List[str]] = ranks or SHORT_RANKS

    return lambda: PokerEnv(
        Game(
            [
                load_ppo_player(starting_model_path, "me"),
                load_ppo_player(opponent_model_path, "opponent1"),
                load_ppo_player(opponent_model_path, "opponent2"),
            ],
            get_card_list(suits_, ranks_),
        ),
        feature_collector=feature_collector,
        reset_threshold=reset_threshold,
    )

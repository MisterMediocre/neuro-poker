"""Classes for modeling a poker game in a Gym environment."""

from pathlib import Path
from typing import Any, Dict, Final, List, Tuple, override

import gymnasium
import numpy as np
from gymnasium import spaces
from pypokerengine.engine.data_encoder import DataEncoder
from pypokerengine.engine.poker_constants import PokerConstants as Const

from neuropoker.game.game import BIG_BLIND_AMOUNT, Game, GameState
from neuropoker.game.utils import STACK, extract_features
from neuropoker.players.ppo import PPOPlayer


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
        self.observation_space = spaces.Box(
            low=0, high=3, shape=(1, num_features), dtype=np.float32
        )
        self.action_space = spaces.Discrete(5)

        # Game
        self.game: Final[Game] = game

        # Stats
        self.game_state: Dict[str, Any] = {}
        self.num_games: int = 0
        self.bad_games: int = 0
        self.total_reward: float = 0
        self.cumulative_reward: float = 0
        self.statistics: Dict[Tuple[str, str], Any] = {}

        self.reset()

    def keep_playing(self, game_state: GameState, break_me: bool) -> GameState:
        """Play a single round/game of Poker.

        Parameters:
            game_state: GameState
                The current game state.
            break_me: bool
                Whether our player should break after the player's turn.
        """
        # game_state: Dict[str, Any] = self.game_state

        while game_state["street"] != Const.Street.FINISHED:
            next_player: int = game_state["next_player"]
            if next_player == self.player_pos and break_me:
                break

            table = game_state["table"]
            street = game_state["street"]
            round_state = DataEncoder.encode_round_state(game_state)
            player = table.seats.players[next_player]
            valid_actions = self.game.emulator.generate_possible_actions(game_state)
            hole_card = DataEncoder.encode_player(
                self.game_state["table"].seats.players[next_player], holecard=True
            )["hole_card"]

            if len(hole_card) != 2:
                raise ValueError(f"Invalid hole card: {hole_card}")

            action, bet = self.game.players[next_player].declare_action(
                valid_actions, hole_card, round_state
            )
            game_state, _event = self.game.emulator.apply_action(
                game_state, action, bet
            )

        # self.game_state = game_state
        return game_state

    @override
    def reset(
        self, *, seed: int | None = None, options: Dict[str, Any] | None = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment.

        Parameters:
            seed: int | None
                The seed for the environment.
            options: Dict[str, Any] | None
                Additional options.

        Returns:
            observation: np.ndarray
                The observation after the reset.
            info: Dict[str, Any]
                Additional information.
        """
        self.num_games += 1
        dealer: int = 1

        self.player_pos: int = self.num_games % 3

        self.game_state, _events = self.game.start_round(
            dealer_button=dealer, seed=seed
        )

        self.game_state = self.keep_playing(self.game_state, break_me=True)

        round_state: Final[Dict[str, Any]] = DataEncoder.encode_round_state(
            self.game_state
        )

        if round_state["street"] is None:
            # Both opponents folded already...
            # print("Both opponents folded already")
            self.bad_games += 1
            return self.reset()

        encoded_player: Final[Dict[str, Any]] = DataEncoder.encode_player(
            self.game_state["table"].seats.players[self.player_pos], holecard=True
        )
        hole_card: Final[List[str]] = encoded_player["hole_card"]

        # print("hole_card:", hole_card)
        # print("self.player_pos:", self.player_pos)

        # print(self.cards)
        extracted_features: np.ndarray = extract_features(hole_card, round_state, "me")
        # print(hole_card)
        # print(round_state)
        # print(extracted_features)
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
        game_state: Final[Dict[str, Any]] = self.game_state
        hole_card: Final[List[str]] = DataEncoder.encode_player(
            self.game_state["table"].seats.players[self.player_pos], holecard=True
        )["hole_card"]

        if len(hole_card) != 2:
            raise ValueError(f"Invalid hole card: {hole_card}")

        # Must be my turn
        valid_actions: List[Dict[str, Any]] = (
            self.game.emulator.generate_possible_actions(game_state)
        )
        # print(valid_actions)
        my_action = None
        bet_amount: float = 0
        street: Final[str] = game_state["street"]

        my_action, bet_amount = PPOPlayer.int_to_action(action, valid_actions)
        self.statistics[(street, my_action)] = (
            self.statistics.get((street, my_action), 0) + 1
        )

        # print(my_action, bet_amount)
        self.game_state, _event = self.game.emulator.apply_action(
            game_state, my_action, bet_amount
        )

        # if street == Const.Street.FLOP or street == Const.Street.PREFLOP or street == Const.Street.TURN or street == Const.Street.RIVER:
        # self.keep_playing(break_me=True) # We want to only play first turn properly, rest is all calls
        # else:
        self.game_state = self.keep_playing(self.game_state, break_me=True)

        round_state = DataEncoder.encode_round_state(self.game_state)
        extracted_features = extract_features(hole_card, round_state, "me")[
            np.newaxis, :
        ]

        if self.game_state["street"] == Const.Street.FINISHED:
            stack = self.game_state["table"].seats.players[self.player_pos].stack
            reward = stack - STACK
            # print("STACK:", stack)
            self.total_reward += reward
            # print(stack - STACK)
            self.cumulative_reward += reward

            # print(self.num_games)
            RESET_THRESHOLD = 3000
            if self.num_games % RESET_THRESHOLD == 0:
                # print("opponent", self.opponent_path)
                # print("opponent2:", self.opponent2_path)
                print("Total games:", self.num_games)
                print("Bad games:", self.bad_games)
                print("Cumulative reward:", self.cumulative_reward)
                print("Total reward:", self.total_reward)
                print("Average reward:", self.total_reward / RESET_THRESHOLD)
                print(
                    "Average rewards since beginning:",
                    self.cumulative_reward / self.num_games,
                )
                print(
                    "bb per 100g:",
                    self.total_reward * 100 / (RESET_THRESHOLD * BIG_BLIND_AMOUNT),
                )
                # print("Statistics:", self.statistics)
                for street_ in ["preflop", "flop", "turn", "river"]:
                    for act in ["fold", "call", "raise"]:
                        print(
                            f"{street_} {act}: {self.statistics.get((street_, act), 0)}"
                        )
                print("")

                stats_file: Final[Path] = Path("sb6_stats.txt")
                with stats_file.open("at", encoding="utf-8") as f:
                    f.write(f"Total games: {self.num_games}\n")
                    f.write(f"Cumulative reward: {self.cumulative_reward}\n")
                    f.write(f"Total reward: {self.total_reward}\n")
                    f.write(f"Average reward: {self.total_reward / RESET_THRESHOLD}\n")
                    f.write(
                        f"Average rewards since beginning: {self.cumulative_reward / self.num_games}\n"
                    )
                    f.write(
                        f"bb per 100g: {self.total_reward * 100 / (RESET_THRESHOLD * BIG_BLIND_AMOUNT)}\n"
                    )
                    f.write("\n")

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

import random
import time
from typing import List

import gym
import numpy as np
from gym import spaces
from pypokerengine.api.emulator import Emulator
from pypokerengine.engine.data_encoder import DataEncoder
from pypokerengine.engine.poker_constants import PokerConstants as Const
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.ppo.ppo import PPO

from neuropoker.game.cards import SHORT_RANKS, SHORTER_SUITS, get_card_list
from neuropoker.game.game import BIG_BLIND_AMOUNT, SMALL_BLIND_AMOUNT, get_deck
from neuropoker.game.utils import NUM_PLAYERS, STACK, extract_features
from neuropoker.players.base import BasePlayer
from neuropoker.players.naive import CallPlayer


def int_to_action(i, valid_actions):
    if i == 0:
        return "fold", 0
    elif i == 1:
        return "call", valid_actions[1]["amount"]
    elif i == 2:
        return "raise", valid_actions[2]["amount"]["min"]
    elif i == 3 or i == 4:
        return "raise", valid_actions[2]["amount"]["min"] * 2
    else:
        raise ValueError("Invalid action")


def load_model_player(model_path, uuid):
    if model_path == "call":
        return CallPlayer()

    model = PPO.load(model_path)
    return ModelPlayer(model, uuid)


class ModelPlayer(BasePlayer):
    def __init__(self, model, uuid):
        self.model = model
        self.uuid = uuid

    def declare_action(self, valid_actions, hole_card, round_state):
        features = extract_features(hole_card, round_state, self.uuid)
        action = self.model.predict(features[np.newaxis, :])[0]
        # print(self.uuid, action, hole_card)
        return int_to_action(action, valid_actions)


class PokerEnv(gym.Env):
    def __init__(self, old_paths: List[str]):
        num_features = 73
        self.observation_space = spaces.Box(
            low=0, high=3, shape=(1, num_features), dtype=np.float32
        )
        self.action_space = spaces.Discrete(5)

        self.old_paths = old_paths

        self.num_games = 0
        self.bad_games = 0
        self.total_reward = 0
        self.cumulative_reward = 0
        self.statistics = {}

        self.emulator = Emulator()
        self.emulator.set_game_rule(
            NUM_PLAYERS,
            STACK,
            SMALL_BLIND_AMOUNT,
            0,  # NO ANTE
        )

        random.seed(time.time())
        self.seed = random.randint(0, 10000)
        print("SEED:", self.seed)

        self.set_players()

        self.cards = get_card_list(SHORTER_SUITS, SHORT_RANKS)

        self.reset()

    def set_players(self):
        self.opponent_path = random.choice(self.old_paths)
        # self.opponent1_path = random.choice(self.old_paths)
        # self.opponent2_path = random.choice(self.old_paths)
        # opponent1 = load_model_player(self.opponent1_path, "opponent1")
        # opponent2 = load_model_player(self.opponent2_path, "opponent2")
        self.opponent1 = load_model_player(self.opponent_path, "opponent1")
        self.opponent2 = load_model_player(self.opponent_path, "opponent2")

    def keep_playing(self, break_me):
        game_state = self.game_state
        while game_state["street"] != Const.Street.FINISHED:
            next_player = game_state["next_player"]
            if next_player == self.player_pos and break_me:
                break

            table = game_state["table"]
            street = game_state["street"]
            round_state = DataEncoder.encode_round_state(game_state)
            player = table.seats.players[next_player]
            valid_actions = self.emulator.generate_possible_actions(game_state)
            hole_card = DataEncoder.encode_player(
                self.game_state["table"].seats.players[next_player], holecard=True
            )["hole_card"]
            assert len(hole_card) == 2

            action, bet = self.players[next_player].declare_action(
                valid_actions, hole_card, round_state
            )
            game_state, _event = self.emulator.apply_action(game_state, action, bet)

        self.game_state = game_state

    def reset(self, **kwargs):
        self.num_games += 1

        dealer = 1
        # seed = (self.num_games//3)%10000
        seed = (self.num_games // 3) + self.seed

        self.player_pos = self.num_games % 3
        self.player_names = ["", "", ""]
        self.player_names[self.player_pos] = "me"
        self.player_names[(self.player_pos + 1) % 3] = "opponent1"
        self.player_names[(self.player_pos + 2) % 3] = "opponent2"

        self.players = [BasePlayer(), BasePlayer(), BasePlayer()]
        self.players[(self.player_pos + 1) % 3] = self.opponent1
        self.players[(self.player_pos + 2) % 3] = self.opponent2

        self.players_info = {
            name: {"stack": STACK, "name": name, "uuid": name}
            for name in self.player_names
        }
        self.initial_state = self.emulator.generate_initial_game_state(
            self.players_info
        )

        self.initial_state["table"].deck = get_deck(self.cards, seed=seed)
        self.initial_state["table"].dealer_btn = dealer

        self.game_state, _event = self.emulator.start_new_round(self.initial_state)
        self.keep_playing(break_me=True)

        round_state = DataEncoder.encode_round_state(self.game_state)
        if round_state["street"] is None:
            # Both opponents folded already...
            # print("Both opponents folded already")
            self.bad_games += 1
            return self.reset()

        encoded_player = DataEncoder.encode_player(
            self.game_state["table"].seats.players[self.player_pos], holecard=True
        )
        hole_card = encoded_player["hole_card"]

        # print("hole_card:", hole_card)
        # print("self.player_pos:", self.player_pos)

        # print(self.cards)
        extracted_features = extract_features(hole_card, round_state, "me")
        # print(hole_card)
        # print(round_state)
        # print(extracted_features)
        extracted_features = extracted_features[np.newaxis, :]

        return extracted_features, {}

    def step(self, action):
        game_state = self.game_state
        hole_card = DataEncoder.encode_player(
            self.game_state["table"].seats.players[self.player_pos], holecard=True
        )["hole_card"]
        assert len(hole_card) == 2

        # Must be my turn
        valid_actions = self.emulator.generate_possible_actions(game_state)
        # print(valid_actions)
        my_action = None
        bet_amount = 0
        street = game_state["street"]

        my_action, bet_amount = int_to_action(action, valid_actions)
        self.statistics[(street, my_action)] = (
            self.statistics.get((street, my_action), 0) + 1
        )

        # print(my_action, bet_amount)
        self.game_state, _event = self.emulator.apply_action(
            game_state, my_action, bet_amount
        )

        # if street == Const.Street.FLOP or street == Const.Street.PREFLOP or street == Const.Street.TURN or street == Const.Street.RIVER:
        # self.keep_playing(break_me=True) # We want to only play first turn properly, rest is all calls
        # else:
        self.keep_playing(break_me=True)

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
                print("opponent", self.opponent_path)
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
                for i, street in enumerate(["preflop", "flop", "turn", "river"]):
                    for act in ["fold", "call", "raise"]:
                        print(f"{street} {act}: {self.statistics.get((i, act), 0)}")
                print("")

                STATS_FILE = "sb6_stats.txt"
                with open(STATS_FILE, "a") as f:
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
                self.set_players()

            # fresh_observation, _ = self.reset()
            return extracted_features, reward / STACK, True, False, {}

        return extracted_features, 0, False, False, {}


FIRST_MODEL_PATH = "models/3p_3s/sb2"
SECOND_MODEL_PATH = "models/3p_3s/sb3"
THIRD_MODEL_PATH = "models/3p_3s/sb4"
FOURTH_MODEL_PATH = "models/3p_3s/sb5"
CURRENT_MODEL_PATH = "models/3p_3s/sb6"

NUM_ENVIRONMENTS = 16
NET_ARCH = [128, 128]


def make_env():
    return lambda: PokerEnv([CURRENT_MODEL_PATH])


if __name__ == "__main__":
    # print("MPS available:", torch.backends.mps.is_available())
    # device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    # print(f"Training on: {device}")

    env = SubprocVecEnv([make_env() for _ in range(NUM_ENVIRONMENTS)])
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        ent_coef=0.01,
        vf_coef=0.7,
        n_steps=256,
        learning_rate=0.0001,
        policy_kwargs=dict(net_arch=NET_ARCH),
    )

    old_model = PPO.load(CURRENT_MODEL_PATH, env=env)
    model.policy.load_state_dict(old_model.policy.state_dict(), strict=True)

    while True:
        model.learn(total_timesteps=100000, reset_num_timesteps=False)
        print("SAVING MODEL")
        model.save(CURRENT_MODEL_PATH)

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import random
import time

from pypokerengine.engine.card import Card
from pypokerengine.engine.player import Player
from pypokerengine.engine.table import Table


from neuropoker.cards import SHORT_SUITS, get_card_list, SHORT_RANKS, SHORTER_SUITS
from neuropoker.game_utils import NUM_PLAYERS, STACK, STREET_MAPPING, extract_features, extract_features_tensor
from neuropoker.game import BIG_BLIND_AMOUNT, SMALL_BLIND_AMOUNT
from neuropoker.game import get_deck
from neuropoker.players.naive import CallPlayer, FoldPlayer
from neuropoker.players.base import BasePlayer

from pypokerengine.api.emulator import Emulator
from pypokerengine.engine.data_encoder import DataEncoder
from pypokerengine.engine.poker_constants import PokerConstants as Const

from stable_baselines3.ppo.ppo import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv


import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class PokerCNNExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super(PokerCNNExtractor, self).__init__(observation_space, features_dim)

        # Observation space dimensions (channels, height, width)
        n_input_channels = observation_space.shape[0]

        # Define the CNN layers
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),  # Conv Layer 1
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # Conv Layer 2
            nn.ReLU(),
            nn.Flatten(),  # Flatten to 1D vector
        )

        # Compute the size of the output after CNN layers
        with torch.no_grad():
            sample_input = torch.zeros((1,) + observation_space.shape)
            n_flatten = self.cnn(sample_input).shape[1]

        # Fully connected layer to produce feature vector
        self.fc = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations):
        x = self.cnn(observations)
        return self.fc(x)




def int_to_action(i, valid_actions):
    if i == 0:
        return "fold", 0
    elif i == 1:
        return "call", valid_actions[1]["amount"]
    elif i == 2:
        return "raise", valid_actions[2]["amount"]["min"]
    elif i == 3 or i == 4 :
        return "raise", valid_actions[2]["amount"]["min"] * 2
    else:
        raise ValueError("Invalid action")


def load_model_player(model_path, uuid):
    model = PPO.load(model_path)
    return ModelPlayer(model, uuid)

class ModelPlayer(BasePlayer):
    def __init__(self, model, uuid):
        self.model = model
        self.uuid = uuid

    def declare_action(self, valid_actions, hole_card, round_state):
        features = extract_features(hole_card, round_state, self.uuid)
        action = self.model.predict(features[np.newaxis, :])[0]
        return int_to_action(action, valid_actions)


class PokerEnv(gym.Env):
    def __init__(self, opponent1:BasePlayer , opponent2: BasePlayer):
        num_cards = len(SHORTER_SUITS) * len(SHORT_RANKS)
        self.observation_space = spaces.Box(low=0, high=3, shape=(4, 8, num_cards), dtype=np.float32)
        self.action_space = spaces.Discrete(5)


        self.num_games = 0
        self.total_reward = 0
        self.cumulative_reward = 0
        self.statistics = {}

        self.emulator = Emulator()
        self.emulator.set_game_rule(
            NUM_PLAYERS, 
            STACK,
            SMALL_BLIND_AMOUNT,
            0, # NO ANTE
        )

        self.players = [BasePlayer(), opponent1, opponent2]
        self.player_names = ["me", "opponent1", "opponent"]
        self.players_info = {name: {"stack": STACK, "name": name, "uuid": name } for name in self.player_names}

        self.cards = get_card_list(SHORTER_SUITS, SHORT_RANKS)

        self.initial_state = self.emulator.generate_initial_game_state(self.players_info)

        random.seed(time.time())
        self.seed = random.randint(0, 10000)
        print("SEED:", self.seed)
        
        self.reset()

    def keep_playing(self, break_me):
        game_state = self.game_state
        while game_state["street"] != Const.Street.FINISHED:
            next_player = game_state["next_player"]
            if next_player == 0 and break_me:
                break

            table = game_state["table"]
            street = game_state["street"]
            round_state = DataEncoder.encode_round_state(game_state)
            player = table.seats.players[next_player]
            valid_actions = self.emulator.generate_possible_actions(game_state)
            hole_card = DataEncoder.encode_player(self.game_state["table"].seats.players[next_player], holecard=True)["hole_card"]

            action, bet = self.players[next_player].declare_action(valid_actions, hole_card, round_state)
            game_state, _event = self.emulator.apply_action(game_state, action, bet)

        self.game_state = game_state

    def reset(self, **kwargs):
        self.num_games+=1

        dealer = self.num_games % 3
        # seed = (self.num_games//3)%10000
        seed = (self.num_games // 3) + self.seed

        self.initial_state["table"].deck = get_deck(self.cards, seed=seed)
        self.initial_state["table"].dealer_btn = dealer

        self.game_state, _event = self.emulator.start_new_round(self.initial_state)
        self.keep_playing(break_me=True)

        round_state = DataEncoder.encode_round_state(self.game_state)
        hole_card = DataEncoder.encode_player(self.game_state["table"].seats.players[0], holecard=True)["hole_card"]
        assert len(hole_card) == 2


        # print(self.cards)
        # extracted_features = extract_features(hole_card, round_state, "me")
        extracted_features = extract_features_tensor(hole_card, round_state, "me")

        # print(hole_card)
        # print(round_state)
        # print(extracted_features)
        # extracted_features = extracted_features[np.newaxis, :]

        return extracted_features, {}


    def step(self, action):
        game_state = self.game_state
        hole_card = DataEncoder.encode_player(self.game_state["table"].seats.players[0], holecard=True)["hole_card"]
        assert len(hole_card) == 2

        # Must be my turn
        valid_actions = self.emulator.generate_possible_actions(game_state)
        # print(valid_actions)
        my_action = None
        bet_amount = 0
        street = game_state["street"]


        my_action, bet_amount = int_to_action(action, valid_actions)
        self.statistics[(street, my_action)] = self.statistics.get((street, my_action), 0) + 1

        # print(my_action, bet_amount)
        self.game_state, _event = self.emulator.apply_action(game_state, my_action, bet_amount)

        # if street == Const.Street.FLOP or street == Const.Street.PREFLOP or street == Const.Street.TURN or street == Const.Street.RIVER:
            # self.keep_playing(break_me=True) # We want to only play first turn properly, rest is all calls
        # else:
        self.keep_playing(break_me=True)

        round_state = DataEncoder.encode_round_state(self.game_state)
        # extracted_features = extract_features(hole_card, round_state, "me")[np.newaxis, :]
        # extracted_features = extract_features_tensor(hole_card, round_state, "me")[np.newaxis, :]
        extracted_features = extract_features_tensor(hole_card, round_state, "me")



        if self.game_state["street"] == Const.Street.FINISHED:
            stack = self.game_state["table"].seats.players[0].stack
            reward = (stack - STACK) 
            # print("STACK:", stack)
            self.total_reward += reward
            # print(stack - STACK)
            self.cumulative_reward += reward

            # print(self.num_games)
            RESET_THRESHOLD = 3000
            if self.num_games % RESET_THRESHOLD == 0:
                print("Total games:", self.num_games)
                print("Cumulative reward:", self.cumulative_reward)
                print("Total reward:", self.total_reward)
                print("Average reward:", self.total_reward / RESET_THRESHOLD)
                print("bb per 100g:", self.total_reward * 100 / (RESET_THRESHOLD * BIG_BLIND_AMOUNT))
                # print("Statistics:", self.statistics)
                for i, street in enumerate(["preflop", "flop", "turn", "river"]):
                    for act in ["fold", "call", "raise"]:
                        print(f"{street} {act}: {self.statistics.get((i, act), 0)}")
                print("")
                self.total_reward = 0
                self.statistics = {}

            # fresh_observation, _ = self.reset()
            return extracted_features, reward/STACK, True, False, {}


        return extracted_features, 0, False, False, {}



MODEL_PATH = "models/3p_3s/sb_cnn"
NUM_ENVIRONMENTS = 8

policy_kwargs = dict(
    features_extractor_class=PokerCNNExtractor,
    features_extractor_kwargs=dict(features_dim=256),
    net_arch=[]  # Skip additional layers since the CNN handles feature extraction
)

def make_env():
    return lambda: PokerEnv(CallPlayer(), CallPlayer())

if __name__ == "__main__":

    print("MPS available:", torch.backends.mps.is_available())
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Training on: {device}")

    env = SubprocVecEnv([make_env() for _ in range(NUM_ENVIRONMENTS)])
    model = PPO(
        'CnnPolicy',  # Use CNN-compatible policy
        env, 
        verbose=1,
        ent_coef=0.01,
        vf_coef=0.7,
        n_steps=256,
        # learning_rate=0.003,
        policy_kwargs=policy_kwargs,
        device=device
    )
    
    old_model = PPO.load(MODEL_PATH, env=env)
    model.policy.load_state_dict(old_model.policy.state_dict(), strict=True)

    while (True):
        model.learn(total_timesteps=100000, reset_num_timesteps=False)
        print("SAVING MODEL")
        model.save(MODEL_PATH)





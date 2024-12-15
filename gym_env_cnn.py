#!/usr/bin/env python3

"""Train a PPO agent, using a CNN, to play poker."""

from pathlib import Path
from typing import Any, Callable, Dict, Final, List, override

import torch
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.ppo.ppo import PPO
from termcolor import colored

from neuropoker.game.cards import SHORT_RANKS, SHORTER_SUITS, get_card_list
from neuropoker.game.game import Game, default_player_stats, merge
from neuropoker.game.gym import PokerEnv
from neuropoker.players.base import BasePlayer
from neuropoker.players.naive import CallPlayer
from neuropoker.players.ppo import PPOPlayer

DEFAULT_MODEL_FILE: Final[Path] = Path("models/3p_3s/sb_cnn")
DEFAULT_CONFIG_FILE: Final[Path] = Path("configs/3p_3s_neat.toml")
DEFAULT_NUM_CORES: Final[int] = 8
DEFAULT_NUM_TIMESTEPS: Final[int] = 100000


def load_model_player(model_path: str | Path, uuid: str) -> BasePlayer:
    """Load a ModelPlayer from a file.

    Parameters:
        model_path: str | Path
            The path to the model file.
        uuid: str
            The UUID of the player.

    Returns:
        player: BasePlayer
            The loaded player.
    """
    if not Path(model_path).exists() or model_path == "call":
        return CallPlayer(uuid)

    model = PPO.load(model_path)
    return PPOPlayer(model, uuid)


class PokerCNNExtractor(BaseFeaturesExtractor):
    def __init__(
        self, observation_space: spaces.Space, features_dim: int = 256
    ) -> None:
        """Initialize the PokerCNNExtractor.

        Parameters:
            observation_space: gym.spaces.Space
                The observation space.
            features_dim: int
                The size of the feature vector.
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
            n_flatten = self.cnn(sample_input).shape[1]

        # Fully connected layer to produce feature vector
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(n_flatten, features_dim),
            torch.nn.ReLU(),
        )

    @override
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Make a forward pass through the network.

        Parameters:
            observations: torch.Tensor
                The input tensor.

        Returns:
            output: torch.Tensor
                The output tensor.
        """
        x: torch.Tensor = self.cnn(observations)
        return self.fc(x)


POLICY_KWARGS: Final[Dict[str, Any]] = {
    "features_extractor_class": PokerCNNExtractor,
    "features_extractor_kwargs": {"features_dim": 256},
    "net_arch": [],  # Skip additional layers since the CNN handles feature extraction
}


def make_env() -> Callable[[], PokerEnv]:
    """Create a poker environment.

    Returns:
        env: () -> PokerEnv
            A function that creates a poker environment.
    """
    return lambda: PokerEnv(
        Game(
            [
                load_model_player(DEFAULT_MODEL_FILE, "me"),
                load_model_player(DEFAULT_MODEL_FILE, "opponent1"),
                load_model_player(DEFAULT_MODEL_FILE, "opponent2"),
            ],
            get_card_list(SHORTER_SUITS, SHORT_RANKS),
        )
    )


def get_device() -> str:
    """Get the device to train on.

    Returns:
        device: str
            The device to train on.
    """
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def main() -> None:
    """Run the script."""

    device: Final[str] = get_device()

    env = SubprocVecEnv([make_env() for _ in range(DEFAULT_NUM_CORES)])
    model = PPO(
        "CnnPolicy",  # Use CNN-compatible policy
        env,
        verbose=1,
        ent_coef=0.01,
        vf_coef=0.7,
        n_steps=256,
        # learning_rate=0.003,
        policy_kwargs=POLICY_KWARGS,
        device=device,
    )

    print(
        colored("------------ gym_env_cnn -------------", color="blue", attrs=["bold"])
    )
    print(colored(f'{"Model file":<12}: ', color="blue") + f"{DEFAULT_MODEL_FILE}")
    print(colored(f'{"Device":<12}: ', color="blue") + f"{device}")

    #
    # Load the model
    #
    model_zip_file: Final[Path] = DEFAULT_MODEL_FILE.with_suffix(".zip")
    if model_zip_file.exists():
        print(colored(f"Loading old model from {model_zip_file}...", "green"))

        # Load the old model and copy the policy weights to the new model
        old_model = PPO.load(DEFAULT_MODEL_FILE, env=env)
        model.policy.load_state_dict(old_model.policy.state_dict(), strict=True)

        # Statically load opponents as the original model
        p2: BasePlayer = load_model_player(DEFAULT_MODEL_FILE, "opponent1")
        p3: BasePlayer = load_model_player(DEFAULT_MODEL_FILE, "opponent2")
    else:
        print(colored("Training new model from scratch...", "green"))

        # Statically load opponents as the original model
        p2: BasePlayer = CallPlayer("opponent1")
        p3: BasePlayer = CallPlayer("opponent2")

    #
    # Train the model
    #
    while True:
        model.learn(total_timesteps=100000, reset_num_timesteps=False)
        print("SAVING MODEL")
        model.save(DEFAULT_MODEL_FILE)

        print("Evaluating model")

        p1 = load_model_player(DEFAULT_MODEL_FILE, "me")
        players: List[BasePlayer] = [p1, p2, p3]

        overall_performance = default_player_stats()
        overall_performance["uuid"] = "me"

        for i in range(0, 3):
            players_: List[BasePlayer] = players[i:] + players[:i]

            game = Game(players_, get_card_list(SHORTER_SUITS, SHORT_RANKS))
            performances = game.play_multiple(num_games=2000, seed=-1)
            overall_performance = merge(overall_performance, performances["me"])

        print("Overall performance:")
        print(
            "Average winning:",
            overall_performance["winnings"] / overall_performance["num_games"],
        )
        print(overall_performance)


if __name__ == "__main__":
    main()

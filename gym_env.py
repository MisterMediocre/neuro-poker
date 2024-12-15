#!/usr/bin/env python3

"""Train a PPO agent to play poker."""

from pathlib import Path
from typing import Callable, Final, List

import torch
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.ppo.ppo import PPO
from termcolor import colored

from neuropoker.game.cards import SHORT_RANKS, SHORTER_SUITS, get_card_list
from neuropoker.game.game import (
    Game,
    default_player_stats,
    merge,
)
from neuropoker.game.gym import PokerEnv
from neuropoker.players.base import BasePlayer
from neuropoker.players.naive import CallPlayer
from neuropoker.players.ppo import PPOPlayer

DEFAULT_MODEL_FILE: Final[Path] = Path("models/3p_3s/sb")
DEFAULT_CONFIG_FILE: Final[Path] = Path("configs/3p_3s_neat.toml")
DEFAULT_NUM_CORES: Final[int] = 16
DEFAULT_NUM_TIMESTEPS: Final[int] = 100000
DEFAULT_NET_ARCH: Final[list[int]] = [128, 128]


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


FIRST_MODEL_PATH: Final[Path] = Path("models/3p_3s/sb2")
SECOND_MODEL_PATH: Final[Path] = Path("models/3p_3s/sb3")
THIRD_MODEL_PATH: Final[Path] = Path("models/3p_3s/sb4")
FOURTH_MODEL_PATH: Final[Path] = Path("models/3p_3s/sb5")
CURRENT_MODEL_PATH: Final[Path] = Path("models/3p_3s/sb6")
CURRENT_MODEL_PATH_BACKUP: Final[Path] = Path("models/3p_3s/sb6_backup")
STATS_FILE: Final[Path] = Path("sb6_stats.txt")

NUM_ENVIRONMENTS: Final[int] = 16
NET_ARCH: Final[List[int]] = [128, 128]


def make_env() -> Callable[[], PokerEnv]:
    """Create a poker environment.

    Returns:
        env: () -> PokerEnv
            A function that creates a poker environment.
    """
    return lambda: PokerEnv(
        Game(
            [
                load_model_player(CURRENT_MODEL_PATH, "me"),
                load_model_player(CURRENT_MODEL_PATH, "opponent1"),
                load_model_player(CURRENT_MODEL_PATH, "opponent2"),
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
    # print("MPS available:", torch.backends.mps.is_available())
    # device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    device: Final[str] = get_device()

    env = SubprocVecEnv([make_env() for _ in range(NUM_ENVIRONMENTS)])
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        ent_coef=0.01,
        vf_coef=0.7,
        n_steps=512,
        learning_rate=0.0001,
        policy_kwargs={"net_arch": NET_ARCH},
        device=torch.device(device),
    )

    print(
        colored("-------------- gym_env ---------------", color="blue", attrs=["bold"])
    )
    print(colored(f'{"Model file":<12}: ', color="blue") + f"{CURRENT_MODEL_PATH}")
    print(colored(f'{"Device":<12}: ', color="blue") + f"{device}")

    if CURRENT_MODEL_PATH.exists():
        print(colored(f"Loading old model from {CURRENT_MODEL_PATH}...", "green"))
        old_model = PPO.load(CURRENT_MODEL_PATH, env=env)
        model.policy.load_state_dict(old_model.policy.state_dict(), strict=True)
    else:
        print(colored("Training new model from scratch...", "green"))

    # Statically load opponents as the original model
    p2: Final[BasePlayer] = load_model_player(CURRENT_MODEL_PATH, "opponent1")
    p3: Final[BasePlayer] = load_model_player(CURRENT_MODEL_PATH, "opponent2")

    while True:
        model.learn(total_timesteps=100000, reset_num_timesteps=False)
        print("SAVING MODEL")
        model.save(CURRENT_MODEL_PATH)

        print("Evaluating model")

        p1 = load_model_player(CURRENT_MODEL_PATH, "me")
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

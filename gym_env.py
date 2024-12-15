#!/usr/bin/env python3

"""Train a PPO agent to play poker."""

import argparse
from pathlib import Path
from typing import Dict, Final, List

import gymnasium
from stable_baselines3.common.vec_env import SubprocVecEnv, VecEnv
from stable_baselines3.ppo.ppo import PPO
from termcolor import colored

from neuropoker.extra.torch import get_device
from neuropoker.game.cards import SHORT_RANKS, SHORTER_SUITS, get_card_list
from neuropoker.game.game import Game, PlayerStats, default_player_stats, merge
from neuropoker.game.gym import make_env
from neuropoker.players.base import BasePlayer
from neuropoker.players.naive import CallPlayer
from neuropoker.players.ppo import PPOPlayer
from neuropoker.players.utils import load_ppo_player

DEFAULT_MODEL_FILE: Final[Path] = Path("models/3p_3s/sb6")
DEFAULT_CONFIG_FILE: Final[Path] = Path("configs/3p_3s_neat.toml")
DEFAULT_NUM_ENVIRONMENTS: Final[int] = 16
DEFAULT_NUM_TIMESTEPS: Final[int] = 100000
DEFAULT_LAYERS: Final[list[int]] = [128, 128]


def get_args() -> argparse.Namespace:
    """Get command line arguments."""
    parser = argparse.ArgumentParser(description="Train a PPO agent to play poker.")

    parser.add_argument(
        "-m",
        "--model-file",
        type=Path,
        default=DEFAULT_MODEL_FILE,
        help=(
            "The path to the model file to train. This model will "
            "be trained during training and updated. "
            f"(default: {DEFAULT_MODEL_FILE})"
        ),
    )
    parser.add_argument(
        "-o",
        "--opponent-model-file",
        type=Path,
        default=DEFAULT_MODEL_FILE,
        help=(
            "The path to the opponent model file to train and play "
            "against. This model will be held static during training "
            f"and evaluation. (default: {DEFAULT_MODEL_FILE})"
        ),
    )
    parser.add_argument(
        "-e",
        "--num-environments",
        type=int,
        default=DEFAULT_NUM_ENVIRONMENTS,
        help=(
            "The number of environments to run in parallel. "
            f"(default: {DEFAULT_NUM_ENVIRONMENTS})"
        ),
    )
    parser.add_argument(
        "-t",
        "--num-timesteps",
        type=int,
        default=DEFAULT_NUM_TIMESTEPS,
        help=(
            "The number of timesteps to train the model for between "
            f"evaluations. (default: {DEFAULT_NUM_TIMESTEPS})"
        ),
    )
    parser.add_argument(
        "-l",
        "--layers",
        type=int,
        nargs="+",
        default=DEFAULT_LAYERS,
        help=(
            "The number of units in each layer of the fully-connected "
            "neural network used by the PPO agent. "
            f"(default: {DEFAULT_LAYERS})"
        ),
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default="auto",
        help="The device to train on (cpu, cuda, mps). (default: auto)",
    )

    return parser.parse_args()


def load_trainee_player(
    env: gymnasium.Env | VecEnv,
    layers: List[int],
    model_path: str | Path | None = None,
    device: str = "auto",
) -> PPOPlayer:
    """Load the model to be trained.

    Parameters:
        env: gymnasium.Env | VecEnv
            The environment to train in.
        layers: List[int]
            The number of nodes in each layer of the neural network.
        model_path: str | Path | None
            The path to the model file, if not training from scratch.
        device: str
            The device to train on.

    Returns:
        player: BasePlayer
            The player to be trained
    """

    model: Final[PPO] = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        ent_coef=0.01,
        vf_coef=0.7,
        n_steps=512,
        learning_rate=0.0001,
        policy_kwargs={"net_arch": layers},
        device=device,
    )

    # If we are loading from an existing model file, load the old model
    # and copy the policy weights to the new model
    if model_path is not None and Path(model_path).with_suffix(".zip").exists():
        print(
            colored("[load_trainee_player]", color="blue")
            + f" Starting training from old model at {model_path}"
        )

        old_model: Final[PPO] = PPO.load(model_path, env=env)
        model.policy.load_state_dict(old_model.policy.state_dict(), strict=True)
    else:
        print(
            colored("[load_trainee_player]", color="blue")
            + " Starting training from scratch"
        )

    return PPOPlayer(model, "me")


def load_opponent_players(
    model_file: str | Path | None = None, num_opponents: int = 2
) -> List[PPOPlayer | CallPlayer]:
    """Load opponents.

    Parameters:
        model_file: str | Path | None
            The path to the model file, if not comparing against a Call baseline.
        num_opponents: int
            The number of opponents to load.

    Returns:
        opponent_players: List[POPlayer | CallPlayer]
            The list of opponents.
    """
    opponent_players: List[PPOPlayer | CallPlayer] = []

    for i in range(num_opponents):
        opponent_player: PPOPlayer | CallPlayer = load_ppo_player(
            model_file, f"opponent{i}"
        )
        opponent_players.append(opponent_player)

    return opponent_players


def main() -> None:
    """Run the script."""
    args: Final[argparse.Namespace] = get_args()

    model_path: Final[Path] = args.model_file
    opponent_path: Final[Path] = args.opponent_model_file
    num_environments: Final[int] = args.num_environments
    num_timesteps: Final[int] = args.num_timesteps
    layers: Final[list[int]] = args.layers
    device: Final[str] = get_device() if args.device == "auto" else args.device

    print(
        colored("-------------- gym_env ---------------", color="blue", attrs=["bold"])
    )
    print(colored(f'{"Model file":<16}: ', color="blue") + f"{model_path}")
    print(colored(f'{"Opponent file":<16}: ', color="blue") + f"{opponent_path}")
    print(colored(f'{"Environments":<16}: ', color="blue") + f"{num_environments}")
    print(colored(f'{"Timesteps":<16}: ', color="blue") + f"{num_timesteps}")
    print(colored(f'{"Layers":<16}: ', color="blue") + f"{layers}")
    print(colored(f'{"Device":<16}: ', color="blue") + f"{device}")
    print()

    env = SubprocVecEnv(
        [
            make_env(
                model_path=model_path,
                reset_threshold=30000,
                suits=SHORTER_SUITS,
                ranks=SHORT_RANKS,
            )
            for _ in range(num_environments)
        ]
    )

    #
    # Load the trainee player
    #
    trainee_player: Final[PPOPlayer] = load_trainee_player(
        env, layers, model_path=model_path, device=device
    )
    print(trainee_player)

    #
    # Load the opponent players
    #
    opponent_players: Final[List[PPOPlayer | CallPlayer]] = load_opponent_players(
        opponent_path, 2
    )
    print(opponent_players)

    #
    # Train the trainee player
    #
    num_epochs: int = 0
    while True:
        num_epochs += 1

        #
        # Train the model
        #
        trainee_player.model.learn(
            total_timesteps=num_timesteps, reset_num_timesteps=False
        )

        #
        # Save the model
        #
        print(
            colored(f"[epoch {num_epochs:>4}]", color="blue")
            + f" Saving model to {model_path}..."
        )
        trainee_player.model.save(model_path)

        #
        # Evaluate the model
        #
        print(colored(f"[epoch {num_epochs:>4}]", "blue") + " Evaluating model...")
        p1: PPOPlayer | CallPlayer = load_ppo_player(model_path, "me")
        if not isinstance(p1, PPOPlayer):
            raise ValueError("Player 1 is not a PPOPlayer")

        players: List[BasePlayer] = [p1, *opponent_players]
        overall_performance: PlayerStats = default_player_stats()
        overall_performance["uuid"] = "me"

        # Try each position
        for i in range(0, 3):
            players_: List[BasePlayer] = players[i:] + players[:i]

            game = Game(players_, get_card_list(SHORTER_SUITS, SHORT_RANKS))
            performances: Dict[str, PlayerStats] = game.play_multiple(
                num_games=2000, seed=-1
            )
            overall_performance = merge(overall_performance, performances["me"])

        #
        # Print evaluation results
        #
        average_winnings: float = (
            overall_performance["winnings"] / overall_performance["num_games"]
        )
        print(
            colored(f"[epoch {num_epochs:>4}]", color="blue")
            + f" Average winnings: {average_winnings}"
        )
        print(
            colored(f"[epoch {num_epochs:>4}]", color="blue")
            + f" Overall performance: {overall_performance}"
        )


if __name__ == "__main__":
    main()

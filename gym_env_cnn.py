#!/usr/bin/env python3

"""Train a PPO agent with a CNN to play poker."""

import argparse
from pathlib import Path
from typing import Any, Dict, Final, List

import gymnasium
from stable_baselines3.common.vec_env import SubprocVecEnv, VecEnv
from stable_baselines3.ppo.ppo import PPO
from termcolor import colored

from neuropoker.extra.torch import get_device
from neuropoker.game.cards import SHORT_RANKS, SHORTER_SUITS, get_card_list
from neuropoker.game.features import (
    CNNFeaturesCollector,
    FeaturesCollector,
)
from neuropoker.game.game import Game, PlayerStats, default_player_stats, merge
from neuropoker.game.gym import PokerCNNExtractor, make_env
from neuropoker.players.base import BasePlayer
from neuropoker.players.naive import CallPlayer
from neuropoker.players.ppo import PPOPlayer
from neuropoker.players.utils import load_ppo_player

DEFAULT_STARTING_MODEL_PATH: Final[Path] = Path("models/3p_3s/sb_cnn")
DEFAULT_OPPONENT_MODEL_PATH: Final[Path] = Path("models/3p_3s/sb_cnn")
DEFAULT_MODEL_DIR: Final[Path] = Path("models/3p_3s/sb_cnn/")

DEFAULT_NUM_ENVIRONMENTS: Final[int] = 8
DEFAULT_NUM_TIMESTEPS: Final[int] = 100000
DEFAULT_NUM_EPOCHS: Final[int] = 100

POLICY_KWARGS: Final[Dict[str, Any]] = {
    "features_extractor_class": PokerCNNExtractor,
    "features_extractor_kwargs": {"features_dim": 256},
    "net_arch": [],  # Skip additional layers since the CNN handles feature extraction
}


def get_args() -> argparse.Namespace:
    """Get command line arguments."""
    parser = argparse.ArgumentParser(description="Train a PPO agent to play poker.")

    parser.add_argument(
        "-s",
        "--starting-model-path",
        type=Path,
        default=DEFAULT_STARTING_MODEL_PATH,
        help=(
            "The path to the starting model to train from. "
            f"(default: {DEFAULT_STARTING_MODEL_PATH})"
        ),
    )
    parser.add_argument(
        "-o",
        "--opponent-model-path",
        type=Path,
        default=DEFAULT_OPPONENT_MODEL_PATH,
        help=(
            "The path to the opponent model file to train and play "
            "against. This model will be held static during training "
            f"and evaluation. (default: {DEFAULT_OPPONENT_MODEL_PATH})"
        ),
    )
    parser.add_argument(
        "-m",
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help=(
            "The path to save trained models to at the end of "
            f"each epoch. (default: {DEFAULT_MODEL_DIR})"
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
        "--num-epochs",
        type=int,
        default=DEFAULT_NUM_EPOCHS,
        help=f"The maximum number of epochs to train for. (default: {DEFAULT_NUM_EPOCHS})",
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
    model_path: str | Path | None = None,
    feature_collector: FeaturesCollector | None = None,
    device: str = "auto",
    verbose: bool = False,
) -> PPOPlayer:
    """Load the model to be trained.

    Parameters:
        env: gymnasium.Env | VecEnv
            The environment to train in.
        model_path: str | Path | None
            The path to the model file, if not training from scratch.
        feature_collector: FeaturesCollector | None
            The feature collector to use.
        device: str
            The device to train on.
        verbose: bool
            Whether to print verbose output.

    Returns:
        player: BasePlayer
            The player to be trained
    """

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

    # If we are loading from an existing model file, load the old model
    # and copy the policy weights to the new model
    if model_path is not None and Path(model_path).with_suffix(".zip").exists():
        if verbose:
            print(
                colored("[load_trainee_player]", color="blue")
                + f" Starting training from old model at {model_path}"
            )

        old_model: Final[PPO] = PPO.load(model_path, env=env)
        model.policy.load_state_dict(old_model.policy.state_dict(), strict=True)
    else:
        if verbose:
            print(
                colored("[load_trainee_player]", color="blue")
                + " Starting training from scratch"
            )

    return PPOPlayer(model, "me", feature_collector=feature_collector)


def load_opponent_players(
    model_file: str | Path | None = None,
    num_opponents: int = 2,
    feature_collector: FeaturesCollector | None = None,
    verbose: bool = False,
) -> List[PPOPlayer | CallPlayer]:
    """Load opponents.

    Parameters:
        model_file: str | Path | None
            The path to the model file, if not comparing against a Call baseline.
        num_opponents: int
            The number of opponents to load.
        feature_collector: FeaturesCollector | None
            The feature collector to use.
        verbose: bool
            Whether to print verbose output.

    Returns:
        opponent_players: List[PPOPlayer | CallPlayer]
            The list of opponents.
    """
    opponent_players: List[PPOPlayer | CallPlayer] = []

    for i in range(num_opponents):
        opponent_player: PPOPlayer | CallPlayer = load_ppo_player(
            model_file,
            f"opponent{i}",
            feature_collector=feature_collector,
            verbose=verbose,
        )
        opponent_players.append(opponent_player)

    return opponent_players


def main() -> None:
    """Run the script."""
    #
    # Process arguments
    #
    args: Final[argparse.Namespace] = get_args()

    starting_model_path: Final[Path] = args.starting_model_path
    opponent_model_path: Final[Path] = args.opponent_model_path
    model_dir: Final[Path] = args.model_dir
    num_environments: Final[int] = args.num_environments
    num_timesteps: Final[int] = args.num_timesteps
    num_epochs: Final[int] = args.num_epochs
    device: Final[str] = get_device() if args.device == "auto" else args.device

    #
    # Print arguments
    #
    print(
        colored("------------ gym_env_cnn -------------", color="blue", attrs=["bold"])
    )
    print(colored("Models", color="blue", attrs=["bold"]))
    print(
        "    "
        + colored(f'{"Starting model":<14}:', color="blue")
        + f" {starting_model_path}"
    )
    print(
        "    "
        + colored(f'{"Opponent model":<14}:', color="blue")
        + f" {opponent_model_path}"
    )
    print("    " + colored(f'{"Output models":<14}:', color="blue") + f" {model_dir}")

    print(colored("Training", color="blue", attrs=["bold"]))
    print(
        "    "
        + colored(f'{"Environments":<14}: ', color="blue")
        + f"{num_environments}"
    )
    print("    " + colored(f'{"Timesteps":<14}: ', color="blue") + f"{num_timesteps}")
    print("    " + colored(f'{"Epochs":<14}: ', color="blue") + f"{num_epochs}")
    print("    " + colored(f'{"Device":<14}: ', color="blue") + f"{device}")
    print()

    #
    # Set up the environments
    #
    print(
        colored(
            f"Setting up {num_environments} environments...",
            color="blue",
            attrs=["bold"],
        )
    )

    feature_collector: Final[FeaturesCollector] = CNNFeaturesCollector()
    env: Final[VecEnv] = SubprocVecEnv(
        [
            make_env(
                starting_model_path=starting_model_path,
                opponent_model_path=opponent_model_path,
                feature_collector=feature_collector,
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
        env,
        model_path=starting_model_path,
        feature_collector=feature_collector,
        device=device,
    )

    #
    # Load the opponent players
    #
    opponent_players: Final[List[PPOPlayer | CallPlayer]] = load_opponent_players(
        opponent_model_path,
        2,
        feature_collector=feature_collector,
    )

    #
    # Print the players
    #
    print()
    print(colored("Players", color="blue", attrs=["bold"]))
    print(
        "   "
        + colored(f"{trainee_player.uuid:<14}: ", color="blue")
        + f" {trainee_player}"
        + f" {colored('(trainee)', color='blue')}"
    )
    for opponent_i, opponent_player in enumerate(opponent_players):
        print(
            "   "
            + colored(f"{opponent_player.uuid:<14}: ", color="blue")
            + f" {opponent_player}"
            + f" {colored('(opponent ' + str(opponent_i + 1) + ')', color='blue')}"
        )
    print()

    #
    # Train the trainee player
    #
    for epoch in range(num_epochs):
        print(
            colored(
                f"--------- epoch {epoch:>4} / {num_epochs:>4} ----------",
                color="blue",
                attrs=["bold"],
            )
        )

        #
        # Train the model
        #
        trainee_player.model.learn(
            total_timesteps=num_timesteps, reset_num_timesteps=False
        )

        #
        # Save the model
        #
        epoch_model_path: Path = model_dir / f"epoch_{epoch}"
        model_dir.mkdir(exist_ok=True, parents=True)

        print(
            colored(f"[epoch {epoch:>4}]", color="blue")
            + f" Saving model to {epoch_model_path}..."
        )
        trainee_player.model.save(epoch_model_path)

        #
        # Evaluate the model
        #
        print(colored(f"[epoch {num_epochs:>4}]", "blue") + " Evaluating model...")
        p1: PPOPlayer | CallPlayer = load_ppo_player(
            epoch_model_path, "me", feature_collector=feature_collector
        )
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

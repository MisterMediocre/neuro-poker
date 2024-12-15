#!/usr/bin/env python3

"""Train a PPO agent with a CNN to play poker."""

import argparse
from pathlib import Path
from typing import Any, Dict, Final

from neuropoker.extra.torch import get_device
from neuropoker.game.features import (
    CNNFeaturesCollector,
)
from neuropoker.game.gym import PokerCNNExtractor
from neuropoker.ppo_bench import PPOBench

# DEFAULT_STARTING_MODEL_PATH: Final[Path] = Path("models/3p_3s/sb_cnn")
# DEFAULT_OPPONENT_MODEL_PATH: Final[Path] = Path("models/3p_3s/sb_cnn")
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
        default=None,
        help=(
            "The path to the starting model to train from. "
            "(default: None, train from scratch)"
        ),
    )
    # parser.add_argument(
    #     "-o",
    #     "--opponent-model-path",
    #     type=Path,
    #     default=DEFAULT_OPPONENT_MODEL_PATH,
    #     help=(
    #         "The path to the opponent model file to train and play "
    #         "against. This model will be held static during training "
    #         f"and evaluation. (default: {DEFAULT_OPPONENT_MODEL_PATH})"
    #     ),
    # )
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


def main() -> None:
    """Run the script."""
    #
    # Process arguments
    #
    args: Final[argparse.Namespace] = get_args()

    starting_model_path: Final[Path] = args.starting_model_path
    # opponent_model_path: Final[Path] = args.opponent_model_path
    model_dir: Final[Path] = args.model_dir
    num_environments: Final[int] = args.num_environments
    num_timesteps: Final[int] = args.num_timesteps
    num_epochs: Final[int] = args.num_epochs
    device: Final[str] = get_device() if args.device == "auto" else args.device

    # #
    # # Print arguments
    # #
    # print(
    #     colored("------------ gym_env_cnn -------------", color="blue", attrs=["bold"])
    # )
    # print(colored("Models", color="blue", attrs=["bold"]))
    # print(
    #     "    "
    #     + colored(f'{"Starting model":<16}:', color="blue")
    #     + f" {starting_model_path}"
    # )
    # # print(
    # #     "    "
    # #     + colored(f'{"Opponent model":<14}:', color="blue")
    # #     + f" {opponent_model_path}"
    # # )
    # print("    " + colored(f'{"Output models":<16}:', color="blue") + f" {model_dir}")

    # print(colored("Training", color="blue", attrs=["bold"]))
    # print(
    #     "    "
    #     + colored(f'{"Environments":<16}: ', color="blue")
    #     + f"{num_environments}"
    # )
    # print("    " + colored(f'{"Timesteps":<16}: ', color="blue") + f"{num_timesteps}")
    # print("    " + colored(f'{"Epochs":<16}: ', color="blue") + f"{num_epochs}")
    # print("    " + colored(f'{"Device":<16}: ', color="blue") + f"{device}")
    # print()

    bench: Final[PPOBench] = PPOBench(
        model_dir,
        starting_model_path=starting_model_path,
        features_collector=CNNFeaturesCollector(),
        model_kwargs={
            "verbose": 1,
            "ent_coef": 0.01,
            "vf_coef": 0.7,
            "n_steps": 256,
            # "learning_rate": 0.003,
        },
        policy_kwargs=POLICY_KWARGS,
        num_environments=num_environments,
        device=device,
    )

    bench.train(num_epochs=num_epochs, num_timesteps=num_timesteps)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

"""Train a PPO agent to play poker."""

import argparse
from pathlib import Path
from typing import Final

from neuropoker.extra.torch import get_device
from neuropoker.game.features import LinearFeaturesCollector
from neuropoker.ppo_bench import PPOBench

# DEFAULT_STARTING_MODEL_PATH: Final[Path] = Path("models/3p_3s/sb6")
# DEFAULT_OPPONENT_MODEL_PATH: Final[Path] = Path("models/3p_3s/sb6")
DEFAULT_MODEL_DIR: Final[Path] = Path("models/3p_3s/sb/")

DEFAULT_NUM_ENVIRONMENTS: Final[int] = 8
DEFAULT_NUM_TIMESTEPS: Final[int] = 100000
DEFAULT_NUM_EPOCHS: Final[int] = 100
DEFAULT_LAYERS: Final[list[int]] = [128, 128]


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
        "-n",
        "--num-epochs",
        type=int,
        default=DEFAULT_NUM_EPOCHS,
        help=f"The maximum number of epochs to train for. (default: {DEFAULT_NUM_EPOCHS})",
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
    layers: Final[list[int]] = args.layers
    device: Final[str] = get_device() if args.device == "auto" else args.device

    # #
    # # Print arguments
    # #
    # print(
    #     colored("-------------- gym_env ---------------", color="blue", attrs=["bold"])
    # )
    # print(colored("Models", color="blue", attrs=["bold"]))
    # print(
    #     "    "
    #     + colored(f'{"Starting model":<14}:', color="blue")
    #     + f" {starting_model_path}"
    # )
    # # print(
    # #     "    "
    # #     + colored(f'{"Opponent model":<14}:', color="blue")
    # #     + f" {opponent_model_path}"
    # # )
    # print("    " + colored(f'{"Output models":<14}:', color="blue") + f" {model_dir}")
    # print("    " + colored(f'{"Layers":<14}: ', color="blue") + f"{layers}")

    # print(colored("Training", color="blue", attrs=["bold"]))
    # print(
    #     "    "
    #     + colored(f'{"Environments":<14}: ', color="blue")
    #     + f"{num_environments}"
    # )
    # print("    " + colored(f'{"Timesteps":<14}: ', color="blue") + f"{num_timesteps}")
    # print("    " + colored(f'{"Epochs":<14}: ', color="blue") + f"{num_epochs}")
    # print("    " + colored(f'{"Device":<14}: ', color="blue") + f"{device}")
    # print()

    bench: Final[PPOBench] = PPOBench(
        model_dir,
        starting_model_path=starting_model_path,
        features_collector=LinearFeaturesCollector(),
        model_kwargs={
            "verbose": 1,
            "ent_coef": 0.01,
            "vf_coef": 0.7,
            "n_steps": 512,
            "learning_rate": 0.0001,
        },
        policy_kwargs={
            "net_arch": layers,
        },
        num_environments=num_environments,
        device=device,
    )

    bench.train(num_epochs=num_epochs, num_timesteps=num_timesteps)


if __name__ == "__main__":
    main()

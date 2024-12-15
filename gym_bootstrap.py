#!/usr/bin/env python3

"""Bootstrap the training of a PPO agent by iteratively improving
the opponent it plays against."""

import argparse
import subprocess
from pathlib import Path
from typing import Final, List

from termcolor import colored

# Parallel environments
DEFAULT_NUM_ENVIRONMENTS: Final[int] = 8

# Timesteps per epoch
DEFAULT_NUM_TIMESTEPS: Final[int] = 100000

# Epochs per iteration
DEFAULT_NUM_EPOCHS: Final[int] = 100
DEFAULT_NUM_ITERATIONS: Final[int] = 6

DEFAULT_MODEL_DIR: Final[Path] = Path("models/3p_3s/sb_bootstrap/")


def get_args() -> argparse.Namespace:
    """Get command line arguments."""

    parser = argparse.ArgumentParser(
        "Bootstrap the training of a PPO agent by iteratively improving "
        "the opponent it plays against."
    )

    parser.add_argument(
        "-e",
        "--num-environments",
        type=int,
        default=DEFAULT_NUM_ENVIRONMENTS,
        help=(
            "The number of parallel environments to run. "
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
        "-i",
        "--num-iterations",
        type=int,
        default=DEFAULT_NUM_ITERATIONS,
        help=(
            "The number of bootstrap iterations to run. "
            f"(default: {DEFAULT_NUM_ITERATIONS})"
        ),
    )
    parser.add_argument(
        "-m",
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help=(
            "The directory to save each iteration's models to. "
            f"(default: {DEFAULT_MODEL_DIR})"
        ),
    )

    # TODO: Other args

    return parser.parse_args()


def get_iteration_dir(model_dir: Path, iteration: int) -> Path:
    """Get the directory for a specific iteration.

    Parameters:
        model_dir: Path
            The prefix for directories to save each iteration's models to.
        iteration: int
            The iteration number.

    Returns:
        iteration_dir: Path
            The directory for the specific iteration.
    """
    return model_dir / f"iteration_{iteration}"


def get_iteration_last_model_path(model_dir: Path, iteration: int) -> Path | None:
    """Get the last model generated for a specific iteration.

    Parameters:
        model_dir: Path
            The prefix for directories to save each iteration's models to.
        iteration: int
            The iteration number.

    Returns:
        model_path: Path | None
            The path to the last model generated for the specific iteration.
    """
    iteration_dir: Final[Path] = get_iteration_dir(model_dir, iteration)

    iteration_model_paths: Final[List[Path]] = sorted(
        iteration_dir.glob("epoch_*"), reverse=True
    )

    if len(iteration_model_paths) == 0:
        return None

    return iteration_model_paths[0]


def is_iteration_finished(model_dir: Path, iteration: int, num_epochs: int) -> bool:
    """Determine if an iteration is complete.

    Parameters:
        model_dir: Path
            The prefix for directories to save each iteration's models to.
        iteration: int
            The current iteration.
        num_epochs: int
            The maximum number of epochs to train for.

    Returns:
        exists: bool
            True if the last model for the iteration exists, False otherwise.
    """
    iteration_model_dir: Final[Path] = get_iteration_dir(model_dir, iteration)
    iteration_last_model_path: Final[Path] = (
        iteration_model_dir / f"epoch_{num_epochs}.zip"
    )
    return iteration_last_model_path.exists()


def main():
    """Run the script."""
    #
    # Process arguments
    #
    args: Final[argparse.Namespace] = get_args()
    model_dir: Final[Path] = args.model_dir
    num_environments: Final[int] = args.num_environments
    num_timesteps: Final[int] = args.num_timesteps
    num_epochs: Final[int] = args.num_epochs
    num_iterations: Final[int] = args.num_iterations

    #
    # Print arguments
    #
    print(
        colored(
            "------------ gym_bootstrap -------------", color="green", attrs=["bold"]
        )
    )
    print(colored("Models", color="green", attrs=["bold"]))
    print(
        "    "
        + colored(f'{"Model dirs":<14}:', color="green")
        + f" {model_dir}/iteration_<iteration>"
    )

    print(colored("Training", color="green", attrs=["bold"]))
    print(
        "    "
        + colored(f'{"Environments":<14}: ', color="green")
        + f"{num_environments}"
    )
    print("    " + colored(f'{"Timesteps":<14}: ', color="green") + f"{num_timesteps}")
    print("    " + colored(f'{"Epochs":<14}: ', color="green") + f"{num_epochs}")
    print(
        "    " + colored(f'{"Iterations":<14}: ', color="green") + f"{num_iterations}"
    )

    for iteration in range(1, num_iterations + 1):
        print()
        print()
        print(
            colored(
                f"====\n================ Iteration {iteration} / {num_iterations} ================\n====",
                color="green",
                attrs=["bold"],
            )
        )
        print()

        if is_iteration_finished(model_dir, iteration, num_epochs):
            print(
                colored(
                    f"Iteration {iteration} already finished, skipping",
                    color="yellow",
                    attrs=["bold"],
                )
            )
            print()
            continue

        command: List[str] = ["./gym_env.py"]

        starting_model_path: Path = (
            get_iteration_dir(model_dir, iteration - 1) / f"epoch_{num_epochs}.zip"
        )

        if starting_model_path.exists():
            command += ["-s", str(starting_model_path)]

        iteration_dir: Path = get_iteration_dir(model_dir, iteration)
        command += ["-m", str(iteration_dir)]

        command += ["-e", str(num_environments)]
        command += ["-t", str(num_timesteps)]
        command += ["-n", str(num_epochs)]

        print(
            colored("Running command:", color="green", attrs=["bold"])
            + f" {' '.join(command)}"
        )
        print()

        subprocess.run(
            command,
            check=True,
        )


if __name__ == "__main__":
    main()
